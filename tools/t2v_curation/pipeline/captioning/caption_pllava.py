import argparse
import os
import sys

import numpy as np
import pandas as pd
from pipeline.scoring.utils import merge_scores
from tqdm import tqdm

import mindspore as ms
import mindspore.dataset as ds
import mindspore.mint as mint
from mindspore.mint.distributed import all_gather, all_gather_object, get_rank, get_world_size, init_process_group

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../.."))
sys.path.insert(0, mindone_lib_path)

from tools.captioners.PLLaVA.tasks.eval.eval_utils import load_video  # noqa: E402
from tools.captioners.PLLaVA.tasks.eval.model_utils import load_pllava, pllava_answer  # noqa: E402


class VideoTextDataset:
    def __init__(self, meta_path):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample["path"]
        return path, index

    def __len__(self):
        return len(self.meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="pretrained_models/pllava-7b",
        help="Path or name of the pretrained PLLaVA model",
    )
    parser.add_argument("--question", type=str, default="Describe the video in detail.")
    parser.add_argument("--num_frames", type=int, default=4, help="Number of frames to sample from video")
    parser.add_argument("--bs", type=int, default=1, help="Batch size")
    parser.add_argument("--skip_if_existing", action="store_true", help="Skip processing if output CSV already exists.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_caption_pllava{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init_process_group()

    rank_id = get_rank()
    rank_size = get_world_size()

    print("Loading PLLaVA model...")
    model, processor = load_pllava(
        args.pretrained_model_name_or_path, args.num_frames, pooling_shape=(args.num_frames, 12, 12)
    )
    model.set_train(False)

    raw_dataset = VideoTextDataset(meta_path)
    dataset = ds.GeneratorDataset(
        source=raw_dataset, column_names=["video_path", "index"], shuffle=False, num_shards=rank_size, shard_id=rank_id
    )
    dataset = dataset.batch(args.bs, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)

    indices_list = []
    caption_list = []

    SYSTEM = (
        "You are a powerful Video Magic ChatBot, a large vision-language assistant.\n"
        "You are able to understand the video content that the user provides and assist the user in a video-language related task.\n"
        "The user might provide you with the video and maybe some extra noisy information to help you out or ask you a question.\n"
        "Make use of the information in a proper way to be competent for the job.\n"
        "### INSTRUCTIONS:\n"
        "1. Follow the user's instruction.\n"
        "2. Be critical yet believe in yourself.\n"
    )

    prompt = SYSTEM + "USER: " + args.question + " </s> USER:<image> ASSISTANT:"

    for batch in tqdm(iterator):
        video_paths = batch["video_path"]
        indices = batch["index"]

        for video_path, idx in zip(video_paths, indices):
            try:
                frames = load_video(video_path, args.num_frames)

                _, output_text = pllava_answer(
                    model,
                    processor,
                    [frames],
                    prompt,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    temperature=1.0,
                )

                cleaned_output = output_text.split("ASSISTANT: ", 1)[1]
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                cleaned_output = ""

            caption_list.append(cleaned_output)
            indices_list.append(idx)

    if rank_size > 1:
        indices_tensor = ms.Tensor(indices_list, dtype=ms.int64)
        indices_all = [ms.Tensor(np.zeros(indices_tensor.shape, dtype=np.int64)) for _ in range(rank_size)]
        all_gather(indices_all, indices_tensor)
        indices_list_all = mint.concat(indices_all, dim=0).asnumpy().tolist()

        captions_all = [None] * rank_size
        all_gather_object(captions_all, caption_list)
        caption_list_all = sum(captions_all, [])

        if rank_id == 0:
            meta_local = merge_scores([(indices_list_all, caption_list_all)], raw_dataset.meta, column="text")
    elif rank_size == 1:
        meta_local = raw_dataset.meta.copy()
        meta_local["text"] = caption_list

    if rank_id == 0:
        meta_local.to_csv(out_path, index=False)
        print(meta_local.head())
        print(f"New meta with captions saved to '{out_path}'.")


if __name__ == "__main__":
    main()
