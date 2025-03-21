import argparse
import os
import sys

import numpy as np
import pandas as pd
from pipeline.scoring.utils import merge_scores
from tqdm import tqdm
from transformers import AutoProcessor

import mindspore as ms
import mindspore.dataset as ds
import mindspore.mint as mint
from mindspore.mint.distributed import all_gather, all_gather_object, get_rank, get_world_size, init_process_group

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../.."))
sys.path.insert(0, mindone_lib_path)

from mindone.transformers import Qwen2VLForConditionalGeneration  # noqa: E402
from mindone.transformers.models.qwen2_vl.qwen_vl_utils import process_vision_info  # noqa: E402


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
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="pretrained_models/Qwen2-VL-7B-Instruct")
    parser.add_argument("--question", type=str, default="Describe the video in detail.")
    parser.add_argument("--height", type=int, default=448, help="resized video height")
    parser.add_argument("--width", type=int, default=672, help="resized video width")
    parser.add_argument("--fps", type=int, default=4, help="fps to sample from video")
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
    out_path = f"{wo_ext}_caption_qwen2vl{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", pynative_synchronize=True)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init_process_group()

    rank_id = get_rank()
    rank_size = get_world_size()

    print("Loading Qwen2VLForConditionalGeneration Model")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path, mindspore_dtype=ms.float16, attn_implementation="flash_attention_2"
    ).set_train(False)
    print("Loading AutoProcessor")
    processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path)

    raw_dataset = VideoTextDataset(meta_path)
    dataset = ds.GeneratorDataset(
        source=raw_dataset, column_names=["video_path", "index"], shuffle=False, num_shards=rank_size, shard_id=rank_id
    )
    dataset = dataset.batch(args.bs, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)

    indices_list = []
    caption_list = []

    for batch in tqdm(iterator):
        video_paths = batch["video_path"]
        indices = batch["index"]

        for video_path, idx in zip(video_paths, indices):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": args.height * args.width,
                            "fps": float(args.fps),
                        },
                        {"type": "text", "text": args.question},
                    ],
                }
            ]

            try:
                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="np",
                )

                for key, value in inputs.items():
                    if isinstance(value, np.ndarray):
                        inputs[key] = ms.Tensor(value)
                    elif isinstance(value, list):
                        inputs[key] = ms.Tensor(value)
                    if inputs[key].dtype == ms.int64:
                        inputs[key] = inputs[key].to(ms.int32)

                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                output_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                output_text = ""

            caption_list.append(output_text)
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
