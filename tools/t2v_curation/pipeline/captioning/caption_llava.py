import argparse
import os
import sys

import numpy as np
import pandas as pd
from pipeline.captioning.utils import set_model_param_dtype
from pipeline.datasets.utils import extract_frames, is_video, pil_loader
from pipeline.scoring.utils import NUM_FRAMES_POINTS, merge_scores
from tqdm import tqdm

import mindspore as ms
import mindspore.dataset as ds
import mindspore.mint as mint
from mindspore.mint.distributed import all_gather, all_gather_object, get_rank, get_world_size, init_process_group

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../.."))
sys.path.insert(0, mindone_lib_path)

from transformers import AutoTokenizer, CLIPImageProcessor  # noqa: E402

from mindone.transformers import LlavaConfig, LlavaForConditionalGeneration  # noqa: E402


class VideoTextDataset:
    def __init__(self, meta_path, num_frames=1):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.points = NUM_FRAMES_POINTS[num_frames]  # llava takes 1 frame ONLY

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample["path"]

        if not is_video(path):
            images = [pil_loader(path)]
        else:
            num_frames = sample["num_frames"] if "num_frames" in sample else None
            images = extract_frames(path, points=self.points, backend="decord", num_frames=num_frames)

        return index, images

    def __len__(self):
        return len(self.meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "--llava_model_path",
        type=str,
        default="pretrained_models/llava-llama-3-8b-v1_1-transformers",
        help="Path or identifier for the Llava model",
    )
    parser.add_argument(
        "--question", type=str, default="Describe the video in detail.", help="Captioning prompt question"
    )
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
    out_path = f"{wo_ext}_caption_llava{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", pynative_synchronize=True)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init_process_group()

    rank_id = get_rank()
    rank_size = get_world_size()

    print("Loading LlavaForConditionalGeneration Model")
    config = LlavaConfig.from_pretrained(args.llava_model_path)
    config.text_config._attn_implementation = "flash_attention_2"
    model = LlavaForConditionalGeneration.from_pretrained(args.llava_model_path, text_config=config.text_config)
    model.set_train(False)
    set_model_param_dtype(model, ms.float16)

    tokenizer = AutoTokenizer.from_pretrained(args.llava_model_path, padding_side="left")
    image_processor = CLIPImageProcessor.from_pretrained(args.llava_model_path)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    raw_dataset = VideoTextDataset(meta_path)
    dataset = ds.GeneratorDataset(
        source=raw_dataset, column_names=["index", "images"], shuffle=False, num_shards=rank_size, shard_id=rank_id
    )

    dataset = dataset.batch(1, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)

    indices_list = []
    caption_list = []

    for batch in tqdm(iterator):
        idx = batch["index"][0]
        images = batch["images"][0]

        prompt = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "<image>\n" + args.question + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        try:
            inputs = tokenizer(
                [prompt],
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="np",
            )
            inputs["input_ids"] = ms.Tensor(inputs["input_ids"], dtype=ms.int32)
            inputs["attention_mask"] = ms.Tensor(inputs["attention_mask"], dtype=ms.bool_)

            # Process the list of images using the image processor.
            img_inputs = image_processor(images=images, return_tensors="np")
            inputs["pixel_values"] = ms.Tensor(img_inputs["pixel_values"], dtype=ms.float16)

            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, use_cache=False)
            output_text = tokenizer.decode(generated_ids[0][2:], skip_special_tokens=True)
        except Exception as e:
            print(f"Error processing video at index {idx}: {e}")
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
