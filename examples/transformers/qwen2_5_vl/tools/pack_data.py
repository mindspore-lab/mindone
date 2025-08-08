import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import binpacking
import numpy as np
from decord import VideoReader
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from mindone.transformers import Qwen2VLImageProcessor


def read_data(file_path):
    """Read JSON or JSONL file"""
    if file_path.endswith((".json", ".jsonl")):
        with open(file_path, "r") as f:
            if file_path.endswith(".json"):
                return json.load(f)
            return [json.loads(line) for line in f]
    raise ValueError("Please provide a .json or .jsonl file")


def write_data(file_path, data):
    """Write data to JSON or JSONL file"""
    with open(file_path, "w", encoding="utf8") as f:
        if file_path.endswith(".json"):
            json.dump(data, f, indent=4, ensure_ascii=False)
        elif file_path.endswith(".jsonl"):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


class DataArguments:
    def __init__(self):
        self.max_pixels = 2048 * 28 * 28
        self.min_pixels = 32 * 28 * 28
        self.video_max_frame_pixels = 576 * 28 * 28
        self.video_min_frame_pixels = 144 * 28 * 28
        self.base_interval = 4
        self.video_min_frames = 4
        self.video_max_frames = 8
        self.data_path = ""


class MultimodalProcessor:
    def __init__(self, data_args, base_processor, device="cpu"):
        self.data_args = data_args
        self.base_processor = base_processor
        self.device = device

    def _configure_processor(self, max_val, min_val):
        processor = deepcopy(self.base_processor)
        processor.max_pixels = max_val
        processor.min_pixels = min_val
        processor.size = {"longest_edge": max_val, "shortest_edge": min_val}
        return processor

    def process_image(self, image_file):
        image_path = os.path.join(self.data_args.data_path, image_file)
        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            return 0
        processor = self._configure_processor(self.data_args.max_pixels, self.data_args.min_pixels)
        image = Image.open(image_path).convert("RGB")
        visual_processed = processor.preprocess(images=image, return_tensors="np")
        return visual_processed["image_grid_thw"].prod().item() // 4

    def process_video(self, video_file):
        video_path = os.path.join(self.data_args.data_path, video_file)
        processor = self._configure_processor(
            self.data_args.video_max_frame_pixels, self.data_args.video_min_frame_pixels
        )
        vr = VideoReader(video_path, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        visual_processed = processor.preprocess(images=None, videos=video, return_tensors="np")
        return visual_processed["video_grid_thw"].prod().item() // 4


def calculate_tokens(conversation, processor, tokenizer):
    total_tokens = 21
    roles = {"human": "user", "gpt": "assistant"}
    for message in conversation["conversations"]:
        role = message["from"]
        text = message["value"]
        conv = [{"role": roles[role], "content": text}]
        encode_id = tokenizer.apply_chat_template(conv, return_tensors="np", add_generation_prompt=False)[0]
        total_tokens += len(encode_id)
    if "image" in conversation:
        images = conversation["image"] if isinstance(conversation["image"], list) else [conversation["image"]]
        for image_file in images:
            total_tokens += processor.process_image(image_file)
    elif "video" in conversation:
        videos = conversation["video"] if isinstance(conversation["video"], list) else [conversation["video"]]
        for video_file in videos:
            total_tokens += processor.process_video(video_file)
    return total_tokens


def pack_data(data_list, pack_length):
    # Extract the length of each data item
    lengths = [data["num_tokens"] for data in data_list]
    grouped_indices = binpacking.to_constant_volume(
        list(enumerate(lengths)), pack_length, weight_pos=1  # Explicitly convert to list
    )
    packed_data = []
    for group in grouped_indices:
        group_data = []
        for index, _ in group:
            new_data = data_list[index].copy()
            new_data.pop("num_tokens", None)
            group_data.append(new_data)
        packed_data.append(group_data)
    return packed_data


def main():
    parser = argparse.ArgumentParser("Convert DocVQA to the training format.")
    parser.add_argument("-a", "--annotation_path", default="./doc_vqa.json", help="path of the annotation.")
    parser.add_argument("-m", "--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct", help="name or path of the model")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--pack_length", default=4096, type=int, help="pack length")
    args = parser.parse_args()

    random.seed(0)

    data_args = DataArguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    base_image_processor = Qwen2VLImageProcessor.from_pretrained(args.model_name)
    print(f"Successfully loaded model components from {args.model_name}")

    processor = MultimodalProcessor(data_args, base_image_processor, device="cpu")

    print(f"Annotation file path: {args.annotation_path}")
    print(f"Image configuration: max_pixels={data_args.max_pixels}, min_pixels={data_args.min_pixels}")
    print(
        f"Video frame configuration: video_max_frame_pixels={data_args.video_max_frame_pixels}, video_min_frame_pixels={data_args.video_min_frame_pixels}"
    )
    data = read_data(args.annotation_path)

    count_file_path = args.annotation_path.replace(".json", "_count.json").replace(".jsonl", "_count.json")
    if os.path.exists(count_file_path):
        print(f"Found pre - calculated token counts, loading data from {count_file_path}.")
        data_with_tokens = read_data(count_file_path)
    else:

        def calculate_and_update(item):
            item["num_tokens"] = calculate_tokens(item, processor, tokenizer)
            return item

        with concurrent.futures.ThreadPoolExecutor() as executor:
            data_with_tokens = list(
                tqdm(executor.map(calculate_and_update, data), total=len(data), desc="Processing data")
            )

        # Save the token count results
        write_data(count_file_path, data_with_tokens)
        print(f"Token counts saved to: {count_file_path}")

    all_packed_results = []

    # Record the start time of binpacking
    start_time = time.time()
    random.shuffle(data_with_tokens)
    for i in range(0, len(data_with_tokens), args.batch_size):
        batch_data = data_with_tokens[i : i + args.batch_size]
        batch_packed_result = pack_data(batch_data, args.pack_length)
        all_packed_results.extend(batch_packed_result)
    # Record the end time of binpacking
    end_time = time.time()

    # Calculate the time spent on binpacking
    binpack_time = end_time - start_time
    print(f"Time spent on binpacking: {binpack_time:.4f} seconds")

    # Save the packed results as a JSON file
    pack_output_path = args.annotation_path.replace(".json", "_pack.json").replace(".jsonl", "_pack.json")
    with open(pack_output_path, "w", encoding="utf-8") as file:
        json.dump(all_packed_results, file, indent=4, ensure_ascii=False)
    print(f"Packed results saved to: {pack_output_path}")


if __name__ == "__main__":
    main()
