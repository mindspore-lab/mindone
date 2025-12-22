import glob
import os
import random
from typing import Optional, Union

import yaml
from datasets import concatenate_datasets, load_dataset
from PIL import Image

from mindone.data import BaseDataset

from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor


class OmniGen2TrainDataset(BaseDataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."

    def __init__(
        self,
        config_path: str,
        tokenizer,
        use_chat_template: bool,
        max_input_pixels: Optional[Union[int, list[int]]] = None,
        max_output_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        img_scale_num: int = 16,
        prompt_dropout_prob: float = 0.0,
        ref_img_dropout_prob: float = 0.0,
    ):
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels

        self.max_side_length = max_side_length
        self.img_scale_num = img_scale_num
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob

        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.use_chat_template = use_chat_template
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)

        data = self._collect_annotations(self.config)

        self.data = data
        self.tokenizer = tokenizer

        self.output_columns = ["input_images", "output_image", "instruction"]

    def _collect_annotations(self, config):
        total_samples = 0
        total_ratio = 0
        json_datasets = []
        for data in config["data"]:
            if os.path.isdir(data["path"]):
                jsonl_files = list(glob.glob(os.path.join(data["path"], "**/*.jsonl"), recursive=True)) + list(
                    glob.glob(os.path.join(data["path"], "**/*.json"), recursive=True)
                )
                json_dataset = load_dataset("json", data_files=jsonl_files, cache_dir=None)["train"]
            else:
                data_ext = os.path.splitext(data["path"])[-1]
                if data_ext in [".json", ".jsonl"]:
                    json_dataset = load_dataset("json", data_files=data["path"], cache_dir=None)["train"]
                elif data_ext in [".yml", ".yaml"]:
                    with open(data["path"], "r") as f:
                        sub_config = yaml.load(f, Loader=yaml.FullLoader)
                        json_dataset = self._collect_annotations(sub_config)
                else:
                    raise NotImplementedError(
                        f'Unknown data file extension: "{data_ext}". '
                        f"Currently, .json, .jsonl .yml .yaml are supported. "
                        "If you are using a supported format, please set the file extension so that the proper parsing "
                        "routine can be called."
                    )
            total_ratio += data["ratio"]
            total_samples += len(json_dataset)
            json_datasets.append(json_dataset)

        for json_dataset in json_datasets:
            target_size = int(len(json_dataset) * data["ratio"] / total_ratio)  # normalize the ratio
            if target_size <= len(json_dataset):
                # Random selection without replacement
                indices = random.sample(range(len(json_dataset)), target_size)
            else:
                # Oversample with replacement
                indices = random.choices(range(len(json_dataset)), k=target_size)
            json_dataset = json_dataset.select(indices)

        json_dataset = concatenate_datasets(json_datasets)
        return json_dataset

    def clean_data_item(self, data_item):
        task_type = data_item["task_type"]
        prefixs = [
            "The image portrays ",
            "The image depicts ",
            "The image captures ",
            "The image highlights ",
            "The image shows ",
            "这张图片展示了",
        ]
        if "text_to_image" in task_type or "t2i" in task_type:
            if random.random() < 0.5:
                for p in prefixs:
                    if p in data_item["instruction"]:
                        data_item["instruction"] = data_item["instruction"].replace(p, "")
                        break
        return data_item

    def apply_chat_template(self, instruction, system_prompt):
        if self.use_chat_template:
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return instruction

    def process_item(self, data_item):
        assert data_item["instruction"] is not None
        data_item = self.clean_data_item(data_item)

        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction = self.apply_chat_template(data_item["instruction"], self.SYSTEM_PROMPT)

        input_images = []
        if not drop_ref_img and "input_images" in data_item and data_item["input_images"] is not None:
            input_images_path = data_item["input_images"]

            max_input_pixels = (
                self.max_input_pixels[len(input_images_path) - 1]
                if isinstance(self.max_input_pixels, list)
                else self.max_input_pixels
            )

            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_image = self.image_processor.preprocess(
                    input_image, max_pixels=max_input_pixels, max_side_length=self.max_side_length
                )
                input_images.append(input_image)

        output_image_path = data_item["output_image"]
        output_image = Image.open(output_image_path).convert("RGB")
        output_image = self.image_processor.preprocess(
            output_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length
        )

        return input_images, output_image, instruction

    def __getitem__(self, index):
        max_retries = 12

        current_index = index
        for attempt in range(max_retries):
            try:
                return self.process_item(self.data[current_index])
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                else:
                    # Try a different index for the next attempt
                    current_index = random.randint(0, len(self.data) - 1)
                    continue

    def __len__(self):
        return len(self.data)

    @staticmethod
    def train_transforms(**kwargs) -> list[dict]:
        pass
