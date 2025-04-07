import os
import random

import numpy as np
from datasets import load_dataset
from PIL import Image


class DatasetFromJson:
    def __init__(
        self,
        json_file: str,
        image_path: str,
        processer,
        image_transform,
        max_input_length_limit: int = 18000,
        condition_dropout_prob: float = 0.1,
    ):
        self.image_transform = image_transform
        self.processer = processer
        self.condition_dropout_prob = condition_dropout_prob
        self.max_input_length_limit = max_input_length_limit

        self.data = load_dataset("json", data_files=json_file)["train"]
        self.image_path = image_path

    def process_image(self, image_file):
        if self.image_path is not None:
            image_file = os.path.join(self.image_path, image_file)
        image = Image.open(image_file).convert("RGB")
        return self.image_transform(image)

    def __getitem__(self, index):
        return self.get_example(index)

    def get_example(self, index):
        if isinstance(index, np.int64):
            index = index.item()
        example = self.data[index]

        instruction, input_images, output_image = (
            example["instruction"],
            example["input_images"],
            example["output_image"],
        )
        if random.random() < self.condition_dropout_prob:
            instruction = "<cfg>"
            input_images = None
        if input_images is not None:
            input_images = [self.process_image(x)[0] for x in input_images]
        mllm_input = self.processer.process_multi_modal_prompt(instruction, input_images)

        output_image = self.process_image(output_image)[0]

        return mllm_input, output_image

    def __len__(self):
        return len(self.data)


class TrainDataCollator:
    def __init__(self, pad_token_id: int, hidden_size: int):
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size

    def create_position(self, attention_mask, num_tokens_for_output_images):
        position_ids = []

        text_length = attention_mask.shape[-1]
        img_length = num_tokens_for_output_images

        for mask in attention_mask:
            temp_l = np.sum(mask)
            temp_position = [0] * (text_length - temp_l) + [i for i in range(temp_l + img_length + 1)]
            position_ids.append(temp_position)

        return np.array(position_ids, dtype=np.int32)

    def create_mask(self, attention_mask, num_tokens_for_output_images):
        extended_mask = []
        # padding_images = []
        text_length = attention_mask.shape[-1]
        img_length = num_tokens_for_output_images
        seq_len = text_length + img_length + 1
        # inx = 0
        for mask in attention_mask:
            temp_l = np.sum(mask)

            pad_l = text_length - temp_l
            temp_mask = np.tril(np.ones((temp_l + 1, temp_l + 1)))

            image_mask = np.zeros((temp_l + 1, img_length))
            temp_mask = np.concatenate([temp_mask, image_mask], axis=-1)

            image_mask = np.ones((img_length, temp_l + img_length + 1))
            temp_mask = np.concatenate([temp_mask, image_mask], axis=0)

            if pad_l > 0:
                pad_mask = np.zeros((temp_l + 1 + img_length, pad_l))
                temp_mask = np.concatenate([pad_mask, temp_mask], axis=-1)

                pad_mask = np.ones((pad_l, seq_len))
                temp_mask = np.concatenate([pad_mask, temp_mask], axis=0)

            extended_mask.append(np.expand_dims(temp_mask, axis=0))
        return np.concatenate(extended_mask, axis=0)

    def pad_input_ids(self, input_ids):
        max_l = max([len(x) for x in input_ids])
        padded_ids = []
        attention_mask = []

        for i in range(len(input_ids)):
            temp_ids = input_ids[i]
            temp_l = len(temp_ids)
            pad_l = max_l - temp_l
            if pad_l == 0:
                attention_mask.append([1] * max_l)
                padded_ids.append(temp_ids)
            else:
                attention_mask.append([0] * pad_l + [1] * temp_l)
                padded_ids.append([self.pad_token_id] * pad_l + temp_ids)

        return np.array(padded_ids, dtype=np.int32), np.array(attention_mask, dtype=np.int32)  # , image_sizes

    def adjust_attention_for_input_images(self, attention_mask, image_sizes):
        for b_inx in image_sizes.keys():
            for start_inx, end_inx in image_sizes[b_inx]:
                attention_mask[b_inx][start_inx:end_inx, start_inx:end_inx] = 1
        return attention_mask

    def __call__(self, col1, col2):
        mllm_inputs = [f for f in col1]

        output_images = [np.expand_dims(f, axis=0) for f in col2]
        output_images = np.concatenate(output_images, axis=0)

        target_img_size = (output_images.shape[-2], output_images.shape[-1])

        num_tokens_for_output_images = []
        num_tokens_for_output_images = target_img_size[0] * target_img_size[1] // 16 // 16
        input_ids = [x["input_ids"] for x in mllm_inputs]
        padded_input_ids, attention_mask = self.pad_input_ids(input_ids)
        position_ids = self.create_position(attention_mask, num_tokens_for_output_images)
        attention_mask = self.create_mask(attention_mask, num_tokens_for_output_images)

        return padded_input_ids, attention_mask, position_ids, output_images
