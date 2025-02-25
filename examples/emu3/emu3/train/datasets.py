# -*- coding: utf-8 -*-

import json
import os.path as osp
import random
from typing import List

from emu3.mllm import Emu3Tokenizer
from emu3.train import DataArguments

import mindspore as ms
from mindspore import ops

from mindone.data import BaseDataset

class Emu3FeatureDataset(BaseDataset):
    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer", split: str = "train", task: str ="img_gen"):
        super().__init__()

        self.args = args
        if split == "train":
            with open(args.train_data_path) as f:
                d = json.load(f)
        elif split == "test":
            with open(args.eval_data_path) as f:
                d = json.load(f)
        else:
            raise ValueError(f"Unknow split value {split} for data loading")

        self.path_prefix = d["prefix"]
        self.filelist = d["path_list"]

        self.tokenizer = tokenizer
        assert not self.args.apply_loss_on_only_vision or not self.args.apply_loss_on_only_text
        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]

        # self.output_columns = ["input_ids", "attention_mask", "position_ids", "past_key_values", "inputs_embeds", "labels"]
        self.task = task # "img_gen" or "vqa"
        self.chat_template="You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:"

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index: int):
        path = osp.join(self.path_prefix, self.filelist[index])
        data = ms.load_checkpoint(path) # {"name": name, "images": token_ids, "texts": prompt}

        image_tokens = data["images"]
        image_prompt = self.format_image_prompt(image_tokens)

        # structure:
        # [BOS] {caption text} [SOV] {meta text} [SOT] {vision tokens} [EOV] [EOS].
        if self.task == "img_gen":
            p_prob = random.random()
            if p_prob < self.args.null_prompt_prob:
                prompt = ""
            else:
                prompt = data["texts"]

            # image generation template
            input = self.tokenizer.bos_token + prompt + image_prompt
        else: # vqa
            response = data["response"]
            input = self.tokenizer.bos_token + self.chat_template.format(image_prompt, prompt) + response

        sample = self.tokenizer(
            input,
            padding="max_length",
            return_token_type_ids=False,
            return_tensors="np",
        ) # keys: "input_ids", "attention_mask"
        # print(sample)

        labels = (ms.Tensor(sample["input_ids"], dtype=ms.int32), )
        if self.args.apply_loss_on_only_vision: # image generation
            labels = ops.where(ops.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)
        elif self.args.apply_loss_on_only_text: # vqa, simply ignore visual ids, ignore special ids
            visual_mask = ops.logical_and(labels >= self.bov, labels <= self.eov)
            special_mask = ms.Tensor([label in self.args.special_token_ids for label in labels], dtype=ms.bool_)
            labels = ops.where((visual_mask or special_mask), self.args.ignore_index, labels)

        sample["labels"] = labels
        for k, v in sample.items():
            if isinstance(sample[k], np.ndarray):
                sample[k] = ms.Tensor(sample[k], dtype=ms.int32)
            sample[k] = v.squeeze(0)

        # return (
        #     sample["input_ids"],
        #     sample["attention_mask"],
        #     None,
        #     None,
        #     None,
        #     sample["labels"]
        # )
        return sample

    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token
            + f"{h}*{w}"
            + self.tokenizer.img_token
            + imgstr
            + self.tokenizer.eol_token
            + self.tokenizer.eof_token
            + self.tokenizer.eoi_token
        )

        return image_prompt

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [self.args.visual_token_pattern.format(token_id=token_id) for token_id in token_row]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr
