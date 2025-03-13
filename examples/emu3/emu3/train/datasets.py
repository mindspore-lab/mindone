# -*- coding: utf-8 -*-

import json
import os.path as osp
import random

import numpy as np
from emu3.mllm import Emu3Tokenizer

import mindspore as ms

from mindone.data import BaseDataset


class Emu3FeatureDataset(BaseDataset):
    def __init__(self, args, tokenizer: "Emu3Tokenizer", split: str = "train"):
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
        if self.args.apply_loss_on_only_vision:
            self.task = "img_gen"
        else:
            self.task = "vqa"
        self.chat_template = "You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:"
        # self.special_token_ids = [
        #     151643,  # pad_token_id
        #     151849,  # bos_token_id
        #     151850,  # eos_token_id
        #     151851,  # img_token_id
        #     151852,  # boi_token_id
        #     151853,  # eoi_token_id
        #     151846,  # eol_token_id
        #     151847,  # eof_token_id
        # ]  # details in Emu3Config
        self.output_columns = ["input_ids", "attention_mask", "labels"]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index: int):
        path = osp.join(self.path_prefix, self.filelist[index])
        data = ms.load_checkpoint(path)  # {"name": name, "images": token_ids, "texts": prompt}
        image_prompt = ""
        if data["images"].dtype == ms.int32:
            image_tokens = data["images"].asnumpy()
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
        else:  # vqa
            prompt = data["texts"]
            response = data["response"]
            vt_prompts = self.chat_template.format(image_prompt=image_prompt, text_prompt=prompt)  # instruction + input vision & text prompts
            input = vt_prompts + response  # instruction + input vision & text prompts + response
        if self.task == "img_gen":
            sample = self.tokenizer(
                input,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
                return_tensors="np",
            )  # keys: "input_ids", "attention_mask"
        else:
            sample = self.tokenizer(
                text=vt_prompts,
                text_pair=response,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
                return_tensors="np",
            )  # keys: "input_ids", "attention_mask"

        labels = sample["input_ids"]
        # mask labels
        if self.args.apply_loss_on_only_vision:  # image generation
            labels = np.where(np.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)
        elif self.args.apply_loss_on_only_text:  # vqa
            prompt_ids = self.tokenizer.encode(vt_prompts)
            response_ids = self.tokenizer.encode(response)
            labels = np.ones_like(sample[input_ids]) * self.args.ignore_index
            padding_start = min(len(prompt_ids) + len(response_ids), labels.shape[-1])
            labels[..., len(prompt_ids) : padding_start] = sample[input_ids][..., len(prompt_ids) : padding_start]

        sample["labels"] = labels
        for k, v in sample.items():
            if k != "attention_mask":
                sample[k] = np.squeeze(sample[k], axis=0).astype(np.int32)
            else:
                sample[k] = np.squeeze(sample[k], axis=0).astype(np.bool_)
        return (
            sample["input_ids"],
            sample["attention_mask"],
            sample["labels"],
        )

    def train_transforms(**kwargs):
        return []

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
