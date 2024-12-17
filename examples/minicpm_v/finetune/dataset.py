import copy
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import ops

# from torch.nn.utils.rnn import pad_sequence
from mindspore.dataset import Dataset

mindone_lib_path = os.path.abspath(os.path.abspath("../../../"))
sys.path.insert(0, mindone_lib_path)

import logging

from mindone.transformers.models.minicpm_v2_6.processing_minicpmv import MiniCPMVProcessor

logger = logging.getLogger(__name__)

llama3_chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"

class SupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        transform,
        tokenizer,
        slice_config,
        llm_type="minicpm",
        patch_size=14,
        query_nums=64,
        batch_vision=False,
        max_length=2048,
    ):
        super(SupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.llm_type = llm_type
        self.patch_size = patch_size
        self.query_nums=query_nums
        self.batch_vision = batch_vision
        self.max_length = max_length
        # self.dataset_column_names = ["input_ids", "position_ids", "labels", "attention_mask", "pixel_values", "tgt_sizes", "image_bound"]
        # self.dataset_output_column_names = ["input_ids", "position_ids", "labels", "attention_mask", "pixel_values", "tgt_sizes", "image_bound"]
        self.dataset_column_names = ["item"]

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx, retry_count=3):
        try:
            if isinstance(self.raw_data[idx]["image"], str):
                images_dict = { "<image>" : Image.open(self.raw_data[idx]["image"]).convert("RGB") }
            elif isinstance(self.raw_data[idx]["image"], Dict):
                ### for multi-images input, the template for every image is <image_xx>, such as <image_00>, <image_01>
                images_dict = {img_name : Image.open(img_path).convert("RGB") for img_name, img_path in self.raw_data[idx]["image"].items()}

            ret = preprocess(
                images_dict,
                self.raw_data[idx]["conversations"],
                self.tokenizer,
                self.transform,
                query_nums=self.query_nums,
                slice_config=self.slice_config,
                llm_type=self.llm_type,
                patch_size=self.patch_size,
                batch_vision=self.batch_vision,
                max_length=self.max_length
            )
            ret = dict(
                input_ids=ret["input_ids"],
                position_ids=ret["position_ids"],
                labels=ret["target"],
                attention_mask=np.ones_like(ret["input_ids"], dtype=np.bool_),
                pixel_values=ret["pixel_values"],
                tgt_sizes=ret["tgt_sizes"],
                image_bound=ret["image_bound"],
            )

            ret = data_collator(ret, max_length = self.max_length)

        except (EOFError, ValueError, OSError) as e:
            # Log and handle EOFError and other file-related errors
            logger.error(f"Data fetch error at index {idx}: {e}")

            if retry_count > 0:
                logger.info(f"Retrying with a different sample. {retry_count} retries left.")
                retry_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(retry_idx, retry_count - 1)
            else:
                # If max retries reached, return a blank or default item
                logger.warning("Max retries reached. Returning a blank entry.")
                return None

        # except:
        #     logger.error(f"data fetch error")
        #     # return self.__getitem__(random.randint(0, len(self)))
        # return (ret["input_ids"], ret["position_ids"], ret["labels"], np.ones_like(ret["input_ids"], dtype=np.bool_), ret["pixel_values"], ret["tgt_sizes"], ret["image_bound"])
        return ret

def data_collator(examples, padding_value=0, max_length=2048):
    def trim_and_pad(seq, batch_first, padding_value):
        # return pad_sequence([s[:max_length] for s in seq], batch_first=True, padding_value=padding_value)
        # return np.stack([s[:max_length] for s in seq])
        return seq

    input_ids = trim_and_pad(
        examples["input_ids"],
        batch_first=True,
        padding_value=padding_value,
    )
    position_ids = trim_and_pad(
        examples["position_ids"],
        batch_first=True,
        padding_value=padding_value,
    )
    targets = trim_and_pad(
        examples["labels"],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = trim_and_pad(
        examples["attention_mask"],
        batch_first=True,
        padding_value=padding_value,
    )
    pixel_values = examples["pixel_values"]
    image_bound = examples["image_bound"]
    tgt_sizes = examples["tgt_sizes"]
    # return {
    #     "input_ids": input_ids,
    #     "position_ids": position_ids,
    #     "labels": targets,
    #     "attention_mask": attention_mask,
    #     "image_bound": image_bound,
    #     "tgt_sizes": tgt_sizes,
    #     "pixel_values": pixel_values,
    # }
    outputs = dict(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=targets,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        tgt_sizes=tgt_sizes,
        image_bound=image_bound,
    )

    return outputs


def conversation_to_ids(conversation, tokenizer, llm_type=None, new_schema=False, max_length=2048):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """
    if llm_type == "llama3":
        input_ids, context, raw_msg = conversation_to_ids_llama3(
            conversation, tokenizer
        )
    elif llm_type == "qwen2":
        input_ids, context, raw_msg = conversation_to_ids_qwen2(
            conversation, tokenizer
        )
    else:
        input_ids, context, raw_msg = conversation_to_ids_minicpm(
            conversation, tokenizer
        )

    ids = np.hstack(input_ids, dtype=np.int32)
    context = np.hstack(context, dtype=np.int8)
    if input_ids.shape[-1] > max_length:
        ids = ids[:max_length]
        context = context[:max_length]
        logger.warning(f"The input length ({input_ids.shape[-1]}) exceeds the model's maximum length ({max_length}), so it has been truncated")

    if np.all(context):
        logger.error("No tokens available to compute loss.")
        raise Exception("No tokens available to compute loss.")

    # build target
    target = np.full_like(ids, -100, dtype=np.int32)

    for i in range(1, len(ids)):
        if context[i] == 0:
            target[i - 1] = ids[i]
        if context[i] == 1 and context[i - 1] == 0:
            if hasattr(tokenizer, "eot_id"):
                target[i - 1] = tokenizer.eot_id
            else:
                target[i - 1] = tokenizer.eos_id

    # build image bound
    if new_schema:
        start_cond = (ids == tokenizer.im_start_id) | (ids == tokenizer.slice_start_id)
        end_cond = (ids == tokenizer.im_end_id) | (ids == tokenizer.slice_end_id)
        image_start_tokens = np.where(start_cond)[0]
        image_start_tokens += 1
        image_end_tokens = np.where(end_cond)[0]
    else:
        image_start_tokens = np.where(ids == tokenizer.im_start_id)[0]
        image_start_tokens += 1
        image_end_tokens = np.where(ids == tokenizer.im_end_id)[0]
    if len(image_start_tokens) != len(image_end_tokens):
        logger.error("image start token != image end tokens")
        raise Exception("image start token != image end tokens")

    if len(image_start_tokens) > 0:
        image_bound = np.hstack(
            [np.expand_dims(image_start_tokens, axis=-1), np.expand_dims(image_end_tokens, axis=-1)]
        )
    else:
        image_bound = []

    position_ids = np.arange(ids.shape[0]).astype(np.int64)
    return {
        "input_ids": ids,
        "target": target,
        "image_bound": image_bound,
        "raw_msg": raw_msg,
        "position_ids": position_ids
    }


def conversation_to_ids_minicpm(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]
        if role == "user":
            prefix = "<用户>"
        else:
            prefix = "<AI>"
        # append eos
        if idx == len(conversation) - 1:
            message = message + tokenizer.eos_token
        prefix_ids = tokenizer.encode(prefix)[1:]  # remove bos
        message_ids = tokenizer.encode(message)[1:]

        input_ids.append(prefix_ids)
        input_ids.append(message_ids)

        context.append(np.ones((len(prefix_ids),), dtype=np.int8))
        if role == "assistant":
            context.append(np.zeros((len(message_ids),), dtype=np.int8))
        else:
            context.append(np.ones((len(message_ids),), dtype=np.int8))

        raw_msg += prefix + message

    return input_ids, context, raw_msg


def conversation_to_ids_llama3(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    raw_msg = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, chat_template=llama3_chat_template,
    )
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, chat_template=llama3_chat_template,
    )
    input_ids = np.array(input_ids)

    start_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    )[0]
    assistant_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )[0]
    end_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    )[0]
    eot_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|eot_id|>"))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx in set((start_header_idxs + end_header_idxs) / 2):
            st = assistant_idx + 3  # assistant<|end_header_id|>\n\n
            for eot_idx in eot_idxs:
                if eot_idx > st:
                    context[st: eot_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)

    return input_ids, context, raw_msg


def conversation_to_ids_qwen2(conversation, tokenizer):
    raw_msg = ""
    chat = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]
        if role == "user":
            prefix = "user"
        else:
            prefix = "assistant"
        chat.append({"role":prefix, "content":message})
        raw_msg += prefix + message
    assert set([i['role'] for i in chat]) & set(['assistant'])

    ret = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    input_ids = np.array(input_ids)

    start_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_start|>'))[0]
    assistant_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('assistant'))[0]
    end_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_end|>'))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx-1 in set(start_idxs):
            st = assistant_idx + 1
            for end_idx in end_idxs:
                if end_idx > st:
                    context[st: end_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)
    return input_ids, context, raw_msg

def trans_fn(x):
    x = np.asarray(x).transpose((2, 0, 1))
    return (x-0.5*255)/(0.5*255)

def preprocess(
    images_dict,
    conversations,
    tokenizer,
    transform,
    query_nums=64,
    slice_config=None,
    llm_type=None,
    patch_size=14,
    batch_vision=False,
    max_length=2048,
):
    """
    single(multi) image(s) preprocess, the image(s) will be placed at the top of the conversation
    """
    conversations = copy.deepcopy(conversations)
    assert len(conversations) > 1, "conversations length must large than 2"
    assert conversations[0]["role"] == "user", "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config
    default_image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
    )
    new_schema = False
    use_image_id = False
    if llm_type=='qwen2':
        new_schema = True
        use_image_id = True
    image_placeholder_dict = {}
    images = []
    image_id_cnt = 0
    for img_name, image in images_dict.items():
        if slice_config:
            source_image, patches, best_grid = slice_image(
                image,
                slice_config["max_slice_nums"],
                slice_config["scale_resolution"],
                slice_config["patch_size"],
            )
            images.append(source_image)
            image_placeholder = default_image_placeholder
            if len(patches) > 0:
                for i in range(len(patches)):
                    for j in range(len(patches[0])):
                        images.append(patches[i][j])
                if use_image_id:
                    image_placeholder = f'{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}' + image_placeholder
                    image_id_cnt += 1
                image_placeholder += get_grid_placeholder(
                    tokenizer, best_grid, query_nums, new_schema = new_schema)
            image_placeholder_dict[img_name] = image_placeholder
        else:
            images.append(image)
            if use_image_id:
                image_placeholder = f'{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}' + image_placeholder
                image_id_cnt += 1
            else:
                image_placeholder = default_image_placeholder
            image_placeholder_dict[img_name] = image_placeholder

    images = [trans_fn(i) for i in images]

    if len(images_dict) == 1 and "<image>" in images_dict:
        if "<image>" in conversations[0]["content"]:
            conversations[0]["content"] = conversations[0]["content"].replace(
                "<image>", image_placeholder
            )
        else:
            conversations[0]["content"] = (
                image_placeholder + "\n" + conversations[0]["content"]
            )
        input_dict = conversation_to_ids(conversations, tokenizer, llm_type, new_schema, max_length)
    else:
        pattern = r'<image_\d+>'
        new_conversations = []
        for conversation in conversations:
            content = conversation['content']
            parts = re.split(f'({pattern})', content)
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                if re.match(pattern, part):
                    if part in image_placeholder_dict:
                        parts[i] = image_placeholder_dict[part]
                    else:
                        raise Exception(f"not found {part} in image dict")
            conversation['content'] = '\n'.join(parts)
            new_conversations.append(conversation)
        conversations = new_conversations

        input_dict = conversation_to_ids(conversations, tokenizer, llm_type, new_schema, max_length)

    if batch_vision:
        tgt_sizes = []
        reshape_images = []
        for image in images:
            H, W = image.shape[1:]
            reshape_image = reshape_by_patch(image, patch_size)
            reshape_images.append(reshape_image)
            tgt_sizes.append([H // patch_size, W // patch_size])
        if tgt_sizes:
            tgt_sizes = np.array(tgt_sizes).astype(np.int32)

        input_dict["pixel_values"] = reshape_images
        input_dict["tgt_sizes"] = tgt_sizes

    else:
        input_dict["pixel_values"] = images
        input_dict["tgt_sizes"] = []

    return input_dict


def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / \
        (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(
            original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num, new_schema=False):
    if new_schema:
        image_placeholder = (
            tokenizer.slice_start + tokenizer.unk_token * query_num + tokenizer.slice_end
        )
    else:
        image_placeholder = (
            tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
        )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    if new_schema:
        slice_placeholder = '\n'.join(slices)
    else:
        slice_placeholder = tokenizer.slice_start + \
        "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    # image = ms.Tensor(image_tensor)
    #
    # c = image.shape[0]
    # h = image.shape[1]
    # w = image.shape[2]
    # image = image.reshape(1, c, h, w)
    #
    # patches = ops.unfold(
    #     image,
    #     (patch_size, patch_size),
    #     stride=(patch_size, patch_size)
    # )

    c = image_tensor.shape[0]
    h = image_tensor.shape[1]
    w = image_tensor.shape[2]

    v_block_num = h // patch_size
    h_block_num = w // patch_size

    patches = image_tensor.reshape(c, v_block_num, patch_size, h_block_num, patch_size)
    patches = np.transpose(patches, (0, 2, 4, 1, 3))
    patches = patches.reshape(c*patch_size*patch_size, -1)

    patches = patches.reshape(image_tensor.shape[0], patch_size, patch_size, -1)
    patches = patches.transpose((0, 1, 3, 2)).reshape(
        image_tensor.shape[0], patch_size, -1)
    return patches
