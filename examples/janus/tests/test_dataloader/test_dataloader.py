import bisect
import json
import os
import sys
from typing import Dict, List
from collections import defaultdict

import mindspore as ms
from mindspore import Tensor, ops
from mindspore.dataset import WeightedRandomSampler, GeneratorDataset

from janus.models import VLChatProcessor
from janus.utils.conversation import get_conv_template
from janus.utils.io import load_pil_images


class  SFTDataset():
    """
    Supervised fine-tune dataset that outputs pure text, multi-modal and text-to-image data with a given sampling ratio
    """
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, meta_path, vl_chat_processor: VLChatProcessor):
        self.vl_chat_processor = vl_chat_processor

        ds_collections = json.loads(open(meta_path).read())
        # Statistics for dataset sizes per task type
        self.dataset_sizes_for_task = defaultdict(int)
        self.num_dataset = len(ds_collections)

        # Initialize dataset metadata storage
        self.ds_meta_data = []  # Metadata for all datasets
        self.roots = []         # Root directories for each dataset
        self.task_types = []    # Collection of task types
        self.dataset_sizes = [] # Sample counts per dataset

        # Iterate through each dataset configuration
        for ds_idx, (ds_name, ds_meta) in enumerate(ds_collections.items()):
            task_type = ds_meta["task_type"]
            assert ds_meta["annotation"].endswith('jsonl'), f"annotation must be jsonl, but go {ds_meta['annotation']}"
            # Load dataset annotations (JSONL format)
            self.ds_meta_data.append(json.loads(open(ds_meta['annotation']).read()))

            # Record root directory path for current dataset
            self.roots.append(ds_meta['root'])

            self.task_types.append(task_type)

            self.dataset_sizes.append(len(self.ds_meta_data[-1]))

            if task_type not in self.dataset_sizes_for_task:
                self.dataset_sizes_for_task[task_type] = 0
            self.dataset_sizes_for_task[task_type] += self.dataset_sizes[-1]

        self.total_dataset_size = sum(len(r) for r in self.ds_meta_data)

        self.cumulative_sizes = self.cumsum(self.ds_meta_data)

        # set abs path for images info in meta_data
        for i in range(self.num_dataset):
            for conversation in self.ds_meta_data[i]:
                for turn in conversation:
                    if "images" not in turn:
                        continue
                    turn["images"] = [os.path.join(self.roots[i], rel_path) for rel_path in turn["images"]]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx ==0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        raw_data = self.ds_meta_data[dataset_idx][sample_idx]

        task_type = self.task_types[dataset_idx]

        if task_type == 0:
            ret = self.multi_modal_get_item(raw_data)
        elif task_type == 1:
            ret = self.pure_text_get_item(raw_data)
        else:
            ret = self.visual_generation_get_item(raw_data)
        return ret

    def __len__(self):
        return self.total_dataset_size

    def multi_modal_get_item(self, conversation: List[Dict]):
        """
        Get item for multi-modal task
        """
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            images=pil_images,
            conversations=conversation,
            force_batchify=False,
        )

        print(f"prompt: {prepare_inputs.sft_format}")

        # add image token to prompt, this should be aligned with vl_chat_processor.add_image_token
        image_tokens = f"{self.vl_chat_processor.image_start_tag}" \
                       f"{self.vl_chat_processor.image_tag * self.vl_chat_processor.num_image_tokens}" \
                       f"{self.vl_chat_processor.image_end_tag}"

        sft_format_with_full_image = prepare_inputs.sft_format.replace(
            self.vl_chat_processor.image_tag, image_tokens, 1
        )

        target_ids = self.get_conversation_targets(sft_format_with_full_image, prepare_inputs.input_ids)

        task_type = Tensor(0, dtype=ms.int32)

        return dict(
            input_ids=prepare_inputs.input_ids,
            pixel_values=prepare_inputs.pixel_values,
            target_ids=target_ids,
            task_type=task_type,
            num_image_tokens=prepare_inputs.num_image_tokens,
        )

    def pure_text_get_item(self, conversation: List[Dict]):
        """
        Get item for pure text task
        """
        # apply sft format
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=self.vl_chat_processor.system_prompt,
        )

        print(f"prompt: {sft_format}")

        # tokenize
        input_ids = self.vl_chat_processor.tokenizer.encode(sft_format)
        input_ids = Tensor(input_ids, dtype=ms.int64)
        target_ids = self.get_conversation_targets(sft_format, input_ids)
        task_type = Tensor(1, dtype=ms.int32)
        return dict(
            input_ids=input_ids,
            target_ids=target_ids,
            task_type=task_type,
            num_image_tokens=Tensor([self.vl_chat_processor.num_image_tokens], dtype=ms.int32),
            pixel_values=ops.zeros((1, *self.vl_chat_processor.image_processor.default_shape)).float(),
        )

    def visual_generation_get_item(self, conversation: List[Dict]):
        """
        Get item for visual generation task
        """
        pil_images = load_pil_images(conversation)
        pixel_values = self.vl_chat_processor.image_processor(pil_images)["pixel_values"]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )

        prompt = sft_format + self.vl_chat_processor.image_start_tag \
                 + self.vl_chat_processor.image_tag * self.vl_chat_processor.num_image_tokens \
                 + self.vl_chat_processor.image_end_tag
        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = Tensor(input_ids, dtype=ms.int64)
        num_image_tokens = Tensor([self.vl_chat_processor.num_image_tokens], dtype=ms.int32)

        task_type = Tensor(2, dtype=ms.int32)
        return dict(
            input_ids=input_ids,
            target_ids=input_ids.clone(),
            pixel_values=pixel_values,
            task_type=task_type,
            num_image_tokens=num_image_tokens,
        )

    def get_conversation_targets(self, prompt, input_ids):
        """
        Get conversation targets. The target_ids is the copy of input_ids, except that the non-answer tokens are masked.
        """
        IGNORE_INDEX = -100
        conv = get_conv_template(self.vl_chat_processor.sft_format)
        tokenizer = self.vl_chat_processor.tokenizer

        target_ids = input_ids.clone()

        user_template = conv.sep + conv.roles[0] + ": "
        assistant_template = conv.sep + conv.roles[1] + ":"
        total_len = int(ops.not_equal(target_ids, self.vl_chat_processor.pad_id).sum())

        cur_len = 1
        target_ids[:cur_len] = IGNORE_INDEX # <bos>
        parts = prompt.split(assistant_template)
        info = parts[0] + assistant_template  # str before first answer
        temp_len = len(tokenizer(info).input_ids) - 1  # remove <bos>
        target_ids[cur_len: cur_len+temp_len] = IGNORE_INDEX
        cur_len += temp_len

        for index in range(1, len(parts)-1):
            info  = parts[index]
            part1, part2 = info.split(user_template)
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len += temp_len
            part = user_template + part2 + assistant_template
            temp_len = len(tokenizer(part).input_ids) - 1
            target_ids[cur_len: cur_len+temp_len] = IGNORE_INDEX
            cur_len += temp_len

        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len += temp_len
        target_ids[cur_len:] = IGNORE_INDEX

        # inspect and check the correctness of masking
        if False:
            z = target_ids.clone()
            z = ops.where(z == IGNORE_INDEX, 1, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target_ids[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch {cur_len} vs {total_len}")
                print(f"WARNING: please check the prompt: {prompt}")
                sys.stdout.flush()

        return target_ids

    @staticmethod
    def collate_fn(prepare_list: List[Dict], batch_info):
        """
        Padding and stacking tensors in a batch
        """
        pad_id = 100002
        image_id = 100581
        image_default_shape = [3, 384, 384]

        batch_size = len(prepare_list)

        n_images = []

        for prepare in prepare_list:
            n_images.append(len(prepare["num_image_tokens"]))

        input_token_max_len = 1024
        max_n_image = max(1, max(n_images))
        batched_input_ids = ops.full((batch_size, input_token_max_len), pad_id).long()
        batched_target_ids = ops.full((batch_size, input_token_max_len), pad_id).long()
        batched_pixel_values = ops.zeros((batch_size, max_n_image, *image_default_shape)).float()
        batched_attention_mask = ops.zeros((batch_size, input_token_max_len)).long()
        batched_images_seq_mask = ops.zeros((batch_size, input_token_max_len)).bool()

        batched_task_type = ops.zeros((batch_size,), dtype=ms.int32)

        for i, prepare in enumerate(prepare_list):
            input_ids = prepare["input_ids"]
            target_ids = prepare["target_ids"]
            pixel_values = prepare["pixel_values"]
            task_type = prepare["task_type"]

            n_image = len(prepare["num_image_tokens"])
            seq_len = len(prepare["input_ids"])

            # left-padding
            batched_attention_mask[i, -seq_len:] = 1
            batched_input_ids[i, -seq_len:] = Tensor(input_ids, dtype=ms.int64)
            batched_target_ids[i, -seq_len:] = Tensor(target_ids, dtype=ms.int64)
            batched_images_seq_mask[i, -seq_len:] = input_ids == image_id
            batched_task_type[i] = task_type


            if n_image > 0:
                batched_pixel_values[i, :n_image] = pixel_values

        return (
            batched_input_ids,
            batched_attention_mask,
            batched_pixel_values,
            batched_images_seq_mask,
            batched_target_ids,
            batched_task_type,
        )


def build_dataloader(meta_path, vl_chat_processor, sample_ratios):
    """
    Build dataloader for supervised fine-tune

    Args:
        meta_path: path to meta file
        vl_chat_processor: vl chat processor
        sample_ratios: sample ratios for each task type
    """
    sft_dataset = SFTDataset(meta_path, vl_chat_processor)
    ds_sample_weights = [0.0]* len(sample_ratios)
    for i in range(len(sample_ratios)):
        ds_sample_weights[i] = sample_ratios[i] \
                               * sft_dataset.dataset_sizes_for_task[i] / sft_dataset.total_dataset_size
    weights = []
    for i in range(sft_dataset.num_dataset):
        task_type = sft_dataset.task_types[i]
        weights += [ds_sample_weights[task_type]] * sft_dataset.dataset_sizes[i]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    dataloader = GeneratorDataset(
        sft_dataset,
        sampler=sampler,
        column_names=["data"],
    )

    dataloader = dataloader.batch(num_parallel_workers=2, batch_size=3, per_batch_map=SFTDataset.collate_fn,
                                  output_columns=["input_ids", "attention_mask", "pixel_values",
                                                  "images_seq_mask", "target_ids", "task_type"])
    print(f'finish build dataloader')
    return dataloader

if __name__ == "__main__":
    meta_path = "./meta_data.jsonl"
    pretrain_model_path = "/mnt/disk2/fredhong/hf_ckpts/Janus-Pro-1B"
    vl_chat_processor = VLChatProcessor.from_pretrained(pretrain_model_path)
    dataloader = build_dataloader(meta_path, vl_chat_processor, [1, 1, 1])
    for data in dataloader.create_dict_iterator():
        print(data)
        break

