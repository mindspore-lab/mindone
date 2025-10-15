import logging
import os
from copy import deepcopy
from typing import Tuple

import numpy as np
from datasets import load_dataset
from janus.models import VLChatProcessor
from janus.utils.io import load_pil_images

import mindspore as ms
from mindspore.dataset.transforms import Compose, vision

logger = logging.getLogger(__name__)


class VqaDataset:
    def __init__(
        self,
        vl_chat_processor: VLChatProcessor = None,
        dataset_name=None,
        data_dir=None,
        max_token_length: int = 1024,
        num_samples: int = -1,
    ) -> None:
        if dataset_name.lower() == "medical-vqa":
            self.image_dir = os.path.join(data_dir, "vqa-rad/images")
            self.dataset = load_dataset(data_dir, split="train")
            # filter data
            self.dataset = self.filter_samples(self.dataset)
        else:
            raise NotImplementedError

        if num_samples > 0:
            logger.info(f"sequential slice dataset samples to {num_samples}")
            self.dataset = self.dataset.select(range(num_samples))

        self.length = len(self.dataset)
        self.vl_chat_processor = vl_chat_processor
        self.max_token_length = max_token_length

    def filter_samples(self, dataset):
        num_src_samples = len(dataset)
        print("num src samples: ", num_src_samples)

        indices_to_keep = []
        for idx, record in enumerate(dataset):
            image_path = os.path.join(self.image_dir, record["image"])
            # filter out record without image
            if os.path.exists(image_path):
                indices_to_keep.append(idx)

        print("num samples after filtering: ", len(indices_to_keep))

        return dataset.select(indices_to_keep)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = self.dataset[int(idx)]
        assert record["conversations"][0]["from"] == "human"
        question = record["conversations"][0]["value"]
        answer = record["conversations"][1]["value"]
        image_path = os.path.join(self.image_dir, record["image"])

        # preprocess
        ds_image_tag = "<image>"  # image tag used in the original dataset
        assert question.count(ds_image_tag) == 1, "the question should contain one image exactly"
        question = question.replace("\n<image>", "").replace("<image>\n", "").replace("<image>", "")
        # janus_image_tag = self.vl_chat_processor.image_tag

        (
            input_ids,
            labels,
            attention_mask,
            image_seq_mask,
            image,
        ) = self.prepare_sft_inputs_and_label(question, answer, image_path)

        task_type = np.array(0, dtype=np.int32)

        return task_type, input_ids, labels, attention_mask, image_seq_mask, image

    @staticmethod
    def create_transform(image_size: int, interpolation: vision.Inter) -> Compose:
        return Compose(
            [
                vision.Resize(image_size, interpolation=interpolation),
                vision.CenterCrop(image_size),
                vision.ToTensor(),
                vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
            ]
        )

    def prepare_sft_inputs_and_label(self, question, answer, image_path):
        # convert to sft prompt
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": answer},
        ]

        vlcp = self.vl_chat_processor
        pil_images = load_pil_images(conversation)

        # apply sft format
        sft_format = vlcp.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vlcp.sft_format,
            system_prompt=vlcp.system_prompt,
        )
        # tokenize
        input_ids = vlcp.tokenizer.encode(sft_format)
        input_ids = np.array(input_ids, dtype=np.int32)

        # add image tokens to the input_ids
        image_token_mask = input_ids == vlcp.image_id
        image_indices = image_token_mask.nonzero()
        input_ids, num_image_tokens = add_image_token(
            vlcp,
            image_indices=image_indices,
            input_ids=input_ids,
        )

        # load images
        image = vlcp.image_processor(pil_images, return_tensors="np")["pixel_values"]
        image = np.stack(image)  # list -> np [1, 3, 384, 384]

        # pad to pre-set max_length or max seq len in the current batch
        padded_input_ids = np.ones((self.max_token_length), dtype=np.int32) * vlcp.pad_id
        attention_mask = np.zeros((self.max_token_length), dtype=np.bool_)
        image_seq_mask = np.zeros((self.max_token_length), dtype=np.bool_)

        seq_len = len(input_ids)
        padded_input_ids[-seq_len:] = input_ids
        attention_mask[-seq_len:] = 1
        image_seq_mask[-seq_len:] = input_ids == vlcp.image_id

        input_ids = padded_input_ids

        # make labels
        # label, only train on answer seq
        ignore_index = -100
        labels = deepcopy(input_ids)
        answer_begin_token = vlcp.tokenizer.vocab.get("<|Assistant|>")
        # shift 2 tokens is the answer first token, e.g.  <Assistant>: Results show
        answer_begin_idx = np.where(input_ids == answer_begin_token)[0] + 2

        # answer_end_token = vclp.tokenizer.eos_token_id
        labels[: int(answer_begin_idx)] = ignore_index

        return input_ids, labels, attention_mask, image_seq_mask, image


def add_image_token(
    vlcp,
    image_indices,
    input_ids,
):
    """

    Args:
        vlcp: chat processor
        image_indices (List[int]): [index_0, index_1, ..., index_j]
        input_ids (np array): [N]

    Returns:
        input_ids (np array): [N + image tokens]
        num_image_tokens (np array): [n_images]
    """

    input_slices = []

    start = 0
    for index in image_indices:
        if vlcp.add_special_token:
            end = int(index + 1)
        else:
            end = int(index)

        # original text tokens
        input_slices.append(input_ids[start:end])

        # add boi, image tokens, eoi and set the mask as False
        input_slices.append(vlcp.image_start_id * np.ones((1), dtype=np.int32))
        # FIXME: allow set num_image_tokens to fit different image size
        input_slices.append(vlcp.image_id * np.ones((vlcp.num_image_tokens,), dtype=np.int32))
        input_slices.append(vlcp.image_end_id * np.ones((1), dtype=np.int32))
        start = int(index + 1)

    # the left part
    input_slices.append(input_ids[start:])

    # concat all slices
    input_ids = np.concatenate(input_slices, axis=0)
    num_image_tokens = np.array([vlcp.num_image_tokens] * len(image_indices), dtype=np.int32)

    return input_ids, num_image_tokens


def create_dataloader_vqa(
    dataset_name: str,
    data_dir: str,
    vl_chat_processor: VLChatProcessor,
    max_token_length: int = 1024,
    num_samples: int = -1,
    batch_size: int = 1,
    shuffle: bool = True,
    num_parallel_workers: int = 8,
    rank: int = 0,
    rank_size: int = 1,
):
    dataset = VqaDataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        vl_chat_processor=vl_chat_processor,
        max_token_length=max_token_length,
        num_samples=num_samples,
    )

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=[
            "task_type",
            "input_ids",
            "labels",
            "attention_mask",
            "image_seq_mask",
            "image",
        ],
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=True,
        num_shards=rank_size,
        shard_id=rank,
    )

    dataloader = dataloader.batch(batch_size, drop_remainder=True)

    return dataloader
