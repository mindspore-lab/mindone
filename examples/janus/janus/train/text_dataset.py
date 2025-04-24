import logging
from copy import deepcopy
from typing import Tuple

import numpy as np
from datasets import load_dataset
from janus.models import VLChatProcessor

import mindspore as ms
from mindspore.dataset.transforms import Compose, vision

logger = logging.getLogger(__name__)


class TextDataset:
    def __init__(
        self,
        vl_chat_processor: VLChatProcessor = None,
        dataset_name=None,
        data_dir=None,
        max_token_length: int = 1024,
        num_samples: int = -1,
        default_image_shape=(1, 3, 384, 384),
    ) -> None:
        if dataset_name.lower() == "pubmedqa":
            self.dataset = load_dataset(data_dir, "default", split="train")
        else:
            raise NotImplementedError

        if num_samples > 0:
            logger.info(f"sequential slice dataset samples to {num_samples}")
            self.dataset = self.dataset.select(range(num_samples))

        self.length = len(self.dataset)
        self.vl_chat_processor = vl_chat_processor
        self.max_token_length = max_token_length
        self.default_image_shape = default_image_shape

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = self.dataset[int(idx)]
        question = record["question"]
        answer = record["long_answer"]

        # process text
        input_ids, labels, attention_mask = self.prepare_sft_inputs_and_label(question, answer)
        task_type = np.array(1, dtype=np.int32)

        # add dummy image and image_seq_mask item to pure text for batching
        image = np.zeros(self.default_image_shape, np.float32)
        image_seq_mask = np.zeros((self.max_token_length), dtype=np.bool_)

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

    def prepare_sft_inputs_and_label(self, question, answer):
        # convert to sft prompt
        conversation = [
            {
                "role": "<|User|>",
                "content": question,
            },
            {"role": "<|Assistant|>", "content": answer},
        ]
        prompt = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=self.vl_chat_processor.system_prompt,
        )

        vlcp = self.vl_chat_processor

        # left padding (default), same as inference. eos will be added
        """
        input_ids = vlcp.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_token_length,
            # padding_side='left',
            truncation=True,
        )
        input_ids = np.array(input_ids, np.int32)
        attention_mask = np.ones(shape=[len(input_ids)], dtype=np.bool_)
        attention_mask[input_ids == vlcp.pad_id] = 0
        """
        inputs = vlcp.tokenizer(
            prompt,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_token_length,
            # padding_side='left',
            truncation=True,
        )
        input_ids = np.array(inputs["input_ids"], dtype=np.int32)
        attention_mask = np.array(inputs["attention_mask"], dtype=np.bool_)

        # make labels
        # label, only train on answer seq
        ignore_index = -100
        labels = deepcopy(input_ids)
        answer_begin_token = vlcp.tokenizer.vocab.get("<|Assistant|>")
        # shift 2 tokens is the answer first token, e.g.  <Assistant>: Results show
        answer_begin_idx = np.where(input_ids == answer_begin_token)[0] + 2

        # answer_end_token = vclp.tokenizer.eos_token_id
        labels[: int(answer_begin_idx)] = ignore_index

        return input_ids, labels, attention_mask


def create_dataloader_text(
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
    dataset = TextDataset(
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
