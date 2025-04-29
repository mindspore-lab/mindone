import logging
from typing import Tuple

import numpy as np
from janus.models import VLChatProcessor
from janus.train.t2i_dataset import TextImageDataset
from janus.train.text_dataset import TextDataset
from janus.train.vqa_dataset import VqaDataset

import mindspore as ms
from mindspore.dataset.transforms import vision

# from PIL import Image


logger = logging.getLogger(__name__)


class UnifiedDataset:
    def __init__(
        self,
        vl_chat_processor: VLChatProcessor,
        max_token_length: int = 1024,
        image_size: int = 384,
        null_prompt_prob: float = 0.0,
        vqa_data_dir: str = None,
        text_qa_data_dir: str = None,
        t2i_csv_path: str = None,
        t2i_data_path: str = None,
        num_samples_vqa: int = -1,
        num_samples_puretext: int = -1,
        num_samples_t2i: int = -1,
    ) -> None:
        self.interpolation_mode = vision.Inter.ANTIALIAS  # vision.Inter.BICUBIC
        self.null_prompt_prob = null_prompt_prob
        self.max_token_length = max_token_length
        self.meta_data = []
        self.image_size = image_size
        self.idx_offsets = []

        # load vqa part
        self.vqa_dataset = VqaDataset(
            dataset_name="medical-vqa",
            data_dir=vqa_data_dir,
            vl_chat_processor=vl_chat_processor,
            max_token_length=max_token_length,
            num_samples=num_samples_vqa,
        )
        data_type_dict = {"data_type": 0}
        dataset = [{**i, **data_type_dict} for i in self.vqa_dataset.dataset]
        self.meta_data += dataset
        offset = len(dataset)
        self.idx_offsets.append(offset)

        # load text dataset
        self.text_dataset = TextDataset(
            dataset_name="pubmedqa",
            data_dir=text_qa_data_dir,
            vl_chat_processor=vl_chat_processor,
            max_token_length=max_token_length,
            num_samples=num_samples_puretext,
        )
        data_type_dict = {"data_type": 1}
        dataset = [{**i, **data_type_dict} for i in self.text_dataset.dataset]
        self.meta_data += dataset
        offset = len(dataset)
        self.idx_offsets.append(offset)

        # load t2i part
        self.t2i_dataset = TextImageDataset(
            csv_path=t2i_csv_path,
            data_dir=t2i_data_path,
            vl_chat_processor=vl_chat_processor,
            max_token_length=max_token_length,
            image_size=image_size,
            null_prompt_prob=null_prompt_prob,
            num_samples=num_samples_t2i,
        )
        data_type_dict = {"data_type": 2}
        dataset = [{**i, **data_type_dict} for i in self.t2i_dataset.dataset]
        self.meta_data += dataset

        self.length = len(self.meta_data)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        record = self.meta_data[idx]
        task_type = record["data_type"]
        if task_type == 0:
            (
                task_type,
                input_ids,
                labels,
                attention_mask,
                image_seq_mask,
                image,
            ) = self.vqa_dataset.__getitem__(idx)
        elif task_type == 1:
            (
                task_type,
                input_ids,
                labels,
                attention_mask,
            ) = self.text_dataset.__getitem__(idx)
            image_seq_mask = np.zeros((self.max_token_length), dtype=np.bool_)
            image = np.zeros(
                (1, 3, self.image_size, self.image_size), dtype=np.float32
            )  # t 3 h w, t is reserved for future video generation
        elif task_type == 2:
            (
                task_type,
                input_ids,
                labels,
                attention_mask,
                image_seq_mask,
                image,
            ) = self.t2i_dataset.__getitem__(idx)
        else:
            raise NotImplementedError
        return task_type, input_ids, labels, attention_mask, image_seq_mask, image


def create_dataloader_unified(
    vl_chat_processor: VLChatProcessor,
    max_token_length: int = 1024,
    image_size: int = 384,
    null_prompt_prob: float = 0.0,
    batch_size: int = 1,
    shuffle: bool = True,
    num_parallel_workers: int = 8,
    rank: int = 0,
    rank_size: int = 1,
    vqa_data_dir: str = None,
    text_qa_data_dir: str = None,
    t2i_csv_path: str = None,
    t2i_data_path: str = None,
    num_samples_vqa: int = -1,
    num_samples_puretext: int = -1,
    num_samples_t2i: int = -1,
):
    dataset = UnifiedDataset(
        vl_chat_processor,
        max_token_length,
        image_size,
        null_prompt_prob,
        vqa_data_dir,
        text_qa_data_dir,
        t2i_csv_path,
        t2i_data_path,
        num_samples_vqa,
        num_samples_puretext,
        num_samples_t2i,
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
