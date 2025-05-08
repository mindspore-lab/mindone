# import bisect

# import numpy as np
from janus.train.t2i_dataset import TextImageDataset
from janus.train.text_dataset import TextDataset
from janus.train.vqa_dataset import VqaDataset

import mindspore as ms
from mindspore.dataset import DistributedSampler, WeightedRandomSampler


class UnifiedDataset:
    """
    Mixed dataset that outputs pure text, multi-modal and text-to-image data.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            leng = len(e)
            r.append(leng + s)
            s += leng
        return r

    def __init__(self, datasets, default_image_shape=(1, 3, 384, 384), max_token_length=1024):
        self.max_token_length = max_token_length
        self.datasets = datasets
        self.default_image_shape = default_image_shape
        self.num_dataset = len(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __getitem__(self, idx):
        # comment for now, as graph mode mixed-data sft for now requires each batch contain all type of tasks, otherwise graph compilation err.
        # dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        # if dataset_idx == 0:
        #     sample_idx = idx
        # else:
        #     sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # fix 1:1:1 data src
        dataset_idx = int(idx % 3)
        sample_idx = int(idx // 3)

        ret = self.datasets[dataset_idx][sample_idx]

        # # add image and image_seq_mask item to pure text for batching in graph mode
        # can rm, this done in text dataset
        # if dataset_idx == 0:
        #     image = np.zeros(self.default_image_shape, np.float32)
        #     image_seq_mask = np.zeros((self.max_token_length), dtype=np.bool_)
        #     ret += (image_seq_mask, image)

        return ret

    def __len__(self):
        return self.cumulative_sizes[-1]


def create_unified_dataloader_weightrandsamp(
    vl_chat_processor,
    t2i_csv_path=None,
    t2i_data_dir=None,
    t2i_parquet_dir=None,
    text_dataset_name="pubmedqa",
    text_data_dir=None,
    vqa_dataset_name="medical-vqa",
    vqa_data_dir=None,
    max_token_length=1024,
    image_size=384,
    null_prompt_prob=0.0,
    batch_size=1,
    num_parallel_workers=8,
    rank=0,
    rank_size=1,
    num_samples=100,
    sample_ratios=(5, 1, 4),
    shuffle=False,
):
    dataset_t2i = TextImageDataset(
        vl_chat_processor=vl_chat_processor,
        csv_path=t2i_csv_path,
        data_dir=t2i_data_dir,
        parquet_dir=t2i_parquet_dir,
        max_token_length=max_token_length,
        image_size=image_size,
        null_prompt_prob=null_prompt_prob,
        num_samples=num_samples,
    )
    dataset_text = TextDataset(
        vl_chat_processor=vl_chat_processor,
        dataset_name=text_dataset_name,
        data_dir=text_data_dir,
        max_token_length=max_token_length,
        num_samples=num_samples,
    )
    dataset_vqa = VqaDataset(
        vl_chat_processor=vl_chat_processor,
        dataset_name=vqa_dataset_name,
        data_dir=vqa_data_dir,
        max_token_length=max_token_length,
        num_samples=num_samples,
    )
    datasets = [dataset_vqa, dataset_text, dataset_t2i]  # keep the right order
    mixed_dataset = UnifiedDataset(
        datasets=datasets,
        default_image_shape=(1, 3, image_size, image_size),
        max_token_length=max_token_length,
    )

    sample_weights = []
    assert len(sample_ratios) == len(datasets)
    for i in range(len(sample_ratios)):
        weight = sample_ratios[i] * len(mixed_dataset) / len(datasets[i])
        sample_weights += [weight] * len(datasets[i])

    sampler = DistributedSampler(num_shards=rank_size, shard_id=rank, shuffle=False)
    sampler.add_child(WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True))

    dataloader = ms.dataset.GeneratorDataset(
        source=mixed_dataset,
        # sampler=sampler,  # comment for now for graph mode mixed-data sft
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
    )

    # TODO rm this when graph comp can be improved
    assert (
        batch_size % 3 == 0
    ), f"graph mode mixed-data sft for now requires each batch contain all type of tasks, otherwise graph compilation err. But got {batch_size}"

    dataloader = dataloader.batch(batch_size, drop_remainder=True)

    return dataloader
