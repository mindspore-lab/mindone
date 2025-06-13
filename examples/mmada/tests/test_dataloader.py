import logging
import math

import torch
from parquet import RefinedWebDataset  # Assuming this is from a 'parquet' library
from parquet.loader import CombinedLoader, create_dataloader
from training.data import Text2ImageDataset
from training.imagenet_dataset import ImageNetDataset

logger = logging.getLogger(__name__)


# Placeholder for config object for testing purposes
class MockConfig:
    def __init__(self):
        self.training = MockTrainingConfig()
        self.dataset = MockDatasetConfig()
        self.experiment = MockExperimentConfig()


class MockTrainingConfig:
    def __init__(self):
        self.batch_size_t2i = 4
        self.batch_size_lm = 4
        self.batch_size_mmu = 4
        self.gradient_accumulation_steps = 1
        self.max_train_steps = 1000


class MockDatasetConfig:
    def __init__(self):
        self.preprocessing = MockPreprocessingConfig()
        self.params = MockDatasetParamsConfig()
        self.gen_type = "imagenet1k"  # or "t2i_parquet", "imagenet1k"
        self.und_type = "captioning"  # or "captioning_parquet"
        self.combined_loader_mode = "max_size_cycle"


class MockExperimentConfig:
    def __init__(self):
        self.max_train_examples_t2i = 1000
        self.max_train_examples_mmu = 1000


class MockPreprocessingConfig:
    def __init__(self):
        self.max_seq_length = 77
        self.resolution = 256


class MockDatasetParamsConfig:
    def __init__(self):
        self.train_t2i_shards_path_or_url = "train_datasets/imagenet-1k/data/train/"
        self.train_mmu_shards_path_or_url = "train_datasets/laion-aesthetics-12m-data/{00000..00000}.tar"
        self.train_lm_shards_path_or_url = "train_datasets/falcon-refinedweb/data/*parquet"
        self.num_workers = 1
        self.shuffle_buffer_size = 1000
        self.pin_memory = False
        self.persistent_workers = False
        self.external_caption_path = ""
        self.external_journeydb_caption_path = ""
        self.external_laion12m_caption_path = ""
        self.external_cc12m_caption_path = ""
        self.add_caption_prompt = False
        self.resolution = 256


# Placeholder for create_imagetext_dataloader
def create_imagetext_dataloader(*args, **kwargs):
    logger.warning(
        "Using a dummy create_imagetext_dataloader. This function needs to be properly implemented for actual use."
    )

    # Return a dummy DataLoader for testing
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {"images": torch.randn(3, 256, 256), "input_ids": torch.randint(0, 100, (77,))}

    return torch.utils.data.DataLoader(DummyDataset(), batch_size=kwargs.get("batch_size", 1))


def create_dataloaders(config):
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_t2i_without_accum = config.training.batch_size_t2i
    total_batch_size_t2i = config.training.batch_size_t2i * config.training.gradient_accumulation_steps

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # Data for generation
    if config.dataset.gen_type == "t2i":
        dataset = Text2ImageDataset(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            tokenizer=None,  # we want to get raw texts
            max_seq_length=preproc_config.max_seq_length,
            num_train_examples=config.experiment.max_train_examples_t2i,
            per_gpu_batch_size=config.training.batch_size_t2i,
            global_batch_size=total_batch_size_t2i_without_accum,
            num_workers=dataset_config.num_workers,
            resolution=preproc_config.resolution,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
            external_caption_path=dataset_config.external_caption_path,
            external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
            external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
            external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
        )
        train_dataloader_t2i = dataset.train_dataloader
        num_update_steps_per_epoch = math.ceil(
            train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    elif config.dataset.gen_type == "t2i_parquet":
        # this part relies on the internal packages, which will not be released
        num_update_steps_per_epoch = math.ceil(config.experiment.max_train_examples_t2i / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

        train_dataloader_t2i = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            batch_size=config.training.batch_size_t2i,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        )

    elif config.dataset.gen_type == "imagenet1k":
        dataset_imagenet = ImageNetDataset(
            dataset_config.train_t2i_shards_path_or_url,
            image_size=preproc_config.resolution,
        )
        sampler = None
        shuffle = True

        train_dataloader_t2i = create_dataloader(
            dataset_imagenet,
            column_names=["image", "input_ids", "class_ids"],
            batch_size=config.training.batch_size_t2i,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=dataset_config.num_workers,
        )
        train_dataloader_t2i = train_dataloader_t2i.create_dict_iterator(num_epochs=1, output_numpy=True)
        train_dataloader_t2i.dataset_size = len(dataset_imagenet) // config.training.batch_size_t2i

        for x in train_dataloader_t2i:
            for k, v in x.items():
                if isinstance(v, (list, tuple)):
                    print(k, len(v))
                else:
                    print(k, v.shape)
            break
        num_update_steps_per_epoch = math.ceil(len(dataset_imagenet) / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    else:
        raise ValueError(f"Unsupported dataset type {config.dataset.gen_type}")  # Changed from config.dataset.type

    total_batch_size_mmu_without_accum = config.training.batch_size_mmu
    # Data for image captioning
    if config.dataset.und_type == "captioning":
        dataset_mmu = Text2ImageDataset(
            train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
            tokenizer=None,  # we want to get raw texts
            max_seq_length=preproc_config.max_seq_length,
            num_train_examples=config.experiment.max_train_examples_mmu,
            per_gpu_batch_size=config.training.batch_size_mmu,
            global_batch_size=total_batch_size_mmu_without_accum,
            num_workers=dataset_config.num_workers,
            resolution=preproc_config.resolution,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
            external_caption_path=dataset_config.external_caption_path,
            external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
            external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
            external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
            is_captioning=True,
            add_caption_prompt=dataset_config.add_caption_prompt,
        )
        train_dataloader_mmu = dataset_mmu.train_dataloader
        train_dataloader_mmu.dataset_size = train_dataloader_mmu.num_batches
        for x in train_dataloader_mmu:
            for k, v in x.items():
                if isinstance(v, (list, tuple)):
                    print(k, len(v))
                else:
                    print(k, v.shape)
            break
    elif config.dataset.und_type == "captioning_parquet":
        train_dataloader_mmu = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
            batch_size=config.training.batch_size_mmu,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            is_captioning=True,
        )

    else:
        raise NotImplementedError(f"Unsupported dataset type {config.dataset.und_type}")

    # LLM pure text dataset: RefinedWeb
    dataset_lm = RefinedWebDataset(
        data_path=dataset_config.train_lm_shards_path_or_url,
        rank=0,
        world_size=1,
        num_workers=dataset_config.num_workers,
    )

    train_dataloader_lm = create_dataloader(
        dataset_lm,
        column_names=["input_ids"],
        batch_size=config.training.batch_size_lm,
        sampler=None,
        num_workers=dataset_config.num_workers,
    )
    train_dataloader_lm = train_dataloader_lm.create_dict_iterator(num_epochs=1, output_numpy=True)
    train_dataloader_lm.dataset_size = len(dataset_lm) // config.training.batch_size_lm

    for x in train_dataloader_lm:
        for k, v in x.items():
            if isinstance(v, (list, tuple)):
                print(k, len(v))
            else:
                print(k, v.shape)
        break
    # Combine these dataloaders into a single iterable model
    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "lm_flow": train_dataloader_lm,
        "mmu_flow": train_dataloader_mmu,
    }

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)
    return combined_dataloader


def main():
    config = MockConfig()
    # Add a dummy max_train_examples_t2i and max_train_examples_mmu to the mock config
    config.experiment.max_train_examples_t2i = 1000
    config.experiment.max_train_examples_mmu = 1000

    combined_dataloader = create_dataloaders(config)

    # You can now iterate through combined_dataloader for testing
    print("Successfully created combined_dataloader. You can now iterate through it for testing.")
    # Example of iterating through a few batches
    for i, batch in enumerate(combined_dataloader):
        print(f"Batch {i}:")
        for key, value in batch.items():
            print(f"  {key}: {value.keys()}")
        if i > 2:  # Process a few batches
            break


if __name__ == "__main__":
    main()
