import os

from .base_depth_dataset import BaseDepthDataset, DatasetMode, get_pred_name
from .kitti_dataset import KITTIDataset
from .nyu_dataset import NYUDataset
from .vkitti_dataset import VirtualKITTIDataset

dataset_name_class_dict = {
    "vkitti": VirtualKITTIDataset,
    "nyu_v2": NYUDataset,
    "kitti": KITTIDataset,
}


def get_dataset(cfg_data_split, base_data_dir: str, mode: DatasetMode, dtype, **kwargs) -> BaseDepthDataset:
    if cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=cfg_data_split.filenames,
            dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
            dtype=dtype,
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
