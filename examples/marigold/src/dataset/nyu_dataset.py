import numpy as np

import mindspore

from .base_depth_dataset import BaseDepthDataset, DatasetMode, DepthFileNameMode


class NYUDataset(BaseDepthDataset):
    def __init__(
        self,
        eigen_valid_mask: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            # NYUv2 dataset parameter
            min_depth=1e-3,
            max_depth=10.0,
            has_filled_depth=True,
            name_mode=DepthFileNameMode.rgb_id,
            **kwargs,
        )

        self.eigen_valid_mask = eigen_valid_mask

    def __getitem__(self, index):
        outputs = super().__getitem__(index)
        if DatasetMode.RGB_ONLY == self.mode:
            outputs["rgb_int"] = outputs["rgb_int"][..., 16:464, 32:608]
            outputs["rgb_norm"] = outputs["rgb_norm"][..., 16:464, 32:608]
        else:
            outputs["rgb_int"] = outputs["rgb_int"][..., 16:464, 32:608]
            outputs["rgb_norm"] = outputs["rgb_norm"][..., 16:464, 32:608]
            outputs["depth_raw_linear"] = outputs["depth_raw_linear"][..., 16:464, 32:608]
            outputs["depth_filled_linear"] = outputs["depth_filled_linear"][..., 16:464, 32:608]
            outputs["valid_mask_raw"] = outputs["valid_mask_raw"][..., 16:464, 32:608]
            outputs["valid_mask_filled"] = outputs["valid_mask_filled"][..., 16:464, 32:608]
            outputs["depth_raw_norm"] = outputs["depth_raw_norm"][..., 16:464, 32:608]
            outputs["depth_filled_norm"] = outputs["depth_filled_norm"][..., 16:464, 32:608]
        return outputs

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode NYU depth
        depth_decoded = depth_in / 1000.0
        return depth_decoded

    def _get_valid_mask(self, depth: mindspore.Tensor):
        valid_mask = super()._get_valid_mask(depth)

        # Eigen crop for evaluation
        if self.eigen_valid_mask:
            eval_mask = np.zeros_like(valid_mask.squeeze())
            eval_mask[45:471, 41:601] = 1
            eval_mask = eval_mask[np.newaxis, ...]
            eval_mask = eval_mask.astype(np.bool_)
            valid_mask = np.logical_and(valid_mask, eval_mask)

        return valid_mask
