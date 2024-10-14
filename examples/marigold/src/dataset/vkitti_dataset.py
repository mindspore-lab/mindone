import numpy as np

import mindspore

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class VirtualKITTIDataset(BaseDepthDataset):
    def __init__(
        self,
        kitti_bm_crop,  # Crop to KITTI benchmark size
        valid_mask_crop,  # Evaluation mask. [None, garg or eigen]
        **kwargs,
    ) -> None:
        super().__init__(
            # virtual KITTI data parameter
            min_depth=1e-5,
            max_depth=80,  # 655.35
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )
        self.kitti_bm_crop = kitti_bm_crop
        self.valid_mask_crop = valid_mask_crop
        assert self.valid_mask_crop in [
            None,
            "garg",  # set evaluation mask according to Garg  ECCV16
            "eigen",  # set evaluation mask according to Eigen NIPS14
        ], f"Unknown crop type: {self.valid_mask_crop}"

        # Filter out empty depth
        self.filenames = [f for f in self.filenames if "None" != f[1]]

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode vKITTI depth
        depth_decoded = depth_in / 100.0
        return depth_decoded

    @staticmethod
    def vkitti_benchmark_crop(input_img):
        """
        Crop images to KITTI benchmark size
        Args:
            `input_img` (numpy.ndarray): Input image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        KB_CROP_HEIGHT = 320
        KB_CROP_WIDTH = 768

        height, width = input_img.shape[-2:]
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)
        if 2 == len(input_img.shape):
            out = input_img[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        elif 3 == len(input_img.shape):
            out = input_img[
                :,
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]
        return out

    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        if self.kitti_bm_crop:
            rgb_data = {k: self.vkitti_benchmark_crop(v) for k, v in rgb_data.items()}
        return rgb_data

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        depth_data = super()._load_depth_data(depth_rel_path, filled_rel_path)
        if self.kitti_bm_crop:
            depth_data = {k: self.vkitti_benchmark_crop(v) for k, v in depth_data.items()}
        return depth_data

    def _get_valid_mask(self, depth: mindspore.Tensor):
        # reference: https://github.com/cleinc/bts/blob/master/pytorch/bts_eval.py
        valid_mask = super()._get_valid_mask(depth)  # [1, H, W]

        if self.valid_mask_crop is not None:
            eval_mask = np.zeros_like(valid_mask.squeeze()).astype(np.bool_)
            gt_height, gt_width = eval_mask.shape

            if "garg" == self.valid_mask_crop:
                eval_mask[
                    int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            elif "eigen" == self.valid_mask_crop:
                eval_mask[
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1

            eval_mask.reshape(valid_mask.shape)
            valid_mask = np.logical_and(valid_mask, eval_mask)
        return valid_mask
