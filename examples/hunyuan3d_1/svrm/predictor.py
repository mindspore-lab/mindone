# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import math
import os
import time

import numpy as np
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import mint
from mindspore.dataset import transforms, vision

from mindone.safetensors.mindspore import load_file

from .ldm.util import instantiate_from_config


class MV23DPredictor(object):
    def __init__(self, ckpt_path, cfg_path, elevation=15, number_view=60, render_size=256) -> None:
        self.elevation = elevation
        self.number_view = number_view
        self.render_size = render_size

        # horizontal spherical rotation
        self.elevation_list = [0, 0, 0, 0, 0, 0, 0]
        self.azimuth_list = [0, 60, 120, 180, 240, 300, 0]

        st = time.time()
        self.model = self.init_model(ckpt_path, cfg_path)
        print(f"=====> mv23d model init time: {time.time() - st}")

        self.input_view_transform = transforms.Compose(
            [
                vision.Resize(504, interpolation=vision.Inter.BICUBIC),
                vision.ToTensor(),
                vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), is_hwc=False),
            ]
        )  # output numpy
        # vision.ToTensor()
        # input is an image of PIL type or a Numpy array in [0, 255] in the format of <H, W, C>,
        # output is a Numpy array in the range of [0.0, 1.0] with the format of <C, H, W>
        # vision.ToTensor change the format from HWC to CHW, so normalize have to specify `is_hwc=False`

    def init_model(self, ckpt_path, cfg_path):
        config = OmegaConf.load(cfg_path)
        model = instantiate_from_config(config.model)  # SVRMModel

        if ckpt_path.endswith(".ckpt"):  # if converted savetensors to ms.ckpt
            state_dict = ms.load_checkpoint(ckpt_path)
        elif ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
        else:
            raise AssertionError(
                f"Cannot recognize checkpoint file {ckpt_path}, only support MS *.ckpt and *.safetensors"
            )

        # check loading keys:
        model_state_dict = {k: v for k, v in model.parameters_and_names()}
        loaded_keys = list(state_dict.keys())
        expexted_keys = list(model_state_dict.keys())
        original_loaded_keys = loaded_keys
        missing_keys = list(set(expexted_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expexted_keys))
        mismatched_keys = []
        for checkpoint_key in original_loaded_keys:
            if (
                checkpoint_key in model_state_dict
                and checkpoint_key in state_dict
                and state_dict[checkpoint_key].shape != model_state_dict[checkpoint_key].shape
            ):
                mismatched_keys.append(
                    (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[checkpoint_key].shape)
                )

        print(
            f"Loading SVRMModel...\nmissing_keys: {missing_keys}, \nunexpected_keys: {unexpected_keys}, \nmismatched_keys: {mismatched_keys}"
        )

        print(f"state_dict.dtype {state_dict[loaded_keys[0]].dtype}")  # float16
        print(f"model.dtype {model.dtype}")
        if state_dict[loaded_keys[0]].dtype != model.dtype:
            model = model.to(state_dict[loaded_keys[0]].dtype)
        print(f"Use {model.dtype} for inference.")
        param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict, strict_load=True)
        print(f"Loaded checkpoint: param_not_load {param_not_load}, ckpt_not_load {ckpt_not_load}")

        model = model.set_train(False)

        print("Load model successfully")

        return model

    def create_camera_to_world_matrix(self, elevation, azimuth, cam_dis=1.5):
        # elevation azimuth are radians
        # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
        # Right-angled Cartesian coordinate system: right x, forward y , up z
        x = np.cos(elevation) * np.cos(azimuth)
        y = np.cos(elevation) * np.sin(azimuth)
        z = np.sin(elevation)

        # Calculate camera position, target, and up vectors
        camera_pos = np.array([x, y, z]) * cam_dis
        target = np.array([0, 0, 0])  # object at world origin
        up = np.array([0, 0, 1])

        # Construct view matrix
        # TODO to confirm Camera coordinate... Unity cam coord??: right x, up y???, forward z(look at obj)
        forward = target - camera_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        new_up = np.cross(right, forward)
        new_up /= np.linalg.norm(new_up)
        cam2world = np.eye(4)
        cam2world[:3, :3] = np.array([right, new_up, -forward]).T
        cam2world[:3, 3] = camera_pos
        return cam2world

    def refine_mask(self, mask, k=16):
        mask /= 255.0
        boder_mask = (mask >= -math.pi / 2.0 / k + 0.5) & (mask <= math.pi / 2.0 / k + 0.5)
        mask[boder_mask] = 0.5 * np.sin(k * (mask[boder_mask] - 0.5)) + 0.5
        mask[mask < -math.pi / 2.0 / k + 0.5] = 0.0
        mask[mask > math.pi / 2.0 / k + 0.5] = 1.0
        return (mask * 255.0).astype(np.uint8)

    def load_images_and_cameras(self, input_imgs, elevation_list, azimuth_list):
        input_image_list = []
        input_cam_list = []
        for input_view_image, elevation, azimuth in zip(input_imgs, elevation_list, azimuth_list):
            input_view_image = self.input_view_transform(input_view_image)[0]  # tuple(CHW)
            input_image_list.append(ms.Tensor(input_view_image))

            input_view_cam_pos = self.create_camera_to_world_matrix(np.radians(elevation), np.radians(azimuth))
            input_view_cam_intrinsic = np.array([35.0 / 32, 35.0 / 32, 0.5, 0.5])
            input_view_cam = ms.Tensor(
                np.concatenate([input_view_cam_pos.reshape(-1), input_view_cam_intrinsic], 0)  # 4*4+4=20
            ).float()
            input_cam_list.append(input_view_cam)

        input_images = mint.stack(input_image_list, dim=0)  # [B,C,H,W]
        input_cams = mint.stack(input_cam_list, dim=0)  # [N, 20]
        return input_images, input_cams

    def load_data(self, input_imgs):
        assert (6 + 1) == len(input_imgs)

        input_images, input_cams = self.load_images_and_cameras(input_imgs, self.elevation_list, self.azimuth_list)
        input_cams[-1, :] = 0  # for user input cond view

        data = {}
        data["input_view"] = input_images.unsqueeze(0)  # 1 7 3 504 504
        data["input_view_cam"] = input_cams.unsqueeze(0)  # 1 7 20
        return data

    def predict(
        self,
        intput_imgs,
        save_dir="outputs/",
        image_input=None,
        target_face_count=10000,
        do_texture_mapping=True,
    ):
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir)

        self.model.export_mesh_with_uv(
            data=self.load_data(intput_imgs),
            out_dir=save_dir,
            target_face_count=target_face_count,
            do_texture_mapping=do_texture_mapping,
        )
