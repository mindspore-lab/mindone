import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(
    os.path.join(__dir__, "../../../../../")
)  # TODO: remove in future when mindone is ready for install
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))  # for loading utils

from transformers import ViTImageProcessor
from utils.camera_util import FOV_to_intrinsics, center_looking_at_camera_pose, get_circular_camera_poses

import mindspore as ms
from mindspore import Tensor
from mindspore.dataset.vision import Inter, Resize, ToPIL


def read_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def read_txt2list(txt_path):
    list_entry = []
    with open(txt_path, "r") as f:
        for line in f:
            x = line[:-1]
            list_entry.append(x)
    return list_entry


def read_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def random_crop_return_params(imgs, height, width):
    """imgs: (b h w c)"""
    assert imgs.shape[1] >= height
    assert imgs.shape[2] >= width
    top = np.random.randint(0, imgs.shape[1] - height + 1)
    left = np.random.randint(
        0, imgs.shape[2] - width + 1
    )  # same as torch left inclusive, right exclusive, caveat: if using random pkg, right is inclusive
    imgs = np.array([img[top : top + height, left : left + width] for img in imgs])
    return imgs, (top, left, height, width)


def crop_with_param(imgs, top, left, height, width):
    return np.array([img[top : top + height, left : left + width] for img in imgs])


class ObjaverseDataset:
    def __init__(
        self,
        root_dir="training_examples/",
        meta_fname="uid_set.pkl",
        input_image_dir="input",
        target_image_dir="input",
        input_view_num=6,
        target_view_num=4,
        input_size=None,
        render_size=None,
        total_view_n=32,
        fov=50,
        camera_rotation=True,
        camera_scaling=False,
    ):
        self.root_dir = Path(root_dir)
        self.input_image_dir = input_image_dir
        self.target_image_dir = target_image_dir
        self.input_view_num = input_view_num
        self.target_view_num = target_view_num
        self.total_view_n = total_view_n
        self.fov = fov
        self.camera_rotation = camera_rotation
        self.camera_scaling = camera_scaling
        self.output_columns = [
            "images",
            "cameras",
            "render_cameras",
            "target_images",
            "target_alphas",
            "render_size",
            "crop_params",
        ]

        if meta_fname == "uid_set.pkl":
            self.paths = read_pickle(os.path.join(root_dir, meta_fname))[-3:]
            # [:1]  # only takes the first scene for debugging
            print("dataset read pickle")
        elif meta_fname.split(".")[-1] == "txt":
            self.paths = read_txt2list(os.path.join(root_dir, meta_fname))
            print("reading the fixed pose target list as the dataset")
        else:
            raise ValueError(f"set up meta_fname {meta_fname} is not matched with the datset proc")

        # dataaug: vit img processor peeled from dino-vit, not learnable thus cannot put in the .construct() unlike torch
        self.img_processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
        self.topil = ToPIL()

        # make tuple as PIL requires
        self.input_size = (input_size, input_size)
        self.render_size = render_size
        print("============= length of dataset %d =============" % len(self.paths))

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color, _is_gt=False):
        """
        replace background pixel with random color in rendering
        """
        pil_img = Image.open(path)  # h w c
        image = np.asarray(pil_img, dtype=np.float32) / 255.0
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)
        return image, alpha

    def prepare_sample_data(self, sample: dict) -> tuple[dict, dict]:
        """
        The prepare_batch_data() in the pl original implmenetaion. Move to here in the dataset, as
        1. let Pil handling and Dino ViT preprocessing input imgs;
        2. ms dataloader only allows tensor data flushing, as defined by the output_columns. Thus cannot put this into construct as pl did.
        """
        lrm_generator_input = {}
        render_gt = {}

        images = sample["input_images"]  # (1 n h w c) in dataloader, (nhwc) if simply _getitem_.

        # requested by topil, which is limited by ms.Resize and pil.resize
        images = np.asarray(images * 255, dtype=np.uint8)

        input_antialias_resizer = Resize(self.input_size, interpolation=Inter.ANTIALIAS)
        images = np.asarray([input_antialias_resizer(self.topil(img)) for img in images])  # img: n h w c
        images = images.astype("float32") / 255.0

        # images = images / 255.0  # for dino-vit processor, it takes fp32 (0, 1)
        images = images.clip(min=0, max=1)

        # requested by vit proc, restore into n c h w
        images = images.transpose(0, 3, 1, 2)  # nhwc -> nchw for antialias input images

        # dino-vit wrapper forward(), moved from the dino-wrapper
        # normalize from fp32 (0, 1) to (-2.1, 2.6)
        images = self.img_processor(
            images=images,
            return_tensors="np",
            do_rescale=False,
            do_resize=False,
        )["pixel_values"]
        lrm_generator_input["images"] = images

        input_c2ws = sample["input_c2ws"].reshape((self.input_view_num, 16))
        input_Ks = sample["input_Ks"].reshape((self.input_view_num, 9))
        target_c2ws = sample["target_c2ws"].reshape((self.target_view_num, 16))
        target_Ks = sample["target_Ks"].reshape((self.target_view_num, 9))
        render_cameras_input = np.concatenate([input_c2ws, input_Ks], axis=-1)
        render_cameras_target = np.concatenate([target_c2ws, target_Ks], axis=-1)
        render_cameras = np.concatenate([render_cameras_input, render_cameras_target], axis=0)  # n_in+n_ta, 25

        input_extrinsics = input_c2ws[:, :12]
        input_intrinsics = np.stack(
            [
                input_Ks[:, 0],
                input_Ks[:, 4],
                input_Ks[:, 2],
                input_Ks[:, 5],
            ],
            axis=-1,
        )
        cameras = np.concatenate([input_extrinsics, input_intrinsics], axis=-1)

        # add noise to input cameras
        cameras = cameras + np.random.rand(*cameras.shape) * 0.04 - 0.02

        lrm_generator_input["cameras"] = cameras.astype("float32")
        lrm_generator_input["render_cameras"] = render_cameras

        # construct target images and alpha channels from input+target
        target_images = np.concatenate([sample["input_images"], sample["target_images"]], axis=0)
        target_alphas = np.concatenate([sample["input_alphas"], sample["target_alphas"]], axis=0)

        target_images = np.asarray(target_images * 255, dtype=np.uint8)
        target_alphas = np.asarray(target_alphas * 255, dtype=np.uint8)

        render_size = np.random.randint(self.render_size, 513)

        # crop and display the correct target img/alpha
        target_antialias_resizer = Resize(render_size, interpolation=Inter.ANTIALIAS)
        target_images = np.asarray([target_antialias_resizer(self.topil(img)) for img in target_images])
        target_images = target_images.astype("float32") / 255.0
        target_images = target_images.clip(min=0, max=1)

        target_alphas = np.asarray(
            [target_antialias_resizer(self.topil(img)) for img in target_alphas]
        )  # (n h w), the resizer squeeze the last dim when it's 1
        target_alphas = target_alphas[..., None]
        target_alphas = target_alphas.astype("float32") / 255.0

        # random crop with get_params implementation
        target_images, crop_params = random_crop_return_params(target_images, self.render_size, self.render_size)
        target_alphas = crop_with_param(target_alphas, *crop_params)

        render_gt["target_images"] = target_images.transpose(
            0, 3, 1, 2
        )  # nhwc -> nchw for calculating loss correctly with the render imgs
        render_gt["target_alphas"] = target_alphas.transpose(0, 3, 1, 2)  # nhwc -> nchw

        lrm_generator_input["render_size"] = render_size
        lrm_generator_input["crop_params"] = crop_params

        return lrm_generator_input, render_gt

    def __getitem__(self, index):
        input_image_path = os.path.join(self.root_dir, self.input_image_dir, self.paths[index])
        target_image_path = os.path.join(self.root_dir, self.target_image_dir, self.paths[index])

        indices = np.random.choice(range(self.total_view_n), self.input_view_num + self.target_view_num, replace=False)
        input_indices = indices[: self.input_view_num]
        target_indices = indices[self.input_view_num :]

        """background color, default: white"""
        bg_white = [1.0, 1.0, 1.0]

        image_list = []
        alpha_list = []
        pose_list = []

        K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join(input_image_path, "meta.pkl"))
        input_cameras = cam_poses
        for idx in input_indices:
            image, alpha = self.load_im(os.path.join(input_image_path, "%03d.png" % idx), bg_white)
            pose = input_cameras[idx]
            pose = np.concatenate([pose, np.asarray([[0, 0, 0, 1]])], axis=0)

            image_list.append(image)
            alpha_list.append(alpha)
            pose_list.append(pose)

        # K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join(input_image_path, 'meta.pkl'))  # duplicate line with above?
        target_cameras = cam_poses
        for idx in target_indices:
            image, alpha = self.load_im(os.path.join(target_image_path, "%03d.png" % idx), bg_white)
            pose = target_cameras[idx]
            pose = np.concatenate([pose, np.asarray([[0, 0, 0, 1]])], axis=0)

            image_list.append(image)
            alpha_list.append(alpha)
            pose_list.append(pose)

        images = np.stack(
            image_list, axis=0, dtype=np.float32
        )  # (6+V, H, W, C), for PIL proc/ms.resizer/cropper in prepare_sample_data(), thus !=[(6+V, 3, H, W)] and it should be uint8 before pass to topil()
        alphas = np.stack(alpha_list, axis=0, dtype=np.float32)  # (6+V, H, W, 1)

        w2cs = np.stack(pose_list, axis=0, dtype=np.float32)  # (6+V, 4, 4)
        c2ws = np.linalg.inv(w2cs).astype(np.float32)

        # random rotation along z axis
        if self.camera_rotation:
            degree = np.random.uniform(0, math.pi * 2)
            rot = np.expand_dims(
                np.asarray(
                    [
                        [np.cos(degree), -np.sin(degree), 0, 0],
                        [np.sin(degree), np.cos(degree), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                ),
                axis=0,
            ).astype(np.float32)
            c2ws = np.matmul(rot, c2ws)

        # random scaling
        if self.camera_scaling and np.random.rand() < 0.5:
            scale = np.random.uniform(0.7, 1.1)
            c2ws[:, :3, 3] *= scale

        # instrinsics of perspective cameras
        K = FOV_to_intrinsics(self.fov)
        Ks = np.tile(np.expand_dims(K, axis=0), (self.input_view_num + self.target_view_num, 1, 1)).astype(np.float32)

        data = {
            "input_images": images[: self.input_view_num],  # (6, H, W, 3)
            "input_alphas": alphas[: self.input_view_num],  # (6, H, W, 1)
            "input_c2ws": c2ws[: self.input_view_num],  # (6, 4, 4)
            "input_Ks": Ks[: self.input_view_num],  # (6, 3, 3)
            # lrm generator input and supervision
            "target_images": images[self.input_view_num :],  # (V, H, W, 3)
            "target_alphas": alphas[self.input_view_num :],  # (V, H, W, 1)
            "target_c2ws": c2ws[self.input_view_num :],  # (V, 4, 4)
            "target_Ks": Ks[self.input_view_num :],  # (V, 3, 3)
        }

        lrm_generator_input, render_gt = self.prepare_sample_data(data)

        return (
            lrm_generator_input["images"],
            lrm_generator_input["cameras"],
            lrm_generator_input["render_cameras"],
            render_gt["target_images"],
            render_gt["target_alphas"],
            lrm_generator_input["render_size"],
            lrm_generator_input["crop_params"],
        )


class ValidationDataset:
    def __init__(
        self,
        root_dir="objaverse/",
        input_view_num=6,
        input_image_size=320,
        fov=30,
    ):
        self.root_dir = Path(root_dir)
        self.input_view_num = input_view_num
        self.input_image_size = input_image_size
        self.fov = fov

        self.paths = sorted(os.listdir(self.root_dir))[-3:]
        print("============= length of dataset %d =============" % len(self.paths))

        cam_distance = 4.0
        azimuths = np.asarray([30, 90, 150, 210, 270, 330])
        elevations = np.asarray([20, -10, 20, -10, 20, -10])
        azimuths = np.deg2rad(azimuths)
        elevations = np.deg2rad(elevations)

        x = cam_distance * np.cos(elevations) * np.cos(azimuths)
        y = cam_distance * np.cos(elevations) * np.sin(azimuths)
        z = cam_distance * np.sin(elevations)

        cam_locations = np.stack([x, y, z], axis=-1)
        cam_locations = Tensor.from_numpy(cam_locations).float()
        c2ws = center_looking_at_camera_pose(cam_locations)
        self.c2ws = c2ws.astype(ms.float32)
        K = FOV_to_intrinsics(self.fov)
        # .unsqueeze(0).tile((6, 1, 1)).float()
        # .astype(np.float32)
        self.Ks = Tensor(np.tile(np.expand_dims(K, axis=0), (6, 1, 1)), ms.float32)

        self.render_c2ws = get_circular_camera_poses(M=8, radius=cam_distance, elevation=20.0).float()
        #  = FOV_to_intrinsics(self.fov)
        # .unsqueeze(0).tile((self.render_c2ws.shape[0], 1, 1)).float()
        self.render_Ks = Tensor(np.tile(np.expand_dims(K, axis=0), (self.render_c2ws.shape[0], 1, 1)), ms.float32)

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        """
        replace background pixel with random color in rendering
        """
        pil_img = Image.open(path)
        pil_img = pil_img.resize((self.input_image_size, self.input_image_size), resample=Image.BICUBIC)

        image = np.asarray(pil_img, dtype=np.float32) / 255.0
        if image.shape[-1] == 4:
            alpha = image[:, :, 3:]
            image = image[:, :, :3] * alpha + color * (1 - alpha)
        else:
            alpha = np.ones_like(image[:, :, :1])

        # comment below as we need (v h w c) in topil, not (v c h w)
        # image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        # alpha = np.asarray(alpha, dtype=np.float32).transpose(2, 0, 1)
        return image, alpha

    def __getitem__(self, index):
        # load data
        input_image_path = os.path.join(self.root_dir, self.paths[index])

        """background color, default: white"""
        bkg_color = [1.0, 1.0, 1.0]

        image_list = []
        alpha_list = []

        for idx in range(self.input_view_num):
            image, alpha = self.load_im(os.path.join(input_image_path, f"{idx: 03d}.png"), bkg_color)
            image_list.append(image)
            alpha_list.append(alpha)

        images = np.stack(image_list, axis=0, dtype=np.float32)  # (6+V, 3, H, W)
        alphas = np.stack(alpha_list, axis=0, dtype=np.float32)  # (6+V, 1, H, W)

        data = {
            "input_images": images,
            "input_alphas": alphas,
            "input_c2ws": self.c2ws,
            "input_Ks": self.Ks,
            "input_image_path": input_image_path,
            "render_c2ws": self.render_c2ws,
            "render_Ks": self.render_Ks,
        }
        return data
