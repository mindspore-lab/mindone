from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import threestudio
from omegaconf import DictConfig
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import get_mvp_matrix, get_projection_matrix, get_ray_directions, get_rays, l2norm_np


@dataclass
class RandomCameraDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64  # for train
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 4
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy


class RandomCameraIterableDataset(Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.heights: List[int] = [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        self.widths: List[int] = [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size] if isinstance(self.cfg.batch_size, int) else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1 and len(self.batch_sizes) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.info("Ignoring resolution_milestones since height and width are not changing")
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0) for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range

        # l2norm_np = np.L2Normalize(axis=-1)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        # self.height = self.heights[size_ind]
        # self.width = self.widths[size_ind]
        # self.batch_size = self.batch_sizes[size_ind]
        # self.directions_unit_focal = self.directions_unit_focals[size_ind]
        # threestudio.info(
        #     f"Uncond NOT updated: height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        # )
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):
        # while True:
        #     yield {}
        return self

    def progressive_view(self, global_step):
        r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        self.elevation_range = [
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        ]
        self.azimuth_range = [
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        ]


class RandomCameraDataset:
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views

            # for val, force low res otherwise oom
            self.cfg.eval_height = 128
            self.cfg.eval_width = 128
        else:
            self.n_views = self.cfg.n_test_views

        azimuth_deg: np.array
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = np.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
        else:
            azimuth_deg = np.linspace(0, 360.0, self.n_views)
        elevation_deg: np.array = np.full_like(azimuth_deg, self.cfg.eval_elevation_deg)
        camera_distances: np.array = np.full_like(elevation_deg, self.cfg.eval_camera_distance)

        elevation = elevation_deg * np.pi / 180
        azimuth = azimuth_deg * np.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: np.array = np.stack(
            [
                camera_distances * np.cos(elevation) * np.cos(azimuth),
                camera_distances * np.cos(elevation) * np.sin(azimuth),
                camera_distances * np.sin(elevation),
            ],
            axis=-1,
        )

        # default scene center at origin
        center: np.array = np.zeros_like(camera_positions)
        # default camera up direction as +z
        up: np.array = np.tile(np.array([0, 0, 1], dtype=np.float32)[None, :], (self.cfg.eval_batch_size, 1))

        fovy_deg: np.array = np.full_like(elevation_deg, self.cfg.eval_fovy_deg)
        fovy = fovy_deg * np.pi / 180
        light_positions: np.array = camera_positions

        lookat: np.array = l2norm_np(center - camera_positions)
        right: np.array = l2norm_np(np.cross(lookat, up))
        up = l2norm_np(np.cross(right, lookat))
        c2w3x4: np.array = np.concatenate(
            [np.stack([right, up, -lookat], axis=-1), camera_positions[:, :, None]],
            axis=-1,
        )
        c2w: np.array = np.concatenate([c2w3x4, np.zeros_like(c2w3x4[:, :1])], axis=1)
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: np.array = 0.5 * self.cfg.eval_height / np.tan(0.5 * fovy)
        directions_unit_focal = get_ray_directions(H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0)
        directions: np.array = np.tile(directions_unit_focal[None, :, :, :], (self.n_views, 1, 1, 1))
        directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length[:, None, None, None]

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: np.array = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: np.array = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.fovy_deg = fovy_deg
        self.output_columns = ["rays_o", "rays_d"]

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        b = {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation_deg": self.elevation_deg[index],
            "azimuth_deg": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "fovy_deg": self.fovy_deg[index],
        }
        # create batch dim
        for key in b:
            b.update({key: np.expand_dims(b[key], axis=0)})
        return b["rays_o"].astype(np.float32), b["rays_d"].astype(np.float32)

    def collate(self, batch):
        return batch


@register("random-camera-datamodule")
class RandomCameraDataModule:
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass
