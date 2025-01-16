import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import threestudio
from omegaconf import DictConfig
from threestudio import register
from threestudio.data.uncond import RandomCameraDataModuleConfig, RandomCameraDataset, RandomCameraIterableDataset
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import get_mvp_matrix, get_projection_matrix, get_rays, l2norm_np


@dataclass
class RandomMultiviewCameraDataModuleConfig(RandomCameraDataModuleConfig):
    relative_radius: bool = True
    n_view: int = 1
    zoom_range: Tuple[float, float] = (1.0, 1.0)


class RandomMultiviewCameraIterableDataset(RandomCameraIterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_range = self.cfg.zoom_range
        self.output_columns = [
            "rays_o",
            "rays_d",
            "mvp_mtx",
            "camera_positions",
            "c2w",
            "light_positions",
            "elevation",
            "azimuth",
            "camera_distances",
            "fovy",
        ]

    def __len__(self):
        return self.cfg.n_view

    # the th dataset collate on empty input and curate data for each batch
    # def collate(
    def __next__(
        self,
    ) -> Tuple:
        threestudio.debug("running in randmview dataset collate")
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # sample elevation angles
        elevation_deg: np.array
        elevation: np.array
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = np.repeat(
                (
                    np.random.rand(real_batch_size) * (self.elevation_range[1] - self.elevation_range[0])
                    + self.elevation_range[0]
                ),
                self.cfg.n_view,
                axis=0,
            )
            elevation = elevation_deg * np.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = np.repeat(
                np.arcsin(
                    2
                    * (
                        np.random.rand(real_batch_size) * (elevation_range_percent[1] - elevation_range_percent[0])
                        + elevation_range_percent[0]
                    )
                    - 1.0
                ),
                self.cfg.n_view,
                axis=0,
            )
            elevation_deg = elevation / np.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: np.array
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            np.random.rand(real_batch_size).reshape(-1, 1) + np.arange(self.cfg.n_view).reshape(1, -1)
        ).reshape(-1) / self.cfg.n_view * (self.azimuth_range[1] - self.azimuth_range[0]) + self.azimuth_range[0]
        azimuth = azimuth_deg * np.pi / 180

        # Different from original
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: np.array = np.repeat(
            np.random.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0],
            self.cfg.n_view,
            axis=0,
        )
        fovy = fovy_deg * np.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: np.array = np.repeat(
            np.random.rand(real_batch_size) * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0],
            self.cfg.n_view,
            axis=0,
        )
        if self.cfg.relative_radius:
            scale = 1 / np.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: np.array = np.repeat(
            np.random.rand(real_batch_size) * (self.zoom_range[1] - self.zoom_range[0]) + self.zoom_range[0],
            self.cfg.n_view,
            axis=0,
        )
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom

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
        up: np.array = np.tile(np.array([0, 0, 1], dtype=np.float32)[None, :], (self.batch_size, 1))

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: np.array = np.repeat(
            np.random.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb - self.cfg.camera_perturb,
            self.cfg.n_view,
            axis=0,
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: np.array = np.repeat(
            np.random.randn(real_batch_size, 3) * self.cfg.center_perturb, self.cfg.n_view, axis=0
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: np.array = np.repeat(
            np.random.randn(real_batch_size, 3) * self.cfg.up_perturb, self.cfg.n_view, axis=0
        )
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: np.array = np.repeat(
            np.random.rand(real_batch_size) * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0],
            self.cfg.n_view,
            axis=0,
        )

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: np.array = l2norm_np(
                camera_positions
                + np.repeat(np.random.randn(real_batch_size, 3), self.cfg.n_view, axis=0)
                * self.cfg.light_position_perturb
            )
            # get light position by scaling light direction by light distance
            light_positions: np.array = light_direction * light_distances[:, None]
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = l2norm_np(camera_positions)
            local_x = l2norm_np(
                np.stack(
                    [local_z[:, 1], -local_z[:, 0], np.zeros_like(local_z[:, 0])],
                    axis=-1,
                )
            )
            local_y = l2norm_np(np.cross(local_z, local_x, axis=-1))
            rot = np.stack([local_x, local_y, local_z], axis=-1)
            light_azimuth = np.repeat(
                np.random.rand(real_batch_size) * np.pi - 2 * np.pi, self.cfg.n_view, axis=0
            )  # [-pi, pi]
            light_elevation = np.repeat(
                np.random.rand(real_batch_size) * np.pi / 3 + np.pi / 6, self.cfg.n_view, axis=0
            )  # [pi/6, pi/2]
            light_positions_local = np.stack(
                [
                    light_distances * np.cos(light_elevation) * np.cos(light_azimuth),
                    light_distances * np.cos(light_elevation) * np.sin(light_azimuth),
                    light_distances * np.sin(light_elevation),
                ],
                axis=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(f"Unknown light sample strategy: {self.cfg.light_sample_strategy}")

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
        focal_length: np.array = 0.5 * self.height / np.tan(0.5 * fovy)
        directions: np.array = np.tile(self.directions_unit_focal[None, :, :, :], (self.batch_size, 1, 1, 1))
        directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length[:, None, None, None]

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx: np.array = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: np.array = get_mvp_matrix(c2w, proj_mtx)

        return (
            rays_o.astype(np.float32),
            rays_d.astype(np.float32),
            mvp_mtx.astype(np.float32),
            camera_positions.astype(np.float32),
            c2w.astype(np.float32),
            light_positions.astype(np.float32),
            elevation_deg.astype(np.float32),
            azimuth_deg.astype(np.float32),
            camera_distances.astype(np.float32),
            fovy_deg.astype(np.float32),
        )


class BatchSampler:
    """
    Batch Sampler
    """

    def __init__(self, lens, batch_size, num_device=1):
        self._lens = lens
        self._batch_size = batch_size * num_device

    def _create_ids(self):
        return list(range(self._lens))

    def __iter__(self):
        ids = self._create_ids()
        batches = [ids[i : i + self._batch_size] for i in range(0, len(ids), self._batch_size)]
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. " "This has some randomness across epochs")


@register("random-multiview-camera-datamodule")
class RandomMultiviewCameraDataModule:
    cfg: RandomMultiviewCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "train"]:
            self.train_dataset = RandomMultiviewCameraIterableDataset(self.cfg)
        if stage in [None, "train", "validate"]:  # TODO to enable this, needs to make the valset operation in np
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass
