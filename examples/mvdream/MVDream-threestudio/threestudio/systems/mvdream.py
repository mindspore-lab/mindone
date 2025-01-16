import logging
from dataclasses import dataclass
from typing import List

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot

import mindspore as ms
from mindspore import Tensor, nn, ops

logger = logging.getLogger("")


@threestudio.register("mvdream-system")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # visualize_samples: bool = False
        use_recompute: bool = False

    cfg: Config

    # a safer recompute wrapper, with this doing val during training won't leads to more mem alloc (3% each val-4view run if .recompute)
    def safe_recompute(self, b):
        if (
            not b._has_config_recompute
        ):  # this is to avoid doing recompute on one instance multiple times which causes leakage
            b.recompute()
        if isinstance(b, nn.CellList):
            self.safe_recompute(b[-1])

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.requires_grad = False
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.prompt_utils = self.prompt_processor()

        # below inputs int not tensor, assign during config of the system obj
        self.width = 64
        self.height = 64

        # grad ckpting to save ram
        if self.cfg.use_recompute:
            self.safe_recompute(self.renderer)  # the grad of the geo/bg inside will get recompute
            threestudio.info("initing recompute, if you see this multiple times then it may coz NMEM LEAKAGE...")
        else:
            threestudio.info("NOT using recompute")

    def construct(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        mvp_mtx: Tensor,
        camera_positions: Tensor,
        c2w: Tensor,
        light_positions: Tensor,
        elevation_deg: Tensor,
        azimuth_deg: Tensor,
        camera_distances: Tensor,
        fovy_deg: Tensor,
    ):
        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "c2w": c2w,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "fovy": fovy_deg,
        }
        out = self.renderer(rays_o, rays_d)
        guidance_out = self.guidance(out["comp_rgb"], self.prompt_utils, **batch)

        loss = 0.0

        for name, value in guidance_out.items():
            # self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        # for now these loss are all 0, no need to train
        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError("Normal is required for orientation loss, no normal is found in the output.")
            loss_orient = (out["weights"] * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2).sum() / (
                out["opacity"] > 0
            ).sum()
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            # self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if hasattr(self.cfg.loss, "lambda_eikonal") and self.C(self.cfg.loss.lambda_eikonal) > 0:
            loss_eikonal = ((ops.norm(out["sdf_grad"], p=2, axis=-1) - 1.0) ** 2).mean()
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int):
        with ms._no_grad():
            out = self.renderer(batch[0], batch[1])
        self.save_image_grid(
            f"val/it{self.true_global_step}-{batch_idx}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "opacity" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def test_step(self, batch, batch_idx):
        out = self.renderer(*batch)
        self.save_image_grid(
            f"test-it{self.true_global_step}/{batch_idx}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_train_start(self):
        for name, value in self.cfg.loss.items():
            logger.info(f"train_params/{name}: {self.C(value)}")

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"test-it{self.true_global_step}",
            f"test-it{self.true_global_step}",
            r"(\d+)\.png",
            # save_format="mp4",
            save_format="gif",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
