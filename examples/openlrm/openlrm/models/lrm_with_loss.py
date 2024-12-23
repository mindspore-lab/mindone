import logging
import os

logger = logging.getLogger("")

import numpy as np
from PIL import Image
from openlrm.losses.perceptual import LPIPSLoss

import mindspore as ms
from mindspore import Tensor, mint, nn
from mindspore.dataset.vision import ToPIL

from . import ModelLRM


class ModelLRMWithLoss(nn.Cell):
    """The training pipeline for instant mesh model."""

    def __init__(
        self,
        cfg,
        # lrm_generator_config=None,
        # lrm_ckpt_path=None,  # these under two args are for loading ckpts
        # input_size=256,
        # render_size=192,
    ):
        super().__init__()
        self.cfg = cfg
        self.lrm_generator = ModelLRM(**cfg.model)
        self.lrm_generator.set_train(True)

        self.input_size = cfg.dataset.source_image_res
        self.render_image_res_low = cfg.dataset.render_image.low
        self.render_image_res_high = cfg.dataset.render_image.high

        self.lpips = LPIPSLoss()
        self.pixel_loss, self.lpips, self.tv_loss = self._build_loss_fn(cfg)
        self.topil = ToPIL()
        self.validation_step_outputs = []

    def _build_loss_fn(self, cfg):
        from openlrm.losses import PixelLoss, LPIPSLoss, TVLoss
        pixel_loss_fn = PixelLoss()
        perceptual_loss_fn = LPIPSLoss(prefech=True)
        tv_loss_fn = TVLoss()
        return pixel_loss_fn, perceptual_loss_fn, tv_loss_fn

    def on_fit_start(self):
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, "images_val"), exist_ok=True)

    #TODO: seems no use
    def prepare_validation_batch_data(self, batch, _use_dataloader=False):
        """Used during eval/inference, cast all np input into Tensors.

        Args:
        batch: data that read in from the val dataset.
        """
        lrm_generator_input = {}
        images = batch["source_image"]
        lrm_generator_input["source_camera"] = batch["source_camera"] # [1, 12+4]
        lrm_generator_input["render_camera"] = batch["render_camera"] # [N, 16+9]
        lrm_generator_input["source_image"] = batch["source_image"] # Tensor [N,C,H,W]

        return lrm_generator_input

    def construct(
        self,
        uid: str,
        source_camera: Tensor,
        render_camera: Tensor,
        source_image: Tensor,
        render_image: Tensor,
        render_anchors: Tensor,
        render_full_resolutions: Tensor,
        render_bg_colors: Tensor
    ) -> Tensor:
        """For training, only return loss."""
        N, M, C, H, W = render_image.shape

        # Infer image2triplane + render views
        outputs = self.lrm_generator(
            image=source_image,
            source_camera=source_camera,
            render_cameras=render_camera,
            render_anchors=render_anchors,
            render_resolutions=render_full_resolutions,
            render_bg_colors=render_bg_colors,
            render_region_size=self.cfg.dataset.render_image.region,
        )

        # compute loss
        loss = self.compute_loss(outputs, render_image)

        return loss

    # TODO: to use
    def forward_nocalloss(
        self, 
        source_camera: Tensor,
        render_camera: Tensor,
        source_image: Tensor,
        render_anchors: Tensor,
        render_full_resolutions: Tensor,
        render_bg_colors: Tensor
    ) -> Tensor:
        """For evaluate()."""
        outputs = self.lrm_generator(
            image=source_image,
            source_camera=source_camera,
            render_cameras=render_camera,
            render_anchors=render_anchors,
            render_resolutions=render_full_resolutions,
            render_bg_colors=render_bg_colors,
            render_region_size=self.cfg.dataset.render_image.region,
        )

        images_rgb, images_depth, images_weight = self.lrm_generator(
            images, cameras, render_cameras, render_size, crop_params=None
        )
        render_images = mint.clamp(outputs["images_rgb"], 0.0, 1.0)
        return render_images

    def compute_loss(self, outputs, render_image):
        # NOTE: the rgb value range of OpenLRM is [0, 1]
        # loss calculation
        loss = 0.
        loss_pixel = None
        loss_perceptual = None
        loss_tv = None

        if self.cfg.train.loss.pixel_weight > 0.:
            loss_pixel = self.pixel_loss_fn(outputs['images_rgb'], render_image)
            loss += loss_pixel * self.cfg.train.loss.pixel_weight
        if self.cfg.train.loss.perceptual_weight > 0.:
            loss_perceptual = self.perceptual_loss_fn(outputs['images_rgb'], render_image)
            loss += loss_perceptual * self.cfg.train.loss.perceptual_weight
        if self.cfg.train.loss.tv_weight > 0.: 
            loss_tv = self.tv_loss_fn(outputs['planes'])
            loss += loss_tv * self.cfg.train.loss.tv_weight

        logger.info(f"loss pixel: {loss_pixel}, loss lpips: {loss_perceptual}, loss tv: {loss_tv}")

        return loss #, loss_pixel, loss_perceptual, loss_tv