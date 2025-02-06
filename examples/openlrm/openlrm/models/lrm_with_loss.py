import logging

logger = logging.getLogger("")

import mindspore as ms
from mindspore import Tensor, mint, nn

from .modeling_lrm import ModelLRM


class ModelLRMWithLoss(nn.Cell):
    """The training pipeline for LRM model."""

    def __init__(self, cfg, use_recompute=False):
        super().__init__()
        self.cfg = cfg
        self.lrm_generator = ModelLRM(**cfg.model, use_recompute=use_recompute)

        self.input_size = cfg.dataset.source_image_res
        self.render_image_res_low = cfg.dataset.render_image.low
        self.render_image_res_high = cfg.dataset.render_image.high

        # losses
        self.pixel_loss, self.lpips, self.tv_loss = self._build_loss_fn(cfg)

        self.validation_step_outputs = []

    def _build_loss_fn(self, cfg):
        from openlrm.losses import LPIPSLoss, PixelLoss, TVLoss

        pixel_loss_fn = PixelLoss()
        perceptual_loss_fn = LPIPSLoss(prefech=True)
        tv_loss_fn = TVLoss()
        return pixel_loss_fn, perceptual_loss_fn, tv_loss_fn

    def construct(
        self,
        source_camera: Tensor,
        render_camera: Tensor,
        source_image: Tensor,
        render_image: Tensor,
        render_anchors: Tensor,
        render_full_resolutions: Tensor,
        render_bg_colors: Tensor,
    ) -> Tensor:
        """For training, only return loss."""

        # Infer image2triplane + render views
        planes, images_rgb = self.lrm_generator(
            image=source_image,
            source_camera=source_camera,
            render_cameras=render_camera,
            render_anchors=render_anchors,
            render_resolutions=render_full_resolutions,
            render_bg_colors=render_bg_colors,
            render_region_size=self.cfg.dataset.render_image.region,
        )

        # compute loss
        loss = self.compute_loss(planes, images_rgb, render_image)

        return loss

    def compute_loss(self, planes, images_rgb, render_image):
        # NOTE: the rgb value range of OpenLRM is [0, 1]
        # loss calculation
        loss = ms.Tensor(0.0)
        loss_pixel = None
        loss_perceptual = None
        loss_tv = None

        if self.cfg.train.loss.pixel_weight > 0.0:
            loss_pixel = self.pixel_loss(images_rgb, render_image)
            loss += loss_pixel * self.cfg.train.loss.pixel_weight
        if self.cfg.train.loss.perceptual_weight > 0.0:
            loss_perceptual = self.lpips(images_rgb, render_image)
            loss += loss_perceptual * self.cfg.train.loss.perceptual_weight
        if self.cfg.train.loss.tv_weight > 0.0:
            loss_tv = self.tv_loss(planes)
            loss += loss_tv * self.cfg.train.loss.tv_weight

        logger.info(f"loss: {loss}, loss pixel: {loss_pixel}, loss lpips: {loss_perceptual}, loss tv: {loss_tv}")

        return loss


class ModelLRMWithLossEval(nn.Cell):
    """The evaluation pipeline for LRM model."""

    def __init__(self, cfg, use_recompute=False):
        super().__init__()
        self.cfg = cfg
        self.lrm_generator = ModelLRM(**cfg.model, use_recompute=use_recompute)

        self.input_size = cfg.dataset.source_image_res
        self.render_image_res_low = cfg.dataset.render_image.low
        self.render_image_res_high = cfg.dataset.render_image.high

        # losses
        self.pixel_loss, self.lpips, self.tv_loss = self._build_loss_fn(cfg)

    def _build_loss_fn(self, cfg):
        from openlrm.losses import LPIPSLoss, PixelLoss, TVLoss

        pixel_loss_fn = PixelLoss()
        perceptual_loss_fn = LPIPSLoss(prefech=True)
        tv_loss_fn = TVLoss()
        return pixel_loss_fn, perceptual_loss_fn, tv_loss_fn

    def forward_nocalloss(
        self,
        source_camera: Tensor,  # [1, 16]
        render_camera: Tensor,  # [1, M, 25]
        source_image: Tensor,  # [1, C, H, W]
        render_size: int,
    ) -> Tensor:
        """For evaluate()."""
        planes = self.lrm_generator.forward_planes(source_image, source_camera)
        N = planes.shape[0]  # N=1
        render_cameras = render_camera.tile((N, 1, 1))  # [N, M, 25]
        render_anchors = mint.zeros((N, render_cameras.shape[1], 2), dtype=ms.float32)
        render_resolutions = mint.ones((N, render_cameras.shape[1], 1), dtype=ms.float32) * render_size
        render_bg_colors = mint.ones((N, render_cameras.shape[1], 1), dtype=ms.float32) * 1.0

        frames = []
        for i in range(render_camera.shape[1]):
            frames.append(
                self.lrm_generator.synthesizer(
                    planes=planes,
                    cameras=render_cameras[:, i : i + 1],
                    anchors=render_anchors[:, i : i + 1],
                    resolutions=render_resolutions[:, i : i + 1],
                    bg_colors=render_bg_colors[:, i : i + 1],
                    region_size=render_size,
                )
            )
        # merge frames
        outputs = mint.cat([r["images_rgb"] for r in frames], dim=1)
        render_images = mint.clamp(outputs, 0.0, 1.0)
        return render_images

    def construct(
        self,
        source_camera: Tensor,
        render_camera: Tensor,
        source_image: Tensor,
        render_image: Tensor,
        render_anchors: Tensor,
        render_full_resolutions: Tensor,
        render_bg_colors: Tensor,
    ) -> Tensor:
        """For evaluation, return loss, pred, and target."""

        # compute loss
        loss, model_pred, target = self.compute_loss(
            source_camera,
            render_camera,
            source_image,
            render_image,
            render_anchors,
            render_full_resolutions,
            render_bg_colors,
        )

        return loss, model_pred, target

    def compute_loss(
        self,
        source_camera: Tensor,
        render_camera: Tensor,
        source_image: Tensor,
        render_image: Tensor,
        render_anchors: Tensor,
        render_full_resolutions: Tensor,
        render_bg_colors: Tensor,
    ):
        # Infer image2triplane + render views
        planes, images_rgb = self.lrm_generator(
            image=source_image,
            source_camera=source_camera,
            render_cameras=render_camera,
            render_anchors=render_anchors,
            render_resolutions=render_full_resolutions,
            render_bg_colors=render_bg_colors,
            render_region_size=self.cfg.dataset.render_image.region,
        )

        # loss calculation
        loss = ms.Tensor(0.0)
        loss_pixel = None
        loss_perceptual = None
        loss_tv = None

        if self.cfg.train.loss.pixel_weight > 0.0:
            loss_pixel = self.pixel_loss(images_rgb, render_image)
            loss += loss_pixel * self.cfg.train.loss.pixel_weight
        if self.cfg.train.loss.perceptual_weight > 0.0:
            loss_perceptual = self.lpips(images_rgb, render_image)
            loss += loss_perceptual * self.cfg.train.loss.perceptual_weight
        if self.cfg.train.loss.tv_weight > 0.0:
            loss_tv = self.tv_loss(planes)
            loss += loss_tv * self.cfg.train.loss.tv_weight

        logger.info(f"loss: {loss}, loss pixel: {loss_pixel}, loss lpips: {loss_perceptual}, loss tv: {loss_tv}")

        return loss, images_rgb, render_image
