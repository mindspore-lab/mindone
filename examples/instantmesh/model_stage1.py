import logging
import os

logger = logging.getLogger("")

import numpy as np
from einops import rearrange
from PIL import Image
from utils.loss_util import LPIPS

import mindspore as ms
from mindspore import Tensor, mint, nn
from mindspore.dataset.vision import ToPIL

from mindone.utils.config import instantiate_from_config


class InstantMeshStage1WithLoss(nn.Cell):
    """The training pipeline for instant mesh model."""

    def __init__(
        self,
        lrm_generator_config=None,
        lrm_ckpt_path=None,  # these under two args are for loading ckpts
        input_size=256,
        render_size=192,
    ):
        super().__init__()

        self.input_size = input_size
        self.render_size = render_size
        self.lrm_generator = instantiate_from_config(lrm_generator_config)

        # load pretrained model
        if lrm_ckpt_path is not None:
            logger.info(f"LOADING lrm ckpts from {lrm_ckpt_path} \ninside model_stage1")
            lrm_ckpt_sdict = ms.load_checkpoint(lrm_ckpt_path)
            start_epoch = int(lrm_ckpt_sdict.get("epoch_num", ms.Tensor(0, ms.int32)).asnumpy().item())
            m, u = ms.load_param_into_net(self.lrm_generator, lrm_ckpt_sdict, strict_load=False)
            self.resume_epoch = start_epoch + 1
        else:
            logger.info(
                "NOT loading ckpt inside model_stage1, will load openlrm model as configured in train.py (if applicable)."
            )

        self.lpips = LPIPS()
        self.topil = ToPIL()
        self.validation_step_outputs = []

    def on_fit_start(self):
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, "images_val"), exist_ok=True)

    def prepare_validation_batch_data(self, batch, render_size, _use_dataloader=False):
        """Used during eval/inference, cast all np input into Tensors.

        Args:
        batch: np array, img that read in from the val dataset, which is np.fp32.
        """
        topil = ToPIL()
        lrm_generator_input = {}

        # cast input images np arr from fp32 to as topil() does not take fp32
        images = (batch["input_images"] * 255).astype("uint8")

        # this for: images = v2.functional.resize(images, self.input_size, interpolation=3, antialias=True).clamp(0, 1)
        images = np.array(
            [(topil(img)).resize((320, 320), Image.LANCZOS) for img in images]
        )  # img: 1 h w c; images: n h w c
        images = images.astype("float32") / 255.0
        images = images.clip(min=0, max=1)
        images = np.expand_dims(images, axis=0)
        images = rearrange(images, "b n h w c -> b n c h w")

        input_c2ws = batch["input_c2ws"].flatten(start_dim=-2)
        input_Ks = batch["input_Ks"].flatten(start_dim=-2)

        input_extrinsics = input_c2ws[:, :12]
        input_intrinsics = mint.stack(
            [
                input_Ks[:, 0],
                input_Ks[:, 4],
                input_Ks[:, 2],
                input_Ks[:, 5],
            ],
            dim=-1,
        )
        cameras = mint.cat([input_extrinsics, input_intrinsics], dim=-1)

        lrm_generator_input["cameras"] = cameras

        render_c2ws = batch["render_c2ws"].flatten(start_dim=-2)
        render_Ks = batch["render_Ks"].flatten(start_dim=-2)
        render_cameras = mint.cat([render_c2ws, render_Ks], dim=-1)
        lrm_generator_input["render_cameras"] = render_cameras

        # create batch dim when not using dataloader, presuming bsize==1
        if not _use_dataloader:
            for k, v in lrm_generator_input.items():
                lrm_generator_input[k] = v.unsqueeze(0)

        # assign the proc images at last, which is left as np array, for the ViTProc in eval.py
        lrm_generator_input["images"] = images
        lrm_generator_input["render_size"] = render_size

        return lrm_generator_input

    def construct(
        self,
        images: Tensor,
        cameras: Tensor,
        render_cameras: Tensor,
        target_images: Tensor,
        target_alphas: Tensor,
        render_size: Tensor,
        crop_params: Tensor,
    ) -> Tensor:
        """For training, only return loss."""
        images_rgb, images_depth, images_weight = self.lrm_generator(
            images, cameras, render_cameras, render_size.item(), crop_params  # to int
        )
        render_images = mint.clamp(images_rgb, 0.0, 1.0)
        render_alphas = mint.clamp(images_weight, 0.0, 1.0)

        loss = self.compute_loss(render_images, render_alphas, target_images, target_alphas)

        return loss

    def forward_nocalloss(self, images: Tensor, cameras: Tensor, render_cameras: Tensor, render_size: int) -> Tensor:
        """For evaluate()."""

        images_rgb, images_depth, images_weight = self.lrm_generator(
            images, cameras, render_cameras, render_size, crop_params=None
        )
        render_images = mint.clamp(images_rgb, 0.0, 1.0)
        render_alphas = mint.clamp(images_weight, 0.0, 1.0)
        return render_images, render_alphas

    def compute_loss(self, render_images, render_alphas, target_images, target_alphas):
        # NOTE: the rgb value range of OpenLRM is [0, 1]

        # render_images = render_out['render_images']
        # # TODO move the transform for gt data target_xx back to the dataset proc, BALANCE cpu/npucore proc for higher eff
        # target_images = render_gt['target_images'].to(render_images.dtype)
        target_images = target_images.to(render_images.dtype)
        # render_images = rearrange(render_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        # target_images = rearrange(target_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        b, n, c, h, w = render_images.shape
        render_images = render_images.reshape(b * n, c, h, w) * 2.0 - 1.0
        b, n, c, h, w = target_images.shape
        target_images = target_images.reshape(b * n, c, h, w) * 2.0 - 1.0

        loss_mse = mint.nn.functional.mse_loss(render_images, target_images)
        loss_lpips = 2.0 * mint.mean(self.lpips(render_images, target_images))
        target_alphas = target_alphas.permute((0, 1, 4, 2, 3))  # b n h w c -> b n c h w
        loss_mask = mint.nn.functional.mse_loss(render_alphas, target_alphas)

        logger.info(f"loss mse: {loss_mse}, loss mask: {loss_mask}, loss lpips: {loss_lpips}")

        loss = loss_mse + loss_mask + loss_lpips

        return loss
