import logging
import os

logger = logging.getLogger("")

import numpy as np
from einops import rearrange
from PIL import Image
# from utils.loss_util import LPIPS
from openlrm.losses.perceptual import LPIPSLoss

import mindspore as ms
from mindspore import Tensor, mint, nn
from mindspore.dataset.vision import ToPIL

from mindone.utils.config import instantiate_from_config
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

    
    def forward_loss_local_step(self, data):

        source_camera = data['source_camera']
        render_camera = data['render_camera']
        source_image = data['source_image']
        render_image = data['render_image']
        render_anchors = data['render_anchors']
        render_full_resolutions = data['render_full_resolutions']
        render_bg_colors = data['render_bg_colors']

        N, M, C, H, W = render_image.shape

        # forward
        outputs = self.model(
            image=source_image,
            source_camera=source_camera,
            render_cameras=render_camera,
            render_anchors=render_anchors,
            render_resolutions=render_full_resolutions,
            render_bg_colors=render_bg_colors,
            render_region_size=self.cfg.dataset.render_image.region,
        )

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

        return outputs, loss, loss_pixel, loss_perceptual, loss_tv

    # TODO
    def train_epoch(self, pbar: tqdm, loader: torch.utils.data.DataLoader, profiler: torch.profiler.profile):
        self.model.train()

        local_step_losses = []
        global_step_losses = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")
        for data in loader:

            logger.debug(f"======== Starting global step {self.global_step} ========")
            with self.accelerator.accumulate(self.model):

                # forward to loss
                outs, loss, loss_pixel, loss_perceptual, loss_tv = self.forward_loss_local_step(data)
                
                # backward
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # track local losses
                local_step_losses.append(torch.stack([
                    _loss.detach() if _loss is not None else ms.Tensor(float('nan'), device=self.device)
                    for _loss in [loss, loss_pixel, loss_perceptual, loss_tv]
                ]))

            # track global step
            if self.accelerator.sync_gradients:
                profiler.step()
                self.scheduler.step()
                logger.debug(f"======== Scheduler step ========")
                self.global_step += 1
                global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
                loss, loss_pixel, loss_perceptual, loss_tv = global_step_loss.unbind()
                loss_kwargs = {
                    'loss': loss.item(),
                    'loss_pixel': loss_pixel.item(),
                    'loss_perceptual': loss_perceptual.item(),
                    'loss_tv': loss_tv.item(),
                }
                self.log_scalar_kwargs(
                    step=self.global_step, split='train',
                    **loss_kwargs
                )
                self.log_optimizer(step=self.global_step, attrs=['lr'], group_ids=[0, 1])
                local_step_losses = []
                global_step_losses.append(global_step_loss)

                # manage display
                pbar.update(1)
                description = {
                    **loss_kwargs,
                    'lr': self.optimizer.param_groups[0]['lr'],
                }
                description = '[TRAIN STEP]' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in description.items() if not math.isnan(v))
                pbar.set_description(description)

                # periodic actions
                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    self.save_checkpoint()
                if self.global_step % self.cfg.val.global_step_period == 0:
                    self.evaluate()
                    self.model.train()
                if self.global_step % self.cfg.logger.image_monitor.train_global_steps == 0:
                    self.log_image_monitor(
                        step=self.global_step, split='train',
                        renders=outs['images_rgb'].detach()[:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        gts=data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    )

                # progress control
                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break

        # track epoch
        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        epoch_loss, epoch_loss_pixel, epoch_loss_perceptual, epoch_loss_tv = epoch_losses.unbind()
        epoch_loss_dict = {
            'loss': epoch_loss.item(),
            'loss_pixel': epoch_loss_pixel.item(),
            'loss_perceptual': epoch_loss_perceptual.item(),
            'loss_tv': epoch_loss_tv.item(),
        }
        self.log_scalar_kwargs(
            epoch=self.current_epoch, split='train',
            **epoch_loss_dict,
        )
        logger.info(
            f'[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: ' + \
                ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in epoch_loss_dict.items() if not math.isnan(v))
        )