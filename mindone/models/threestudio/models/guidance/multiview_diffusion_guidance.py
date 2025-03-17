import sys
from dataclasses import dataclass
from typing import Any, Optional

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import Tensor, mint, ops

sys.path.insert(0, "../MVDream/")  # relative to the run path of `launch.py`
from mvdream.camera_utils import normalize_camera
from mvdream.model_zoo import build_model


@threestudio.register("multiview-diffusion-guidance")
class MultiviewDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model_name: str = "sd-v2.1-base-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        ckpt_path: Optional[str] = None  # path to local checkpoint (None for loading from url)
        guidance_scale: float = 50.0
        grad_clip: Optional[Any] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5

    cfg: Config

    def configure(self) -> None:
        threestudio.info("Loading Multiview Diffusion ...")

        self.model = build_model(self.cfg.model_name, ckpt_path=self.cfg.ckpt_path)
        for p in self.model.get_parameters():
            p.requires_grad = False

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        threestudio.info("Loaded Multiview Diffusion!")
        # self.mse_loss = nn.MSELoss(reduction="sum")

    def get_camera_cond(
        self,
        camera: Tensor,
        fovy=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(f"Unknown camera_condition_type={self.cfg.camera_condition_type}")
        return camera

    def encode_images(self, imgs: Tensor) -> Tensor:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))  # [B, 4, 32, 32]
        return latents

    def construct(
        self,
        rgb: Tensor,
        prompt_utils: PromptProcessorOutput,
        elevation: Tensor,
        azimuth: Tensor,
        camera_distances: Tensor,
        c2w: Tensor,
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        **kwargs,
    ):
        camera = c2w
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        if text_embeddings is None:  # FIXME make suere the prompt_utils works here
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Tensor  # B 4 64 64
            if rgb_as_latents:
                latents = ops.interpolate(rgb_BCHW, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = ops.interpolate(
                    rgb_BCHW, (self.cfg.image_size, self.cfg.image_size), mode="bilinear", align_corners=False
                )
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t = ops.randint(self.min_step, self.max_step + 1, (1,), dtype=ms.int32)
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = mint.full([1], timestep, dtype=ms.int32)
        t_expand = t.tile((text_embeddings.shape[0],))

        # predict the noise residual with unet, NO grad!
        with ms._no_grad():
            # add noise
            noise = mint.normal(size=latents.shape)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = mint.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.tile((2, 1)).to(text_embeddings.dtype)
                context = {"context": text_embeddings, "camera": camera, "num_frames": self.cfg.n_view}
            else:
                context = {"context": text_embeddings}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)  # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.cfg.recon_loss:
            # reconstruct x0, using the pretrained 4-view mview diffusion
            latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_text)
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(
                    -1, self.cfg.n_view, *latents_recon_nocfg.shape[1:]
                )
                latents_recon_reshape = latents_recon.view(-1, self.cfg.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std((1, 2, 3, 4), keepdims=True) + 1e-8) / (
                    latents_recon_reshape.std((1, 2, 3, 4), keepdims=True) + 1e-8
                )

                latents_recon_adjust = latents_recon * factor.squeeze(1).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = (
                    self.cfg.recon_std_rescale * latents_recon_adjust + (1 - self.cfg.recon_std_rescale) * latents_recon
                )

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * F.mse_loss(latents, latents_recon, reduction="sum") / latents.shape[0]

            # this only needs to be logged for comparison purposes
            # grad = ops.grad(self.loss_func)(latents, latents_recon)
            # grad = (0.5 / latents.shape[0]) * ops.grad(self.mse_loss)(latents, latents_recon)

        else:
            # Original SDS
            # w(t), sigma_t^2
            w = 1 - self.alphas_cumprod[t]
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = mint.nan_to_num(grad)

            target = (
                latents - grad
            )  # it can be derived from the below eq, if trgt is assigned like this, the grad w/ mse will be grad above
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * mint.nn.functional.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return {
            "loss_sds": loss,
            # "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        threestudio.debug(f"in guidance, now global step: {global_step}")
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
