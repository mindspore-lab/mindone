from abc import ABC, abstractmethod

import numpy as np

from mindspore import Tensor, mint, tensor

from ..models.mmdit import MMDiTModel


def pack(x: Tensor, patch_size: int = 2) -> Tensor:
    b, c, t, h, w = x.shape
    # b c t (h ph) (w pw) -> b (t h w) (c ph pw)
    return (
        x.reshape(b, c, t, h // patch_size, patch_size, w // patch_size, patch_size)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(b, -1, c * patch_size * patch_size)
    )


def get_oscillation_gs(guidance_scale: float, i: int, force_num=10):
    """
    get oscillation guidance for cfg.

    Args:
        guidance_scale: original guidance value
        i: denoising step
        force_num: before which don't apply oscillation
    """
    return guidance_scale if i < force_num or (i >= force_num and i % 2 == 0) else 1.0


class Denoiser(ABC):
    @abstractmethod
    def denoise(self, model: MMDiTModel, img: np.ndarray, timesteps: list[float], guidance: float, **kwargs) -> Tensor:
        """Denoise the input."""

    @abstractmethod
    def prepare_guidance(self, text: list[str], **kwargs) -> tuple[list[str], dict[str, Tensor]]:
        """Prepare the guidance for the model. This method will alter the text."""


class I2VDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, img: Tensor, timesteps: list[float], guidance: float, **kwargs) -> Tensor:
        guidance_img = kwargs.pop("guidance_img")

        # cond ref arguments
        masks = kwargs.pop("masks")
        masked_ref = kwargs.pop("masked_ref")

        # oscillation guidance
        text_osci = kwargs.pop("text_osci", False)
        image_osci = kwargs.pop("image_osci", False)
        scale_temporal_osci = kwargs.pop("scale_temporal_osci", False)

        # patch size
        patch_size = kwargs.pop("patch_size", 2)

        guidance_vec = tensor(np.full((img.shape[0],), guidance), dtype=model.dtype)
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # timesteps
            t_vec = tensor(np.full((img.shape[0],), t_curr), dtype=model.dtype)
            b, c, t, w, h = masked_ref.shape
            cond = mint.cat((masks, masked_ref), dim=1)
            cond = pack(cond, patch_size=patch_size)
            kwargs["cond"] = mint.cat([cond, cond, mint.zeros_like(cond)], dim=0)

            # forward preparation
            cond_x = img[: len(img) // 3]

            img = mint.cat([cond_x, cond_x, cond_x], dim=0)
            # forward
            pred = model(
                img=img,
                **kwargs,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            # prepare guidance
            text_gs = get_oscillation_gs(guidance, i) if text_osci else guidance
            image_gs = get_oscillation_gs(guidance_img, i) if image_osci else guidance_img
            cond, uncond, uncond_2 = pred.chunk(3, dim=0)
            if image_gs > 1.0 and scale_temporal_osci:
                # image_gs decrease with each denoising step
                step_upper_image_gs = np.linspace(image_gs, 1.0, len(timesteps))[i]
                # image_gs increase along the temporal axis of the latent video
                image_gs = np.tile(np.linspace(1.0, step_upper_image_gs, t)[None, None, :, None, None], (b, c, 1, h, w))
                image_gs = pack(tensor(image_gs, dtype=model.dtype), patch_size=patch_size)

            # update
            pred = uncond_2 + image_gs * (uncond - uncond_2) + text_gs * (cond - uncond)
            pred = mint.cat([pred, pred, pred], dim=0)

            img = img + (t_prev - t_curr) * pred

        img = img[: len(img) // 3]

        return img

    def prepare_guidance(self, text: list[str], **kwargs) -> tuple[list[str], dict[str, Tensor]]:
        ret = {}

        neg = kwargs.get("neg", None)
        ret["guidance_img"] = kwargs.pop("guidance_img")

        t5_emb = kwargs.get("t5_emb", None)
        neg_t5_emb = kwargs.get("neg_t5_emb", None)
        clip_emb = kwargs.get("clip_emb", None)
        neg_clip_emb = kwargs.get("neg_clip_emb", None)

        # text
        if text:
            if not neg:
                neg = [""] * len(text)
            text = text + neg + neg
        else:
            text = {
                "t5": np.concatenate((t5_emb, neg_t5_emb, neg_t5_emb)),
                "clip": np.concatenate((clip_emb, neg_clip_emb, neg_clip_emb)),
            }
        return text, ret


class DistilledDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, img: Tensor, timesteps: list[float], guidance: float, **kwargs) -> Tensor:
        guidance_vec = tensor(np.full((img.shape[0],), guidance), dtype=model.dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            # timesteps
            t_vec = tensor(np.full((img.shape[0],), t_curr), dtype=model.dtype)
            # forward
            pred = model(img=img, **kwargs, timesteps=t_vec, guidance=guidance_vec)
            # update
            img = img + (t_prev - t_curr) * pred
        return img

    def prepare_guidance(self, text: list[str], **kwargs) -> tuple[list[str], dict[str, Tensor]]:
        return text, {}
