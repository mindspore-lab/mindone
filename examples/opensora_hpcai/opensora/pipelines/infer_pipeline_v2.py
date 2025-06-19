import os
from math import ceil, sqrt
from typing import Literal, Optional, Tuple, Union

import numpy as np
from einops import rearrange, repeat

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn, tensor

from ..models.mmdit import MMDiTModel
from ..utils.inference import collect_references_batch, prepare_inference_condition
from ..utils.sampling import SamplingOption
from .denoisers import Denoiser
from .utils_v2 import get_res_lin_function, time_shift

__all__ = ["InferPipelineV2"]


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    num_frames: int,
    shift_alpha: Optional[float] = None,
    base_shift: float = 1,
    max_shift: float = 3,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = np.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        if shift_alpha is None:
            # estimate mu based on linear estimation between two points
            # spatial scale
            shift_alpha = get_res_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
            # temporal scale
            shift_alpha *= sqrt(num_frames)
        # calculate shifted timesteps
        timesteps = time_shift(shift_alpha, timesteps)

    return timesteps.tolist()


def unpack(x: Tensor, height: int, width: int, num_frames: int, patch_size: int = 2) -> Tensor:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    h, w = ceil(height / D), ceil(width / D)
    # b (t h w) (c ph pw) -> b c t (h ph) (w pw)
    return (
        x.reshape(x.shape[0], num_frames, h, w, -1, patch_size, patch_size)
        .permute(0, 4, 1, 2, 5, 3, 6)
        .reshape(x.shape[0], -1, num_frames, h * patch_size, w * patch_size)
    )


class InferPipelineV2:
    def __init__(
        self,
        model: MMDiTModel,
        model_ae,
        denoiser: Denoiser,
        t5_encoder: Optional[nn.Cell] = None,
        clip_encoder: Optional[nn.Cell] = None,
        num_inference_steps=50,
    ):
        self.model = model
        self.vae = model_ae
        self.denoiser = denoiser
        self.t5_enc = t5_encoder
        self.clip_enc = clip_encoder
        self.num_steps = num_inference_steps

    def vae_decode(
        self,
        x: Tensor,
        num_frames: int,
        cond_type: Literal["t2i", "t2v", "i2v", "v2v"] = "t2i",
        is_causal_vae: bool = False,
    ) -> Tensor:
        x = self.vae.decode(x.to(self.vae.dtype)).to(mstype.float32)
        x = x[:, :, :num_frames]  # image

        # remove the duplicate frames
        if not is_causal_vae:
            if cond_type == "i2v_head":
                pad_len = self.vae.compression[0] - 1
                x = x[:, :, pad_len:]
            elif cond_type == "i2v_tail":
                pad_len = self.vae.compression[0] - 1
                x = x[:, :, :-pad_len]
            elif cond_type == "i2v_loop":
                pad_len = self.vae.compression[0] - 1
                x = x[:, :, pad_len:-pad_len]

        return x

    @staticmethod
    def _prepare_inputs(
        t5,
        clip,
        img: np.ndarray,
        prompt: Union[str, list[str]],
        t5_emb: Optional[Tensor] = None,
        clip_emb: Optional[Tensor] = None,
        patch_size: int = 2,
        dtype: mstype.Type = mstype.float32,
    ) -> dict[str, Tensor]:
        """
        Prepare the input for the model.

        Args:
            t5 (HFEmbedder): The T5 model.
            clip (HFEmbedder): The CLIP model.
            img (Tensor): The image tensor.
            prompt (str | list[str]): The prompt(s).

        Returns:
            dict[str, Tensor]: The input dictionary.

            img_ids: used for positional embedding in T,H,W dimensions later
            text_ids: for positional embedding, but set to 0 for now since our text encoder already encodes positional information
        """
        bs = len(prompt or t5_emb)
        t, h, w = img.shape[-3:]

        img = rearrange(img, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=patch_size, pw=patch_size)
        if img.shape[0] != bs:
            img = repeat(img, "b ... -> (repeat b) ...", repeat=bs // img.shape[0])

        img_ids = np.zeros((t, h // patch_size, w // patch_size, 3), dtype=np.int32)
        img_ids[..., 0] = img_ids[..., 0] + np.arange(t)[:, None, None]
        img_ids[..., 1] = img_ids[..., 1] + np.arange(h // patch_size)[None, :, None]
        img_ids[..., 2] = img_ids[..., 2] + np.arange(w // patch_size)[None, None, :]
        img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

        # Encode the tokenized prompts
        txt, vec = t5_emb, clip_emb
        if prompt:
            assert t5 is not None and clip is not None
            txt = t5(prompt)
            vec = clip(prompt)

        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)

        txt_ids = np.zeros((bs, txt.shape[1], 3), dtype=np.int32)

        return {
            "img": tensor(img, dtype=dtype),
            "img_ids": tensor(img_ids, dtype=dtype),
            "txt": tensor(txt, dtype=dtype),
            "txt_ids": tensor(txt_ids, dtype=dtype),
            "y_vec": tensor(vec, dtype=dtype),
        }

    @staticmethod
    def _get_noise(
        num_samples: int, height: int, width: int, num_frames: int, patch_size: int = 2, channel: int = 16
    ) -> np.ndarray:
        """
        Generate a noise tensor.

        Args:
            num_samples (int): Number of samples.
            height (int): Height of the noise tensor.
            width (int): Width of the noise tensor.
            num_frames (int): Number of frames.

        Returns:
            np.ndarray: The noise tensor.
        """
        D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))  # FIXME
        # FIXME: numpy generator with seed
        return np.random.randn(
            num_samples,
            channel,
            num_frames,
            # allow for packing
            patch_size * ceil(height / D),
            patch_size * ceil(width / D),
        ).astype(np.float32)

    def __call__(
        self,
        text: list[str],
        neg_text: list[str],
        t5_emb: Optional[np.ndarray],
        neg_t5_emb: Optional[np.ndarray],
        clip_emb: Optional[np.ndarray],
        neg_clip_emb: Optional[np.ndarray],
        opt: SamplingOption,
        neg: list[str] = None,
        frames_mask: Optional[Tensor] = None,
        additional_kwargs: Optional[dict] = None,
        patch_size: int = 2,
        cond_type: Literal["t2i", "t2v", "i2v", "v2v"] = "t2i",
        references: Optional[list[str]] = None,
    ) -> Tuple[Union[Tensor, None], Tensor]:
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        # step 1: adjust the number of frames based on whether VAE is causal or not
        if opt.is_causal_vae:
            num_frames = 1 if opt.num_frames == 1 else (opt.num_frames - 1) // opt.temporal_reduction + 1
        else:
            num_frames = 1 if opt.num_frames == 1 else opt.num_frames // opt.temporal_reduction

        # step 2: generate noise
        z = self._get_noise(
            len(text or t5_emb),
            opt.height,
            opt.width,
            num_frames,
            # seed,
            patch_size=patch_size,
            channel=self.model.in_channels // (patch_size**2),
        )

        # step 3: collect references
        if cond_type != "t2v" and references:
            references = collect_references_batch(
                references,
                cond_type,
                self.vae,
                (opt.height, opt.width),
                is_causal=opt.is_causal_vae,
            )
        elif cond_type != "t2v":
            print(
                "your csv file doesn't have a ref column or is not processed properly. will default to cond_type t2v!"
            )
            cond_type = "t2v"

        # step 4: generate timesteps
        timesteps = get_schedule(
            self.num_steps,
            (z.shape[-1] * z.shape[-2]) // patch_size**2,
            num_frames,
            shift=opt.shift,
            shift_alpha=opt.flow_shift,
        )

        # step 5: prepare classifier-free guidance data (method specific)
        out, additional_inp = self.denoiser.prepare_guidance(
            text=text,
            neg=neg,
            guidance_img=opt.guidance_img,
            t5_emb=t5_emb,
            neg_t5_emb=neg_t5_emb,
            clip_emb=clip_emb,
            neg_clip_emb=neg_clip_emb,
        )
        if isinstance(out, dict):
            t5_emb, clip_emb = out["t5"], out["clip"]
        else:
            text = out

        # step 6: prepare inputs
        inp = self._prepare_inputs(
            self.t5_enc,
            self.clip_enc,
            z,
            prompt=text,
            t5_emb=t5_emb,
            clip_emb=clip_emb,
            patch_size=patch_size,
            dtype=self.model.dtype,
        )
        inp.update(additional_inp)

        # step 7: prepare references
        if opt.method == "i2v":
            masks, masked_ref = prepare_inference_condition(
                z, cond_type, ref_list=references, causal=opt.is_causal_vae, dtype=self.model.dtype
            )
            inp["masks"] = masks
            inp["masked_ref"] = masked_ref

        x = self.denoiser.denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            guidance=opt.guidance,
            text_osci=opt.text_osci,
            image_osci=opt.image_osci,
            scale_temporal_osci=(
                opt.scale_temporal_osci and "i2v" in cond_type
            ),  # don't use temporal osci for v2v or t2v
            flow_shift=opt.flow_shift,
            patch_size=patch_size,
        )

        latents = unpack(x, opt.height, opt.width, num_frames, patch_size=patch_size).to(mstype.float32)

        # replace for image condition
        if cond_type == "i2v_head":
            latents[0, :, :1] = references[0][0]
        elif cond_type == "i2v_tail":
            latents[0, :, -1:] = references[0][0]
        elif cond_type == "i2v_loop":
            latents[0, :, :1] = references[0][0]
            latents[0, :, -1:] = references[0][1]

        if self.vae is not None:
            video = self.vae_decode(
                latents, num_frames=opt.num_frames, cond_type=cond_type, is_causal_vae=opt.is_causal_vae
            )
            return video, latents
        else:
            return None, latents
