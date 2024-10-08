# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------

import logging
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import mindspore as ms
import mindspore.dataset.vision as vision
from mindspore import Tensor, ops
from mindspore.dataset.vision import Inter

from mindone.diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, LCMScheduler, UNet2DConditionModel
from mindone.diffusers.utils import BaseOutput

from .util.ensemble import ensemble_depth
from .util.image_util import chw2hwc, colorize_depth_maps, get_tv_resample_method, resize_max_res


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for marigold depth estimation pipeline.
    """

    depth_np: np.ndarray
    depth_colored: Union[Image.Image, None]
    uncertainty: Union[np.ndarray, None]


class MarigoldPipeline(DiffusionPipeline):
    """
    Marigold depth estimation pipeline.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
        is_ms_ckpt: Optional[bool] = False,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None
        self.is_ms_ckpt = is_ms_ckpt

    def __call__(
        self,
        input_image: Union[Image.Image, Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Optional[int] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MarigoldDepthOutput:
        # use default setting if not provided
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # check denoising steps
        self._check_inference_step(denoising_steps)

        # get resample method
        resample_method = get_tv_resample_method(resample_method)

        # ----------------- Preprocess input -----------------
        # if input is PIL image, convert to Tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = np.array(input_image)
            rgb = rgb[np.newaxis, :, :, :]  # [1, H, W, rgb]
            # resize the longer edge to 768
            if processing_res > 0:
                rgb = resize_max_res(
                    rgb,
                    max_edge_resolution=processing_res,
                    resample_method=resample_method,
                    stick_size64=self.is_ms_ckpt,
                )
            rgb = rgb.transpose(0, 3, 1, 2)  # [rgb, H, W]
            rgb = Tensor(rgb, dtype=self.dtype)
        elif isinstance(input_image, Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image)}")
        input_size = rgb.shape
        assert len(rgb.shape) == 4 and rgb.shape[1] == 3, f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # normalize to [-1, 1]
        rgb_norm = (rgb / 255.0 * 2.0 - 1.0).astype(self.dtype)  # [0, 255] -> [-1, 1]
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Prediction -----------------
        # build ensemble dataset
        duplicated_rgb = [rgb_norm for _ in range(ensemble_size)]

        # predict depth for each ensemble image
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(duplicated_rgb, desc=" " * 2 + "Inference batches", leave=False)
        else:
            iterable = duplicated_rgb
        for rgb_tensor in iterable:
            ops.stop_gradient(rgb_tensor)
            depth_pred_raw = self.single_infer(
                rgb_in=rgb_tensor,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            depth_pred_ls.append(depth_pred_raw)
        depth_preds = ops.Concat(0)(depth_pred_ls)
        depth_preds = depth_preds.asnumpy()

        # ----------------- Ensemble -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depth(
                depth_preds,
                scale_invariant=self.scale_invariant,
                shift_invariant=self.shift_invariant,
                max_res=50,
                **(ensemble_kwargs or {}),
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None

        # convert to numpy to process and save
        depth_pred = depth_pred.squeeze()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze()

        resample_method_dict = {
            "bilinear": Inter.BILINEAR,
            "bicubic": Inter.BICUBIC,
            "nearest": Inter.NEAREST,
            "nearest-exact": Inter.NEAREST,
        }

        # resize to input size
        if match_input_res:
            resize_op = vision.Resize((input_size[2], input_size[3]), resample_method_dict.get(resample_method, None))
            depth_pred = resize_op(depth_pred)

        # clip to [0, 1]
        depth_pred = np.clip(depth_pred, 0, 1)  # [H, W, 1] -> [1, H, W]

        # colorize depth map
        if color_map is not None:
            depth_colored = colorize_depth_maps(
                depth_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return MarigoldDepthOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check the number of denoising steps for inference by the scheduler.
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps.")
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode an empty text prompt once for all inference.
        """
        prompt = ""
        if self.is_ms_ckpt:
            text_inputs = self.tokenizer.tokenize(prompt)
            self.empty_text_embed = self.text_encoder(text_inputs)
        else:
            text_inputs = self.tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            text_input_ids = text_input_ids.numpy()
            text_input_ids = Tensor(text_input_ids, dtype=ms.int32)
            self.empty_text_embed = self.text_encoder(text_input_ids)[0]

    def single_infer(
        self,
        rgb_in: Tensor,
        num_inference_steps: int,
        generator: Optional[int],
        show_pbar: bool,
    ) -> Tensor:
        """
        Perform an individual depth prediction without ensembling.
        """

        # set inference steps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps  # [T]

        # encode RGB image
        rgb_latent = self.encode_rgb(rgb_in)

        # initialize depth latent with standard normal
        depth_latent = ops.StandardNormal()(rgb_latent.shape)  # [B, 4, h, w]
        depth_latent = depth_latent.to(self.dtype)

        # get empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = ops.Tile()(self.empty_text_embed, (rgb_latent.shape[0], 1, 1))

        # denoise
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = ops.Concat(1)([rgb_latent, depth_latent])  # first RGB then depth

            t = ops.Tile()(t, (rgb_latent.shape[0],))

            # predict noise
            if self.is_ms_ckpt:
                noise_pred = self.unet(unet_input, t, batch_empty_text_embed)  # [B, 4, h, w]
            else:
                noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed)[0]  # [B, 4, h, w]

            # update depth latent with noise
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent)[0]

        depth = self.decode_depth(depth_latent)

        # clip to [-1, 1]
        depth = ops.clip_by_value(depth, -1.0, 1.0)
        # convert to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth

    def encode_rgb(self, rgb_in: Tensor) -> Tensor:
        """
        Encode RGB image to latent representation.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = ops.Split(1, 2)(moments)
        # scale
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: Tensor) -> Tensor:
        """
        Decode depth latent representation to depth map.
        """
        # unscale
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # output mean
        depth_mean = ops.ReduceMean(keep_dims=True)(stacked, 1)
        return depth_mean
