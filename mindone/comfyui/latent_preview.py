import logging

import comfy.model_management
import comfy.utils
import folder_paths
from comfy.cli_args import LatentPreviewMethod, args
from comfy.taesd.taesd import TAESD
from PIL import Image

import mindspore
from mindspore import mint

MAX_PREVIEW_RESOLUTION = args.preview_size


def preview_to_image(latent_image):
    latents_ubyte = ((latent_image + 1.0) / 2.0).clamp(0, 1).mul(0xFF)  # change scale from -1..1 to 0..1  # to 0..255
    if comfy.model_management.directml_enabled:
        latents_ubyte = latents_ubyte.to(dtype=mindspore.uint8)
    latents_ubyte = latents_ubyte.to(
        dtype=mindspore.uint8, non_blocking=comfy.model_management.device_supports_non_blocking(None)
    )

    return Image.fromarray(latents_ubyte.numpy())


class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("JPEG", preview_image, MAX_PREVIEW_RESOLUTION)


class TAESDPreviewerImpl(LatentPreviewer):
    def __init__(self, taesd):
        self.taesd = taesd

    def decode_latent_to_preview(self, x0):
        x_sample = self.taesd.decode(x0[:1])[0].movedim(0, 2)
        return preview_to_image(x_sample)


class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None):
        self.latent_rgb_factors = mindspore.tensor(latent_rgb_factors).transpose(0, 1)
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = mindspore.tensor(latent_rgb_factors_bias)

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype)

        if x0.ndim == 5:
            x0 = x0[0, :, 0]
        else:
            x0 = x0[0]

        latent_image = mint.functional.linear(
            x0.movedim(0, -1), self.latent_rgb_factors, bias=self.latent_rgb_factors_bias
        )
        # latent_image = x0[0].permute(1, 2, 0) @ self.latent_rgb_factors

        return preview_to_image(latent_image)


def get_previewer(device, latent_format):
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer methods
        taesd_decoder_path = None
        if latent_format.taesd_decoder_name is not None:
            taesd_decoder_path = next(
                (
                    fn
                    for fn in folder_paths.get_filename_list("vae_approx")
                    if fn.startswith(latent_format.taesd_decoder_name)
                ),
                "",
            )
            taesd_decoder_path = folder_paths.get_full_path("vae_approx", taesd_decoder_path)

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB

        if method == LatentPreviewMethod.TAESD:
            if taesd_decoder_path:
                taesd = TAESD(None, taesd_decoder_path, latent_channels=latent_format.latent_channels)
                previewer = TAESDPreviewerImpl(taesd)
            else:
                logging.warning(
                    "Warning: TAESD previews enabled, but could not find models/vae_approx/{}".format(
                        latent_format.taesd_decoder_name
                    )
                )

        if previewer is None:
            if latent_format.latent_rgb_factors is not None:
                previewer = Latent2RGBPreviewer(latent_format.latent_rgb_factors, latent_format.latent_rgb_factors_bias)
    return previewer


def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer(None, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    return callback
