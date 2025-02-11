import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from huggingface_hub import hf_hub_download
from imwatermark import WatermarkEncoder

import mindspore as ms
from mindspore import ops

from mindone.safetensors.mindspore import load_file as load_sft

from .model import Flux, FluxParams
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from .modules.conditioner import HFEmbedder

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: Optional[str]
    ae_path: Optional[str]
    repo_id: Optional[str]
    repo_flow: Optional[str]
    repo_ae: Optional[str]


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: List[str], unexpected: List[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        logger.info(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        logger.info("\n" + "-" * 79 + "\n")
        logger.info(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        logger.info(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        logger.info(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(name: str, hf_download: bool = True):
    # Loading Flux
    logger.info("Init model")
    ckpt_path = configs[name].ckpt_path
    if ckpt_path is None and configs[name].repo_id is not None and configs[name].repo_flow is not None and hf_download:
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    model = Flux(configs[name].params)

    for p in model.get_parameters():
        p.set_dtype(ms.bfloat16)

    if ckpt_path is not None:
        logger.info("Loading checkpoint")
        sd = load_sft(ckpt_path)
        missing, unexpected = ms.load_param_into_net(model, sd, strict_load=False)
        print_load_warning(missing, unexpected)
    return model


def load_t5(max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, mindspore_dtype=ms.bfloat16)


def load_clip() -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, mindspore_dtype=ms.bfloat16)


def load_ae(name: str, hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if ckpt_path is None and configs[name].repo_id is not None and configs[name].repo_ae is not None and hf_download:
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # Loading the autoencoder
    logger.info("Init AE")
    ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path)
        missing, unexpected = ms.load_param_into_net(ae, sd, strict_load=False)
        print_load_warning(missing, unexpected)
    return ae


class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: ms.Tensor) -> ms.Tensor:
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, RGB, H, W) in range [-1, 1]

        Returns:
            same as input but watermarked
        """
        image = 0.5 * image + 0.5
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        n, b, c, h, w = image.shape
        image_np = (255 * image).reshape(-1, c, h, w).permute(0, 2, 3, 1).numpy()[:, :, :, ::-1]
        # (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        # watermarking libary expects input as cv2 BGR format
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = ms.Tensor.from_numpy(image_np[:, :, :, ::-1]).reshape(n, b, h, w, c).permute(0, 1, 4, 2, 3)
        image = ops.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        image = 2 * image - 1
        return image


# A fixed 48-bit message that was choosen at random
WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)
