import logging
import os

from transformers import PretrainedConfig

import mindspore as ms
from mindspore import nn, ops

from ..layers.operation_selector import get_split_op
from .autoencoder_kl import AutoencoderKL as AutoencoderKL_SD
from .vae_temporal import VAE_Temporal_SD  # noqa: F401

__all__ = ["AutoencoderKL"]


_logger = logging.getLogger(__name__)
SD_CONFIG = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
}
SDXL_CONFIG = SD_CONFIG.copy()
SDXL_CONFIG.update({"resolution": 512})


class AutoencoderKL(AutoencoderKL_SD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = get_split_op()

    def init_from_ckpt(self, path, ignore_keys=list()):
        if not os.path.exists(path):
            raise ValueError(
                "Maybe download failed. Please download the VAE encoder from https://huggingface.co/stabilityai/sd-vae-ft-ema"
            )
        param_dict = ms.load_checkpoint(path)
        param_not_load, ckpt_not_load = ms.load_param_into_net(self, param_dict, strict_load=True)
        if param_not_load or ckpt_not_load:
            _logger.warning(
                f"{param_not_load} in network is not loaded or {ckpt_not_load} in checkpoint is not loaded!"
            )

    def encode_with_moments_output(self, x):
        """For latent caching usage"""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = self.split(moments, moments.shape[1] // 2, 1)
        logvar = ops.clip_by_value(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)

        return mean, std


class VideoAutoencoderKL(nn.Cell):
    """
    Spatial VAE
    """

    def __init__(
        self,
        config=SDXL_CONFIG,
        ckpt_path=None,
        micro_batch_size=None,
    ):
        super().__init__()

        self.module = AutoencoderKL_SD(
            ddconfig=config,
            embed_dim=config["z_channels"],
            ckpt_path=ckpt_path,
        )

        self.out_channels = config["z_channels"]  # self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size

        # FIXME: "scaling_factor": 0.13025 is set in
        # https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/blob/main/vae/config.json.
        # This is a mistake made during the training of OpenSora v1.2.
        # To re-use the trained model, we need to keep this mistake.
        # For training, we should refine to 0.13025.
        self.scale_factor = 0.18215

    @staticmethod
    def rearrange_in(x):
        B, C, T, H, W = x.shape
        # (b c t h w) -> (b t c h w)
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (B * T, C, H, W))

        return x

    @staticmethod
    def rearrange_out(x, B):
        # x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        BT, C, H, W = x.shape
        T = BT // B
        x = ops.reshape(x, (B, T, C, H, W))
        x = ops.transpose(x, (0, 2, 1, 3, 4))

        return x

    def encode(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Return:
            (B C T H W)

        NOTE: remind to use stop gradient when invoke it
        """
        # is_video = (x.ndim == 5)

        B = x.shape[0]
        # x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.rearrange_in(x)

        if self.micro_batch_size is None:
            # x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
            x = self.module.encode(x) * self.scale_factor
        else:
            bs = self.micro_batch_size
            # Not sure whether to enter the for loop because of dynamic shape,
            # avoid initialize x_out as an empty list
            x_out = [self.module.encode(x[:bs]) * self.scale_factor]
            # FIXME: supported in graph mode? or use split
            for i in range(bs, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.encode(x_bs) * self.scale_factor
                x_out.append(x_bs)
            x = ops.cat(x_out, axis=0)

        # x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        x = self.rearrange_out(x, B=B)

        return x

    def decode(self, x, **kwargs):
        # is_video = (x.ndim == 5)

        B = x.shape[0]
        # x: (B, Z, T, H, W)
        # x = rearrange(x, "B Z T H W -> (B T) Z H W")
        x = self.rearrange_in(x)

        if self.micro_batch_size is None:
            x = self.module.decode(x / self.scale_factor)
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.decode(x_bs / self.scale_factor)
                x_out.append(x_bs)
            x = ops.cat(x_out, axis=0)

        # x = rearrange(x, "(B T) Z H W -> B Z T H W", B=B)
        x = self.rearrange_out(x, B=B)

        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size


class VideoAutoencoderPipelineConfig(PretrainedConfig):
    model_type = "VideoAutoencoderPipeline"

    def __init__(
        self,
        vae_2d=None,
        vae_temporal=None,
        from_pretrained=None,
        freeze_vae_2d=False,
        cal_loss=False,
        micro_frame_size=None,
        shift=0.0,
        scale=1.0,
        **kwargs,
    ):
        self.vae_2d = vae_2d
        self.vae_temporal = vae_temporal
        self.from_pretrained = from_pretrained
        self.freeze_vae_2d = freeze_vae_2d
        self.cal_loss = cal_loss
        self.micro_frame_size = micro_frame_size
        self.shift = shift
        self.scale = scale
        super().__init__(**kwargs)


def build_module_from_config(config):
    """
    config dict format:
        - type: model class name
        - others: model init args
    """
    cfg = config.copy()
    name = cfg.pop("type")
    kwargs = cfg

    # FIXME: use importlib with path
    module = eval(name)(**kwargs)
    return module


class VideoAutoencoderPipeline(nn.Cell):
    """
    Main model for spatial vae + tempral vae
    """

    # config_class = VideoAutoencoderPipelineConfig
    def __init__(self, config: VideoAutoencoderPipelineConfig):
        super().__init__()
        self.spatial_vae = build_module_from_config(config.vae_2d)
        self.temporal_vae = build_module_from_config(config.vae_temporal)

        self.cal_loss = config.cal_loss
        self.micro_frame_size = config.micro_frame_size
        self.micro_z_frame_size = self.temporal_vae.get_latent_size([config.micro_frame_size, None, None])[0]

        if config.freeze_vae_2d:
            for param in self.spatial_vae.get_parameters():
                param.requires_grad = False

        self.out_channels = self.temporal_vae.out_channels

        # normalization parameters
        scale = ms.Tensor(config.scale)
        shift = ms.Tensor(config.shift)
        if len(scale.shape) > 0:
            scale = scale[None, :, None, None, None]
        if len(shift.shape) > 0:
            shift = shift[None, :, None, None, None]
        self.scale = ms.Parameter(scale, requires_grad=False)
        self.shift = ms.Parameter(shift, requires_grad=False)

    def encode(self, x):
        x_z = self.spatial_vae.encode(x)

        if self.micro_frame_size is None:
            posterior_mean, posterior_logvar = self.temporal_vae._encode(x_z)
            z = self.temporal_vae.sample(posterior_mean, posterior_logvar)
        else:
            z_list = []
            # TODO: there is a bug in torch impl. need to concat posterior as well. But ot save memory for concatnated posterior. Let's remain unchange.
            for i in range(0, x_z.shape[2], self.micro_frame_size):
                x_z_bs = x_z[:, :, i : i + self.micro_frame_size]
                posterior_mean, posterior_logvar = self.temporal_vae._encode(x_z_bs)
                z_bs = self.temporal_vae.sample(posterior_mean, posterior_logvar)
                z_list.append(z_bs)
            z = ops.cat(z_list, axis=2)
            if self.cal_loss:
                raise ValueError(
                    "Please fix the bug of posterior concatenation for temporal vae training with micro_frame_size"
                )

        if self.cal_loss:
            return z, posterior_mean, posterior_logvar, x_z
        else:
            return (z - self.shift) / self.scale

    def decode(self, z, num_frames=None):
        if not self.cal_loss:
            z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.micro_frame_size is None:
            x_z = self.temporal_vae.decode(z, num_frames=num_frames)
            x = self.spatial_vae.decode(x_z)
        else:
            x_z_list = []
            for i in range(0, z.shape[2], self.micro_z_frame_size):
                z_bs = z[:, :, i : i + self.micro_z_frame_size]
                x_z_bs = self.temporal_vae.decode(z_bs, num_frames=min(self.micro_frame_size, num_frames))
                x_z_list.append(x_z_bs)
                num_frames -= self.micro_frame_size
            x_z = ops.cat(x_z_list, axis=2)
            x = self.spatial_vae.decode(x_z)

        if self.cal_loss:
            return x, x_z
        else:
            return x

    def construct(self, x):
        z, posterior_mean, posterior_logvar, x_z = self.encode(x)
        x_rec, x_z_rec = self.decode(z, num_frames=x_z.shape[2])
        return x_rec, x_z_rec, z, posterior_mean, posterior_logvar, x_z

    def get_latent_size(self, input_size):
        if self.micro_frame_size is None or input_size[0] is None:
            return self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(input_size))
        else:
            sub_input_size = [self.micro_frame_size, input_size[1], input_size[2]]
            sub_latent_size = self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(sub_input_size))
            sub_latent_size[0] = sub_latent_size[0] * (input_size[0] // self.micro_frame_size)
            remain_temporal_size = [input_size[0] % self.micro_frame_size, None, None]
            if remain_temporal_size[0] > 0:
                remain_size = self.temporal_vae.get_latent_size(remain_temporal_size)
                sub_latent_size[0] += remain_size[0]
            return sub_latent_size

    def get_temporal_last_layer(self):
        return self.temporal_vae.decoder.conv_out.conv.weight


def OpenSoraVAE_V1_2(
    micro_batch_size=4,
    micro_frame_size=17,
    ckpt_path=None,
    freeze_vae_2d=False,
    cal_loss=False,
):
    vae_2d = dict(
        type="VideoAutoencoderKL",
        config=SDXL_CONFIG,
        micro_batch_size=micro_batch_size,
    )
    vae_temporal = dict(
        type="VAE_Temporal_SD",
        from_pretrained=None,
    )
    shift = (-0.10, 0.34, 0.27, 0.98)
    scale = (3.85, 2.32, 2.33, 3.06)
    kwargs = dict(
        vae_2d=vae_2d,
        vae_temporal=vae_temporal,
        freeze_vae_2d=freeze_vae_2d,
        cal_loss=cal_loss,
        micro_frame_size=micro_frame_size,
        shift=shift,
        scale=scale,
    )

    config = VideoAutoencoderPipelineConfig(**kwargs)
    model = VideoAutoencoderPipeline(config)

    if ckpt_path is not None:
        sd = ms.load_checkpoint(ckpt_path)
        pu, cu = ms.load_param_into_net(model, sd, strict_load=False)
        print(f"Net param not loaded : {pu}")
        print(f"Checkpoint param not loaded : {cu}")

    return model
