import logging
import os

from transformers import PretrainedConfig

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.communication import get_group_size

from ...acceleration.communications import GatherFowardSplitBackward, SplitFowardGatherBackward
from ...acceleration.parallel_states import get_sequence_parallel_group
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
        scale_factor=0.18215,
        use_recompute=False,
        micro_batch_parallel=False,
        sample_deterministic=False,
    ):
        super().__init__()

        self.module = AutoencoderKL_SD(
            ddconfig=config,
            embed_dim=config["z_channels"],
            ckpt_path=ckpt_path,
            use_recompute=use_recompute,
            sample_deterministic=sample_deterministic,
        )

        self.out_channels = config["z_channels"]  # self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size
        self.micro_batch_parallel = micro_batch_parallel
        if self.micro_batch_parallel:
            sp_group = get_sequence_parallel_group()
            _logger.info(f"Initialize Spatial VAE model with parallel group `{sp_group}`.")
            self.sp_size = get_group_size(sp_group)
            self.split_forward_gather_backward = SplitFowardGatherBackward(dim=0, grad_scale="down", group=sp_group)
            self.gather_forward_split_backward = GatherFowardSplitBackward(dim=0, grad_scale="up", group=sp_group)
            # TODO: drop the assertion once conv3d support fp32, test with test suites
            assert self.micro_batch_size == 1

        # FIXME: "scaling_factor": 0.13025 is set in
        # https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/blob/main/vae/config.json.
        # This is a mistake made during the training of OpenSora v1.2.
        # To re-use the trained model, we need to keep this mistake.
        # For training, we should refine to 0.13025.
        self.scale_factor = scale_factor
        self.split = get_split_op()
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
        # B C T H W -> (B T) C H W
        x = self.rearrange_in(x)

        pad_num = None
        if self.micro_batch_parallel:
            # select part of x for micro_batch
            pad_num = self.get_pad_num(x.shape[0])
            if pad_num > 0:
                x = mint.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, 0, pad_num))
            x = self.split_forward_gather_backward(x)

        if self.micro_batch_size is None:
            x_out = self.module.encode(x) * self.scale_factor
        else:
            bs = self.micro_batch_size
            x_out = self.module.encode(x[:bs]) * self.scale_factor
            for i in range(bs, x.shape[0], bs):
                x_cur = self.module.encode(x[i : i + bs]) * self.scale_factor
                x_out = ops.cat((x_out, x_cur), axis=0)

        if self.micro_batch_parallel:
            x_out = self.gather_forward_split_backward(x_out)
            if pad_num > 0:
                x_out = x_out.narrow(0, 0, x_out.shape[0] - pad_num)

        # (B T) C H W -> B C T H W
        x_out = self.rearrange_out(x_out, B=B)

        return x_out

    def decode(self, x, **kwargs):
        # is_video = (x.ndim == 5)

        B = x.shape[0]
        # x: (B, Z, T, H, W)
        # B Z T H W -> (B T) Z H W
        x = self.rearrange_in(x)

        if self.micro_batch_size is None:
            x_out = self.module.decode(x / self.scale_factor)
        else:
            mbs = self.micro_batch_size

            x_out = self.module.decode(x[:mbs] / self.scale_factor)
            for i in range(mbs, x.shape[0], mbs):
                x_cur = self.module.decode(x[i : i + mbs] / self.scale_factor)
                x_out = ops.cat((x_out, x_cur), axis=0)

        # (B T) Z H W -> B Z T H W
        x_out = self.rearrange_out(x_out, B=B)

        return x_out

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size

    def get_pad_num(self, dim_size: int) -> int:
        pad = (self.sp_size - (dim_size % self.sp_size)) % self.sp_size
        return pad


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
        concat_posterior=False,
        shift=0.0,
        scale=1.0,
        micro_frame_parallel=False,
        sample_deterministic=False,
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
        self.concat_posterior = (concat_posterior,)
        self.micro_frame_parallel = micro_frame_parallel
        self.sample_deterministic = sample_deterministic
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
        print(f"micro_frame_size: {self.micro_frame_size}, micro_z_frame_size: {self.micro_z_frame_size}")
        self.micro_frame_parallel = config.micro_frame_parallel
        self.sample_deterministic = config.sample_deterministic

        if config.freeze_vae_2d:
            for param in self.spatial_vae.get_parameters():
                param.requires_grad = False

        self.out_channels = self.temporal_vae.out_channels
        self.split = get_split_op()

        # normalization parameters
        scale = ms.Tensor(config.scale)
        shift = ms.Tensor(config.shift)
        if len(scale.shape) > 0:
            scale = scale[None, :, None, None, None]
        if len(shift.shape) > 0:
            shift = shift[None, :, None, None, None]
        self.scale = ms.Parameter(scale, requires_grad=False)
        self.shift = ms.Parameter(shift, requires_grad=False)
        self.freeze_vae_2d = config.freeze_vae_2d
        self.concat_posterior = config.concat_posterior

        if self.micro_frame_parallel:
            sp_group = get_sequence_parallel_group()
            _logger.info(f"Initialize Temporal VAE model with parallel group `{sp_group}`.")
            self.sp_size = get_group_size(sp_group)
            self.split_forward_gather_backward = SplitFowardGatherBackward(dim=2, grad_scale="down", group=sp_group)
            self.gather_forward_split_backward = GatherFowardSplitBackward(dim=2, grad_scale="up", group=sp_group)
            if self.cal_loss:
                raise NotImplementedError("Not Supported yet.")

    def encode(self, x):
        if self.freeze_vae_2d:
            x_z = ops.stop_gradient(self.spatial_vae.encode(x))
        else:
            x_z = self.spatial_vae.encode(x)

        if self.micro_frame_parallel:
            # TODO: drop assertion and add padding
            assert x_z.shape[2] % self.sp_size == 0
            if self.micro_frame_size is not None:
                assert x_z.shape[2] % self.micro_frame_size == 0
            x_z = self.split_forward_gather_backward(x_z)

        if self.micro_frame_size is None:
            posterior_mean, posterior_logvar = self.temporal_vae._encode(x_z)
            if self.sample_deterministic:
                z_out = posterior_mean
            else:
                z_out = self.temporal_vae.sample(posterior_mean, posterior_logvar)

            if self.cal_loss:
                return z_out, posterior_mean, posterior_logvar, x_z
            else:
                if self.micro_frame_parallel:
                    z_out = self.gather_forward_split_backward(z_out)
                return (z_out - self.shift) / self.scale
        else:
            # x_z: (b z t h w)
            mfs = self.micro_frame_size
            if self.cal_loss:
                # TODO: fix the bug in torch, output concat of the splitted posteriors instead of the last split
                posterior_mean, posterior_logvar = self.temporal_vae._encode(x_z[:, :, :mfs])
                if self.sample_deterministic:
                    z_out = posterior_mean
                else:
                    z_out = self.temporal_vae.sample(posterior_mean, posterior_logvar)
                for i in range(mfs, x_z.shape[2], mfs):
                    posterior_mean, posterior_logvar = self.temporal_vae._encode(x_z[:, :, i : i + mfs])
                    if self.sample_deterministic:
                        z_cur = posterior_mean
                    else:
                        z_cur = self.temporal_vae.sample(posterior_mean, posterior_logvar)
                    z_out = ops.cat((z_out, z_cur), axis=2)

                return z_out, posterior_mean, posterior_logvar, x_z
            else:
                # no posterior cache to reduce memory in inference
                z_out = self.temporal_vae.encode(x_z[:, :, :mfs])
                for i in range(mfs, x_z.shape[2], mfs):
                    z_cur = self.temporal_vae.encode(x_z[:, :, i : i + mfs])
                    z_out = ops.cat((z_out, z_cur), axis=2)

                if self.micro_frame_parallel:
                    z_out = self.gather_forward_split_backward(z_out)

                return (z_out - self.shift) / self.scale

    def decode(self, z, num_frames=None):
        if not self.cal_loss:
            z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.micro_frame_size is None:
            x_z_out = self.temporal_vae.decode(z, num_frames=num_frames)
            x = self.spatial_vae.decode(x_z_out)
            if self.cal_loss:
                return x, x_z_out
            else:
                return x
        else:
            mz = self.micro_z_frame_size
            remain_frames = num_frames if self.micro_frame_size > num_frames else self.micro_frame_size
            x_z_out = self.temporal_vae.decode(z[:, :, :mz], num_frames=remain_frames)
            num_frames -= self.micro_frame_size

            for i in range(mz, z.shape[2], mz):
                remain_frames = num_frames if self.micro_frame_size > num_frames else self.micro_frame_size
                x_z_cur = self.temporal_vae.decode(z[:, :, i : i + mz], num_frames=remain_frames)
                x_z_out = ops.cat((x_z_out, x_z_cur), axis=2)
                num_frames -= self.micro_frame_size

            x = self.spatial_vae.decode(x_z_out)

            if self.cal_loss:
                return x, x_z_out
            else:
                return x

    def construct(self, x):
        # assert self.cal_loss, "This method is only available when cal_loss is True"
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
    micro_batch_parallel=False,
    micro_frame_parallel=False,
    ckpt_path=None,
    vae2d_ckpt_path=None,
    freeze_vae_2d=False,
    cal_loss=False,
    use_recompute=False,
    sample_deterministic=False,
):
    """
    ckpt_path: path to the checkpoint of the overall model (vae2d + temporal vae)
    vae_2d_ckpt_path: path to the checkpoint of the vae 2d model. It will only be loaded when `ckpt_path` not provided.
    """

    if isinstance(micro_batch_size, int):
        if micro_batch_size <= 0:
            micro_batch_size = None
    if isinstance(micro_frame_size, int):
        if micro_frame_size <= 0:
            micro_frame_size = None

    vae_2d = dict(
        type="VideoAutoencoderKL",
        config=SDXL_CONFIG,
        micro_batch_size=micro_batch_size,
        micro_batch_parallel=micro_batch_parallel,
        use_recompute=use_recompute,
        sample_deterministic=sample_deterministic,
    )
    vae_temporal = dict(
        type="VAE_Temporal_SD",
        from_pretrained=None,
        use_recompute=use_recompute,
        sample_deterministic=sample_deterministic,
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
        micro_frame_parallel=micro_frame_parallel,
        sample_deterministic=sample_deterministic,
    )

    config = VideoAutoencoderPipelineConfig(**kwargs)
    model = VideoAutoencoderPipeline(config)

    # load model weights
    if (ckpt_path is not None) and (os.path.exists(ckpt_path)):
        sd = ms.load_checkpoint(ckpt_path)

        # remove the added prefix in the trained checkpoint
        pnames = list(sd.keys())
        for pn in pnames:
            new_pn = pn.replace("autoencoder.", "").replace("_backbone.", "")
            sd[new_pn] = sd.pop(pn)

        pu, cu = ms.load_param_into_net(model, sd, strict_load=False)
        print(f"Net param not loaded : {pu}")
        print(f"Checkpoint param not loaded : {cu}")
    elif (vae2d_ckpt_path is not None) and (os.path.exists(vae2d_ckpt_path)):
        sd = ms.load_checkpoint(vae2d_ckpt_path)
        # TODO: add spatial_vae prefix to the param name
        pu, cu = ms.load_param_into_net(model.spatial_vae, sd, strict_load=False)
    else:
        _logger.warning("VAE checkpoint is NOT loaded!")

    return model
