import logging
import os
from typing import Tuple

from opensora.acceleration.parallel_states import get_sequence_parallel_state

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.diffusers import __version__
from mindone.diffusers.models.modeling_utils import load_state_dict
from mindone.diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, _add_variant, _get_model_file

from ..modeling_videobase import VideoBaseAE
from ..modules.conv import CausalConv3d
from ..modules.ops import nonlinearity
from ..utils.model_utils import resolve_str_to_obj

logger = logging.getLogger(__name__)


class CausalVAEModel(VideoBaseAE):
    """
    The default vales are set to be the same as those used in OpenSora v1.1
    """

    def __init__(
        self,
        lr: float = 1e-5,  # ignore
        hidden_size: int = 128,
        z_channels: int = 4,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = [],
        dropout: float = 0.0,
        resolution: int = 256,
        double_z: bool = True,
        embed_dim: int = 4,
        num_res_blocks: int = 2,
        q_conv: str = "CausalConv3d",
        encoder_conv_in: str = "Conv2d",
        encoder_conv_out: str = "CausalConv3d",
        encoder_attention: str = "AttnBlock3DFix",
        encoder_resnet_blocks: Tuple[str] = (
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        encoder_spatial_downsample: Tuple[str] = (
            "Downsample",
            "Downsample",
            "Downsample",
            "",
        ),
        encoder_temporal_downsample: Tuple[str] = (
            "",
            "TimeDownsampleRes2x",
            "TimeDownsampleRes2x",
            "",
        ),
        encoder_mid_resnet: str = "ResnetBlock3D",
        decoder_conv_in: str = "CausalConv3d",
        decoder_conv_out: str = "CausalConv3d",
        decoder_attention: str = "AttnBlock3DFix",
        decoder_resnet_blocks: Tuple[str] = (
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        decoder_spatial_upsample: Tuple[str] = (
            "",
            "SpatialUpsample2x",
            "SpatialUpsample2x",
            "SpatialUpsample2x",
        ),
        decoder_temporal_upsample: Tuple[str] = ("", "", "TimeUpsampleRes2x", "TimeUpsampleRes2x"),
        decoder_mid_resnet: str = "ResnetBlock3D",
        use_quant_layer: bool = True,
        ckpt_path=None,
        ignore_keys=[],
        monitor=None,
        use_fp16=False,
        upcast_sigmoid=False,
    ):
        super().__init__()
        dtype = ms.float16 if use_fp16 else ms.float32

        self.encoder = Encoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=encoder_conv_in,
            conv_out=encoder_conv_out,
            attention=encoder_attention,
            resnet_blocks=encoder_resnet_blocks,
            spatial_downsample=encoder_spatial_downsample,
            temporal_downsample=encoder_temporal_downsample,
            mid_resnet=encoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            double_z=double_z,
            dtype=dtype,
            upcast_sigmoid=upcast_sigmoid,
        )

        self.decoder = Decoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=decoder_conv_in,
            conv_out=decoder_conv_out,
            attention=decoder_attention,
            resnet_blocks=decoder_resnet_blocks,
            spatial_upsample=decoder_spatial_upsample,
            temporal_upsample=decoder_temporal_upsample,
            mid_resnet=decoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            dtype=dtype,
            upcast_sigmoid=upcast_sigmoid,
        )
        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.exp = mint.exp
        self.stdnormal = mint.normal
        self.depend = ops.Depend() if get_sequence_parallel_state() else None

        # self.encoder.recompute()
        # self.decoder.recompute()
        self.tile_sample_min_size = 256
        self.tile_sample_min_size_t = 33
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(hidden_size_mult) - 1)))
        # t_down_ratio = [i for i in encoder_temporal_downsample if len(i) > 0]
        # self.tile_latent_min_size_t = int((self.tile_sample_min_size_t - 1) / (2 ** len(t_down_ratio))) + 1
        self.tile_latent_min_size_t = 16
        self.tile_overlap_factor = 0.125
        self.use_tiling = False
        self.use_quant_layer = use_quant_layer
        if self.use_quant_layer:
            quant_conv_cls = resolve_str_to_obj(q_conv)
            self.quant_conv = quant_conv_cls(2 * z_channels, 2 * embed_dim, 1)
            self.post_quant_conv = quant_conv_cls(embed_dim, z_channels, 1)

    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder]
        return [self.encoder]

    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]

    # rewrite class method to allow the state dict as input
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        state_dict = kwargs.pop("state_dict", None)  # additional key argument
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        mindspore_dtype = kwargs.pop("mindspore_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )

        # load model
        model_file = None
        if from_flax:
            raise NotImplementedError("loading flax checkpoint in mindspore model is not yet supported.")
        else:
            if state_dict is None:  # edits: only search for model_file if state_dict is not provided
                if use_safetensors:
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            token=token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            commit_hash=commit_hash,
                        )
                    except IOError as e:
                        if not allow_pickle:
                            raise e
                        pass
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                    )

            model = cls.from_config(config, **unused_kwargs)
            if state_dict is None:  # edits: only load model_file if state_dict is None
                state_dict = load_state_dict(model_file, variant=variant)
            model._convert_deprecated_attention_blocks(state_dict)

            model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                model,
                state_dict,
                model_file,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )

            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }

        if mindspore_dtype is not None and not isinstance(mindspore_dtype, ms.Type):
            raise ValueError(
                f"{mindspore_dtype} needs to be of type ms.Type, e.g. ms.float16, but is {type(mindspore_dtype)}."
            )
        elif mindspore_dtype is not None:
            model = model.to(mindspore_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.set_train(False)
        if output_loading_info:
            return model, loading_info

        return model

    def init_from_vae2d(self, path):
        # default: tail init
        # path: path to vae 2d model ckpt
        vae2d_sd = ms.load_checkpoint(path)
        vae_2d_keys = list(vae2d_sd.keys())
        vae_3d_keys = list(self.parameters_dict().keys())

        # 3d -> 2d
        map_dict = {
            "conv.weight": "weight",
            "conv.bias": "bias",
        }

        new_state_dict = {}
        for key_3d in vae_3d_keys:
            if key_3d.startswith("loss"):
                continue

            # param name mapping from vae-3d to vae-2d
            key_2d = key_3d
            for kw in map_dict:
                key_2d = key_2d.replace(kw, map_dict[kw])

                assert key_2d in vae_2d_keys, f"Key {key_2d} ({key_3d}) should be in 2D VAE"

            # set vae 3d state dict
            shape_3d = self.parameters_dict()[key_3d].shape
            shape_2d = vae2d_sd[key_2d].shape
            if "bias" in key_2d:
                assert shape_3d == shape_2d, f"Shape mismatch for key {key_3d} ({key_2d})"
                new_state_dict[key_3d] = vae2d_sd[key_2d]
            elif "norm" in key_2d:
                assert shape_3d == shape_2d, f"Shape mismatch for key {key_3d} ({key_2d})"
                new_state_dict[key_3d] = vae2d_sd[key_2d]
            elif "conv" in key_2d or "nin_shortcut" in key_2d:
                if shape_3d[:2] != shape_2d[:2]:
                    logger.info(key_2d, shape_3d, shape_2d)
                w = vae2d_sd[key_2d]
                new_w = mint.zeros(shape_3d, dtype=w.dtype)
                # tail initialization
                new_w[:, :, -1, :, :] = w  # cin, cout, t, h, w

                new_w = ms.Parameter(new_w, name=key_3d)

                new_state_dict[key_3d] = new_w
            elif "attn_1" in key_2d:
                new_val = vae2d_sd[key_2d].expand_dims(axis=2)
                new_param = ms.Parameter(new_val, name=key_3d)
                new_state_dict[key_3d] = new_param
            else:
                raise NotImplementedError(f"Key {key_3d} ({key_2d}) not implemented")

            m, u = ms.load_param_into_net(self, new_state_dict)
            if len(m) > 0:
                logger.info("net param not loaded: ", m)
            if len(u) > 0:
                logger.info("checkpoint param not loaded: ", u)

    def init_from_ckpt(self, path, ignore_keys=list()):
        # TODO: support auto download pretrained checkpoints
        sd = ms.load_checkpoint(path)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    logger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        if "ema_state_dict" in sd and len(sd["ema_state_dict"]) > 0 and os.environ.get("NOT_USE_EMA_MODEL", 0) == 0:
            logger.info("Load from ema model!")
            sd = sd["ema_state_dict"]
            sd = {key.replace("module.", ""): value for key, value in sd.items()}
        elif "state_dict" in sd:
            logger.info("Load from normal model!")
            if "gen_model" in sd["state_dict"]:
                sd = sd["state_dict"]["gen_model"]
            else:
                sd = sd["state_dict"]

        ms.load_param_into_net(self, sd, strict_load=False)
        logger.info(f"Restored from {path}")

    def _encode(self, x):
        # return latent distribution, N(mean, logvar)
        h = self.encoder(x)
        if self.use_quant_layer:
            h = self.quant_conv(h)
        mean, logvar = mint.split(h, [h.shape[1] // 2, h.shape[1] // 2], dim=1)

        return mean, logvar

    def sample(self, mean, logvar):
        # sample z from latent distribution
        logvar = mint.clamp(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)
        z = mean + std * self.stdnormal(size=mean.shape)

        return z

    def encode(self, x):
        if self.use_tiling and (
            x.shape[-1] > self.tile_sample_min_size
            or x.shape[-2] > self.tile_sample_min_size
            or x.shape[-3] > self.tile_sample_min_size_t
        ):
            posterior_mean, posterior_logvar = self.tiled_encode(x)
        else:
            # embedding, get latent representation z
            posterior_mean, posterior_logvar = self._encode(x)
        z = self.sample(posterior_mean, posterior_logvar)

        return z

    def tiled_encode2d(self, x):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = ()
        tile = None
        for i in range(0, x.shape[3], overlap_size):
            row = ()
            if self.depend is not None:
                x = self.depend(x, tile)
            for j in range(0, x.shape[4], overlap_size):
                if self.depend is not None:
                    x = self.depend(x, tile)
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                if self.use_quant_layer:
                    tile = self.quant_conv(tile)
                row += (tile,)
            rows += (row,)

        result_rows = ()
        for i, row in enumerate(rows):
            result_row = ()
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row += (tile[:, :, :, :row_limit, :row_limit],)
            result_rows += (mint.cat(result_row, dim=4),)

        moments = mint.cat(result_rows, dim=3)
        return moments

    def tiled_encode(self, x):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_sample_min_size_t - 1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [[t_chunk_idx[i], t_chunk_idx[i + 1] + 1] for i in range(len(t_chunk_idx) - 1)]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        moments = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start:end]
            if idx != 0:
                moment = self.tiled_encode2d(chunk_x)[:, :, 1:]
            else:
                moment = self.tiled_encode2d(chunk_x)
            moments.append(moment)
        moments = mint.cat(moments, dim=2)
        mean, logvar = mint.split(moments, [moments.shape[1] // 2, moments.shape[1] // 2], dim=1)
        return mean, logvar

    def decode(self, z):
        if self.use_tiling and (
            z.shape[-1] > self.tile_latent_min_size
            or z.shape[-2] > self.tile_latent_min_size
            or z.shape[-3] > self.tile_latent_min_size_t
        ):
            return self.tiled_decode(z)
        if self.use_quant_layer:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def tiled_decode2d(self, z):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                if self.use_quant_layer:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(mint.cat(result_row, dim=4))

        dec = mint.cat(result_rows, dim=3)
        return dec

    def tiled_decode(self, x):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_latent_min_size_t - 1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [[t_chunk_idx[i], t_chunk_idx[i + 1] + 1] for i in range(len(t_chunk_idx) - 1)]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        dec_ = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start:end]
            if idx != 0:
                dec = self.tiled_decode2d(chunk_x)[:, :, 1:]
            else:
                dec = self.tiled_decode2d(chunk_x)
            dec_.append(dec)
        dec_ = mint.cat(dec_, dim=2)
        return dec_

    def construct(self, input):
        # overall pass, mostly for training
        posterior_mean, posterior_logvar = self._encode(input)
        z = self.sample(posterior_mean, posterior_logvar)

        recons = self.decode(z)

        return recons, posterior_mean, posterior_logvar

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def blend_v(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def validation_step(self, batch_idx):
        raise NotImplementedError


class Encoder(nn.Cell):
    """
    default value aligned to v1.1 vae config.json
    """

    def __init__(
        self,
        z_channels: int = 4,
        hidden_size: int = 128,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (),
        conv_in: str = "Conv2d",
        conv_out: str = "CausalConv3d",
        attention: str = "AttnBlock3D",  # already fixed, same as AttnBlock3DFix
        resnet_blocks: Tuple[str] = (
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        spatial_downsample: Tuple[str] = (
            "Downsample",
            "Downsample",
            "Downsample",
            "",
        ),
        temporal_downsample: Tuple[str] = (
            "",
            "TimeDownsampleRes2x",
            "TimeDownsampleRes2x",
            "",
        ),
        mid_resnet: str = "ResnetBlock3D",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        double_z: bool = True,
        upcast_sigmoid=False,
        dtype=ms.float32,
        **ignore_kwargs,
    ):
        """
        ch: hidden size, i.e. output channels of the first conv layer. typical: 128
        out_ch: placeholder, not used in Encoder
        hidden_size_mult: channel multiply factors for each res block, also determine the number of res blocks.
            Each block will be applied with spatial downsample x2 except for the last block.
            In total, the spatial downsample rate = 2**(len(hidden_size_mult)-1)
        resolution: spatial resolution, 256
        time_compress: the begging `time_compress` blocks will be applied with temporal downsample x2.
            In total, the temporal downsample rate = 2**time_compress
        """
        super().__init__()
        assert len(resnet_blocks) == len(hidden_size_mult), print(hidden_size_mult, resnet_blocks)
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        self.dtype = dtype
        self.upcast_sigmoid = (upcast_sigmoid,)

        # 1. Input conv
        self.conv_in_name = conv_in
        if conv_in == "Conv2d":
            self.conv_in = nn.Conv2d(3, hidden_size, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        elif conv_in == "CausalConv3d":
            self.conv_in = CausalConv3d(
                3,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            raise NotImplementedError

        # 2. Downsample
        curr_res = resolution
        in_ch_mult = (1,) + tuple(hidden_size_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.CellList(auto_prefix=False)
        self.downsample_flag = [0] * self.num_resolutions
        self.time_downsample_flag = [0] * self.num_resolutions
        for i_level in range(self.num_resolutions):
            block = nn.CellList()
            attn = nn.CellList()
            block_in = hidden_size * in_ch_mult[i_level]  # input channels
            block_out = hidden_size * hidden_size_mult[i_level]  # output channels
            for i_block in range(self.num_res_blocks):
                block.append(
                    resolve_str_to_obj(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        dtype=self.dtype,
                        upcast_sigmoid=upcast_sigmoid,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(resolve_str_to_obj(attention)(block_in, dtype=self.dtype))

            down = nn.Cell()
            down.block = block
            down.attn = attn

            # do spatial downsample according to config
            if spatial_downsample[i_level]:
                down.downsample = resolve_str_to_obj(spatial_downsample[i_level])(block_in, block_in, dtype=self.dtype)
                curr_res = curr_res // 2
                self.downsample_flag[i_level] = 1
            else:
                # TODO: still need it for 910b in new MS version?
                down.downsample = nn.Identity()

            # do temporal downsample according to config
            if temporal_downsample[i_level]:
                # TODO: add dtype support?
                down.time_downsample = resolve_str_to_obj(temporal_downsample[i_level])(block_in, block_in)
                self.time_downsample_flag[i_level] = 1
            else:
                # TODO: still need it for 910b in new MS version?
                down.time_downsample = nn.Identity()

            down.update_parameters_name(prefix=self.param_prefix + f"down.{i_level}.")
            self.down.append(down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            dtype=self.dtype,
            upcast_sigmoid=upcast_sigmoid,
        )
        self.mid.attn_1 = resolve_str_to_obj(attention)(block_in, dtype=self.dtype)
        self.mid.block_2 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            dtype=self.dtype,
            upcast_sigmoid=upcast_sigmoid,
        )
        self.mid.update_parameters_name(prefix=self.param_prefix + "mid.")

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # self.norm_out = Normalize(block_in, extend=True)

        assert conv_out == "CausalConv3d", "Only CausalConv3d is supported for conv_out"
        self.conv_out = resolve_str_to_obj(conv_out)(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    # copied from models.causalvideovae.model.modules.conv
    def rearrange_in(self, x):
        # b c f h w -> b f c h w
        B, C, F, H, W = x.shape
        x = mint.permute(x, (0, 2, 1, 3, 4))
        # -> (b*f c h w)
        x = ops.reshape(x, (-1, C, H, W))

        return x

    # copied from models.causalvideovae.model.modules.conv
    def rearrange_out(self, x, F):
        BF, D, H_, W_ = x.shape
        # (b*f D h w) -> (b f D h w)
        x = ops.reshape(x, (BF // F, F, D, H_, W_))
        # -> (b D f h w)
        x = mint.permute(x, (0, 2, 1, 3, 4))

        return x

    def construct(self, x):
        # downsampling
        if self.conv_in_name != "Conv2d":
            hs = self.conv_in(x)
        else:
            F = x.shape[-3]
            x = self.rearrange_in(x)
            x = self.conv_in(x)
            hs = self.rearrange_out(x, F)

        h = hs
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # import pdb; pdb.set_trace()
                h = self.down[i_level].block[i_block](hs)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs = h
            # if hasattr(self.down[i_level], "downsample"):
            #    if not isinstance(self.down[i_level].downsample, nn.Identity):
            if self.downsample_flag[i_level]:
                hs = self.down[i_level].downsample(hs)
            # if hasattr(self.down[i_level], "time_downsample"):
            #    if not isinstance(self.down[i_level].time_downsample, nn.Identity):
            if self.time_downsample_flag[i_level]:
                hs_down = self.down[i_level].time_downsample(hs)
                hs = hs_down

        # middle
        # h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h, upcast=self.upcast_sigmoid)
        h = self.conv_out(h)
        return h


class Decoder(nn.Cell):
    """
    default value aligned to v1.1 vae config.json
    """

    def __init__(
        self,
        z_channels: int = 4,
        hidden_size: int = 128,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (),
        conv_in: str = "CausalConv3d",
        conv_out: str = "CausalConv3d",
        attention: str = "AttnBlock3D",  # already fixed, same as AttnBlock3DFix
        resnet_blocks: Tuple[str] = (
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
            "ResnetBlock3D",
        ),
        spatial_upsample: Tuple[str] = ("", "SpatialUpsample2x", "SpatialUpsample2x", "SpatialUpsample2x"),
        temporal_upsample: Tuple[str] = ("", "", "TimeUpsampleRes2x", "TimeUpsampleRes2x"),
        mid_resnet: str = "ResnetBlock3D",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        double_z: bool = True,
        upcast_sigmoid=False,
        dtype=ms.float32,
        **ignore_kwargs,
    ):
        super().__init__()

        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        self.dtype = dtype
        self.upcast_sigmoid = upcast_sigmoid

        # 1. decode input z conv
        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = hidden_size * hidden_size_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # self.z_shape = (1, z_channels, curr_res, curr_res)
        # logger.info("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        assert conv_in == "CausalConv3d", "Only CausalConv3d is supported for conv_in in Decoder currently"
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, padding=1)

        # 2. middle
        self.mid = nn.Cell()
        self.mid.block_1 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in, out_channels=block_in, dropout=dropout, dtype=self.dtype
        )
        self.mid.attn_1 = resolve_str_to_obj(attention)(block_in, dtype=self.dtype)
        self.mid.block_2 = resolve_str_to_obj(mid_resnet)(
            in_channels=block_in, out_channels=block_in, dropout=dropout, dtype=self.dtype
        )
        self.mid.update_parameters_name(prefix=self.param_prefix + "mid.")

        # 3. upsampling
        self.up = nn.CellList(auto_prefix=False)
        self.upsample_flag = [0] * self.num_resolutions
        self.time_upsample_flag = [0] * self.num_resolutions
        # i_level: 3 -> 2 -> 1 -> 0
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.CellList()
            attn = nn.CellList()
            block_out = hidden_size * hidden_size_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    resolve_str_to_obj(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        dtype=self.dtype,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(resolve_str_to_obj(attention)(block_in, dtype=self.dtype))
            up = nn.Cell()
            up.block = block
            up.attn = attn
            # do spatial upsample x2 except for the first block
            if spatial_upsample[i_level]:
                up.upsample = resolve_str_to_obj(spatial_upsample[i_level])(block_in, block_in, dtype=self.dtype)
                curr_res = curr_res * 2
                self.upsample_flag[i_level] = 1
            else:
                up.upsample = nn.Identity()
            # do temporal upsample x2 in the bottom tc blocks
            if temporal_upsample[i_level]:
                # TODO: support dtype?
                up.time_upsample = resolve_str_to_obj(temporal_upsample[i_level])(block_in, block_in)
                self.time_upsample_flag[i_level] = 1
            else:
                up.time_upsample = nn.Identity()

            up.update_parameters_name(prefix=self.param_prefix + f"up.{i_level}.")
            if len(self.up) != 0:
                self.up.insert(0, up)
            else:
                self.up.append(up)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # self.norm_out = Normalize(block_in, extend=True)

        assert conv_out == "CausalConv3d", "Only CausalConv3d is supported for conv_out in Decoder currently"
        self.conv_out = CausalConv3d(block_in, 3, kernel_size=3, padding=1)

    def construct(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        i_level = self.num_resolutions
        while i_level > 0:
            i_level -= 1
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            # if hasattr(self.up[i_level], 'upsample'):
            #    if not isinstance(self.up[i_level].upsample, nn.Identity):
            if self.upsample_flag[i_level]:
                h = self.up[i_level].upsample(h)

            # if hasattr(self.up[i_level], 'time_upsample'):
            #    if not isinstance(self.up[i_level].time_upsample, nn.Identity):
            if self.time_upsample_flag[i_level]:
                h = self.up[i_level].time_upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h, upcast=self.upcast_sigmoid)
        h = self.conv_out(h)
        return h
