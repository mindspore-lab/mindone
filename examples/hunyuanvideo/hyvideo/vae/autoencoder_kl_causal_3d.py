import logging
from typing import Dict, Tuple, Union

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config

# Use this to be compatible with the original diffusers.
from mindone.diffusers.loaders.single_file_model import FromOriginalModelMixin as FromOriginalVAEMixin
from mindone.diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from mindone.diffusers.models.modeling_utils import ModelMixin

from .vae import DecoderCausal3D, EncoderCausal3D

logger = logging.getLogger(__name__)


class AutoencoderKLCausal3D(ModelMixin, ConfigMixin, FromOriginalVAEMixin):
    r"""
    A VAE model with KL loss for encoding images/videos into latents and decoding latent representations into images/videos.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlockCausal3D",),
        up_block_types: Tuple[str] = ("UpDecoderBlockCausal3D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        sample_tsize: int = 64,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        spatial_compression_ratio: int = 8,
        time_compression_ratio: int = 4,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()

        self.time_compression_ratio = time_compression_ratio

        self.encoder = EncoderCausal3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            time_compression_ratio=time_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.decoder = DecoderCausal3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            time_compression_ratio=time_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = nn.Conv3d(
            2 * latent_channels, 2 * latent_channels, kernel_size=1, has_bias=True, pad_mode="valid"
        )
        self.post_quant_conv = nn.Conv3d(
            latent_channels, latent_channels, kernel_size=1, has_bias=True, pad_mode="valid"
        )

        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // time_compression_ratio

        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

        self.exp = ops.Exp()
        self.stdnormal = ops.StandardNormal()

        print("D--: tile_latent_min size  ", self.tile_latent_min_tsize, self.tile_latent_min_size)

    def _set_gradient_checkpointing(self, module, value=False):
        raise NotImplementedError

    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger videos.
        """
        self.enable_spatial_tiling(use_tiling)
        self.enable_temporal_tiling(use_tiling)

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.disable_spatial_tiling()
        self.disable_temporal_tiling()

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: nn.Cell, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.cells_and_names():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.cells_and_names():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: nn.Cell, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            for sub_name, child in module.cells_and_names():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.cells_and_names():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor, _remove_lora=True)

    def sample(self, mean, logvar):
        # sample z from latent distribution
        logvar = mint.clamp(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)
        z = mean + std * self.stdnormal(mean.shape)

        return z

    def encode(self, x: ms.Tensor, sample_posterior=True) -> ms.Tensor:
        """
        Encode a batch of images/videos into latents.

        Args:
            x (`ms.Tensor`): Input batch of images/videos.
            sample_posterior (`bool`, *optional*, defaults to `True`):
                Whether to return a posterior distribution or a posterior mean.
        """
        assert len(x.shape) == 5, "The input tensor should have 5 dimensions."

        if self.use_temporal_tiling and x.shape[2] > self.tile_sample_min_tsize:
            return self.temporal_tiled_encode(x)

        if self.use_spatial_tiling and (
            x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
        ):
            return self.spatial_tiled_encode(x)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in mint.split(x, 1)]
            h = mint.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior_mean, posterior_logvar = mint.split(moments, [moments.shape[1] // 2, moments.shape[1] // 2], dim=1)

        if sample_posterior:
            z = self.sample(posterior_mean, posterior_logvar)
        else:
            z = posterior_mean

        return z

    def _decode(self, z: ms.Tensor) -> ms.Tensor:
        assert len(z.shape) == 5, "The input tensor should have 5 dimensions."

        if self.use_temporal_tiling and z.shape[2] > self.tile_latent_min_tsize:
            return self.temporal_tiled_decode(z)

        if self.use_spatial_tiling and (
            z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
        ):
            return self.spatial_tiled_decode(z)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        return dec

    def decode(
        self,
        z: ms.Tensor,
    ) -> ms.Tensor:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in mint.split(z, 1)]
            decoded = mint.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        return decoded

    def blend_v(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_t(self, a: ms.Tensor, b: ms.Tensor, blend_extent: int) -> ms.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def spatial_tiled_encode(
        self, x: ms.Tensor, sample_posterior: bool = True, return_moments: bool = False
    ) -> ms.Tensor:
        r"""Encode a batch of images/videos using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image/videos size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`ms.Tensor`): Input batch of images/videos.
            sample_posterior (`bool`, *optional*, defaults to `True`):
                Whether to return a posterior distribution or a posterior mean.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split video into tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[-2], overlap_size):
            row = []
            for j in range(0, x.shape[-1], overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
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
            result_rows.append(mint.cat(result_row, dim=-1))

        moments = mint.cat(result_rows, dim=-2)
        if return_moments:
            return moments

        posterior_mean, posterior_logvar = mint.split(moments, [moments.shape[1] // 2, moments.shape[1] // 2], dim=1)
        if sample_posterior:
            z = self.sample(posterior_mean, posterior_logvar)
        else:
            z = posterior_mean
        return z

    def spatial_tiled_decode(self, z: ms.Tensor) -> ms.Tensor:
        r"""
        Decode a batch of images/videos using a tiled decoder.

        Args:
            z (`ms.Tensor`): Input batch of latent vectors.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[-2], overlap_size):
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
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
            result_rows.append(mint.cat(result_row, dim=-1))

        dec = mint.cat(result_rows, dim=-2)
        return dec

    def temporal_tiled_encode(self, x: ms.Tensor, sample_posterior: bool = True) -> ms.Tensor:
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent

        # Split the video into tiles and encode them separately.
        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size
            ):
                tile = self.spatial_tiled_encode(tile, return_moments=True)
            else:
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])

        moments = mint.cat(result_row, dim=2)
        posterior_mean, posterior_logvar = mint.split(moments, [moments.shape[1] // 2, moments.shape[1] // 2], dim=1)
        if sample_posterior:
            z = self.sample(posterior_mean, posterior_logvar)
        else:
            z = posterior_mean
        return z

    def temporal_tiled_decode(self, z: ms.Tensor) -> ms.Tensor:
        # Split z into overlapping tiles and decode them separately.

        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size
            ):
                decoded = self.spatial_tiled_decode(tile)
            else:
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])

        dec = mint.cat(result_row, dim=2)
        return dec

    def construct(
        self,
        sample: ms.Tensor,
        sample_posterior: bool = False,
    ):
        r"""
        Args:
            sample (`ms.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
        """
        x = sample
        z = self.encode(x, sample_posterior=sample_posterior)
        dec = self.decode(z)

        return dec

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query,
        key, value) are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.cells():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def load_state_dict(self, state_dict):
        param_not_load, ckpt_not_load = ms.load_param_into_net(self, state_dict)
        logger.info(
            "Net params not load: {}, Total net params not loaded: {}".format(param_not_load, len(param_not_load))
        )
        logger.info(
            "Ckpt params not load: {}, Total ckpt params not loaded: {}".format(ckpt_not_load, len(ckpt_not_load))
        )
