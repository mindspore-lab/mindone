import logging
import math
import os
from collections import deque
from typing import List

from opensora.npu_config import npu_config

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import HeUniform, Uniform

from mindone.diffusers import __version__
from mindone.diffusers.configuration_utils import register_to_config
from mindone.diffusers.models.modeling_utils import load_state_dict
from mindone.diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, _add_variant, _get_model_file

from ..modeling_videobase import VideoBaseAE
from ..modules import (
    CausalConv3d,
    Conv2d,
    HaarWaveletTransform3D,
    InverseHaarWaveletTransform3D,
    Normalize,
    ResnetBlock2D,
    ResnetBlock3D,
    nonlinearity,
)
from ..registry import ModelRegistry
from ..utils.model_utils import resolve_str_to_obj

logger = logging.getLogger(__name__)


class Encoder(VideoBaseAE):
    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        num_resblocks: int = 2,
        energy_flow_hidden_size: int = 64,
        dropout: float = 0.0,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        norm_type: str = "groupnorm",
        l1_dowmsample_block: str = "Downsample",
        l1_downsample_wavelet: str = "HaarWaveletTransform2D",
        l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
        l2_downsample_wavelet: str = "HaarWaveletTransform3D",
        dtype=ms.float32,
    ) -> None:
        super().__init__()

        self.down1 = nn.SequentialCell(
            Conv2d(
                24,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode="pad",
                has_bias=True,
                weight_init=HeUniform(negative_slope=math.sqrt(5)),
                bias_init=Uniform(scale=1 / math.sqrt(24 * 3 * 3)),
            ).to_float(dtype),
            *[
                ResnetBlock2D(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    dtype=dtype,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l1_dowmsample_block)(in_channels=base_channels, out_channels=base_channels, dtype=dtype),
        )
        self.down2 = nn.SequentialCell(
            Conv2d(
                base_channels + energy_flow_hidden_size,
                base_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode="pad",
                has_bias=True,
                weight_init=HeUniform(negative_slope=math.sqrt(5)),
                bias_init=Uniform(scale=1 / math.sqrt((base_channels + energy_flow_hidden_size) * 3 * 3)),
            ).to_float(dtype),
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 2,
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                    dtype=dtype,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l2_dowmsample_block)(base_channels * 2, base_channels * 2, dtype=dtype),
        )
        # Connection
        if l1_dowmsample_block == "Downsample":  # Bad code. For temporal usage.
            l1_channels = 12
        else:
            l1_channels = 24

        self.connect_l1 = Conv2d(
            l1_channels,
            energy_flow_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode="pad",
            has_bias=True,
            weight_init=HeUniform(negative_slope=math.sqrt(5)),
            bias_init=Uniform(scale=1 / math.sqrt(l1_channels * 3 * 3)),
        ).to_float(dtype)
        self.connect_l2 = Conv2d(
            24,
            energy_flow_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode="pad",
            has_bias=True,
            weight_init=HeUniform(negative_slope=math.sqrt(5)),
            bias_init=Uniform(scale=1 / math.sqrt(24 * 3 * 3)),
        ).to_float(dtype)
        # Mid
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 2 + energy_flow_hidden_size,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
                dtype=dtype,
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
                dtype=dtype,
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, resolve_str_to_obj(attention_type)(in_channels=base_channels * 4, norm_type=norm_type, dtype=dtype)
            )
        self.mid = nn.SequentialCell(*mid_layers)
        self.norm_out = Normalize(base_channels * 4, norm_type=norm_type)
        self.conv_out = CausalConv3d(base_channels * 4, latent_dim * 2, kernel_size=3, stride=1, padding=1)

        self.wavelet_transform_in = HaarWaveletTransform3D()
        self.wavelet_transform_l1 = resolve_str_to_obj(l1_downsample_wavelet)()
        self.wavelet_transform_l2 = resolve_str_to_obj(l2_downsample_wavelet)()

    def construct(self, x):
        coeffs = self.wavelet_transform_in(x)
        l1_coeffs = coeffs[:, :3]
        l1_coeffs = self.wavelet_transform_l1(l1_coeffs)
        l1 = self.connect_l1(l1_coeffs)
        l2_coeffs = self.wavelet_transform_l2(l1_coeffs[:, :3])
        l2 = self.connect_l2(l2_coeffs)

        h = self.down1(coeffs)
        h = mint.cat([h, l1], dim=1)
        h = self.down2(h)
        h = mint.cat([h, l2], dim=1)
        h = self.mid(h)

        if npu_config is None:
            h = self.norm_out(h)
        else:
            h = npu_config.run_group_norm(self.norm_out, h)

        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, (l1_coeffs, l2_coeffs)


class Decoder(VideoBaseAE):
    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        num_resblocks: int = 2,
        dropout: float = 0.0,
        energy_flow_hidden_size: int = 128,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        norm_type: str = "groupnorm",
        t_interpolation: str = "nearest",
        connect_res_layer_num: int = 1,
        l1_upsample_block: str = "Upsample",
        l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
        l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
        l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
        dtype=ms.float32,
    ) -> None:
        super().__init__()

        self.energy_flow_hidden_size = energy_flow_hidden_size

        self.conv_in = CausalConv3d(latent_dim, base_channels * 4, kernel_size=3, stride=1, padding=1)
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
                dtype=dtype,
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
                dtype=dtype,
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, resolve_str_to_obj(attention_type)(in_channels=base_channels * 4, norm_type=norm_type, dtype=dtype)
            )
        self.mid = nn.SequentialCell(*mid_layers)
        self.up2 = nn.SequentialCell(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    dropout=dropout,
                    norm_type=norm_type,
                    dtype=dtype,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l2_upsample_block)(
                base_channels * 4, base_channels * 4, t_interpolation=t_interpolation, dtype=dtype
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                dropout=dropout,
                norm_type=norm_type,
                dtype=dtype,
            ),
        )
        self.up1 = nn.SequentialCell(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (4 if i == 0 else 2),
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                    dtype=dtype,
                )
                for i in range(num_resblocks)
            ],
            resolve_str_to_obj(l1_upsample_block)(
                in_channels=base_channels * 2, out_channels=base_channels * 2, dtype=dtype
            ),
            ResnetBlock3D(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                dropout=dropout,
                norm_type=norm_type,
                dtype=dtype,
            ),
        )
        self.layer = nn.SequentialCell(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (2 if i == 0 else 1),
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    dtype=dtype,
                )
                for i in range(2)
            ],
        )
        # Connection
        if l1_upsample_block == "Upsample":  # Bad code. For temporal usage.
            l1_channels = 12
        else:
            l1_channels = 24
        self.connect_l1 = nn.SequentialCell(
            *[
                ResnetBlock3D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    dtype=dtype,
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(
                energy_flow_hidden_size,
                l1_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode="pad",
                has_bias=True,
                weight_init=HeUniform(negative_slope=math.sqrt(5)),
                bias_init=Uniform(scale=1 / math.sqrt(energy_flow_hidden_size * 3 * 3)),
            ).to_float(dtype),
        )
        self.connect_l2 = nn.SequentialCell(
            *[
                ResnetBlock3D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    dtype=dtype,
                )
                for _ in range(connect_res_layer_num)
            ],
            Conv2d(
                energy_flow_hidden_size,
                24,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode="pad",
                has_bias=True,
                weight_init=HeUniform(negative_slope=math.sqrt(5)),
                bias_init=Uniform(scale=1 / math.sqrt(energy_flow_hidden_size * 3 * 3)),
            ).to_float(dtype),
        )
        # Out
        self.norm_out = Normalize(base_channels, norm_type=norm_type)
        self.conv_out = Conv2d(
            base_channels,
            24,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode="pad",
            has_bias=True,
            weight_init=HeUniform(negative_slope=math.sqrt(5)),
            bias_init=Uniform(scale=1 / math.sqrt(base_channels * 3 * 3)),
        ).to_float(dtype)

        self.inverse_wavelet_transform_out = InverseHaarWaveletTransform3D()
        self.inverse_wavelet_transform_l1 = resolve_str_to_obj(l1_upsample_wavelet)()
        self.inverse_wavelet_transform_l2 = resolve_str_to_obj(l2_upsample_wavelet)()

    def construct(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        l2_coeffs = self.connect_l2(h[:, -self.energy_flow_hidden_size :])
        l2 = self.inverse_wavelet_transform_l2(l2_coeffs)

        h = self.up2(h[:, : -self.energy_flow_hidden_size])

        l1_coeffs = h[:, -self.energy_flow_hidden_size :]
        l1_coeffs = self.connect_l1(l1_coeffs)
        l1_coeffs[:, :3] = l1_coeffs[:, :3] + l2
        l1 = self.inverse_wavelet_transform_l1(l1_coeffs)

        h = self.up1(h[:, : -self.energy_flow_hidden_size])

        h = self.layer(h)
        h = npu_config.run_group_norm(self.norm_out, h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h[:, :3] = h[:, :3] + l1
        dec = self.inverse_wavelet_transform_out(h)
        return dec, (l1_coeffs, l2_coeffs)


@ModelRegistry.register("WFVAE")
class WFVAEModel(VideoBaseAE):
    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        encoder_num_resblocks: int = 2,
        encoder_energy_flow_hidden_size: int = 64,
        decoder_num_resblocks: int = 2,
        decoder_energy_flow_hidden_size: int = 128,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
        t_interpolation: str = "nearest",
        connect_res_layer_num: int = 1,
        scale: List[float] = [0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215],
        shift: List[float] = [0, 0, 0, 0, 0, 0, 0, 0],
        # Module config
        l1_dowmsample_block: str = "Downsample",
        l1_downsample_wavelet: str = "HaarWaveletTransform2D",
        l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
        l2_downsample_wavelet: str = "HaarWaveletTransform3D",
        l1_upsample_block: str = "Upsample",
        l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
        l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
        l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
        dtype=ms.float32,
        use_recompute=False,
    ) -> None:
        super().__init__()
        self.use_tiling = False

        # Hardcode for now
        self.t_chunk_enc = 8
        self.t_chunk_dec = 2
        self.t_upsample_times = 4 // 2

        self.use_quant_layer = False
        self.encoder = Encoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=encoder_num_resblocks,
            energy_flow_hidden_size=encoder_energy_flow_hidden_size,
            dropout=dropout,
            use_attention=use_attention,
            norm_type=norm_type,
            l1_dowmsample_block=l1_dowmsample_block,
            l1_downsample_wavelet=l1_downsample_wavelet,
            l2_dowmsample_block=l2_dowmsample_block,
            l2_downsample_wavelet=l2_downsample_wavelet,
            attention_type=attention_type,
            dtype=dtype,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=decoder_num_resblocks,
            energy_flow_hidden_size=decoder_energy_flow_hidden_size,
            dropout=dropout,
            use_attention=use_attention,
            norm_type=norm_type,
            t_interpolation=t_interpolation,
            connect_res_layer_num=connect_res_layer_num,
            l1_upsample_block=l1_upsample_block,
            l1_upsample_wavelet=l1_upsample_wavelet,
            l2_upsample_block=l2_upsample_block,
            l2_upsample_wavelet=l2_upsample_wavelet,
            attention_type=attention_type,
            dtype=dtype,
        )

        # Set cache offset for trilinear lossless upsample.
        self._set_cache_offset([self.decoder.up2, self.decoder.connect_l2, self.decoder.conv_in, self.decoder.mid], 1)
        self._set_cache_offset(
            [self.decoder.up2[-2:], self.decoder.up1, self.decoder.connect_l1, self.decoder.layer],
            self.t_upsample_times,
        )

        self.exp = mint.exp
        self.stdnormal = ops.standard_normal

        self.update_parameters_name()  # update parameter names to solve pname mismatch
        if use_recompute:
            self.recompute(self.encoder)
            self.recompute(self.decoder)
            if self.use_quant_layer:
                self.recompute(self.quant_conv)
                self.recompute(self.post_quant_conv)

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute(parallel_optimizer_comm_recompute=True)
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            b.add_flags(output_no_recompute=True)

    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder]
        return [self.encoder]

    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]

    def _empty_causal_cached(self, parent):
        for name, module in parent.cells_and_names():
            if hasattr(module, "causal_cached"):
                module.causal_cached = deque()

    def _set_causal_cached(self, enable_cached=True):
        for name, module in self.cells_and_names():
            if hasattr(module, "enable_cached"):
                module.enable_cached = enable_cached

    def _set_cache_offset(self, modules, cache_offset=0):
        for module in modules:
            for submodule in module.cells():
                if hasattr(submodule, "cache_offset"):
                    submodule.cache_offset = cache_offset

    def _set_first_chunk(self, is_first_chunk=True):
        for _, module in self.cells_and_names():
            if hasattr(module, "is_first_chunk"):
                module.is_first_chunk = is_first_chunk

    def build_chunk_start_end(self, t, decoder_mode=False):
        start_end = [[0, 1]]
        start = 1
        end = start
        while True:
            if start >= t:
                break
            end = min(t, end + (self.t_chunk_dec if decoder_mode else self.t_chunk_enc))
            start_end.append([start, end])
            start = end
        return start_end

    def encode(self, x, sample_posterior=True):
        posterior_mean, posterior_logvar, _ = self._encode(x)
        if sample_posterior:
            z = self.sample(posterior_mean, posterior_logvar)
        else:
            z = posterior_mean

        return z

    def _encode(self, x):
        self._empty_causal_cached(self.encoder)
        self._set_first_chunk(True)

        if self.use_tiling:
            h = self.tile_encode(x)
            w_coeffs = None
        else:
            h, w_coeffs = self.encoder(x)
            if self.use_quant_layer:
                h = self.quant_conv(h)
        posterior_mean, posterior_logvar = mint.split(h, [h.shape[1] // 2, h.shape[1] // 2], dim=1)
        return posterior_mean, posterior_logvar, w_coeffs

    def tile_encode(self, x):
        b, c, t, h, w = x.shape

        start_end = self.build_chunk_start_end(t)
        result = []
        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)
            chunk = x[:, :, start:end, :, :]
            chunk = self.encoder(chunk)[0]
            if self.use_quant_layer:
                chunk = self.quant_conv(chunk)
            result.append(chunk)

        return mint.cat(result, dim=2)

    def decode(self, z):
        dec, _ = self._decode(z)

        return dec

    def _decode(self, z):
        self._empty_causal_cached(self.decoder)
        self._set_first_chunk(True)

        if self.use_tiling:
            dec = self.tile_decode(z)
            w_coeffs = None
        else:
            if self.use_quant_layer:
                z = self.post_quant_conv(z)
            dec, w_coeffs = self.decoder(z)

        return dec, w_coeffs

    def tile_decode(self, x):
        b, c, t, h, w = x.shape

        start_end = self.build_chunk_start_end(t, decoder_mode=True)

        result = []
        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)

            if end + 1 < t:
                chunk = x[:, :, start : end + 1, :, :]
            else:
                chunk = x[:, :, start:end, :, :]

            if self.use_quant_layer:
                chunk = self.post_quant_conv(chunk)
            chunk = self.decoder(chunk)[0]

            if end + 1 < t:
                chunk = chunk[:, :, :-4]
                result.append(chunk)
            else:
                result.append(chunk)

        return mint.cat(result, dim=2)

    def sample(self, mean, logvar):
        # sample z from latent distribution
        logvar = mint.clamp(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)
        z = mean + std * self.stdnormal(mean.shape)

        return z

    def construct(self, input, sample_posterior=True):
        # overall pass, mostly for training
        posterior_mean, posterior_logvar, encoder_w_coeffs = self._encode(input)
        if sample_posterior:
            z = self.sample(posterior_mean, posterior_logvar)
        else:
            z = posterior_mean

        recons, decoder_w_coeffs = self._decode(z)
        if encoder_w_coeffs is not None and decoder_w_coeffs is not None:
            assert len(encoder_w_coeffs) == 2 and len(decoder_w_coeffs) == 2
            e_l1, e_l2 = encoder_w_coeffs
            d_l1, d_l2 = decoder_w_coeffs
            w_coeffs = [e_l1, d_l1, e_l2, d_l2]
        else:
            w_coeffs = None

        return recons, posterior_mean, posterior_logvar, w_coeffs

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling
        self._set_causal_cached(use_tiling)

    def disable_tiling(self):
        self.enable_tiling(False)

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
        ignore_prefix = kwargs.pop("ignore_prefix", None)

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
            if ignore_prefix is not None:
                assert len(ignore_prefix) > 0, "the ignore_prefix must not be empty"
                num_params = len(state_dict)
                state_dict = dict(
                    [
                        (k, v)
                        for k, v in state_dict.items()
                        if not any([k.startswith(prefix) for prefix in ignore_prefix])
                    ]
                )
                logger.info(
                    f"Excluding the parameters with prefix in {ignore_prefix}: exclude {num_params - len(state_dict)} out of {num_params} params"
                )
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
