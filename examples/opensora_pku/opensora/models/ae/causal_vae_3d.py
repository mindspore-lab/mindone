import numpy as np

import mindspore as ms
from mindspore import nn, ops

from .modules import (
    CausalConv3d,
    Normalize,
    ResnetBlock3D,
    SpatialDownsample2x,
    SpatialUpsample2x,
    TimeDownsample2x,
    TimeUpsample2x,
    make_attn,
    nonlinearity,
)


class CausalVAEModel(nn.Cell):
    def __init__(
        self,
        ddconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        colorize_nlabels=None,
        monitor=None,
        use_fp16=False,
        upcast_sigmoid=False,
    ):
        super().__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        # print("D--: ddconfig: ", ddconfig)

        self.encoder = Encoder(dtype=self.dtype, upcast_sigmoid=upcast_sigmoid, **ddconfig)
        self.decoder = Decoder(dtype=self.dtype, upcast_sigmoid=upcast_sigmoid, **ddconfig)
        assert ddconfig["double_z"]
        if ddconfig["split_time_upsample"]:
            print("Exclude first frame from time upsample")
        self.quant_conv = CausalConv3d(
            2 * ddconfig["z_channels"],
            2 * embed_dim,
            1,
        )
        self.post_quant_conv = CausalConv3d(
            embed_dim,
            ddconfig["z_channels"],
            1,
        )
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", ms.ops.standard_normal(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.split = ops.Split(axis=1, output_num=2)
        self.exp = ops.Exp()
        self.stdnormal = ops.StandardNormal()

        # self.encoder.recompute()
        # self.decoder.recompute()

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

            assert key_2d in vae_2d_keys, f"Key {key_2d} ({key_3d}) not found in 2D VAE"

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
                    print(key_2d, shape_3d, shape_2d)
                w = vae2d_sd[key_2d]
                new_w = ms.ops.zeros(shape_3d, dtype=w.dtype)
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
                print("net param not loaded: ", m)
            if len(u) > 0:
                print("checkpoint param not loaded: ", u)

    def init_from_ckpt(self, path, ignore_keys=list(), remove_prefix=["first_stage_model.", "autoencoder."]):
        # TODO: support auto download pretrained checkpoints
        sd = ms.load_checkpoint(path)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        ms.load_param_into_net(self, sd, strict_load=False)
        print(f"Restored from {path}")

    def _encode(self, x):
        # return latent distribution, N(mean, logvar)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = self.split(moments)

        return mean, logvar

    def sample(self, mean, logvar):
        # sample z from latent distribution
        logvar = ops.clip_by_value(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)
        z = mean + std * self.stdnormal(mean.shape)

        return z

    def encode(self, x):
        # embedding, get latent representation z
        posterior_mean, posterior_logvar = self._encode(x)
        z = self.sample(posterior_mean, posterior_logvar)

        return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def construct(self, input):
        # overall pass, mostly for training
        posterior_mean, posterior_logvar = self._encode(input)
        z = self.sample(posterior_mean, posterior_logvar)

        recons = self.decode(z)

        return recons, posterior_mean, posterior_logvar


class Encoder(nn.Cell):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla3D",  # diff 3d
        dtype=ms.float32,
        time_compress=2,  # diff 3d
        upcast_sigmoid=False,
        **ignore_kwargs,
    ):
        """
        ch: hidden size, i.e. output channels of the first conv layer. typical: 128
        out_ch: placeholder, not used in Encoder
        ch_mult: channel multiply factors for each res block, also determine the number of res blocks.
            Each block will be applied with spatial downsample x2 except for the last block. In total, the spatial downsample rate = 2**(len(ch_mult)-1)
        resolution: spatial resolution, 256
        time_compress: the begging `time_compress` blocks will be applied with temporal downsample x2. In total, the temporal downsample rate = 2**time_compress
        """
        # TODO: For input to AttnBlock3D, T=1, it's better to squeeze and use 2D AttnBlock
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.time_compress = time_compress
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.dtype = dtype
        self.upcast_sigmoid = (upcast_sigmoid,)

        # downsampling
        # diff 3d
        self.conv_in = CausalConv3d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.CellList(auto_prefix=False)
        for i_level in range(self.num_resolutions):
            block = nn.CellList()
            attn = nn.CellList()
            block_in = ch * in_ch_mult[i_level]  # input channels
            block_out = ch * ch_mult[i_level]  # output channels
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        dtype=self.dtype,
                        upcast_sigmoid=upcast_sigmoid,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dtype=self.dtype))
            down = nn.Cell()
            down.block = block
            down.attn = attn
            # do spatial downsample except for the last block
            if i_level != self.num_resolutions - 1:
                down.downsample = SpatialDownsample2x(block_in, block_in, dtype=self.dtype)
                curr_res = curr_res // 2
            else:
                down.downsample = nn.Identity()
            # do temporal downsample for beginning tc blocks
            if i_level < self.time_compress:
                down.time_downsample = TimeDownsample2x()
            else:
                down.time_downsample = nn.Identity()

            down.update_parameters_name(prefix=self.param_prefix + f"down.{i_level}.")
            self.down.append(down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            dtype=self.dtype,
            upcast_sigmoid=upcast_sigmoid,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dtype=self.dtype)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            dtype=self.dtype,
            upcast_sigmoid=upcast_sigmoid,
        )
        self.mid.update_parameters_name(prefix=self.param_prefix + "mid.")

        # end
        self.norm_out = Normalize(block_in, extend=True)
        self.conv_out = CausalConv3d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def construct(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            if i_level < self.time_compress:
                hs.append(self.down[i_level].time_downsample(hs[-1]))

        # import pdb
        # pdb.set_trace()
        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h, upcast=self.upcast_sigmoid)
        h = self.conv_out(h)
        return h


class Decoder(nn.Cell):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla3D",
        time_compress=2,
        split_time_upsample=True,  # TODO: ablate
        dtype=ms.float32,
        upcast_sigmoid=False,
        **ignorekwargs,
    ):
        super().__init__()
        # if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.time_compress = time_compress
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.dtype = dtype
        self.upcast_sigmoid = upcast_sigmoid

        # compute in_ch_mult, block_in and curr_res at lowest res
        # in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, padding=1)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in, out_channels=block_in, dropout=dropout, dtype=self.dtype)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, dtype=self.dtype)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in, out_channels=block_in, dropout=dropout, dtype=self.dtype)
        self.mid.update_parameters_name(prefix=self.param_prefix + "mid.")

        # upsampling
        self.up = nn.CellList(auto_prefix=False)
        # i_level: 3 -> 2 -> 1 -> 0
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.CellList()
            attn = nn.CellList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        dtype=self.dtype,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, dtype=self.dtype))
            up = nn.Cell()
            up.block = block
            up.attn = attn
            # do spatial upsample x2 except for the first block
            if i_level != 0:
                up.upsample = SpatialUpsample2x(block_in, block_in, dtype=self.dtype)
                curr_res = curr_res * 2
            else:
                up.upsample = nn.Identity()
            # do temporal upsample x2 in the bottom tc blocks. TODO: choices for block positions to be injected with temporal upsample.
            if i_level > self.num_resolutions - 1 - self.time_compress and i_level != 0:
                up.time_upsample = TimeUpsample2x(exclude_first_frame=split_time_upsample)
            else:
                up.time_upsample = nn.Identity()

            up.update_parameters_name(prefix=self.param_prefix + f"up.{i_level}.")
            if len(self.up) != 0:
                self.up.insert(0, up)
            else:
                self.up.append(up)

        # end
        self.norm_out = Normalize(block_in, extend=True)

        self.conv_out = CausalConv3d(block_in, out_ch, kernel_size=3, padding=1)

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
            if i_level != 0:
                h = self.up[i_level].upsample(h)

            if i_level > self.num_resolutions - 1 - self.time_compress and i_level != 0:
                h = self.up[i_level].time_upsample(h)

        # end
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h, upcast=self.upcast_sigmoid)
        h = self.conv_out(h)
        if self.tanh_out:
            h = ops.tanh(h)
        return h
