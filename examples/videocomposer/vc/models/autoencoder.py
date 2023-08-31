import numpy as np

import mindspore as ms
from mindspore import nn, ops

from ..utils.pt2ms import load_pt_weights_in_model

__all__ = [
    "get_first_stage_encoding",
    "DiagonalGaussianDistribution",
    "AutoencoderKL",
]


def nonlinearity(x):
    # swish
    return x * ops.sigmoid(x)


def normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


# @torch.no_grad()
def get_first_stage_encoding(encoder_posterior):
    scale_factor = 0.18215
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample()
    elif isinstance(encoder_posterior, ms.Tensor):
        z = encoder_posterior
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    return scale_factor * z


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = ops.chunk(parameters, 2, axis=1)
        self.logvar = ops.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = ops.exp(0.5 * self.logvar)
        self.var = ops.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = ops.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * ops.randn(self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return ms.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * ops.sum(ops.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * ops.sum(
                    ops.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return ms.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * ops.sum(logtwopi + self.logvar + ops.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


class ResnetBlock(nn.Cell):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        if temb_channels > 0:
            self.temb_proj = nn.Dense(temb_channels, out_channels, has_bias=True)
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=True
                )

    def construct(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=True)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=True)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=True)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=True
        )

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.transpose(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = ops.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = ops.softmax(w_, axis=2)
        # todo: bmm(fp16), softmax(fp32), scale before bmm

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.transpose(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = ops.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Upsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
            )

    def construct(self, x):
        x = ops.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2), mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, pad_mode="pad", padding=0, has_bias=True
            )

    def construct(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = ops.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = ops.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Encoder(nn.Cell):
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
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"  # noqa
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = []
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Cell()
            down.block = nn.CellList(block)
            down.attn = nn.CellList(attn)
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.down = nn.CellList(self.down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # end
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )

    def construct(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
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
        attn_type="vanilla",
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"  # noqa
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)  # noqa
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # upsampling
        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Cell()
            up.block = nn.CellList(block)
            up.attn = nn.CellList(attn)
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        self.up = nn.CellList(self.up)

        # end
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True)

    def construct(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = ops.tanh(h)
        return h


class AutoencoderKL(nn.Cell):
    def __init__(
        self,
        ddconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=None,
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        ema_decay=None,
        learn_logvar=False,
    ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1, has_bias=True)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1, has_bias=True)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.colorize = ms.Parameter(
                ms.Tensor(np.random.randn(3, colorize_nlabels, 1, 1), dtype=ms.float32), requires_grad=False
            )
        if monitor is not None:
            self.monitor = monitor
        self.use_ema = ema_decay is not None
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=None):
        def prune_weights(sd):
            # find keys start with prefix "first_stage_model" and remove the prefix.
            sd_new = {}
            for k in sd.keys():
                if k.find("first_stage_model") >= 0:
                    k_new = k.split("first_stage_model.")[-1]
                    sd_new[k_new] = sd[k]
            return sd_new

        load_pt_weights_in_model(self, path, (prune_weights,))

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def construct(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).float()
        return x

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(ops.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
            if log_ema or self.use_ema:
                with self.ema_scope():
                    xrec_ema, posterior_ema = self(x)
                    if x.shape[1] > 3:
                        # colorize with random projection
                        assert xrec_ema.shape[1] > 3
                        xrec_ema = self.to_rgb(xrec_ema)
                    log["samples_ema"] = self.decode(ops.randn_like(posterior_ema.sample()))
                    log["reconstructions_ema"] = xrec_ema
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.colorize = ms.Parameter(
                ms.Tensor(np.randn(3, x.shape[1], 1, 1), dtype=ms.float32), requires_grad=False
            )
        x = ops.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


class IdentityFirstStage(nn.Cell):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def construct(self, x, *args, **kwargs):
        return x
