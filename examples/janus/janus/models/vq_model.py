# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from dataclasses import dataclass, field
from typing import List

import mindspore as ms
from mindspore import Parameter, mint, nn, ops
from mindspore.common.initializer import Uniform

from .compat import GroupNorm, normalize_l2


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0


class Encoder(nn.Cell):
    def __init__(
        self,
        in_channels=3,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        z_channels=256,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = mint.nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.CellList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Cell()
            # res & attn
            res_block = nn.CellList()
            attn_block = nn.CellList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            else:
                conv_block.downsample = nn.Identity()
            # TODO: uncomment if ckpt loading not compelte
            # conv_block.update_parameters_name(prefix=self.param_prefix + f"down.{i_level}.")
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.CellList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = mint.nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def construct(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Cell):
    def __init__(
        self,
        z_channels=256,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        out_channels=3,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        # z to block_in
        self.conv_in = mint.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.CellList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.CellList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Cell()
            # res & attn
            res_block = nn.CellList()
            attn_block = nn.CellList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            else:
                conv_block.upsample = nn.Identity()
            # TODO: uncomment if ckpt loading not compelte
            # conv_block.update_parameters_name(prefix=self.param_prefix + f"up.{i_level}.")
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = mint.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight

    def construct(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Cell):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        # TODO: re-write the cell to map panme from ms to torch: embedding_table -> weight.
        self.embedding = nn.Embedding(self.n_e, self.e_dim, embedding_table=Uniform(scale=1.0 / self.n_e))
        if self.l2_norm:
            self.embedding.embedding_table.set_data(normalize_l2(self.embedding.embedding_table.value(), dim=-1))
        if self.show_usage:
            self.codebook_used = Parameter(ops.zeros(65536), requires_grad=False)

    def construct(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        # "b c h w -> b h w c"
        z = ops.transpose(z, (0, 2, 3, 1))
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = normalize_l2(z, dim=-1)
            z_flattened = normalize_l2(z_flattened, dim=-1)
            embedding = normalize_l2(self.embedding.embedding_table, dim=-1)
        else:
            embedding = self.embedding.embedding_table
        # TODO: double check
        d = (
            ops.sum(z_flattened**2, dim=1, keepdim=True)
            + ops.sum(embedding**2, dim=1)
            - 2 * ops.matmul(z_flattened, ops.transpose(self.embedding.embedding_table, (1, 0)))
            # "bd,dn->bn", z_flattened, torch.einsum("n d -> d n", embedding)
        )

        min_encoding_indices = ops.argmin(d, axis=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None

        # compute loss for embedding
        if self.training:
            raise NotImplementedError
            vq_loss = ops.mean((z_q - ops.stop_gradient(z)) ** 2)
            commit_loss = self.beta * ops.mean((ops.stop_gradient(z_q) - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients
        z_q = z + ops.stop_gradient(z_q - z)

        # reshape back to match original input shape
        # "b h w c -> b c h w"
        z_q = ops.transpose(z_q, (0, 3, 1, 2))  # (b h w c) -> (b c h w)

        return (
            z_q,
            (vq_loss, commit_loss, entropy_loss),
            (perplexity, min_encodings, min_encoding_indices),
        )

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            # TODO: improve l2norm accuracy in bf16
            embedding = normalize_l2(self.embedding.embedding_table, dim=-1)
        else:
            embedding = self.embedding.embedding_table
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2)
            else:
                z_q = z_q.view(shape)
        return z_q


class ResnetBlock(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        norm_type="group",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = mint.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = mint.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = mint.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = mint.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def construct(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
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
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = ops.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = ops.Softmax(axis=2)(w_)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = ops.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def nonlinearity(x):
    # swish
    # TODO: maybe cast to fp32
    return x * (ops.sigmoid(x))


def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        # TODO: check mint GroupNorm accuracy
        # return mint.nn.GroupNorm(
        return GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == "batch":
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = mint.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def construct(self, x):
        # x = F.interpolate(x.to(ms.float32), scale_factor=2.0, mode="nearest").to(
        # TODO: if use amp, need to ensure interpolation is computed in fp32
        in_dtype = x.dtype
        if x.dtype != ms.float32:
            x = x.to(ms.float32)
        # x = ops.ResizeNearestNeighbor(out_shape)(x)
        x = ops.interpolate(x, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)
        if x.dtype != in_dtype:
            x = x.to(in_dtype)

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = mint.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def construct(self, x):
        if self.with_conv:
            pad = ((0, 0), (0, 0), (0, 1), (0, 1))
            x = nn.Pad(paddings=pad)(x)
            x = self.conv(x)
        else:
            x = ops.AvgPool(kernel_size=2, stride=2)(x)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = ops.softmax(flat_affinity, dim=-1)
    log_probs = ops.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = ops.mean(target_probs, dim=0)
    avg_entropy = -ops.sum(avg_probs * ops.log(avg_probs + 1e-5))
    sample_entropy = -ops.mean(ops.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


class VQModel(nn.Cell):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            ch_mult=config.encoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )
        self.decoder = Decoder(
            ch_mult=config.decoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )

        self.quantize = VectorQuantizer(
            config.codebook_size,
            config.codebook_embed_dim,
            config.commit_loss_beta,
            config.entropy_loss_ratio,
            config.codebook_l2_norm,
            config.codebook_show_usage,
        )
        self.quant_conv = mint.nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = mint.nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def construct(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def load_from_checkpoint(self, ckpt_path):
        # mainly used in unit test
        parameter_dict = dict()
        if ckpt_path.endswith(".bin"):
            import torch

            sd = torch.load(ckpt_path)
            # filter to keep gen_vision_model params only and remove prefix
            pnames = [p for p in sd]
            for p in pnames:
                if "gen_vision_model" not in p:
                    sd.pop(p)
                else:
                    # remove prefix
                    new_pname = p.replace("gen_vision_model.", "")
                    # special: weight (pt) - > embedding_table (ms)
                    if "embedding.weight" in p:
                        new_pname = new_pname.replace("embedding.weight", "embedding.embedding_table")

                    sd[new_pname] = sd.pop(p)

            param_dtype = tuple(self.get_parameters())[0].dtype
            print("Get vq param dtype: ", param_dtype)

            for pname in sd:
                # print(pname, sd[pname].shape, sd[pname].dtype)
                np_val = sd[pname].cpu().detach().float().numpy()
                # TODO: support bf16 param loading
                parameter_dict[pname] = ms.Parameter(ms.Tensor(np_val, dtype=param_dtype))

        elif ckpt_path.endswith(".ckpt"):
            parameter_dict = ms.load_checkpoint(ckpt_path)
        else:
            raise ValueError("Unsupported checkpoint format")

        param_not_load, ckpt_not_load = ms.load_param_into_net(self, parameter_dict, strict_load=True)
        print("Net params not load: {}, Total net params not loaded: {}".format(param_not_load, len(param_not_load)))
        print("Ckpt params not load: {}, Total ckpt params not loaded: {}".format(ckpt_not_load, len(ckpt_not_load)))


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_16(**kwargs):
    return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))


VQ_models = {"VQ-16": VQ_16}
