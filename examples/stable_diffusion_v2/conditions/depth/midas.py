import collections
import math
import os

from utils.download import download_checkpoint

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import Normal, initializer

__all__ = ["MiDaS", "midas_v3_dpt_large"]

_CKPT_URL = {
    "midas_v3_dpt_large": "https://download.mindspore.cn/toolkits/mindone/stable_diffusion/depth_estimator/midas_v3_dpt_large-c8fd1049.ckpt"
}


class SelfAttention(nn.Cell):
    def __init__(self, dim, num_heads):
        assert dim % num_heads == 0
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # layers
        self.to_qkv = nn.Dense(dim, dim * 3)
        self.proj = nn.Dense(dim, dim)

    def construct(self, x):
        b, l, c = x.shape
        n, d = self.num_heads, self.head_dim

        # compute query, key, value
        q, k, v = self.to_qkv(x).view(b, l, n * 3, d).chunk(3, axis=2)

        # compute attention
        # ops.einsum('binc,bjnc->bnij', q, k)
        attn = self.scale * ops.bmm(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        attn = ops.softmax(attn, axis=-1).astype(attn.dtype)

        # gather context
        # ops.einsum('bnij,bjnc->binc', attn, v)
        x = ops.bmm(attn, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = x.reshape(b, l, c)

        # output
        x = self.proj(x)
        return x


class GELU(nn.Cell):
    def __init__(self):
        super(GELU, self).__init__()

    def construct(self, x):
        return x * 0.5 * (1.0 + ops.erf(x / ops.sqrt(ms.Tensor(2.0))))


class AttentionBlock(nn.Cell):
    def __init__(self, dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # layers
        self.norm1 = nn.LayerNorm((dim,))
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm((dim,))
        self.mlp = nn.SequentialCell(
            collections.OrderedDict(
                [
                    ("0", nn.Dense(dim, dim * 4)),
                    ("1", GELU()),  # use self-defined GELU to keep the results consistent with torch
                    ("2", nn.Dense(dim * 4, dim)),
                ]
            )
        )

    def construct(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class conv_nd(nn.Cell):
    def __init__(self, dims, *args, **kwargs):
        super().__init__()
        if dims == 1:
            self.conv = nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            self.conv = nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            self.conv = nn.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

    def construct(self, x, emb=None, context=None):
        x = self.conv(x)
        return x


class VisionTransformer(nn.Cell):
    def __init__(
        self, image_size=384, patch_size=16, dim=1024, out_dim=1000, num_heads=16, num_layers=24, dtype=ms.float32
    ):
        assert image_size % patch_size == 0
        super(VisionTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = (image_size // patch_size) ** 2

        # embeddings
        self.patch_embedding = conv_nd(
            3, 3, dim, kernel_size=patch_size, stride=patch_size, has_bias=True, pad_mode="pad"
        )
        self.cls_embedding = Parameter(ms.Tensor(ops.zeros((1, 1, dim)), dtype))
        self.pos_embedding = Parameter(initializer(Normal(sigma=0.02), (1, self.num_patches + 1, dim), ms.float32))

        # blocks
        self.blocks = nn.SequentialCell(
            collections.OrderedDict([(str(i), AttentionBlock(dim, num_heads)) for i in range(num_layers)])
        )

        self.norm = nn.LayerNorm((dim,))

        # head
        self.head = nn.Dense(dim, out_dim)

    def construct(self, x):
        b = x.shape[0]

        # embeddings
        x = ops.flatten(self.patch_embedding(x), start_dim=2).permute(0, 2, 1)
        x = ops.concat([self.cls_embedding.repeat(b, axis=0), x], axis=1)
        x = x + self.pos_embedding

        # blocks
        x = self.blocks(x)
        x = self.norm(x)

        # head
        x = self.head(x)
        return x


class ResidualBlock(nn.Cell):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.dim = dim

        # layers
        self.residual = nn.SequentialCell(
            collections.OrderedDict(
                [
                    ("0", nn.ReLU()),
                    ("1", conv_nd(2, dim, dim, 3, padding=1, has_bias=True, pad_mode="pad")),
                    ("2", nn.ReLU()),
                    ("3", conv_nd(2, dim, dim, 3, padding=1, has_bias=True, pad_mode="pad")),
                ]
            )
        )

    def construct(self, x):
        return x + self.residual(x)


class FusionBlock(nn.Cell):
    def __init__(self, dim):
        super(FusionBlock, self).__init__()
        self.dim = dim

        # layers
        self.layer1 = ResidualBlock(dim)
        self.layer2 = ResidualBlock(dim)
        self.conv_out = conv_nd(2, dim, dim, 1, has_bias=True, pad_mode="pad")

    def construct(self, *xs):
        assert len(xs) in (1, 2), "invalid number of inputs"
        if len(xs) == 1:
            x = self.layer2(xs[0])
        else:
            x = self.layer2(xs[0] + self.layer1(xs[1]))
        size = x.shape[2:]
        size = [s * 2 for s in size]
        x = ops.interpolate(x, size=size, mode="bilinear", align_corners=True)
        x = self.conv_out(x)
        return x


class MiDaS(nn.Cell):
    r"""MiDaS v3.0 DPT-Large from ``https://github.com/isl-org/MiDaS''.
    Monocular depth estimation using dense prediction transformers.
    """

    def __init__(
        self,
        image_size=384,
        patch_size=16,
        dim=1024,
        neck_dims=[256, 512, 1024, 1024],
        fusion_dim=256,
        num_heads=16,
        num_layers=24,
        dtype=ms.float32,
    ):
        assert image_size % patch_size == 0
        assert num_layers % 4 == 0
        super(MiDaS, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.neck_dims = neck_dims
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = (image_size // patch_size) ** 2

        # embeddings
        self.patch_embedding = conv_nd(
            2, 3, dim, kernel_size=patch_size, stride=patch_size, has_bias=True, pad_mode="pad"
        )
        self.cls_embedding = Parameter(ms.Tensor(ops.zeros((1, 1, dim)), dtype))
        self.pos_embedding = Parameter(initializer(Normal(sigma=0.02), (1, self.num_patches + 1, dim), ms.float32))

        # blocks
        stride = num_layers // 4
        self.blocks = [
            nn.SequentialCell([AttentionBlock(dim, num_heads) for _ in range(i * stride, (i + 1) * stride)])
            for i in range(4)
        ]
        self.blocks = nn.CellList(self.blocks)

        # stage1 (4x)
        self.fc1 = nn.SequentialCell(collections.OrderedDict([("0", nn.Dense(dim * 2, dim)), ("1", GELU())]))
        self.conv1 = nn.SequentialCell(
            collections.OrderedDict(
                [
                    ("0", conv_nd(2, dim, neck_dims[0], 1, has_bias=True, pad_mode="pad")),
                    ("1", nn.Conv2dTranspose(neck_dims[0], neck_dims[0], 4, stride=4, has_bias=True, pad_mode="pad")),
                    ("2", conv_nd(2, neck_dims[0], fusion_dim, 3, padding=1, has_bias=False, pad_mode="pad")),
                ]
            )
        )
        self.fusion1 = FusionBlock(fusion_dim)

        # stage2 (8x)
        self.fc2 = nn.SequentialCell(collections.OrderedDict([("0", nn.Dense(dim * 2, dim)), ("1", GELU())]))
        self.conv2 = nn.SequentialCell(
            collections.OrderedDict(
                [
                    ("0", conv_nd(2, dim, neck_dims[1], 1, has_bias=True, pad_mode="pad")),
                    ("1", nn.Conv2dTranspose(neck_dims[1], neck_dims[1], 2, stride=2, has_bias=True, pad_mode="pad")),
                    ("2", conv_nd(2, neck_dims[1], fusion_dim, 3, padding=1, has_bias=False, pad_mode="pad")),
                ]
            )
        )
        self.fusion2 = FusionBlock(fusion_dim)

        # stage3 (16x)
        self.fc3 = nn.SequentialCell(collections.OrderedDict([("0", nn.Dense(dim * 2, dim)), ("1", GELU())]))
        self.conv3 = nn.SequentialCell(
            collections.OrderedDict(
                [
                    ("0", conv_nd(2, dim, neck_dims[2], 1, has_bias=True, pad_mode="pad")),
                    ("1", conv_nd(2, neck_dims[2], fusion_dim, 3, padding=1, has_bias=False, pad_mode="pad")),
                ]
            )
        )
        self.fusion3 = FusionBlock(fusion_dim)

        # stage4 (32x)
        self.fc4 = nn.SequentialCell(collections.OrderedDict([("0", nn.Dense(dim * 2, dim)), ("1", GELU())]))
        self.conv4 = nn.SequentialCell(
            collections.OrderedDict(
                [
                    ("0", conv_nd(2, dim, neck_dims[3], 1, has_bias=True, pad_mode="pad")),
                    (
                        "1",
                        conv_nd(2, neck_dims[3], neck_dims[3], 3, stride=2, padding=1, has_bias=True, pad_mode="pad"),
                    ),
                    ("2", conv_nd(2, neck_dims[3], fusion_dim, 3, padding=1, has_bias=False, pad_mode="pad")),
                ]
            )
        )
        self.fusion4 = FusionBlock(fusion_dim)

        # head
        # self.head = nn.SequentialCell(collections.OrderedDict([
        #     ("0", conv_nd(2, fusion_dim, fusion_dim // 2, 3, padding=1, has_bias=True, pad_mode="pad")),
        #     ("1", nn.Upsample(scale_factor=2.0, mode='area')),
        #     ("2", conv_nd(2, fusion_dim // 2, 32, 3, padding=1, has_bias=True, pad_mode="pad")),
        #     ("3", nn.ReLU()),
        #     ("4", nn.Conv2dTranspose(32, 1, 1, has_bias=True, pad_mode="pad")),
        #     ("5", nn.ReLU())
        #     ]))
        self.head = nn.CellList(
            [
                conv_nd(2, fusion_dim, fusion_dim // 2, 3, padding=1, has_bias=True, pad_mode="pad"),
                nn.Identity(),
                conv_nd(2, fusion_dim // 2, 32, 3, padding=1, has_bias=True, pad_mode="pad"),
                nn.ReLU(),
                nn.Conv2dTranspose(32, 1, 1, has_bias=True, pad_mode="pad"),
                nn.ReLU(),
            ]
        )

    def construct(self, x):
        b, _, h, w = x.shape
        p = self.patch_size
        # assert h % p == 0 and w % p == 0, f"Image size ({w}, {h}) is not divisible by patch size ({p}, {p})"
        hp, wp, grid = h // p, w // p, self.image_size // p

        # embeddings
        pos_embedding = ops.concat(
            [
                self.pos_embedding[:, :1],
                ops.interpolate(
                    self.pos_embedding[:, 1:].reshape(1, grid, grid, -1).permute(0, 3, 1, 2),
                    size=(hp, wp),
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .reshape(1, hp * wp, -1),
            ],
            axis=1,
        )

        x = ops.flatten(self.patch_embedding(x), start_dim=2).permute(0, 2, 1)
        x = ops.concat([self.cls_embedding.repeat(b, axis=0), x], axis=1)
        x = x + pos_embedding

        # stage1
        x = self.blocks[0](x)
        x1 = ops.concat([x[:, 1:], x[:, :1].expand_as(x[:, 1:])], axis=-1)
        x1 = nn.Unflatten(2, (hp, wp))(self.fc1(x1).permute(0, 2, 1))
        x1 = self.conv1(x1)

        # stage2
        x = self.blocks[1](x)
        x2 = ops.concat([x[:, 1:], x[:, :1].expand_as(x[:, 1:])], axis=-1)
        x2 = nn.Unflatten(2, (hp, wp))(self.fc2(x2).permute(0, 2, 1))
        x2 = self.conv2(x2)

        # stage3
        x = self.blocks[2](x)
        x3 = ops.concat([x[:, 1:], x[:, :1].expand_as(x[:, 1:])], axis=-1)
        x3 = nn.Unflatten(2, (hp, wp))(self.fc3(x3).permute(0, 2, 1))
        x3 = self.conv3(x3)

        # stage4
        x = self.blocks[3](x)
        x4 = ops.concat([x[:, 1:], x[:, :1].expand_as(x[:, 1:])], axis=-1)
        x4 = nn.Unflatten(2, (hp, wp))(self.fc4(x4).permute(0, 2, 1))
        x4 = self.conv4(x4)
        # fusion
        x4 = self.fusion4(x4)
        x3 = self.fusion3(x4, x3)
        x2 = self.fusion2(x3, x2)
        x1 = self.fusion1(x2, x1)

        # head
        x = self.head[0](x1)
        size = x.shape[2:]
        size = [s * 2 for s in size]
        x = ops.interpolate(x, size=size, mode="bilinear", align_corners=True)
        x = self.head[2](x)
        x = self.head[3](x)
        x = self.head[4](x)
        x = self.head[5](x)

        return x


def midas_v3_dpt_large(pretrained=False, **kwargs):
    cfg = dict(
        image_size=384,
        patch_size=16,
        dim=1024,
        neck_dims=[256, 512, 1024, 1024],
        fusion_dim=256,
        num_heads=16,
        num_layers=24,
    )
    if "ckpt_path" in kwargs:
        ckpt_path = kwargs["ckpt_path"]
        del kwargs["ckpt_path"]
    else:
        ckpt_path = None
    cfg.update(**kwargs)
    model = MiDaS(**cfg)

    if pretrained:
        # download and load checkpoint
        if not os.path.exists(ckpt_path):
            download_checkpoint(_CKPT_URL["midas_v3_dpt_large"], "models/depth_estimator")

        state = ms.load_checkpoint(ckpt_path)
        for pname, p in model.parameters_and_names():
            if p.name != pname and (p.name not in state and pname in state):
                param = state.pop(pname)
                state[p.name] = param  # classifier.conv.weight -> weight; classifier.conv.bias -> bias
        param_not_load, _ = ms.load_param_into_net(model, state)
        if len(param_not_load):
            print("Params not load: {}".format(param_not_load))
    return model


if __name__ == "__main__":
    model = midas_v3_dpt_large(pretrained=False)
