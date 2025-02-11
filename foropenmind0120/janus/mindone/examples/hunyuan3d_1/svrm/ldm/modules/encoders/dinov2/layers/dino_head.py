# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal, initializer


class DINOHead(nn.Cell):  # NOTE: no use yet
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer.weight_g.set_data(initializer("ones"))
        # from torch.nn.utils import weight_norm # Deprecated
        # self.last_layer = weight_norm(nn.Dense(bottleneck_dim, out_dim, has_bias=False))
        # self.last_layer.weight_g.set_data(initializer("ones"))
        self.last_layer = nn.Dense(bottleneck_dim, out_dim, has_bias=False)
        self.last_layer.weight.set_data(initializer("ones"))

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            weight = initializer(TruncatedNormal(sigma=0.02, mean=0.0, a=-2.0, b=2.0), m.weight.shape)
            m.weight.set_data(weight)
            if isinstance(m, nn.Dense) and m.bias is not None:
                bias_weight = initializer("zeros", m.bias.shape)
                m.bias.set_data(bias_weight)

    def construct(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == ms.float16 else 1e-12
        x = x / (x.norm(eps, dim=-1) + eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Dense(in_dim, bottleneck_dim, has_bias=bias)
    else:
        layers = [nn.Dense(in_dim, hidden_dim, has_bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU(approximate=False))
        for _ in range(nlayers - 2):
            layers.append(nn.Dense(hidden_dim, hidden_dim, has_bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU(approximate=False))
        layers.append(nn.Dense(hidden_dim, bottleneck_dim, has_bias=bias))
        return nn.SequentialCell(*layers)
