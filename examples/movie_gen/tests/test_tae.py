import numpy as np
import sys
sys.path.insert(0, '.')

from mg.models.tae.modules import (
    Conv2_5d,
    Decoder,
    Encoder,
    ResnetBlock,
    SpatialAttnBlock,
    SpatialAttnBlockV2,
    SpatialDownsample,
    SpatialUpsample,
    TemporalAttnBlock,
    TemporalDownsample,
    TemporalUpsample,
)
from mg.models.tae.tae import SDXL_CONFIG, TemporalAutoencoder

import mindspore as ms


def test_conv25d():
    in_shape = (B, C, T, H, W) = (2, 3, 16, 256, 256)
    cout = 128
    x = np.random.normal(size=in_shape).astype(np.float32)

    x = ms.Tensor(x)
    conv2d = Conv2_5d(C, cout, 3)

    y = conv2d(x)

    print(y.shape)


def test_resnetblock():
    in_shape = (B, C, T, H, W) = (1, 64, 4, 32, 32)
    cout = C
    x = np.random.normal(size=in_shape).astype(np.float32)

    rb = ResnetBlock(
        in_channels=C,
        out_channels=cout,
        dropout=0.0,
    )

    x = ms.Tensor(x)
    y = rb(x)

    print(y.shape)
    print(y.mean(), y.std())


def test_spatial_attn():
    in_shape = (B, C, T, H, W) = (1, 64, 4, 32, 32)
    cout = C
    x = np.random.normal(size=in_shape).astype(np.float32)

    # TODO: compare time cost for v1 and v2
    # sa = SpatialAttnBlock(C)
    sa = SpatialAttnBlockV2(C)


    x = ms.Tensor(x)
    y = sa(x)

    print(y.shape)
    print(y.mean(), y.std())


def test_temporal_attn():
    in_shape = (B, C, T, H, W) = (1, 64, 4, 32, 32)
    cout = C
    x = np.random.normal(size=in_shape).astype(np.float32)

    # TODO: compare time cost for v1 and v2
    ta = TemporalAttnBlock(C)


    x = ms.Tensor(x)
    y = ta(x)

    print(y.shape)
    print(y.mean(), y.std())


def test_spatial_downsample():
    # in_shape = (B, C, T, H, W) = (1, 64, 1, 32, 32)
    in_shape = (B, C, T, H, W) = (1, 64, 4, 32, 32)
    x = np.random.normal(size=in_shape).astype(np.float32)
    sd = SpatialDownsample(C, True)

    x = ms.Tensor(x)
    y = sd(x)

    print(y.shape)


def test_spatial_upsample():
    # in_shape = (B, C, T, H, W) = (1, 64, 1, 32, 32)
    in_shape = (B, C, T, H, W) = (1, 64, 4, 32, 32)
    x = np.random.normal(size=in_shape).astype(np.float32)
    su = SpatialUpsample(C, True)

    x = ms.Tensor(x)
    y = su(x)

    print(y.shape)


def test_temporal_downsample():
    # in_shape = (B, C, T, H, W) = (1, 64, 1, 32, 32)
    in_shape = (B, C, T, H, W) = (1, 64, 1, 32, 32)
    x = np.random.normal(size=in_shape).astype(np.float32)
    td = TemporalDownsample(C)

    print(x[0, 0, :, 0, 0])
    x = ms.Tensor(x)
    y = td(x)

    print(y[0, 0, :, 0, 0])
    print(y.shape)


def test_temporal_upsample():
    # in_shape = (B, C, T, H, W) = (1, 64, 1, 32, 32)
    in_shape = (B, C, T, H, W) = (1, 64, 4, 32, 32)
    x = np.random.normal(size=in_shape).astype(np.float32)
    tu = TemporalUpsample(C)

    print(x[0, 0, :, 0, 0])
    x = ms.Tensor(x)
    y = tu(x)

    print(y[0, 0, :, 0, 0])
    print(y.shape)


def test_encoder():
    # in_shape = (B, C, T, H, W) = (1, 64, 1, 32, 32)
    in_shape = (B, C, T, H, W) = (1, 3, 1, 64, 64)
    x = np.random.normal(size=in_shape).astype(np.float32)
    enc = Encoder(**SDXL_CONFIG)

    x = ms.Tensor(x)
    y = enc(x)

    print(y.shape)


def test_decoder():
    # in_shape = (B, C, T, H, W) = (1, 64, 1, 32, 32)
    in_shape = (B, C, T, H, W) = (1, 4, 4, 16, 16)
    x = np.random.normal(size=in_shape).astype(np.float32)
    dec = Decoder(**SDXL_CONFIG)

    x = ms.Tensor(x)
    y = dec(x)

    print(y.shape)


def test_tae_encode():
    in_shape = (B, C, T, H, W) = (1, 3, 8, 64, 64)
    x = np.random.normal(size=in_shape).astype(np.float32)
    x = ms.Tensor(x)

    tae = TemporalAutoencoder(config=SDXL_CONFIG)
    y = tae.encode(x)

    print(y.shape)

def test_tae_decode():
    # in_shape = (B, C, T, H, W) = (1, 3, 1, 64, 64)
    in_shape = (B, C, T, H, W) = (1, 4, 1, 8, 8)
    x = np.random.normal(size=in_shape).astype(np.float32)
    x = ms.Tensor(x)

    tae = TemporalAutoencoder(config=SDXL_CONFIG)
    y = tae.decode(x)

    print(y.shape)


def test_tae_rec():
    in_shape = (B, C, T, H, W) = (1, 3, 16, 64, 64)
    x = np.random.normal(size=in_shape).astype(np.float32)
    x = ms.Tensor(x)

    tae = TemporalAutoencoder(config=SDXL_CONFIG)
    y = tae(x)

    print(y[0].shape)


if __name__ == "__main__":
    ms.set_context(mode=1)

    # test_conv25d()
    # test_resnetblock()
    # test_spatial_attn()
    # test_temporal_attn()
    # test_spatial_downsample()
    # test_temporal_downsample()
    # test_encoder()

    # test_temporal_upsample()
    # test_spatial_upsample()
    # test_decoder()
    # test_tae_encode()
    # test_tae_decode()
    test_tae_rec()
