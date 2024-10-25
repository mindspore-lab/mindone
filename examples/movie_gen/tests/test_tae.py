import numpy as np
import mindspore as ms
from mg.models.tae.modules import Conv2_5d, ResnetBlock, SpatialAttnBlock, SpatialAttnBlockV2, TemporalAttnBlock, TemporalUpsample, TemporalDownsample, SpatialDownsample, Encoder

from mg.models.tae.tae import SDXL_CONFIG

def test_conv25d():
    in_shape = (B, C, T, H, W) = (2, 3, 16, 256, 256)
    cout = 128
    x = np.random.normal(size=in_shape).astype(np.float32)

    ms.set_context(mode=0)
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
                    dropout=0.,
                )

    ms.set_context(mode=0)
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

    ms.set_context(mode=0)

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

    ms.set_context(mode=0)

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



if __name__ == "__main__":
    # test_conv25d()
    # test_resnetblock()
    # test_spatial_attn()
    # test_temporal_attn()
    # test_spatial_downsample()
    # test_temporal_downsample()
    # test_temporal_upsample()
    test_encoder()


