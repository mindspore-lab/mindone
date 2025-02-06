import os
import sys

import numpy as np
from PIL import Image

import mindspore as ms

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../../"))
sys.path.append(mindone_lib_path)

from mg.models.tae.modules import (
    Conv2_5d,
    Decoder,
    Encoder,
    ResnetBlock,
    SpatialAttnBlockV2,
    SpatialDownsample,
    SpatialUpsample,
    TemporalAttnBlock,
    TemporalDownsample,
    TemporalUpsample,
)
from mg.models.tae.sd3_vae import SD3d5_VAE
from mg.models.tae.tae import SDXL_CONFIG, TAE_CONFIG, TemporalAutoencoder


def get_input_image(img_path="../videocomposer/demo_video/moon_on_water.jpg", W=128, H=128):
    target_size = (H, W)

    # read image using PIL and preprocess
    image = Image.open(img_path).convert("RGB")
    image = image.resize(target_size)
    pixel_values = np.array(image, dtype=np.float32)
    pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)

    pixel_values = pixel_values.transpose(2, 0, 1)

    return pixel_values


def save_output_image(image_array, output_path="tests/tmp_output.png"):
    image_array = image_array.transpose((1, 2, 0))
    image_array = ((image_array + 1) * 127.5).astype(np.uint8)
    image_array = np.clip(image_array, 0, 255)

    image = Image.fromarray(image_array)

    image.save(output_path)
    print(f"image saved in {output_path}")


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
    in_shape = (B, C, T, H, W) = (1, 16, 5, 32, 32)
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
    TAE_CONFIG["attn_type"] = "spat_only"
    tae = TemporalAutoencoder(config=TAE_CONFIG)
    tae.load_pretrained("models/tae_vae2d.ckpt")

    # in_shape = (B, C, T, H, W) = (1, 3, 16, 64, 64)
    in_shape = (B, C, T, H, W) = (1, 3, 1, 128, 128)
    x = np.random.normal(size=in_shape).astype(np.float32)
    img = get_input_image(H=H, W=W)
    x[0, :, 0, :, :] = img
    x = ms.Tensor(x)

    y = tae(x)

    print(y[0].shape)
    save_output_image(y[0].numpy()[0, :, 0, :, :], "tests/tmp_tae_output.png")


def test_sd3d5_vae():
    vae = SD3d5_VAE(sample_deterministic=True)
    vae.load_pretrained("models/sd3.5_vae.ckpt")

    in_shape = (BT, C, H, W) = (1, 3, 128, 128)
    x = np.random.normal(size=in_shape).astype(np.float32)
    img = get_input_image(H=H, W=W)
    x[0] = img

    x = ms.Tensor(x)

    outputs = vae(x)
    recons = outputs[0]
    print(recons.shape)

    # save to image
    # TODO: there are some noise here
    save_output_image(recons.numpy()[0])

    print(recons.sum())


def test_blend():
    ms.set_context(mode=1)
    tae = TemporalAutoencoder(config=TAE_CONFIG, use_tile=True, encode_tile=32, decode_tile=32, decode_overlap=16)

    in_shape = (B, C, T, H, W) = (1, 1, 12, 1, 1)
    x = np.random.normal(size=in_shape).astype(np.float32)
    x = ms.Tensor(x)

    out = tae.blend_slices(x, slice_len=4, overlap_len=2)

    print(out.shape)


def test_tae_tile():
    tae = TemporalAutoencoder(config=TAE_CONFIG, use_tile=True, encode_tile=32, decode_tile=32, decode_overlap=16)

    # in_shape = (B, C, T, H, W) = (1, 3, 16, 64, 64)
    in_shape = (B, C, T, H, W) = (1, 3, 96, 32, 32)
    # in_shape = (B, C, T, H, W) = (1, 3, 64+16, 64, 64)

    x = np.random.normal(size=in_shape).astype(np.float32)
    x = ms.Tensor(x)

    y = tae(x)

    print(y[0].shape)

    # check correctness of blend


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
    # test_tae_rec()
    # test_tae_tile()
    # test_blend()

    test_sd3d5_vae()
