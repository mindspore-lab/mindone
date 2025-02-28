import os
import sys

import numpy as np
import PIL.Image

import mindspore as ms
from mindspore import Tensor

sys.path.append(".")
from janus.models.vq_model import VQ_16
from PIL import Image

from mindspore.dataset.vision import Inter

np.random.seed(42)


def _diff_res(ms_val, pt_val, eps=1e-8):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


def set_model_param_dtype(model, dtype=ms.bfloat16, keep_norm_fp32=False):
    if model is not None:
        assert isinstance(model, ms.nn.Cell)

        k_num, c_num = 0, 0
        for _, p in model.parameters_and_names():
            # filter norm/embedding position_ids param
            if keep_norm_fp32 and ("norm" in p.name):
                # print(f"param {p.name} keep {p.dtype}") # disable print
                k_num += 1
            elif "position_ids" in p.name:
                k_num += 1
            else:
                c_num += 1
                p.set_dtype(dtype)

        print(f"Convert '{type(model).__name__}' param to {dtype}, keep/modify num {k_num}/{c_num}.")

    return model


def test_decode(pt_ckpt=None, pt_np=None, dtype=ms.float32, visualize=False):
    (B, C, H, W) = (1, 8, 24, 24)
    # shape = (B, C, H, W) = (1, 8, 12, 12)
    if pt_np:
        pt_data = np.load(pt_np)
        z = pt_data["quant"]
        code = pt_data["code"] if "code" in pt_data else None
    else:
        z = np.random.normal(size=(B, C, H, W)).astype(np.float32)
        code = np.random.randint(10000, size=(1, B * H * W))  # 576
    decode_from_code = True

    vq = VQ_16()
    vq.set_train(False)
    if dtype != ms.float32:
        set_model_param_dtype(vq, dtype=dtype, keep_norm_fp32=False)
    if pt_ckpt:
        vq.load_from_checkpoint(pt_ckpt)

    if decode_from_code:
        out = vq.decode_code(Tensor(code).to(ms.int32), shape=(B, C, H, W))
    else:
        out = vq.decode(Tensor(z, dtype=dtype))

    print(out.shape)
    print("sum and std", out.sum(), out.std())
    print("min and max", out.min(), out.max())

    if pt_np:
        pt_out = pt_data["dec"]
        print("pt min max: ", pt_out.min(), pt_out.max())
        diff = _diff_res(out.asnumpy(), pt_out)
        print(diff)

    if visualize:
        dec = out
        parallel_size, c, img_size, _ = dec.shape
        dec = dec.float().asnumpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        os.makedirs("generated_samples", exist_ok=True)
        for i in range(parallel_size):
            save_path = os.path.join("generated_samples", "vq_dec_{}.jpg".format(i))
            PIL.Image.fromarray(visual_img[i]).save(save_path)
            print("img saved in ", save_path)

    return out.asnumpy()


def test_encode(pt_ckpt=None, amp=False):
    # shape = (B, C, H, W) = (1, 8, 24, 24)
    (B, C, H, W) = (1, 3, 64, 64)
    vq = VQ_16()
    vq.set_train(False)
    x = np.random.normal(size=(B, C, H, W)).astype(np.float32)
    out = vq.encode(ms.Tensor(x))[0]

    print(out.shape)
    print(out.sum(), out.std())

    return out.asnumpy()


def get_image():
    image_path = "images/doge.png"
    size = (384, 384)
    image = Image.open(image_path).convert("RGB")
    image = ms.dataset.vision.Resize(size, interpolation=Inter.ANTIALIAS)(image)
    image = np.array(image)
    image = (image / 255.0) * 2 - 1
    image = np.transpose(image, (2, 0, 1))
    image = image[None, ...]  # add bs, n_images dimension

    return image


def test_rec(pt_ckpt=None, pt_np=None, dtype=ms.float32, visualize=False):
    # shape = (B, C, H, W) = (1, 3, 384, 384)
    # shape = (B, C, H, W) = (1, 3, 64, 64)
    # x = np.random.normal(size=(B, C, H, W)).astype(np.float32)

    x = get_image()

    x = np.array([x[0], x[0]])

    vq = VQ_16()
    vq.set_train(False)
    if dtype != ms.float32:
        set_model_param_dtype(vq, dtype=dtype, keep_norm_fp32=False)
    if pt_ckpt:
        vq.load_from_checkpoint(pt_ckpt)

    z, emb_loss, info = vq.encode(Tensor(x, dtype=dtype))
    bs = z.shape[0]
    image_tokens = info[-1].reshape(bs, -1)
    print("encoded  z: ", z.shape, z.mean(), z)
    print("encoded  image tokens: ", image_tokens.shape)

    out = vq.decode(z)

    print(out.shape)
    print("sum, std: ", out.sum(), out.std())
    print("min max: ", out.min(), out.max())

    if pt_np:
        pt_data = np.load(pt_np)
        pt_out = pt_data["dec"]
        print("pt min max: ", pt_out.min(), pt_out.max())
        diff = _diff_res(out.asnumpy(), pt_out)
        print(diff)

    if visualize:
        dec = out
        parallel_size, c, img_size, _ = dec.shape
        dec = dec.float().asnumpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        os.makedirs("generated_samples", exist_ok=True)
        for i in range(parallel_size):
            save_path = os.path.join("generated_samples", "vq_rec_{}.jpg".format(i))
            PIL.Image.fromarray(visual_img[i]).save(save_path)
            print("img saved in ", save_path)

    return out.asnumpy()


if __name__ == "__main__":
    ms.set_context(mode=0)
    # test_encode()
    test_decode(
        "ckpts/Janus-Pro-1B/pytorch_model.bin",
        pt_np="tests/vq_dec_io.npz",
        dtype=ms.bfloat16,
        visualize=True,
    )
    # test_decode("ckpts/Janus-Pro-1B/pytorch_model.bin", pt_np='tests/vq_dec_io.npz')
    # test_decode("ckpts/Janus-Pro-1B/pytorch_model.bin", pt_np='tests/vq_dec_io_fp32.npz', dtype=ms.float32)
    # test_decode("ckpts/Janus-Pro-1B/pytorch_model.bin", pt_np='tests/vq_dec_io_bf16.npz', dtype=ms.bfloat16)
