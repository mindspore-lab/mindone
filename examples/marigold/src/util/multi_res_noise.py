import math

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


def multi_res_noise_like(
    x,
    dtype,
    strength=0.9,
    downscale_strategy="original",
    generator=None,
):
    if isinstance(strength, Tensor):
        strength = ops.reshape(strength, (-1, 1, 1, 1))
    b, c, w, h = x.shape

    up_sampler = nn.Upsample(size=(w, h), mode="bilinear")
    noise = ops.standard_normal(x.shape, seed=generator)

    if "original" == downscale_strategy:
        for i in range(10):
            r = (
                ops.uniform((1,), minval=Tensor(2.0, dtype), maxval=Tensor(4.0, dtype), seed=generator)
                if generator
                else ops.uniform((1,), minval=Tensor(2.0, dtype), maxval=Tensor(4.0, dtype))
            )  # Random scaling factor
            w, h = max(1, int(w / (r.item() ** i))), max(1, int(h / (r.item() ** i)))
            new_noise = ops.standard_normal((b, c, w, h), seed=generator)
            noise += up_sampler(new_noise) * (strength**i)
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    elif "every_layer" == downscale_strategy:
        for i in range(int(math.log2(min(w, h)))):
            w, h = max(1, int(w / 2)), max(1, int(h / 2))
            new_noise = ops.standard_normal((b, c, w, h), seed=generator)
            noise += up_sampler(new_noise) * (strength**i)
    elif "power_of_two" == downscale_strategy:
        for i in range(10):
            r = 2
            w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
            new_noise = ops.standard_normal((b, c, w, h), seed=generator)
            noise += up_sampler(new_noise) * (strength**i)
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    elif "random_step" == downscale_strategy:
        for i in range(10):
            r = (
                ops.uniform((1,), minval=Tensor(2.0, dtype), maxval=Tensor(4.0, dtype), seed=generator)
                if generator
                else ops.uniform((1,), minval=Tensor(2.0, dtype), maxval=Tensor(4.0, dtype))
            )  # Random scaling factor
            w, h = max(1, int(w / r.item())), max(1, int(h / r.item()))
            new_noise = ops.standard_normal((b, c, w, h), seed=generator)
            noise += up_sampler(new_noise) * (strength**i)
            if w == 1 or h == 1:
                break  # Lowest resolution is 1x1
    else:
        raise ValueError(f"unknown downscale strategy: {downscale_strategy}")

    noise = noise / noise.std()  # Scaled back to roughly unit variance
    noise = noise.to(dtype)
    return noise


# if __name__ == "__main__":
#     mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE, device_target="CPU")
#     x = Tensor([[[[1, 2], [3, 4]]]], mindspore.float32)
#     shape = (1,)
#     minval = Tensor(1, mindspore.int32)
#     maxval = Tensor(2, mindspore.int32)
#     output = ops.uniform(shape, minval, maxval, seed=5, dtype=mindspore.int32)
#     print(output)
#     noise = multi_res_noise_like(x)
#     print(noise)
#     noise = multi_res_noise_like(x, downscale_strategy="every_layer")
#     print(noise)
#     noise = multi_res_noise_like(x, downscale_strategy="power_of_two")
#     print(noise)
#     noise = multi_res_noise_like(x, downscale_strategy="random_step")
#     print(noise)
#     noise = multi_res_noise_like(x, downscale_strategy="unknown")
#     print(noise)
