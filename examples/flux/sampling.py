import math
from typing import Callable, Dict, List, Union

import numpy as np

import mindspore as ms
from mindspore import Tensor, ops

from .model import Flux
from .modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    dtype: ms.Type,
    seed: int,
):
    shape = (
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
    )

    generator = np.random.default_rng(seed)
    noise = ms.Tensor.from_numpy(generator.standard_normal(size=shape)).to(dtype=dtype)

    return noise


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: Union[str, List[str]]) -> Dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    # img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    b, nh, nw = img.shape[0], h // 2, w // 2
    img = img.reshape(b, c, nh, 2, nw, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, nh * nw, -1)

    if img.shape[0] == 1 and bs > 1:
        img = img.tile((bs,) + (1,) * (img.ndim - 1))

    img_ids = ops.zeros((h // 2, w // 2, 3))
    img_ids[..., 1] = img_ids[..., 1] + ops.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + ops.arange(w // 2)[None, :]
    img_ids = img_ids.reshape(1, -1, img_ids.shape[-1]).tile((bs, 1, 1))

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = txt.tile((bs,) + (1,) * (txt.ndim - 1))
    txt_ids = ops.zeros((bs, txt.shape[1], 3))

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = vec.tile((bs,) + (1,) * (vec.ndim - 1))

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return float(math.exp(mu)) / (float(math.exp(mu)) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> List[float]:
    # extra step for zero
    timesteps = ms.Tensor.from_numpy(np.linspace(1, 0, num_steps + 1))

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: List[float],
    guidance: float = 4.0,
):
    # this is ignored for schnell
    guidance_vec = ops.full((img.shape[0],), guidance, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = ops.full((img.shape[0],), t_curr, dtype=img.dtype)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    b, _, d = x.shape
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)
    c = d // 4

    x = x.reshape(b, h, w, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, 2 * h, 2 * w)

    return x
