# TODO
import importlib
import os
import time
from inspect import isfunction

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mindspore as ms
from mindspore import _no_grad, mint, nn, ops
from mindspore.nn.optim import Adam, AdamWeightDecay, Momentum, Optimizer

_MIN_FP16 = ms.tensor(np.finfo(np.float16).min, dtype=ms.float16)
_MIN_FP32 = ms.tensor(np.finfo(np.float32).min, dtype=ms.float32)
_MIN_FP64 = ms.tensor(np.finfo(np.float64).min, dtype=ms.float64)
_MIN_BF16 = ms.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=ms.bfloat16)
_MAX_FP16 = ms.tensor(np.finfo(np.float16).max, dtype=ms.float16)
_MAX_FP32 = ms.tensor(np.finfo(np.float32).max, dtype=ms.float32)
_MAX_FP64 = ms.tensor(np.finfo(np.float64).max, dtype=ms.float64)
_MAX_BF16 = ms.tensor(float.fromhex("0x1.fe00000000000p+127"), dtype=ms.bfloat16)


def dtype_to_min(dtype):
    if dtype == ms.float16:
        return _MIN_FP16
    if dtype == ms.float32:
        return _MIN_FP32
    if dtype == ms.float64:
        return _MIN_FP64
    if dtype == ms.bfloat16:
        return _MIN_BF16
    else:
        raise ValueError(f"Only support get minimum value of (float16, ), but got {dtype}")


def dtype_to_max(dtype):
    if dtype == ms.float16:
        return _MAX_FP16
    if dtype == ms.float32:
        return _MAX_FP32
    if dtype == ms.float64:
        return _MAX_FP64
    if dtype == ms.bfloat16:
        return _MAX_BF16
    else:
        raise ValueError(f"Only support get maximum value of (float16, ), but got {dtype}")


@ms.jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)


def pil_rectangle_crop(im):
    width, height = im.size  # Get dimensions

    if width <= height:
        left = 0
        right = width
        top = (height - width) / 2
        bottom = (height + width) / 2
    else:
        top = 0
        bottom = height
        left = (width - height) / 2
        bottom = (width + height) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def add_margin(pil_img, color, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result


def load_and_preprocess(interface, input_im):
    """
    :param input_im (PIL Image).
    :return image (H, W, 3) array in [0, 1].
    """
    # See https://github.com/Ir1d/image-background-remove-tool
    image = input_im.convert("RGB")

    image_without_background = interface([image])[0]
    image_without_background = np.array(image_without_background)
    est_seg = image_without_background > 127
    image = np.array(image)
    foreground = est_seg[:, :, -1].astype(np.bool_)
    image[~foreground] = [255.0, 255.0, 255.0]
    x, y, w, h = cv2.boundingRect(foreground.astype(np.uint8))
    image = image[y : y + h, x : x + w, :]
    image = Image.fromarray(np.array(image))

    # resize image such that long edge is 512
    image.thumbnail([200, 200], Image.Resampling.LANCZOS)
    image = add_margin(image, (255, 255, 255), size=256)
    image = np.array(image)
    return image


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = ms.Tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, ms.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, ms.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return mint.mean(tensor, dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum([p.size for p in model.get_parameters()])
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


''' no use
#TODO
class AdamWwithEMAandWings(optim.Optimizer):
    # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,  # TODO: check hyperparameters before using
                 weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,   # ema decay to match previous code
                 ema_power=1., param_names=()):
        """AdamW that saves EMA versions of the parameters."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
        # defaults = dict(learning_rate=lr, betas=betas, eps=eps,
        #                 weight_decay=weight_decay, amsgrad=amsgrad, ema_decay=ema_decay,
        #                 ema_power=ema_power, param_names=param_names)
        # super().__init__(params, defaults)
        super().__init__(learning_rate=lr, parameters=params, weight_decay=weight_decay)

        self.beta1 = Tensor(np.array([betas[0]]).astype(np.float32))
        self.beta2 = Tensor(np.array([betas[1]]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.ema_decay = Tensor(np.array([ema_decay]).astype(np.float32))
        # self.exp_avg = self.clone_state(prefix="adam_m", init="zeros")
        # self.exp_avg_sq = self.clone_state(prefix="adam_v", init="zeros")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.group_params:
        # for group in self.param_groups:
            group.setdefault('amsgrad', False)

    # @torch.no_grad()
    def construct(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.group_params:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            ema_params_with_grad = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            ema_decay = group['ema_decay']
            ema_power = group['ema_power']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of parameter values
                    state['param_exp_avg'] = p.detach().float().clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                ema_params_with_grad.append(state['param_exp_avg'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            AdamWeightDecay(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=False)

            cur_ema_decay = min(ema_decay, 1 - state['step'] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

        return loss

'''
