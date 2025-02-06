import contextlib
import copy
import logging
import math
import os
import random
import time
from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Queue
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

import mindspore as ms
from mindspore import context, nn, ops
from mindspore.amp import DynamicLossScaler, StaticLossScaler, all_finite
from mindspore.common import dtype as mstype
from mindspore.common.api import _pynative_executor
from mindspore.communication import get_group_size, get_local_rank, get_rank, init
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from mindone.diffusers._peft import set_peft_model_state_dict
from mindone.diffusers.models.model_loading_utils import silence_mindspore_logger
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.trainers.zero import ZeroHelper, prepare_network

from .models import UNet2DConditionModel
from .schedulers import SchedulerMixin
from .utils import convert_state_dict_to_diffusers, convert_state_dict_to_peft

logger = logging.getLogger(__name__)


def set_seed(seed=42, rank=0):
    ms.set_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026  # noqa: E501
    # we do not expand alpha/sigma which is redundant for the broadcast shape is actually timesteps.shape
    alpha = sqrt_alphas_cumprod[timesteps].float()
    sigma = sqrt_one_minus_alphas_cumprod[timesteps].float()

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def compute_dream_and_update_latents(
    unet: UNet2DConditionModel,
    noise_scheduler: SchedulerMixin,
    timesteps: ms.Tensor,
    noise: ms.Tensor,
    noisy_latents: ms.Tensor,
    target: ms.Tensor,
    encoder_hidden_states: ms.Tensor,
    dream_detail_preservation: float = 1.0,
) -> Tuple[Optional[ms.Tensor], Optional[ms.Tensor]]:
    """
    Implements "DREAM (Diffusion Rectification and Estimation-Adaptive Models)" from http://arxiv.org/abs/2312.00210.
    DREAM helps align training with sampling to help training be more efficient and accurate at the cost of an extra
    forward step without gradients.

    Args:
        `unet`: The state unet to use to make a prediction.
        `noise_scheduler`: The noise scheduler used to add noise for the given timestep.
        `timesteps`: The timesteps for the noise_scheduler to user.
        `noise`: A tensor of noise in the shape of noisy_latents.
        `noisy_latents`: Previously noise latents from the training loop.
        `target`: The ground-truth tensor to predict after eps is removed.
        `encoder_hidden_states`: Text embeddings from the text model.
        `dream_detail_preservation`: A float value that indicates detail preservation level.
          See reference.

    Returns:
        `tuple[ms.Tensor, ms.Tensor]`: Adjusted noisy_latents and target.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod[timesteps, None, None, None]
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # The paper uses lambda = sqrt(1 - alpha) ** p, with p = 1 in their experiments.
    dream_lambda = sqrt_one_minus_alphas_cumprod**dream_detail_preservation

    pred = unet(noisy_latents, timesteps, encoder_hidden_states)[0]

    _noisy_latents, _target = (None, None)
    if noise_scheduler.config["prediction_type"] == "epsilon":
        predicted_noise = pred
        delta_noise = ops.stop_gradient(noise - predicted_noise)
        delta_noise = delta_noise * dream_lambda
        _noisy_latents = noisy_latents.add(sqrt_one_minus_alphas_cumprod * delta_noise)
        _target = target.add(delta_noise)
    elif noise_scheduler.config["prediction_type"] == "v_prediction":
        raise NotImplementedError("DREAM has not been implemented for v-prediction")
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config['prediction_type']}")

    return _noisy_latents, _target


def cast_training_params(model: Union[nn.Cell, List[nn.Cell]], dtype=ms.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.get_parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.set_dtype(dtype)


def _set_state_dict_into_text_encoder(lora_state_dict: Dict[str, ms.Tensor], prefix: str, text_encoder: nn.Cell):
    """
    Sets the `lora_state_dict` into `text_encoder` coming from `transformers`.

    Args:
        lora_state_dict: The state dictionary to be set.
        prefix: String identifier to retrieve the portion of the state dict that belongs to `text_encoder`.
        text_encoder: Where the `lora_state_dict` is to be set.
    """

    text_encoder_state_dict = {
        f'{k.replace(prefix, "")}': v for k, v in lora_state_dict.items() if k.startswith(prefix)
    }
    text_encoder_state_dict = convert_state_dict_to_peft(convert_state_dict_to_diffusers(text_encoder_state_dict))
    set_peft_model_state_dict(text_encoder, text_encoder_state_dict, adapter_name="default")


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = ops.normal(mean=logit_mean, stddev=logit_std, shape=(batch_size,))
        u = ops.sigmoid(u)
    elif weighting_scheme == "mode":
        u = ops.rand(batch_size)
        u = 1 - u - mode_scale * (ops.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = ops.rand(batch_size)
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = ops.ones_like(sigmas)
    return weighting


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        parameters: Iterable[ms.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        foreach: bool = False,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
    ):
        """
        Args:
            parameters (Iterable[mindspore.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            foreach (bool): Use torch._foreach functions for updating shadow parameters. Should be faster.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """
        parameters = list(parameters)
        self.shadow_params = [p.clone() for p in parameters]

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`
        self.foreach = foreach

        self.model_cls = model_cls
        self.model_config = model_config

    @classmethod
    def from_pretrained(cls, path, model_cls, foreach=False) -> "EMAModel":
        _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
        model = model_cls.from_pretrained(path)

        ema_model = cls(model.get_parameters(), model_cls=model_cls, model_config=model.config, foreach=foreach)

        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        model = self.model_cls.from_config(self.model_config)
        state_dict = self.state_dict()
        state_dict.pop("shadow_params", None)

        model.register_to_config(**state_dict)
        self.copy_to(model.get_parameters())
        model.save_pretrained(path)

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    def step(self, parameters: Iterable[ms.Parameter]):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        if self.foreach:
            raise NotImplementedError("Not Implemeneted for `foreach` branch.")
        else:
            for s_param, param in zip(self.shadow_params, parameters):
                # TODO: gather parameters if they are sharded.
                if param.requires_grad:
                    ops.assign_sub(s_param, one_minus_decay * (s_param - param))
                else:
                    ops.assign(s_param, param)

    def copy_to(self, parameters: Iterable[ms.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `mindspore.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = list(parameters)
        if self.foreach:
            raise NotImplementedError("Not Implemeneted for `foreach` branch.")
        else:
            for s_param, param in zip(self.shadow_params, parameters):
                ops.assign(param, s_param)

    def pin_memory(self) -> None:
        r"""
        Move internal buffers of the ExponentialMovingAverage to pinned memory. Useful for non-blocking transfers for
        offloading EMA params to the host.
        """

        raise NotImplementedError("Not Implemeneted for `pin_memory`.")

    def to(self, dtype=None, non_blocking=False) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`."""
        # .to() on the tensors handles None correctly
        raise NotImplementedError("Not Implemeneted for `to`.")

    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict. This method is used by accelerate during
        checkpointing to save the ema state dict.
        """
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            self.shadow_params = shadow_params
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            if not all(isinstance(p, ms.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")


def is_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.distributed:
        init()
        args.local_rank = get_local_rank()
        args.world_size = get_group_size()
        args.rank = get_rank()
        ms.context.set_auto_parallel_context(
            device_num=args.world_size,
            global_rank=args.rank,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )

    device = f"{ms.get_context('device_target')}:{ms.get_context('device_id')}"
    args.device = device
    return device


@ms.jit_class
class AttrJitWrapper:
    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])


class CompileProgressBar:
    def __init__(self, duration, enable):
        self.duration = duration
        self.enable = enable
        self.q: Queue = None
        self.p: Process = None

    def compile_progress_bar(self):
        pb = tqdm(total=self.duration, bar_format="{l_bar}{bar}| [{elapsed}<{remaining}]")
        while True:
            if self.q.empty():
                time.sleep(1)
                if pb.last_print_n < self.duration:
                    pb.update(1)
                else:
                    pb.refresh(lock_args=pb.lock_args)
            else:
                if self.q.get():
                    pb.update(self.duration - pb.last_print_n)
                pb.close()
                break

    def __enter__(self):
        if self.enable:
            self.q = Queue()
            self.p = Process(target=self.compile_progress_bar)
            self.p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            if exc_type:
                logger.error(f"Oops! Error happens when compiling. {exc_type}: {exc_val}.")
                self.q.put(False)
            else:
                self.q.put(True)
            self.p.join()
            self.p.close()
            self.q.close()


def maybe_compile(m: nn.Cell, enable_progress_bar: bool, *model_args, **model_kwargs):
    if os.getenv("MS_JIT") != "0" and context._get_mode() == context.GRAPH_MODE:
        logger.info(f"Compiling {m.__class__.__name__}...")
        estimated_duration = sum(p.numel() for p in m.get_parameters()) * 2e-7
        with CompileProgressBar(estimated_duration, enable_progress_bar):
            compile_begin = time.perf_counter()
            m.compile(*model_args, **model_kwargs)
            compile_end = time.perf_counter()
        logger.info(f"Compiling is finished, elapsed time {compile_end - compile_begin:.2f} s")


def _grad_accum(cumulative_grad, grad):
    return ops.assign_add(cumulative_grad, grad)


def _grad_clear(cumulative_grad):
    return ops.assign(cumulative_grad, ops.zeros_like(cumulative_grad))


@ms.jit_class
class GradAccumulator:
    """
    A gradient accumulator tries to mock the behavior of PyTorch's backward.

    In PyTorch, after calling `loss.backward()`, the calculated gradients are accumulated to
    the 'grad' attribute of `nn.Parameter`. In distributed scenarios, DDP handles gradient synchronization
    across each device, ensuring that the 'grad' attribute of `nn.Parameter` always receives synchronized gradients.

    This class mocks the above process with the function 'ops.value_and_grad'. Here is an example in Mindspore
    corresponds to the basic PyTorch autograd:

    >>> model = nn.Dense(10, 10)
    >>> # MindSpore
    >>> grad_accumulator = GradAccumulator(model.parameters, gradient_accumulation_steps=None)
    >>> forward_and_backward = ops.value_and_grad(model, None, weights=model.parameters, has_aux=True)
    >>> (loss, logits), grads = forward_and_backward(*inputs)
    >>> grads = grad_accumulator.step(grads)
    >>> # PyTorch
    >>> (loss, logits) = model(*inputs)
    >>> loss.backward()

    The main difference is that in PyTorch, gradients are assigned to the 'grad' attribute of `nn.Parameter`,
    while in MindSpore, they are returned explicitly as a Tensor.

    Moreover, there is a context named 'no_sync' in PyTorch. Within this context, gradients will be accumulated
    on module variables, which will later be synchronized in the first forward-backward pass exiting the context.
    This feature solves the slowdown problem in gradient accumulation by eliminating unnecessary synchronization.
    It's our default behavior. We provide a full example here:

    >>> iters_to_accumulate = 4
    >>> # MindSpore
    >>> grad_accumulator = GradAccumulator(model.parameters, gradient_accumulation_steps=iters_to_accumulate)
    >>> forward_and_backward = ops.value_and_grad(model, None, weights=model.parameters, has_aux=True)
    >>> (loss, logits), grads = forward_and_backward(*inputs)
    >>> grads = grad_accumulator.step(grads)
    >>> # PyTorch
    >>> cm = nullcontext if (i + 1) % iters_to_accumulate == 0 else model.no_sync
    >>> with cm:
    >>>     (loss, logits) = model(*inputs)
    >>>     loss.backward()

    Args:
        params (mindspore.ParameterTuple): The parameters reqires gradient.
        gradient_accumulation_steps (Optional[int]): The number of gradient accumulation steps.
        **kwargs: Additional keyword arguments. Available keyword arguments:
            sync_with_dataloader (`bool`): Whether the gradients should be synced at the end of the dataloader
                iteration and the number of total steps reset.
            length_of_dataloader (`int`): The length of dataloader.
    """

    def __init__(self, params: ms.ParameterTuple, gradient_accumulation_steps: Optional[int], **kwargs):
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode == context.ParallelMode.STAND_ALONE:
            self.grad_reducer = nn.Identity()
        elif parallel_mode in (context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL):
            self.grad_reducer = nn.DistributedGradReducer(params)
        else:
            raise NotImplementedError(f"Unsupported parallel mode: {parallel_mode}")

        if gradient_accumulation_steps is None or gradient_accumulation_steps == 1:
            self.grads = ms.ParameterTuple(())  # placeholder
            self.step = self._dummy_step
            self.zero_grad = self._dummy_zero_grad
            self.sync_with_dataloader = False
            self.length_of_dataloader = -1
        elif gradient_accumulation_steps > 1:
            self.grads = params.clone(prefix="grads", init="zeros")
            self.step = self._step
            self.zero_grad = self._zero_grad
            self.sync_with_dataloader = kwargs.pop("sync_with_dataloader", True)
            self.length_of_dataloader = kwargs.pop("length_of_dataloader", -1)
        else:
            raise ValueError(f"'gradient_accumulation_steps' must be positive, but got {gradient_accumulation_steps}")

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_idx = ms.Parameter(ms.Tensor(0, ms.int64), name="batch_idx", requires_grad=False)
        self.sync_gradients = ms.Parameter(ms.Tensor(True), name="sync_gradients", requires_grad=False)
        self.hyper_map = ops.HyperMap()

        if self.sync_with_dataloader and self.length_of_dataloader <= 0:
            raise ValueError(
                f"The gradients will be synchronized at the end of the dataloader iteration, so you must pass in "
                f"a positive 'length_of_dataloader', but got {self.length_of_dataloader=}. Alternatively, you can "
                f"set 'sync_with_dataloader=False' to disable gradient synchronization, but it's not recommended."
            )

    def _dummy_step(self, grads):
        return self.grad_reducer(grads)

    def _dummy_zero_grad(self):
        return True  # Just return something to make the dumb-ass compiler happy

    def _do_sync(self):
        """Sets the right `sync_gradients` context and either resets or increases `self.batch_idx`"""
        end_of_dataloader = self.batch_idx + 1 == self.length_of_dataloader
        if self.sync_with_dataloader and end_of_dataloader:
            ops.assign(self.batch_idx, 0)
            ops.assign(self.sync_gradients, True)
        else:
            ops.assign_add(self.batch_idx, 1)
            ops.assign(self.sync_gradients, (self.batch_idx % self.gradient_accumulation_steps) == 0)
        return True  # Just return something to make the dumb-ass compiler happy

    def _step(self, grads):
        self._do_sync()
        # Accumulates gradients. Only accumulate locally, no sync.
        self.hyper_map(_grad_accum, self.grads, grads)

        if self.sync_gradients:
            return self.grad_reducer(self.grads)
        else:
            return self.grads

    def _zero_grad(self):
        self.hyper_map(_grad_clear, self.grads)
        return True  # Just return something to make the dumb-ass compiler happy


@ms.jit_class
class GradClipper:
    def __init__(self, max_grad_norm):
        if max_grad_norm is None:
            self.clip_grad_norm = self._dummy_clip
        elif max_grad_norm > 0:
            self.clip_grad_norm = self._clip
        else:
            raise ValueError(f"'max_grad_norm' must be positive, but got {max_grad_norm}")
        self.max_grad_norm = max_grad_norm

    def _dummy_clip(self, grads):
        return grads

    def _clip(self, grads):
        return ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)


@ms.jit_class
class GradScaler:
    def __init__(self, loss_scaler):
        if loss_scaler is None:
            self.loss_scaler = StaticLossScaler(1.0)
            self.step = self._always_opt_step
        elif isinstance(loss_scaler, StaticLossScaler):
            self.loss_scaler = loss_scaler
            self.step = self._always_opt_step
        elif isinstance(loss_scaler, DynamicLossScaler):
            self.loss_scaler = loss_scaler
            self.step = self._maybe_opt_step
        else:
            raise NotImplementedError(f"Unsupported loss scaler: {type(loss_scaler)}")
        self.all_finite = ms.Parameter(ms.Tensor(True), name="all_finite", requires_grad=False)

    def scale(self, inputs):
        return self.loss_scaler.scale(inputs)

    def unscale(self, inputs):
        return self.loss_scaler.unscale(inputs)

    def _always_opt_step(self, optimizer: nn.Optimizer, grads):
        optimizer(grads)
        return True

    def _maybe_opt_step(self, optimizer: nn.Optimizer, grads):
        ops.assign(self.all_finite, all_finite(grads))
        if self.all_finite:
            optimizer(grads)
            return True
        else:
            # Since 'optimizer.step()' is skipped, we do not perform 'scheduler.step()' following accelerator.
            # But someone using native Pytorch might do that.
            # BTW, should we give users an overflow warning?
            return False

    def update(self):
        return self.loss_scaler.adjust(self.all_finite)


class TrainStep(nn.Cell, metaclass=ABCMeta):
    """
    A base class for training steps in MindSpore.

    This class provides a basic framework for training neural networks. It takes care of
    gradient accumulation, scaling, and clipping, as well as optimizer updates.

    Args:
        model (nn.Cell): The neural network model to be trained.
        optimizer (nn.Optimizer): The optimizer used for updating model parameters.
        loss_scaler (Optional[Union[StaticLossScaler, DynamicLossScaler]]): The loss scaler to apply during training.
        max_grad_norm (Optional[float]): The maximum gradient norm for gradient clipping.
        gradient_accumulation_steps (Optional[int]): The number of gradient accumulation steps.
        **kwargs: Additional keyword arguments. Available keyword arguments:
            gradient_accumulation_kwargs: Additional keyword arguments for the `GradAccumulator`.

    Attributes:
        model (nn.Cell): The neural network model.
        optimizer (nn.Optimizer): The optimizer.
        parameters (list): The parameters of the optimizer.
        grad_scaler (GradScaler): The gradient scaler.
        grad_clipper (GradClipper): The gradient clipper.
        grad_accumulator (GradAccumulator): The gradient accumulator.

    Properties:
        sync_gradients (bool): Indicates whether gradients are synchronized.

    Methods:
        scale_loss: Scales the loss according to the gradient accumulation steps.
        unscale_loss: Unscales the loss.
        forward: Abstract method for forward pass.
        forward_and_backward: The function for performing forward and backward passes.
        construct: Constructs the training step.

    Raises:
        NotImplementedError: If the forward method is not implemented in subclasses.

    Note:
        - This class is abstract, meaning you must subclass it to create a specific training step.
        - When implementing the forward method, the users must call 'self.scale_loss' at the end.

    Examples:
        1. Basic usage. Only the forward method implementation.

            >>> class MyAwesomeTrainStep(TrainStep):
            >>>     def forward(self, x):
            >>>         y = self.model(x)
            >>>         loss = ops.sum(y)
            >>>         loss = self.scale_loss(loss)
            >>>         return loss, y
            >>>
            >>> model = nn.Dense(10, 10)
            >>> optim = nn.AdamWeightDecay(model.trainable_params())
            >>> train_step = MyAwesomeTrainStep(
            >>>     model,
            >>>     optim,
            >>>     loss_scaler=DynamicLossScaler(2.0**16, 2, 2000),
            >>>     max_grad_norm=1.0,
            >>>     gradient_accumulation_steps=2,
            >>>     gradient_accumulation_kwargs={"length_of_dataloader": 3}
            >>> )
            >>>
            >>> for epoch in range(2):
            >>>     for batch in range(3):
            >>>         inputs = ops.randn(8, 10)
            >>>         outputs = train_step(inputs)

        2. Advanced usage. Multiple model needs to overload __init__ method.

            >>> class MyAwesomeTrainStep(TrainStep):
            >>>     def __init__(self, text_encoder, unet, optim):
            >>>         super().__init__(unet, optim)
            >>>         self.unet = self.model
            >>>         self.text_encoder = text_encoder
            >>>
            >>>     def forward(self, x, t):
            >>>         e = self.text_encoder(t)
            >>>         y = self.unet(x, e)
            >>>         loss = ops.sum(y)
            >>>         loss = self.scale_loss(loss)
            >>>         return loss, y
            >>> # Then you can launch the training as usual.
    """

    def __init__(
        self,
        model: nn.Cell,
        optimizer: nn.Optimizer,
        loss_scaler: Optional[Union[StaticLossScaler, DynamicLossScaler]] = None,
        max_grad_norm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model.set_grad()  # Why do we need call 'set_grad()'?
        if model.jit_config_dict:
            self.set_jit_config(model.jit_config_dict)
        self.optimizer = optimizer
        self.parameters = optimizer.parameters

        self.grad_scaler = GradScaler(loss_scaler)
        self.grad_clipper = GradClipper(max_grad_norm)

        gradient_accumulation_kwargs = kwargs.pop("gradient_accumulation_kwargs", {})
        self.grad_accumulator = GradAccumulator(
            self.parameters, gradient_accumulation_steps, **gradient_accumulation_kwargs
        )

        self.forward_and_backward = ops.value_and_grad(self.forward, None, weights=self.parameters, has_aux=True)

    @property
    def sync_gradients(self):
        return self.grad_accumulator.sync_gradients

    def scale_loss(self, loss):
        loss = loss / self.grad_accumulator.gradient_accumulation_steps
        loss = self.grad_scaler.scale(loss)
        return loss

    def unscale_loss(self, loss):
        return self.grad_scaler.unscale(loss)

    @abstractmethod
    def forward(self, *args, **kwargs):
        # You need to scale the loss when performing the model forward pass to create scaled gradients.
        # Do **NOT** forget to include 'loss = self.scale_loss(loss)' after loss calculation!
        ...

    def construct(self, *inputs):
        outputs, grads = self.forward_and_backward(*inputs)
        grads = self.grad_accumulator.step(grads)

        if self.sync_gradients:
            # Scaled loss creates scaled gradients. Unscales the gradients.
            grads = self.grad_scaler.unscale(grads)

            # Since the gradients are unscaled, clips as usual.
            grads = self.grad_clipper.clip_grad_norm(grads)

            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.grad_scaler.step(self.optimizer, grads)

            # Updates the scale for next iteration.
            self.grad_scaler.update()

            # Clear the gradients of accumulator's assigned params.
            self.grad_accumulator.zero_grad()

        # The first item of outputs is loss. Unscales the loss for outside logging.
        loss = self.unscale_loss(outputs[0])
        outputs = (loss,) + outputs[1:]
        return outputs


@ms.jit_class
class pynative_no_grad(contextlib.ContextDecorator):
    """
    Context Manager to disable gradient calculation. When enter this context, we will disable calculate
    gradient. When exit this context, we will resume its prev state.
    Currently, it can use both in Pynative and Graph mode. It also can be used as decorator.

    For mindone.diffusers, it is used in PyNative training to decorate the part of calculation that
    does not require gradients, e.g. vae.encode_images or text_encoder.encode_prompts where does not
    need to train VAE or text-encoders.
    """

    def __init__(self):
        self.is_pynative_mode = context.get_context("mode") == context.PYNATIVE_MODE or os.getenv("MS_JIT") == "0"
        self.prev_state = False

    def __enter__(self):
        if self.is_pynative_mode:
            self.prev_state = _pynative_executor.enable_grad()
            _pynative_executor.set_enable_grad(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_pynative_mode:
            _pynative_executor.set_enable_grad(self.prev_state)
        return False


# Adapted from mindone.trainers.zero.prepare_train_network
def prepare_train_network(
    network: nn.Cell,
    optimizer: nn.Optimizer,
    scale_sense: Union[float, nn.Cell] = 1.0,
    updates: int = 0,
    drop_overflow_update: bool = True,
    gradient_accumulation_steps: int = 1,
    clip_grad: bool = False,
    clip_norm: float = 1.0,
    verbose: bool = False,
    zero_stage: int = 0,
    optimizer_offload: bool = False,
    op_group: str = None,
    dp_group: str = None,
    comm_fusion: dict = None,
    parallel_modules=None,
):
    """
    Prepare network and optimizer for distributed training.

    Args:
        network (`nn.Cell`): train network, not include grad function,
            grad function must be built after rewrite train network.
        optimizer (`nn.Optimizer`): Must be the subclass of MindSpore Optimizer.
        scale_sense (Union[Tensor, Cell]): If this value is a Cell, it will be called
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.
        zero_stage (`int`, *optional*): Stage setting of ZeRO, default is 0.
        optimizer_offload (`bool`, *optional*): Only take effect when optimizer is AdamWeightDecay, default is False.
        op_group (`str`, *optional*): The name of the optimizer parallel communication group, default is None.
        dp_group (`str`, *optional*): The name of the data parallel communication group, default is None.
        comm_fusion (`dict`, *optional*): A dict contains the types and configurations
            for setting the communication fusion, default is None, turn off the communication fusion. If set a dict,
            turn on the communication fusion.
            Examples: {"allreduce": {"openstate": True, "bucket_size": 5e8},
                       "reduce_scatter": {"openstate": True, "bucket_size": 5e8},
                       "allgather": {"openstate": False, "bucket_size": 5e8},}
        parallel_modules (`dict`, *optional*): A dict of Cells could split parameters in zero3, default is None.
            If None, use `PARALLEL_MODULES` from `mindone.models.modules.parallel`.
    """
    if zero_stage not in [0, 1, 2, 3]:
        raise ValueError("Not support zero_stage {zero_stage}")
    if op_group is None:
        logger.warning("Not set zero group, set it WORLD_COMM_GROUP.")
        op_group = GlobalComm.WORLD_COMM_GROUP
    if op_group != GlobalComm.WORLD_COMM_GROUP and dp_group is None:
        raise ValueError("op_group {op_group} and dp_group {dp_group} not full network hccl group coverage")

    is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
    if not is_parallel and zero_stage == 0:
        logger.info("No need prepare train_network with zero.")
        zero_helper = None
    else:
        network = prepare_network(network, zero_stage, op_group, parallel_modules=parallel_modules)
        zero_helper = ZeroHelper(optimizer, zero_stage, op_group, dp_group, optimizer_offload, comm_fusion)

    if isinstance(scale_sense, float):
        scale_sense = ms.Tensor(scale_sense, ms.float32)
    train_network = DiffusersTrainOneStepWrapper(
        network,
        optimizer,
        scale_sense=scale_sense,
        updates=updates,
        drop_overflow_update=drop_overflow_update,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        clip_norm=clip_norm,
        verbose=verbose,
        zero_helper=zero_helper,
    )
    return train_network


class DiffusersTrainOneStepWrapper(TrainOneStepWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def use_zero(self):
        return self.zero_helper is not None and self.zero_stage != 0

    def need_save_optimizer(self, args):
        # TODO: Now we save optimizer in every process, try to save depend on self.zero_helper.op_group
        return True if self.use_zero else is_local_master(args)

    def save_state(self, args, output_dir, optimizer_state_filter=lambda x: True):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving current state to {output_dir}")

        # Optimizer states
        if self.use_zero:
            os.makedirs(os.path.join(output_dir, "mindspore_model"), exist_ok=True)
            optimizer_file = os.path.join(output_dir, "mindspore_model", f"zero_pp_{args.local_rank}_optim_states.ckpt")
        elif self.need_save_optimizer(args):
            optimizer_file = os.path.join(output_dir, "optimizer.ckpt")
        ms.save_checkpoint(self.optimizer, optimizer_file, choice_func=optimizer_state_filter)

        # Loss Scaler states
        loss_scaler_file = os.path.join(output_dir, "loss_scaler.ckpt")
        loss_scaler_states = {"scale_sense": self.scale_sense}
        if self.loss_scaling_manager:
            loss_scaler_states.update(
                {
                    "cur_iter": self.loss_scaling_manager.cur_iter,
                    "last_overflow_iter": self.loss_scaling_manager.last_overflow_iter,
                }
            )
        ms.save_checkpoint(loss_scaler_states, loss_scaler_file)

    def load_state(self, args, input_dir, optimizer_state_filter=lambda x: True):
        # Optimizer states
        optimizer_file = (
            os.path.join(input_dir, "mindspore_model", f"zero_pp_{args.local_rank}_optim_states.ckpt")
            if self.use_zero
            else os.path.join(input_dir, "optimizer.ckpt")
        )
        optimizer_state_dict = ms.load_checkpoint(optimizer_file)

        with silence_mindspore_logger():
            param_not_load, ckpt_not_load = ms.load_param_into_net(
                self.optimizer, optimizer_state_dict, strict_load=True
            )

        param_not_load = list(filter(lambda x: optimizer_state_filter(x), param_not_load))
        if param_not_load or ckpt_not_load:
            logger.warning(
                f"Loading checkpoint into optimizer returns param_not_load:{param_not_load} \nand ckpt_not_load:{ckpt_not_load}"
            )

        # Loss Scaler states
        loss_scaler_file = os.path.join(input_dir, "loss_scaler.ckpt")
        loss_scaler_state_dict = ms.load_checkpoint(loss_scaler_file)

        scale_sense = loss_scaler_state_dict.get("scale_sense", ms.Tensor(1.0, dtype=mstype.float32))
        cur_iter = loss_scaler_state_dict.get("cur_iter", None)
        last_overflow_iter = loss_scaler_state_dict.get("last_overflow_iter", None)

        ops.assign(self.scale_sense, scale_sense)
        if cur_iter is not None and last_overflow_iter is not None:
            ops.assign(self.loss_scaling_manager.cur_iter, cur_iter)
            ops.assign(self.loss_scaling_manager.last_overflow_iter, last_overflow_iter)
