import importlib
import inspect
import logging
import random
import sys
from typing import Callable, List, Optional, Tuple, Union

import torch

import mindspore as ms
from mindspore import nn, ops

from mindone.diffusers.models.modeling_utils import ModelMixin
from mindone.diffusers.pipelines.pipeline_utils import DiffusionPipeline
from mindone.diffusers.schedulers.scheduling_utils import SchedulerMixin
from mindone.diffusers.utils.mindspore_utils import randn_tensor as original_randn_tensor

global_rng = random.Random()

logger = logging.getLogger("PipelinesUnitTest")

THRESHOLD_FP16 = 5e-2
THRESHOLD_FP32 = 5e-3
THRESHOLD_PIXEL = 20.0


# copied from mindone.diffusers.models.modeling_utils
def get_pt2ms_mappings(m):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    for name, cell in m.cells_and_names():
        if isinstance(cell, (nn.Conv1d, nn.Conv1dTranspose)):
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: ops.expand_dims(x, axis=-2)
        elif isinstance(cell, nn.Embedding):
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
            if isinstance(cell, (nn.BatchNorm2d,)):
                mappings[f"{name}.running_mean"] = f"{name}.moving_mean", lambda x: x
                mappings[f"{name}.running_var"] = f"{name}.moving_variance", lambda x: x
                mappings[f"{name}.num_batches_tracked"] = None, lambda x: x
    return mappings


# adapted from mindone.diffusers.models.modeling_utils
def convert_state_dict(m, state_dict_pt):
    mappings = get_pt2ms_mappings(m)
    state_dict_ms = {}
    for name_pt, data_pt in state_dict_pt.items():
        name_ms, data_mapping = mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = ms.Parameter(data_mapping(ms.Tensor.from_numpy(data_pt.numpy())), name=name_ms)
        if name_ms is not None:
            state_dict_ms[name_ms] = data_ms
    return state_dict_ms


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]
    dtype = torch.float32 if dtype == ms.float32 else torch.float16

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return ms.Tensor(latents.numpy())


def get_module(module_path):
    path, cls_name = module_path.rsplit(".", 1)
    return getattr(importlib.import_module(path), cls_name)


def get_pipeline_components(components, pipeline_config):
    for name, pt_module, ms_module, init_kwargs in pipeline_config:
        torch.manual_seed(0)
        pt_module_cls = get_module(pt_module)
        ms_module_cls = get_module(ms_module)

        if "pretrained_model_name_or_path" in init_kwargs:
            pt_modules_instance = pt_module_cls.from_pretrained(**init_kwargs)
            ms_modules_instance = (
                pt_modules_instance if pt_module == ms_module else ms_module_cls.from_pretrained(**init_kwargs)
            )
        elif "unet" in init_kwargs:
            init_kwargs["unet"] = components["unet"][0]
            pt_modules_instance = pt_module_cls.from_unet(**init_kwargs)
            init_kwargs["unet"] = components["unet"][1]
            ms_modules_instance = ms_module_cls.from_unet(**init_kwargs)
        else:
            pt_modules_instance = pt_module_cls(**init_kwargs)
            ms_modules_instance = ms_module_cls(**init_kwargs)

        if hasattr(pt_modules_instance, "state_dict"):
            missing_keys, unexpected_keys = ms.load_param_into_net(
                ms_modules_instance,
                convert_state_dict(ms_modules_instance, pt_modules_instance.state_dict()),
                strict_load=True,
            )
            if missing_keys or unexpected_keys:
                logger.warning(
                    f"When load state_dict of '{pt_module}' to encounterpart mindspore model:\n"
                    f"Missing keys: {missing_keys}\n"
                    f"Unexpected keys: {unexpected_keys}\n"
                )

        components[name] = pt_modules_instance, ms_modules_instance

    pt_components = {key: value[0] if value is not None else None for key, value in components.items()}
    ms_components = {key: value[1] if value is not None else None for key, value in components.items()}

    return pt_components, ms_components


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


def randn_tensor_replace(randn_tensor_func: Callable):
    maybe_pipelines = inspect.getmembers(sys.modules["mindone.diffusers.pipelines"], inspect.isclass)
    pipelines = filter(
        lambda x: issubclass(x[1], DiffusionPipeline) and x[0] != "DiffusionPipeline",
        maybe_pipelines,
    )

    maybe_schedulers = inspect.getmembers(sys.modules["mindone.diffusers.schedulers"], inspect.isclass)
    schedulers = filter(
        lambda x: issubclass(x[1], SchedulerMixin) and x[0] != "SchedulerMixin",
        maybe_schedulers,
    )

    maybe_vaes = inspect.getmembers(sys.modules["mindone.diffusers.models.autoencoders"], inspect.isclass)
    vaes = filter(
        lambda x: issubclass(x[1], ModelMixin) and x[0] != "ModelMixin",
        maybe_vaes,
    )

    for _, pipeline in pipelines:
        if hasattr(sys.modules[pipeline.__module__], "randn_tensor"):
            sys.modules[pipeline.__module__].randn_tensor = randn_tensor_func

    for _, scheduler in schedulers:
        if hasattr(sys.modules[scheduler.__module__], "randn_tensor"):
            sys.modules[scheduler.__module__].randn_tensor = randn_tensor_func

    for _, vae in vaes:
        vae = vae()
        if hasattr(vae, "diag_gauss_dist"):
            sys.modules[vae.diag_gauss_dist.__module__].randn_tensor = randn_tensor_func


class PipelineTesterMixin:
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        randn_tensor_replace(randn_tensor)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        randn_tensor_replace(original_randn_tensor)
