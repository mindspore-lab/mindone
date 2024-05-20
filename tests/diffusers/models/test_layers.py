import importlib
import inspect
import logging
import sys
import unittest

import numpy as np
import torch
from diffusers.utils import BaseOutput
from parameterized import parameterized

import mindspore as ms
from mindspore import nn, ops

sys.path.append(".")
from .test_layers_cases import ALL_CASES

logger = logging.getLogger("ModulesUnitTest")


PT_DTYPE_MAPPING = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}


MS_DTYPE_MAPPING = {
    "fp16": ms.float16,
    "fp32": ms.float32,
}


TORCH_FP16_BLACKLIST = (
    "LayerNorm",
    "Timesteps",
    "AvgPool2d",
    "Upsample2D",
    "ResnetBlock2D",
    "FirUpsample2D",
    "FirDownsample2D",
    "KDownsample2D",
    "AutoencoderTiny",
)


def get_pt2ms_mappings(m):
    # copied from mindone.diffusers.models.modeling_utils
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    for name, cell in m.cells_and_names():
        if isinstance(cell, nn.Conv1d):
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


def convert_state_dict(m, state_dict_pt):
    # copied from mindone.diffusers.models.modeling_utils
    mappings = get_pt2ms_mappings(m)
    state_dict_ms = {}
    for name_pt, data_pt in state_dict_pt.items():
        name_ms, data_mapping = mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = data_mapping(ms.Parameter(ms.Tensor.from_numpy(data_pt.numpy())))
        if name_ms is not None:
            state_dict_ms[name_ms] = data_ms
    return state_dict_ms


def get_modules(pt_module, ms_module, dtype, *args, **kwargs):
    ms_dtype = pt_dtype = dtype

    pt_path, pt_cls_name = pt_module.rsplit(".", 1)
    ms_path, ms_cls_name = ms_module.rsplit(".", 1)
    pt_module_cls = getattr(importlib.import_module(pt_path), pt_cls_name)
    ms_module_cls = getattr(importlib.import_module(ms_path), ms_cls_name)

    pt_modules_instance = pt_module_cls(*args, **kwargs)
    ms_modules_instance = ms_module_cls(*args, **kwargs)

    if dtype == "fp16":
        pt_modules_instance = pt_modules_instance.to(torch.float16)
        ms_modules_instance = set_dtype(ms_modules_instance, ms.float16)
    elif dtype == "fp32":
        pt_modules_instance = pt_modules_instance.to(torch.float32)
        ms_modules_instance = set_dtype(ms_modules_instance, ms.float32)
    else:
        raise NotImplementedError(f"Dtype {dtype} for model is not implemented")

    missing_keys, unexpected_keys = ms.load_param_into_net(
        ms_modules_instance, convert_state_dict(ms_modules_instance, pt_modules_instance.state_dict()), strict_load=True
    )
    if missing_keys or unexpected_keys:
        logger.warning(
            f"When load state_dict of '{pt_module}' to encounterpart mindspore model:\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}\n"
        )

    pt_modules_instance.eval()
    ms_modules_instance.set_train(False)

    if dtype == "fp32":
        return pt_modules_instance, ms_modules_instance, pt_dtype, ms_dtype

    # Some torch modules do not support fp16 in CPU, converted to fp32 instead.
    for _, submodule in pt_modules_instance.named_modules():
        if submodule.__class__.__name__ in TORCH_FP16_BLACKLIST:
            logger.warning(
                f"Model '{pt_module}' has submodule {submodule.__class__.__name__} which doens't support fp16, converted to fp32 instead."
            )
            pt_modules_instance = pt_modules_instance.to(torch.float32)
            pt_dtype = "fp32"
            break

    return pt_modules_instance, ms_modules_instance, pt_dtype, ms_dtype


def set_dtype(model, dtype):
    for p in model.get_parameters():
        p = p.set_dtype(dtype)
    return model


def common_parse_args(pt_dtype, ms_dtype, *args, **kwargs):
    dtype_mappings = {
        "fp32": np.float32,
        "fp16": np.float16,
    }

    # parse args
    pt_inputs_args = tuple()
    ms_inputs_args = tuple()
    for x in args:
        if isinstance(x, np.ndarray):
            if x.dtype in (np.float16, np.float32, np.float64):
                px = x.astype(dtype_mappings[pt_dtype])
                mx = x.astype(dtype_mappings[ms_dtype])
            else:
                px = mx = x

            pt_inputs_args += (torch.from_numpy(px),)
            ms_inputs_args += (ms.Tensor.from_numpy(mx),)
        else:
            pt_inputs_args += (x,)
            ms_inputs_args += (x,)

    # parse kwargs
    pt_inputs_kwargs = dict()
    ms_inputs_kwargs = dict()
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            if v.dtype in (np.float16, np.float32, np.float64):
                px = v.astype(dtype_mappings[pt_dtype])
                mx = v.astype(dtype_mappings[ms_dtype])
            else:
                px = mx = v

            pt_inputs_kwargs[k] = torch.from_numpy(px)
            ms_inputs_kwargs[k] = ms.Tensor.from_numpy(mx)
        else:
            pt_inputs_kwargs[k] = v
            ms_inputs_kwargs[k] = v

    return pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs


def compute_diffs(pt_outputs: torch.Tensor, ms_outputs: ms.Tensor, relative=True):
    if isinstance(pt_outputs, BaseOutput):
        pt_outputs = tuple(pt_outputs.values())
    elif not isinstance(pt_outputs, (tuple, list)):
        pt_outputs = (pt_outputs,)
    if not isinstance(ms_outputs, (tuple, list)):
        ms_outputs = (ms_outputs,)

    diffs = []
    for p, m in zip(pt_outputs, ms_outputs):
        if isinstance(p, BaseOutput):
            p = tuple(p.values())[0]

        p = p.detach().cpu().numpy()
        m = m.asnumpy()

        d = np.abs(p - m)

        if relative:
            eps = np.abs(p).mean() * 0.05 + 1e-9
            d = d / (np.abs(p) + eps)

        diffs.append(d.mean())

    return diffs


# TODO: decouple with torch, maybe a feasible solution is fixing seed
# and comparing with fixed expected result, just like what diffusers does
class ModulesTest(unittest.TestCase):
    # 1% relative error when FP32 and 2% when FP16
    eps = 0.01

    @parameterized.expand(ALL_CASES)
    def test_named_modules_with_graph_fp32(
        self,
        name,
        pt_module,
        ms_module,
        init_args,
        init_kwargs,
        inputs_args,
        inputs_kwargs,
    ):
        dtype = "fp32"
        ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)

        (
            pt_model,
            ms_model,
            pt_dtype,
            ms_dtype,
        ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
        pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = common_parse_args(
            pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
        )

        with torch.no_grad():
            pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
        ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

        diffs = compute_diffs(pt_outputs, ms_outputs)

        eps = self.eps * 2 if pt_dtype == "fp16" or ms_dtype == "fp16" else self.eps
        self.assertTrue(
            (np.array(diffs) < eps).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {eps}"
        )

    @parameterized.expand(ALL_CASES)
    def test_named_modules_with_graph_fp16(
        self,
        name,
        pt_module,
        ms_module,
        init_args,
        init_kwargs,
        inputs_args,
        inputs_kwargs,
    ):
        dtype = "fp16"
        ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)

        (
            pt_model,
            ms_model,
            pt_dtype,
            ms_dtype,
        ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
        pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = common_parse_args(
            pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
        )

        if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
            pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
            ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

        with torch.no_grad():
            pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
        ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

        diffs = compute_diffs(pt_outputs, ms_outputs)

        eps = self.eps * 2 if pt_dtype == "fp16" or ms_dtype == "fp16" else self.eps
        self.assertTrue(
            (np.array(diffs) < eps).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {eps}"
        )

    @parameterized.expand(ALL_CASES)
    def test_named_modules_with_pynative_fp32(
        self,
        name,
        pt_module,
        ms_module,
        init_args,
        init_kwargs,
        inputs_args,
        inputs_kwargs,
    ):
        dtype = "fp32"
        ms.set_context(mode=ms.PYNATIVE_MODE, jit_syntax_level=ms.STRICT)

        (
            pt_model,
            ms_model,
            pt_dtype,
            ms_dtype,
        ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
        pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = common_parse_args(
            pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
        )

        with torch.no_grad():
            pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
        ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

        diffs = compute_diffs(pt_outputs, ms_outputs)

        eps = self.eps * 2 if pt_dtype == "fp16" or ms_dtype == "fp16" else self.eps
        self.assertTrue(
            (np.array(diffs) < eps).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {eps}"
        )

    @parameterized.expand(ALL_CASES)
    def test_named_modules_with_pynative_fp16(
        self,
        name,
        pt_module,
        ms_module,
        init_args,
        init_kwargs,
        inputs_args,
        inputs_kwargs,
    ):
        dtype = "fp16"
        ms.set_context(mode=ms.PYNATIVE_MODE, jit_syntax_level=ms.STRICT)

        (
            pt_model,
            ms_model,
            pt_dtype,
            ms_dtype,
        ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
        pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = common_parse_args(
            pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
        )

        if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
            pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
            ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

        with torch.no_grad():
            pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
        ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

        diffs = compute_diffs(pt_outputs, ms_outputs)

        eps = self.eps * 2 if pt_dtype == "fp16" or ms_dtype == "fp16" else self.eps
        self.assertTrue(
            (np.array(diffs) < eps).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {eps}"
        )

    @parameterized.expand(
        [
            ["T5FilmDecoder_graph_fp32", 0, "fp32"],
            ["T5FilmDecoder_graph_fp16", 0, "fp16"],
            ["T5FilmDecoder_pynative_fp32", 1, "fp32"],
            ["T5FilmDecoder_pynative_fp16", 1, "fp16"],
        ]
    )
    def test_t5_film_decoder(self, name, mode, dtype):
        ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)

        # init model
        pt_module = "diffusers.models.transformers.t5_film_transformer.T5FilmDecoder"
        ms_module = f"mindone.{pt_module}"

        init_args = ()
        init_kwargs = {
            "input_dims": 32,
            "d_model": 64,
            "num_heads": 4,
            "d_ff": 64,
            "targets_length": 8,
        }

        pt_model, ms_model, pt_dtype, ms_dtype = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)

        # get inputs
        inputs_args = (
            np.random.randn(2, 64, 64).astype(np.float32),
            np.random.randn(2, 64).astype(np.int32),
            np.random.randn(2, 8, 64).astype(np.float32),
            np.random.randn(2, 8).astype(np.int32),
        )
        inputs_kwargs = {
            "decoder_input_tokens": np.random.randn(2, 8, 32).astype(np.float32),
            "decoder_noise_time": np.array([0.99, 0.50]).astype(np.float32),
        }

        pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = common_parse_args(
            pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
        )
        pt_inputs_kwargs["encodings_and_masks"] = (
            (pt_inputs_args[0], pt_inputs_args[1]),
            (pt_inputs_args[2], pt_inputs_args[3]),
        )
        ms_inputs_kwargs["encodings_and_masks"] = (
            (ms_inputs_args[0], ms_inputs_args[1]),
            (ms_inputs_args[2], ms_inputs_args[3]),
        )

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs_kwargs)
        ms_outputs = ms_model(**ms_inputs_kwargs)

        diffs = compute_diffs(pt_outputs, ms_outputs)

        eps = self.eps * 2 if pt_dtype == "fp16" or ms_dtype == "fp16" else self.eps
        self.assertTrue(
            (np.array(diffs) < eps).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {eps}"
        )
