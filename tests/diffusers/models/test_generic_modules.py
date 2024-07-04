# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import inspect

import numpy as np
import pytest
import torch

import mindspore as ms

from .modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from .modules_test_cases import ALL_CASES

THRESHOLD_FP16 = 1e-2
THRESHOLD_FP32 = 5e-3


PT_DTYPE_MAPPING = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}


MS_DTYPE_MAPPING = {
    "fp16": ms.float16,
    "fp32": ms.float32,
}


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs",
    ALL_CASES,
)
def test_named_modules_with_graph_fp32(
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
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    diffs = compute_diffs(pt_outputs, ms_outputs)

    assert (
        np.array(diffs) < THRESHOLD_FP32
    ).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD_FP32}"


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs",
    ALL_CASES,
)
def test_named_modules_with_graph_fp16(
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
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    diffs = compute_diffs(pt_outputs, ms_outputs)

    assert (
        np.array(diffs) < THRESHOLD_FP16
    ).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD_FP16}"


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs",
    ALL_CASES,
)
def test_named_modules_with_pynative_fp32(
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
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    diffs = compute_diffs(pt_outputs, ms_outputs)

    assert (
        np.array(diffs) < THRESHOLD_FP32
    ).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD_FP32}"


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs",
    ALL_CASES,
)
def test_named_modules_with_pynative_fp16(
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
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    diffs = compute_diffs(pt_outputs, ms_outputs)

    assert (
        np.array(diffs) < THRESHOLD_FP16
    ).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD_FP16}"
