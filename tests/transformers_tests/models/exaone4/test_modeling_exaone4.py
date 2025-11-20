"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/exaone4/test_modeling_exaone4.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
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
from transformers import Exaone4Config

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)

from ...causal_lm_tester import CausalLMModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 8e-3}
MODES = [1]


class Exaone4ModelTester(CausalLMModelTester):
    config_class = Exaone4Config

    def __init__(self, parent=None):
        super().__init__(parent)


model_tester = Exaone4ModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


TEST_CASES = [
    [
        "Exaone4Model",
        "transformers.Exaone4Model",
        "mindone.transformers.Exaone4Model",
        (config,),
        {},
        (inputs_dict["input_ids"], inputs_dict["attention_mask"]),
        {},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "Exaone4ForCausalLM",
        "transformers.Exaone4ForCausalLM",
        "mindone.transformers.Exaone4ForCausalLM",
        (config,),
        {},
        (inputs_dict["input_ids"], inputs_dict["attention_mask"]),
        {},
        {
            "logits": "logits",
        },
    ],
    [
        "Exaone4ForSequenceClassification",
        "transformers.Exaone4ForSequenceClassification",
        "mindone.transformers.Exaone4ForSequenceClassification",
        (config,),
        {},
        (inputs_dict["input_ids"], inputs_dict["attention_mask"]),
        {},
        {
            "logits": "logits",
        },
    ],
    [
        "Exaone4ForTokenClassification",
        "transformers.Exaone4ForTokenClassification",
        "mindone.transformers.Exaone4ForTokenClassification",
        (config,),
        {},
        (inputs_dict["input_ids"], inputs_dict["attention_mask"]),
        {},
        {
            "logits": "logits",
        },
    ],
    [
        "Exaone4ForQuestionAnswering",
        "transformers.Exaone4ForQuestionAnswering",
        "mindone.transformers.Exaone4ForQuestionAnswering",
        (config,),
        {},
        (inputs_dict["input_ids"], inputs_dict["attention_mask"]),
        {},
        {
            "start_logits": "start_logits",
            "end_logits": "end_logits",
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [
        case
        + [
            dtype,
        ]
        + [
            mode,
        ]
        for case in TEST_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    ms.set_context(mode=mode)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    # set `hidden_dtype` if requiring, for some modules always compute in float
    # precision and require specific `hidden_dtype` to cast before return
    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)
    # print("ms:", ms_outputs)
    # print("pt:", pt_outputs)

    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            # print("===map", pt_key, ms_idx)
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[ms_idx]
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )
