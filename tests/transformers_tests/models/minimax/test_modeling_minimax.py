"""
Adapted from https://github.com/huggingface/transformers/blob/a6393e7d28e652c598ced79f0107f1eff370df1b/tests/models/minimax/test_modeling_minimax.py
"""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

from typing import Union

import numpy as np
import pytest
import transformers
from torch import inference_mode, load
from transformers.testing_utils import slow

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules

from ...causal_lm_tester import CausalLMModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-7, "fp16": 1e-3, "bf16": 5e-3}

if transformers.__version__ >= "4.53.0":
    from transformers import MiniMaxConfig

    from mindone.transformers import MiniMaxForCausalLM

    class MiniMaxModelTester(CausalLMModelTester):
        config_class = MiniMaxConfig

        def __init__(self, parent=None, layer_types=None, block_size=3, attn_implementation: str = "eager"):
            super().__init__(parent)
            self.layer_types = layer_types
            self.block_size = block_size
            self.forced_config_args += ["attn_implementation"]
            self.attn_implementation = attn_implementation

    _CASES = []
    for attn_impl in ["eager", "sdpa"]:  # "flash_attention" is not supported on CPU in PyTorch
        model_tester = MiniMaxModelTester(attn_implementation=attn_impl)
        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()
        _CASES.append(
            [
                "MiniMaxForCausalLM",
                "transformers.MiniMaxForCausalLM",
                "mindone.transformers.MiniMaxForCausalLM",
                (config,),
                {},
                (),
                inputs_dict,
                {"logits": "logits"},
            ]
        )

    @pytest.mark.parametrize(
        "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map", _CASES
    )
    @pytest.mark.parametrize("dtype", DTYPE_AND_THRESHOLDS.keys())
    @pytest.mark.skipif(transformers.__version__ < "4.53.0", reason="need to set specific transformers version")
    def test_named_modules(
        name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
    ):
        pt_model, ms_model, pt_dtype, ms_dtype = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
        pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
            pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
        )

        with inference_mode():
            pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
        ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)
        if outputs_map:
            pt_outputs_n = []
            ms_outputs_n = []
            for pt_key, ms_idx in outputs_map.items():
                pt_output = getattr(pt_outputs, pt_key)
                ms_output = getattr(ms_outputs, ms_idx)
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

    @pytest.mark.parametrize("attn_impl", [None, "flash_attention_2"], ids=["default (sdpa)", "flash_attention_2"])
    @pytest.mark.skipif(transformers.__version__ < "4.53.0", reason="need to set specific transformers version")
    @slow
    def test_minimax_forward(attn_impl: Union[str, None]):
        model = MiniMaxForCausalLM.from_pretrained(
            "hf-internal-testing/MiniMax-tiny", attn_implementation=attn_impl, mindspore_dtype=ms.float16
        )

        suffix = "_fa" if attn_impl == "flash_attention_2" else ""
        expected_logits = load(f"minimax_logits{suffix}.pt", map_location="cpu")
        expected_ids = load(f"minimax_ids{suffix}.pt", map_location="cpu")

        dummy_input = ms.tensor([[500, 44, 930, 1925, 355, 330, 1115]] * 2, dtype=ms.int64)
        dummy_mask = ms.tensor([[1, 1, 1, 1, 1, 1, 1]] * 2, dtype=ms.int64)

        logits = model(dummy_input, attention_mask=dummy_mask).logits
        diffs = compute_diffs(expected_logits, logits)
        THRESHOLD = DTYPE_AND_THRESHOLDS["fp16"]
        assert (np.array(diffs) < THRESHOLD).all(), f"Logits have diff bigger than {THRESHOLD}."

        output_ids = model.generate(dummy_input, attention_mask=dummy_mask, max_new_tokens=20, do_sample=False)
        assert (output_ids.numpy() == expected_ids.numpy()).all(), "Output ids are not the same."
