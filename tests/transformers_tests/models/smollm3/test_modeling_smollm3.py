# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the MindSpore SmolLM3 model."""
import inspect

import numpy as np
import pytest
import torch
import transformers

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

# default config of HuggingFaceTB/SmolLM3-3B is bf16
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]

if transformers.__version__ >= "4.54.1":
    from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

    class SmolLM3ModelTester:
        def __init__(
            self,
            batch_size=5,
            seq_length=20,
        ):
            self.batch_size = batch_size
            self.seq_length = seq_length

        def get_config(self):
            return SmolLM3Config()

        def prepare_config_and_inputs(self):
            config = self.get_config()
            vocab_size = config.vocab_size
            input_ids = ids_numpy([self.batch_size, self.seq_length], vocab_size)
            attention_mask = np.tril(np.ones_like(input_ids))

            return config, input_ids, attention_mask

    model_tester = SmolLM3ModelTester()
    config, input_ids, attention_mask = model_tester.prepare_config_and_inputs()

    SMOLLM3_CASES = [
        [
            "SmolLM3ForCausalLM",
            "transformers.SmolLM3ForCausalLM",
            "mindone.transformers.SmolLM3ForCausalLM",
            (config,),
            {},
            (),
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            {
                "logits": 0,
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
            for case in SMOLLM3_CASES
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

        ms_inputs_kwargs.update({"use_cache": False})

        with torch.no_grad():
            pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
        ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)
        if outputs_map:
            pt_outputs_n = []
            ms_outputs_n = []
            for pt_key, ms_idx in outputs_map.items():
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
