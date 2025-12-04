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

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]

if transformers.__version__ >= "4.53.0":
    from transformers import ArceeConfig

    class ArceeModelTester:
        config_class = ArceeConfig

        def __init__(
            self,
            batch_size=1,
            seq_length=7,
            use_input_mask=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            vocab_size=99,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rope_scaling={
                "type": "yarn",
                "rope_type": "yarn",
                "factor": 2.0,
                "beta_fast": 16.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "original_max_position_embeddings": 128,
            },
            hidden_act="relu2",
            rms_norm_eps=1e-5,
            mlp_bias=False,
            attention_bias=False,
            initializer_range=0.02,
            use_cache=False,
        ):
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.use_input_mask = use_input_mask
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            self.pad_token_id = pad_token_id
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.head_dim = head_dim
            self.max_position_embeddings = max_position_embeddings
            self.rope_theta = rope_theta
            self.rope_scaling = rope_scaling
            self.hidden_act = hidden_act
            self.rms_norm_eps = rms_norm_eps
            self.mlp_bias = mlp_bias
            self.attention_bias = attention_bias
            self.initializer_range = initializer_range
            self.use_cache = use_cache

        def prepare_config_and_inputs(self):
            input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = np.tril(np.ones_like(input_ids))

            config = self.get_config()

            # set _attn_implementation to eager because flash-attention is not supported for torch in cpu
            config._attn_implementation = "eager"

            return config, input_ids, input_mask

        def get_config(self):
            return self.config_class(
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                rope_theta=self.rope_theta,
                rope_scaling=self.rope_scaling,
                hidden_act=self.hidden_act,
                rms_norm_eps=self.rms_norm_eps,
                mlp_bias=self.mlp_bias,
                attention_bias=self.attention_bias,
                initializer_range=self.initializer_range,
                use_cache=self.use_cache,
            )

    model_tester = ArceeModelTester()
    config, input_ids, input_mask = model_tester.prepare_config_and_inputs()

    LLAMA_CASES = [
        [
            "ArceeForCausalLM",
            "transformers.ArceeForCausalLM",
            "mindone.transformers.ArceeForCausalLM",
            (config,),
            {},
            (),
            {
                "input_ids": input_ids,
                "attention_mask": input_mask,
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
            for case in LLAMA_CASES
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

        if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
            pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
            ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

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
