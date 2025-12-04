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

if transformers.__version__ >= "4.51.0":
    from transformers import Llama4Config

    class Llama4ModelTester:
        config_class = Llama4Config

        def __init__(
            self,
            batch_size=13,
            seq_length=7,
            is_training=True,
            use_input_mask=True,
            boi_token_index=1001,
            eoi_token_index=1002,
            image_token_index=1003,
            text_config={
                "attention_bias": False,
                "attention_chunk_size": 128,
                "attention_dropout": 0.0,
                "bos_token_id": 1,
                "eos_token_id": [2],
                "for_llm_compressor": False,
                "hidden_size": 32,
                "intermediate_size": 64,
                "intermediate_size_mlp": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "hidden_act": "silu",
                "initializer_range": 0.02,
                "max_position_embeddings": 256,
                "model_type": "llama4_text",
                "no_rope_layers": [],
                "num_experts_per_tok": 1,
                "num_local_experts": 2,
                "output_router_logits": False,
                "pad_token_id": 0,
                "rms_norm_eps": 1e-05,
                "rope_scaling": {
                    "factor": 1.0,
                    "high_freq_factor": 1.0,
                    "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 256,
                    "rope_type": "llama3",
                },
                "rope_theta": 10000.0,
                "router_aux_loss_coef": 0.0,
                "router_jitter_noise": 0.0,
                "use_cache": False,
                "use_qk_norm": False,
                "vocab_size": 99,
            },
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "hidden_act": "gelu",
                "image_size": 64,
                "num_channels": 3,
                "patch_size": 8,
                "pixel_shuffle_ratio": 1.0,
                "projector_input_dim": 32,
                "projector_output_dim": 32,
                "multi_modal_projector_bias": False,
                "projector_dropout": 0.0,
                "vision_output_dim": 32,
                "vision_feature_layer": -1,
                "vision_feature_select_strategy": "default",
                "initializer_range": 0.02,
                "norm_eps": 1e-05,
                "rope_theta": 10000,
            },
        ):
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_mask = use_input_mask
            self.boi_token_index = boi_token_index
            self.eoi_token_index = eoi_token_index
            self.image_token_index = image_token_index
            self.text_config = text_config
            self.vision_config = vision_config

        def prepare_config_and_inputs(self):
            input_ids = ids_numpy([self.batch_size, self.seq_length], self.text_config["vocab_size"])

            input_mask = None
            if self.use_input_mask:
                input_mask = np.tril(np.ones_like(input_ids))

            config = self.get_config()

            # set _attn_implementation to eager because flash-attention is not supported for torch in cpu
            config._attn_implementation = "eager"

            return config, input_ids, input_mask

        def get_config(self):
            return self.config_class(
                boi_token_index=self.boi_token_index,
                eoi_token_index=self.eoi_token_index,
                image_token_index=self.image_token_index,
                text_config=self.text_config,
                vision_config=self.vision_config,
            )

    model_tester = Llama4ModelTester()
    config, input_ids, input_mask = model_tester.prepare_config_and_inputs()

    LLAMA_CASES = [
        [
            "Llama4ForConditionalGeneration",
            "transformers.Llama4ForConditionalGeneration",
            "mindone.transformers.Llama4ForConditionalGeneration",
            (config,),
            {},
            (),
            {
                "input_ids": input_ids,
                "attention_mask": input_mask,
            },
            {
                "logits": 0,  # key: torch attribute, value: mindspore idx
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
