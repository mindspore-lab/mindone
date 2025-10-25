"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/vjepa2/test_modeling_vjepa2.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

# NOTE: use transformers >=4.53.0
# pip install -U git+https://github.com/huggingface/transformers

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
from tests.transformers_tests.models.modeling_common import floats_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 6e-3}
MODES = [1]
# NOTE: all compare with torch fp32 since LayerNorm does not supported in CPU with fp16/bf16

if transformers.__version__ >= "4.52.3":
    from transformers import VJEPA2Config

    class VJEPA2ModelTester:
        def __init__(
            self,
            num_channels=3,
            patch_size=16,
            crop_size=64,
            frames_per_clip=16,
            tubelet_size=2,
            hidden_size=256,
            in_chans=3,
            num_attention_heads=4,
            num_hidden_layers=2,
            drop_path_rate=0.0,
            mlp_ratio=4.0,
            layer_norm_eps=1e-6,
            qkv_bias=True,
            attention_probs_dropout_prob=0.0,
            hidden_act="gelu",
            initializer_range=0.02,
            pred_hidden_size=16,
            pred_mlp_ratio=4.0,
            pred_num_attention_heads=4,
            pred_num_hidden_layers=2,
            pred_num_mask_tokens=10,
            pred_zero_init_mask_tokens=True,
            hidden_dropout_prob=0.0,
            image_size=64,
            use_SiLU=False,
            wide_SiLU=True,
            attn_implementation="eager",
        ):
            self.num_channels = num_channels
            self.patch_size = patch_size
            self.num_attention_heads = num_attention_heads
            self.num_hidden_layers = num_hidden_layers
            self.in_chans = in_chans
            self.qkv_bias = qkv_bias
            self.tubelet_size = tubelet_size
            self.hidden_size = hidden_size
            self.frames_per_clip = frames_per_clip
            self.crop_size = crop_size
            self.drop_path_rate = drop_path_rate
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.image_size = image_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.mlp_ratio = mlp_ratio
            self.pred_hidden_size = pred_hidden_size
            self.pred_mlp_ratio = pred_mlp_ratio
            self.pred_num_attention_heads = pred_num_attention_heads
            self.pred_num_hidden_layers = pred_num_hidden_layers
            self.pred_num_mask_tokens = pred_num_mask_tokens
            self.pred_zero_init_mask_tokens = pred_zero_init_mask_tokens
            self.use_SiLU = use_SiLU
            self.wide_SiLU = wide_SiLU
            self.attn_implementation = attn_implementation

        def prepare_config_and_inputs(self):
            config = self.get_config()
            num_frames = 16
            pixel_values = floats_numpy([1, num_frames, self.num_channels, self.crop_size, self.crop_size])

            return config, pixel_values

        def get_config(self):
            return VJEPA2Config(
                patch_size=self.patch_size,
                num_attention_heads=self.num_attention_heads,
                num_hidden_layers=self.num_hidden_layers,
                in_chans=self.in_chans,
                qkv_bias=self.qkv_bias,
                tubelet_size=self.tubelet_size,
                hidden_size=self.hidden_size,
                frames_per_clip=self.frames_per_clip,
                crop_size=self.crop_size,
                drop_path_rate=self.drop_path_rate,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                image_size=self.image_size,
                initializer_range=self.initializer_range,
                layer_norm_eps=self.layer_norm_eps,
                mlp_ratio=self.mlp_ratio,
                pred_hidden_size=self.pred_hidden_size,
                pred_mlp_ratio=self.pred_mlp_ratio,
                pred_num_attention_heads=self.pred_num_attention_heads,
                pred_num_hidden_layers=self.pred_num_hidden_layers,
                pred_num_mask_tokens=self.pred_num_mask_tokens,
                pred_zero_init_mask_tokens=self.pred_zero_init_mask_tokens,
                use_SiLU=self.use_SiLU,
                wide_SiLU=self.wide_SiLU,
                attn_implementation=self.attn_implementation,
            )

    model_tester = VJEPA2ModelTester()
    (
        config,
        pixel_values,
    ) = model_tester.prepare_config_and_inputs()

    VJEPA2_CASES = [
        [
            "VJEPA2Model",
            "transformers.VJEPA2Model",
            "mindone.transformers.VJEPA2Model",
            (config,),
            {},
            (pixel_values,),
            {},
            {"last_hidden_state": 0},
        ],
        [
            "VJEPA2ForVideoClassification",
            "transformers.VJEPA2ForVideoClassification",
            "mindone.transformers.VJEPA2ForVideoClassification",
            (config,),
            {},
            (pixel_values,),
            {},
            {"logits": 0},
        ],
    ]

    # transformers need >= 4.53.0.dev3
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
            for case in VJEPA2_CASES
            for dtype in DTYPE_AND_THRESHOLDS.keys()
            for mode in MODES
        ],
    )
    @pytest.mark.skipif(transformers.__version__ < "4.52.3", reason="need to set specific transformers version")
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
