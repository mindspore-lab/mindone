"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/tvp/test_modeling_tvp.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import numpy as np
import pytest
import torch
from transformers import ResNetConfig, TvpConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


# Copied from test.models.videomae.test_modeling_videomae.VideoMAEModelTester with VideoMAE->TVP
class TVPModelTester:
    def __init__(
        self,
        batch_size=1,
        seq_length=2,
        alpha=1.0,
        beta=0.1,
        visual_prompter_type="framepad",
        visual_prompter_apply="replace",
        num_frames=2,
        max_img_size=448,
        visual_prompt_size=96,
        vocab_size=100,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=30,
        max_grid_col_position_embeddings=30,
        max_grid_row_position_embeddings=30,
        hidden_dropout_prob=0.1,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        pad_token_id=0,
        type_vocab_size=2,
        attention_probs_dropout_prob=0.1,
    ):
        self.batch_size = batch_size
        self.input_id_length = seq_length
        self.seq_length = seq_length + 10 + 784  # include text prompt length and visual input length
        self.alpha = alpha
        self.beta = beta
        self.visual_prompter_type = visual_prompter_type
        self.visual_prompter_apply = visual_prompter_apply
        self.num_frames = num_frames
        self.max_img_size = max_img_size
        self.visual_prompt_size = visual_prompt_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_grid_col_position_embeddings = max_grid_col_position_embeddings
        self.max_grid_row_position_embeddings = max_grid_row_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.type_vocab_size = type_vocab_size
        self.is_training = False
        self.num_channels = 3

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.input_id_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.input_id_length])
        pixel_values = floats_numpy(
            [self.batch_size, self.num_frames, self.num_channels, self.max_img_size, self.max_img_size]
        )

        config = self.get_config()

        return (config, input_ids, pixel_values, attention_mask)

    def get_config(self):
        resnet_config = ResNetConfig(
            num_channels=3,
            embeddings_size=64,
            hidden_sizes=[64, 128],
            depths=[2, 2],
            hidden_act="relu",
            out_features=["stage2"],
            out_indices=[2],
        )
        return TvpConfig(
            backbone_config=resnet_config,
            backbone=None,
            alpha=self.alpha,
            beta=self.beta,
            visual_prompter_type=self.visual_prompter_type,
            visual_prompter_apply=self.visual_prompter_apply,
            num_frames=self.num_frames,
            max_img_size=self.max_img_size,
            visual_prompt_size=self.visual_prompt_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_grid_col_position_embeddings=self.max_grid_col_position_embeddings,
            max_grid_row_position_embeddings=self.max_grid_row_position_embeddings,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            type_vocab_size=self.type_vocab_size,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, pixel_values, attention_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "pixel_values": pixel_values, "attention_mask": attention_mask}
        return config, inputs_dict


model_tester = TVPModelTester()
(
    config,
    inputs_dict,
) = model_tester.prepare_config_and_inputs_for_common()


TEST_CASES = [
    [
        "TvpModel",
        "transformers.TvpModel",
        "mindone.transformers.TvpModel",
        (config,),
        {},
        (),
        inputs_dict,
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "TvpForVideoGrounding",
        "transformers.TvpForVideoGrounding",
        "mindone.transformers.TvpForVideoGrounding",
        (config,),
        {},
        (),
        inputs_dict,
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
