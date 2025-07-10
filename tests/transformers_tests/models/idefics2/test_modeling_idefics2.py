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
from transformers import Idefics2Config

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-6, "fp16": 5e-3, "bf16": 5e-2}
# Since some operators not supported in CPU for fp16, all evaluation is under **ms.precision vs torch.float32**
MODES = [1]


class Idefics2ModelTester:
    def __init__(
        self,
        batch_size=1,
        seq_length=7,
        # For common tests
        is_training=False,
        use_attention_mask=True,
        use_labels=False,
        use_cache=False,
        output_attentions=False,
        # For net config
        vocab_size=99,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        hidden_act="silu",
        max_position_embeddings=512,
        use_sliding_window=False,
        attn_implementation="eager",
        # vision config
        image_size=64,
        # perceiver config
        resampler_n_latents=1,  # must be 1
        resampler_depth=3,
        resampler_n_heads=8,
        resampler_head_dim=12,
        qk_layer_norms_perceiver=True,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        # For common tests
        self.seq_length = self.seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.use_sliding_window = use_sliding_window
        self.attn_implementation = attn_implementation
        self.image_size = image_size
        # perceiver config
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.resampler_head_dim = resampler_head_dim
        self.qk_layer_norms_perceiver = qk_layer_norms_perceiver

    def get_large_model_config(self):
        return Idefics2Config.from_pretrained("HuggingFaceM4/idefics2-8b")

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy([self.batch_size, self.seq_length], vocab_size=2)

        image_batch_size = 1
        num_images = 1
        num_channels = 3
        height = self.image_size
        width = self.image_size
        pixel_values = ids_numpy([image_batch_size, num_images, num_channels, height, width], vocab_size=256)
        pixel_values = (pixel_values.astype(np.float32) / 255.0) * 2 - 1  # in range [-1, 1]

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        config = Idefics2Config(
            use_cache=self.use_cache,
            vision_config=dict(  # Idefics2VisionConfig
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_channels=3,
                image_size=self.image_size,
                patch_size=32,
                attn_implementation=self.attn_implementation,
            ),
            perceiver_config=dict(  # Idefics2PerceiverConfig
                hidden_size=self.hidden_size,
                num_key_value_heads=self.num_key_value_heads,
                resampler_n_latents=self.resampler_n_latents,
                resampler_depth=self.resampler_depth,
                resampler_n_heads=self.resampler_n_heads,
                resampler_head_dim=self.resampler_head_dim,
                qk_layer_norms_perceiver=self.qk_layer_norms_perceiver,
                attn_implementation=self.attn_implementation,
            ),
            text_config=dict(  # MistralConfig
                model_type="mistral",
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.max_position_embeddings,
                use_cache=self.use_cache,
                num_expert_per_tok=2,
                num_local_experts=2,
                attn_implementation=self.attn_implementation,
            ),
            attn_implementation=self.attn_implementation,
        )

        return config


model_tester = Idefics2ModelTester()
(
    config,
    input_ids,
    attention_mask,
    pixel_values,
) = model_tester.prepare_config_and_inputs()


TEST_CASES = [
    [  # VQA
        "Idefics2Model",
        "transformers.Idefics2Model",
        "mindone.transformers.Idefics2Model",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        },
        {
            "last_hidden_state": 0,  # text_model, i.e., MistralModel
            "image_hidden_states": -1,  # vision_modal, i.e., Idefics2VisionTransformer
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
