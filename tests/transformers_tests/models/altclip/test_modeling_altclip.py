"""Adapted from tests for Cohere2; adds AltCLIP parity tests."""

import inspect

import numpy as np
import pytest
import torch
from transformers import AltCLIPConfig, AltCLIPTextConfig, AltCLIPVisionConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
# To avoid current graph-mode limitations, test in pynative mode only
MODES = [1]


class AltCLIPModelTester:
    def __init__(
        self,
        batch_size=2,
        seq_length=7,
        image_size=32,
        is_training=False,
        use_input_mask=True,
        use_labels=False,
        # text config
        text_vocab_size=99,
        text_hidden_size=32,
        text_intermediate_size=37,
        text_num_hidden_layers=2,
        text_num_attention_heads=4,
        text_hidden_act="gelu",
        text_hidden_dropout_prob=0.0,
        text_attention_probs_dropout_prob=0.0,
        text_max_position_embeddings=64,
        # vision config
        vision_hidden_size=32,
        vision_intermediate_size=37,
        vision_projection_dim=16,
        vision_num_hidden_layers=2,
        vision_num_attention_heads=4,
        vision_num_channels=3,
        vision_patch_size=16,
        vision_hidden_act="quick_gelu",
        vision_attention_dropout=0.0,
        # top-level altclip config
        projection_dim=16,
        logit_scale_init_value=2.6592,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_size = image_size
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels

        # text
        self.text_vocab_size = text_vocab_size
        self.text_hidden_size = text_hidden_size
        self.text_intermediate_size = text_intermediate_size
        self.text_num_hidden_layers = text_num_hidden_layers
        self.text_num_attention_heads = text_num_attention_heads
        self.text_hidden_act = text_hidden_act
        self.text_hidden_dropout_prob = text_hidden_dropout_prob
        self.text_attention_probs_dropout_prob = text_attention_probs_dropout_prob
        self.text_max_position_embeddings = text_max_position_embeddings

        # vision
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_projection_dim = vision_projection_dim
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_num_channels = vision_num_channels
        self.vision_patch_size = vision_patch_size
        self.vision_hidden_act = vision_hidden_act
        self.vision_attention_dropout = vision_attention_dropout

        # top-level
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    def prepare_config_and_inputs(self):
        # text inputs
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.text_vocab_size)
        attention_mask = None
        if self.use_input_mask:
            attention_mask = np.ones_like(input_ids)

        # vision inputs
        pixel_values = floats_numpy(
            (self.batch_size, self.vision_num_channels, self.image_size, self.image_size), scale=1.0
        )

        # configs
        text_cfg = AltCLIPTextConfig(
            vocab_size=self.text_vocab_size,
            hidden_size=self.text_hidden_size,
            num_hidden_layers=self.text_num_hidden_layers,
            num_attention_heads=self.text_num_attention_heads,
            intermediate_size=self.text_intermediate_size,
            hidden_act=self.text_hidden_act,
            hidden_dropout_prob=self.text_hidden_dropout_prob,
            attention_probs_dropout_prob=self.text_attention_probs_dropout_prob,
            max_position_embeddings=self.text_max_position_embeddings,
            use_cache=True,
            project_dim=self.projection_dim,
        )
        vision_cfg = AltCLIPVisionConfig(
            hidden_size=self.vision_hidden_size,
            intermediate_size=self.vision_intermediate_size,
            projection_dim=self.vision_projection_dim,
            num_hidden_layers=self.vision_num_hidden_layers,
            num_attention_heads=self.vision_num_attention_heads,
            num_channels=self.vision_num_channels,
            image_size=self.image_size,
            patch_size=self.vision_patch_size,
            hidden_act=self.vision_hidden_act,
            attention_dropout=self.vision_attention_dropout,
        )
        config = AltCLIPConfig.from_text_vision_configs(
            text_cfg, vision_cfg, projection_dim=self.projection_dim, logit_scale_init_value=self.logit_scale_init_value
        )
        return config, text_cfg, vision_cfg, input_ids, attention_mask, pixel_values


model_tester = AltCLIPModelTester()
config, text_config, vision_config, input_ids, attention_mask, pixel_values = model_tester.prepare_config_and_inputs()


ALTCLIP_CASES = [
    # Text model
    [
        "AltCLIPTextModel",
        "transformers.AltCLIPTextModel",
        "mindone.transformers.AltCLIPTextModel",
        (text_config,),
        {},
        (input_ids,),
        {"attention_mask": attention_mask},
        {"last_hidden_state": 0, "pooler_output": 1},
    ],
    # Vision model
    [
        "AltCLIPVisionModel",
        "transformers.AltCLIPVisionModel",
        "mindone.transformers.AltCLIPVisionModel",
        (vision_config,),
        {},
        (pixel_values,),
        {},
        {"last_hidden_state": 0, "pooler_output": 1},
    ],
    # Full AltCLIP model (compare logits)
    [
        "AltCLIPModel",
        "transformers.AltCLIPModel",
        "mindone.transformers.AltCLIPModel",
        (config,),
        {},
        (),
        {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values},
        {"logits_per_image": 0, "logits_per_text": 1},
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [case + [dtype] + [mode] for case in ALTCLIP_CASES for dtype in DTYPE_AND_THRESHOLDS.keys() for mode in MODES],
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

    # Some modules always compute in float and may require specific hidden dtype
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
    assert (
        np.array(diffs) < THRESHOLD
    ).all(), f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
