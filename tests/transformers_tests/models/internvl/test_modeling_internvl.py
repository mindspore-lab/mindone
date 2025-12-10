"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/internvl/test_modeling_internvl.py."""
import numpy as np
import pytest
import torch
from transformers import InternVLConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]  # not support graph mode yet


class InternVLModelTester:
    def __init__(
        self,
        image_token_id=151667,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        downsample_ratio=0.5,
        text_config={
            "model_type": "qwen2",
            "seq_length": 7,
            "is_training": False,  # inference only
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": False,  # inference only
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 580,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        is_training=False,  # inference only
        vision_config={
            "image_size": [8, 8],
            "patch_size": [4, 4],
            "num_channels": 3,
            "is_training": False,  # inference only
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_dropout": 0.1,
            "projection_dropout": 0.1,
            "initializer_range": 0.02,
            "norm_type": "layer_norm",
            "layer_norm_eps": 1e-6,
            "use_qk_norm": False,
            "attention_bias": False,
            "use_mask_token": False,
            "use_absolute_position_embeddings": True,
            "layer_scale_init_value": 0.1,
            "use_mean_pooling": True,
        },
        attn_implementation="eager",
    ):
        self.image_token_id = image_token_id
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.downsample_ratio = downsample_ratio
        self.text_config = text_config
        self.vision_config = vision_config
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 8
        self.seq_length = seq_length
        self.encoder_seq_length = self.seq_length
        self.attn_implementation = attn_implementation

    def get_config(self):
        return InternVLConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
            downsample_ratio=self.downsample_ratio,
            attn_implementation=self.attn_implementation,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"][0],
                self.vision_config["image_size"][1],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_numpy([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 2
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        input_ids[input_ids == config.image_token_id] = self.pad_token_id

        # Add some image tokens
        num_image_tokens = 4
        input_ids[:, :num_image_tokens] = config.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


model_tester = InternVLModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


_CASES = [
    [
        "InternVLForConditionalGeneration",
        "transformers.InternVLForConditionalGeneration",
        "mindone.transformers.InternVLForConditionalGeneration",
        (config,),
        {},
        (),
        inputs_dict,
        {"logits": "logits"},
    ],
    [
        "InternVLModel",
        "transformers.InternVLModel",
        "mindone.transformers.InternVLModel",
        (config,),
        {},
        (),
        inputs_dict,
        {"last_hidden_state": "last_hidden_state"},
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [case + [dtype] + [mode] for case in _CASES for dtype in DTYPE_AND_THRESHOLDS.keys() for mode in MODES],
)
def test_named_modules(
    name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype, mode
):
    ms.set_context(mode=mode)

    (pt_model, ms_model, pt_dtype, ms_dtype) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
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

