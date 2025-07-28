"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/llava_next_video/test_modeling_llava_next_video.py."""
import numpy as np
import pytest
import torch
from transformers import LlavaNextVideoConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]  # not support graph mode yet


class LlavaNextVideoVisionText2TextModelTester:
    def __init__(
        self,
        ignore_index=-100,
        image_token_index=0,
        video_token_index=1,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        text_config={
            "model_type": "llama",
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
            "pad_token_id": 2,
        },
        is_training=False,  # inference only
        vision_config={
            "image_size": 8,
            "patch_size": 4,
            "num_channels": 3,
            "is_training": False,  # inference only
            "hidden_size": 32,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        attn_implementation="eager",
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
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
        self.image_size = 30

        self.image_grid_pinpoints = [[16, 16]]
        self.num_image_tokens = 24
        self.num_video_tokens = 8
        self.seq_length = seq_length + self.num_image_tokens + self.num_video_tokens
        self.attn_implementation = attn_implementation

    def get_config(self):
        return LlavaNextVideoConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            video_token_index=self.video_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
            image_grid_pinpoints=self.image_grid_pinpoints,
            video_seq_length=self.num_video_tokens,
            image_seq_length=self.num_image_tokens,
            attn_implementation=self.attn_implementation,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy(
            [
                self.batch_size,
                5,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        pixel_values_videos = floats_numpy(
            [
                self.batch_size,
                8,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values, pixel_values_videos

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_values_videos = self.prepare_config_and_inputs()
        input_ids = ids_numpy([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 2
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)

        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[input_ids == config.video_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_index
        input_ids[:, self.num_image_tokens : self.num_video_tokens + self.num_image_tokens] = config.video_token_index

        inputs_dict = {
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_sizes": np.array(
                [[self.vision_config["image_size"], self.vision_config["image_size"]]] * self.batch_size
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


model_tester = LlavaNextVideoVisionText2TextModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


_CASES = [
    [
        "LlavaNextVideoForConditionalGeneration",
        "transformers.LlavaNextVideoForConditionalGeneration",
        "mindone.transformers.LlavaNextVideoForConditionalGeneration",
        (config,),
        {},
        (),
        inputs_dict,
        {"logits": "logits"},
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
