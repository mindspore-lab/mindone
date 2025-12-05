# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import XCLIPConfig, XCLIPProcessor, XCLIPTextConfig, XCLIPVisionConfig
from transformers.testing_utils import slow

from mindspore import tensor

from mindone.transformers import XCLIPModel
from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy, random_attention_mask

# `XCLIPModel` and `test_inference_interpolate_pos_encoding` have higher errors
DTYPE_AND_THRESHOLDS = {"fp32": 3e-3, "fp16": 2e-2, "bf16": 0.1}


class XCLIPVisionModelTester:
    def __init__(
        self,
        parent=None,
        batch_size=8,
        image_size=30,
        patch_size=2,
        num_channels=3,
        num_frames=8,  # important; the batch size * time must be divisible by the number of frames
        is_training=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        mit_hidden_size=64,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.mit_hidden_size = mit_hidden_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy(
            [self.batch_size * self.num_frames, self.num_channels, self.image_size, self.image_size]
        )
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return XCLIPVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_frames=self.num_frames,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            mit_hidden_size=self.mit_hidden_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class XCLIPTextModelTester:
    def __init__(
        self,
        parent=None,
        batch_size=8,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return XCLIPTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class XCLIPModelTester:
    def __init__(
        self,
        parent=None,
        text_kwargs=None,
        vision_kwargs=None,
        projection_dim=64,
        mit_hidden_size=64,
        is_training=True,
    ):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.projection_dim = projection_dim
        self.mit_hidden_size = mit_hidden_size
        self.text_model_tester = XCLIPTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = XCLIPVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, _ = self.vision_model_tester.prepare_config_and_inputs()
        pixel_values = floats_numpy(
            [
                self.vision_model_tester.batch_size,
                self.vision_model_tester.num_frames,
                self.vision_model_tester.num_channels,
                self.vision_model_tester.image_size,
                self.vision_model_tester.image_size,
            ]
        )

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return XCLIPConfig.from_text_vision_configs(
            self.text_model_tester.get_config(),
            self.vision_model_tester.get_config(),
            projection_dim=self.projection_dim,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "return_loss": True,
        }
        return config, inputs_dict


_CASES = [
    [
        "XCLIPVisionModel",
        "transformers.XCLIPVisionModel",
        "mindone.transformers.XCLIPVisionModel",
        XCLIPVisionModelTester().prepare_config_and_inputs_for_common(),
        {"last_hidden_state": "last_hidden_state"},
    ],
    [
        "XCLIPTextModel",
        "transformers.XCLIPTextModel",
        "mindone.transformers.XCLIPTextModel",
        XCLIPTextModelTester().prepare_config_and_inputs_for_common(),
        {"last_hidden_state": "last_hidden_state"},
    ],
    [
        "XCLIPModel",
        "transformers.XCLIPModel",
        "mindone.transformers.XCLIPModel",
        XCLIPModelTester().prepare_config_and_inputs_for_common(),
        {"logits_per_video": "logits_per_video", "logits_per_text": "logits_per_text"},
    ],
]

_CASES = [
    [module, pt_module, ms_module, (config,), {}, (), inputs_dict, outputs]
    for module, pt_module, ms_module, (config, inputs_dict), outputs in _CASES
]


@pytest.mark.parametrize("name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map", _CASES)
@pytest.mark.parametrize("dtype", DTYPE_AND_THRESHOLDS.keys())
def test_named_modules(
    name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
):
    pt_model, ms_model, pt_dtype, ms_dtype = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
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


# We will verify our results on a spaghetti video
def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti_8_frames.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)


@slow
def test_inference():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    model_name = "microsoft/xclip-base-patch32"
    model = XCLIPModel.from_pretrained(model_name)
    processor = XCLIPProcessor.from_pretrained(model_name)

    video = prepare_video()
    inputs = processor(
        text=["playing sports", "eating spaghetti", "go shopping"], videos=video, return_tensors="np", padding=True
    )
    inputs = {k: tensor(v) for k, v in inputs.items()}

    # forward pass
    outputs = model(**inputs)

    # verify the logits
    assert outputs.logits_per_video.shape == (inputs["pixel_values"].shape[0], inputs["input_ids"].shape[0])
    assert outputs.logits_per_text.shape == (inputs["input_ids"].shape[0], inputs["pixel_values"].shape[0])

    expected_logits = np.array([[14.0181, 20.2771, 14.4776]])
    diffs = compute_diffs(expected_logits, outputs.logits_per_video)
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"


@slow
def test_inference_interpolate_pos_encoding():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    # XCLIP models have an `interpolate_pos_encoding` argument in their forward method,
    # allowing to interpolate the pre-trained position embeddings in order to use
    # the model on higher resolutions. The DINO model by Facebook AI leverages this
    # to visualize self-attention on higher resolution images.
    model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")

    processor = XCLIPProcessor.from_pretrained(
        "microsoft/xclip-base-patch32", size=180, crop_size={"height": 180, "width": 180}
    )

    video = prepare_video()
    inputs = processor(text="what's in the video", videos=video, return_tensors="np")
    inputs = {k: tensor(v) for k, v in inputs.items()}

    # interpolate_pos_encodiung false should return value error
    with pytest.raises(ValueError):
        model(**inputs, interpolate_pos_encoding=False)
    # forward pass
    outputs = model(**inputs, interpolate_pos_encoding=True)

    # verify the logits
    assert outputs.vision_model_output.last_hidden_state.shape == (8, 26, 768)

    expected_slice = np.array([[0.0126, 0.2109, 0.0609], [0.0448, 0.5862, -0.1688], [-0.0881, 0.8525, -0.3044]])
    diffs = compute_diffs(expected_slice, outputs.vision_model_output.last_hidden_state[0, :3, :3])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"
