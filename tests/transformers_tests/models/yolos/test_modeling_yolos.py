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
import requests
import torch
from PIL import Image
from transformers import YolosConfig
from transformers.testing_utils import slow

import mindspore as ms
from mindspore import mint

from mindone.transformers import AutoImageProcessor, YolosForObjectDetection
from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 2e-4, "fp16": 2e-3, "bf16": 1e-2}


class YolosModelTester:
    def __init__(
        self,
        parent=None,
        batch_size=13,
        image_size=[30, 30],
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
        n_targets=8,
        num_detection_tokens=10,
        attn_implementation="eager",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        self.n_targets = n_targets
        self.num_detection_tokens = num_detection_tokens
        self.attn_implementation = attn_implementation
        # we set the expected sequence length (which is used in several tests)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token) + num_detection_tokens
        num_patches = (image_size[1] // patch_size) * (image_size[0] // patch_size)
        self.expected_seq_len = num_patches + 1 + self.num_detection_tokens

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])

        labels = None
        if self.use_labels:
            # labels is a list of Dict (each Dict being the labels for a given example in the batch)
            labels = []
            for i in range(self.batch_size):
                target = {}
                target["class_labels"] = mint.randint(0, self.num_labels, (self.n_targets,))
                target["boxes"] = mint.rand(self.n_targets, 4)
                labels.append(target)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return YolosConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            num_detection_tokens=self.num_detection_tokens,
            num_labels=self.num_labels,
            attn_implementation=self.attn_implementation,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


model_tester = YolosModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()

_CASES = [
    [
        "YolosForObjectDetection",
        "transformers.YolosForObjectDetection",
        "mindone.transformers.YolosForObjectDetection",
        (config,),
        {},
        (),
        inputs_dict,
        {"logits": "logits", "last_hidden_state": "last_hidden_state"},
    ],
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


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@slow
def test_inference_object_detection_head():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")

    image = prepare_img()
    inputs = image_processor(images=image, return_tensors="np")

    # forward pass
    outputs = model(ms.tensor(inputs.pixel_values))

    # verify outputs
    expected_shape = (1, 100, 92)
    np.testing.assert_equal(outputs.logits.shape, expected_shape)

    expected_slice_logits = np.array(
        [[-23.7219, -10.3165, -14.9083], [-41.5429, -15.2403, -24.1478], [-29.3909, -12.7173, -19.4650]]
    )
    expected_slice_boxes = np.array([[0.2536, 0.5449, 0.4643], [0.2037, 0.7735, 0.3672], [0.7692, 0.4056, 0.4549]])
    diffs = compute_diffs(expected_slice_logits, outputs.logits[0, :3, :3])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"
    diffs = compute_diffs(expected_slice_boxes, outputs.pred_boxes[0, :3, :3])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"

    # verify postprocessing
    results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=[image.size[::-1]])[0]
    expected_scores = np.array([0.9991, 0.9801, 0.9978, 0.9875, 0.9848])
    expected_labels = [75, 75, 17, 63, 17]
    expected_slice_boxes = np.array([331.8438, 80.5440, 369.9546, 188.0579])

    assert len(results["scores"]) == 5
    assert results["labels"].tolist() == expected_labels

    diffs = compute_diffs(expected_scores, results["scores"])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"
    diffs = compute_diffs(expected_slice_boxes, results["boxes"][0, :])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"
