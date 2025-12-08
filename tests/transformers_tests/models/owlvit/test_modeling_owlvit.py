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

import random

import numpy as np
import pytest
import requests
import torch
from PIL import Image
from transformers import OwlViTConfig, OwlViTTextConfig, OwlViTVisionConfig
from transformers.testing_utils import slow

import mindspore as ms

from mindone.transformers import OwlViTForObjectDetection, OwlViTModel, OwlViTProcessor
from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 1e-3, "fp16": 2e-3, "bf16": 2e-2}


def get_rng():
    return random.Random(9)


class OwlViTVisionModelTester:
    def __init__(
        self,
        parent=None,
        batch_size=12,
        image_size=32,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
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
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy(
            [self.batch_size, self.num_channels, self.image_size, self.image_size], rng=get_rng()
        )
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return OwlViTVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class OwlViTTextModelTester:
    def __init__(
        self,
        parent=None,
        batch_size=12,
        num_queries=4,
        seq_length=16,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=16,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_queries = num_queries
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
        input_ids = ids_numpy([self.batch_size * self.num_queries, self.seq_length], self.vocab_size, rng=get_rng())
        input_mask = None

        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size * self.num_queries, self.seq_length])

        if input_mask is not None:
            num_text, seq_length = input_mask.shape

            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(num_text,))
            for idx, start_index in enumerate(rnd_start_indices):
                input_mask[idx, :start_index] = 1
                input_mask[idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return OwlViTTextConfig(
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


class OwlViTModelTester:
    def __init__(self, parent=None, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = OwlViTTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = OwlViTVisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training
        self.text_config = self.text_model_tester.get_config().to_dict()
        self.vision_config = self.vision_model_tester.get_config().to_dict()
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return OwlViTConfig.from_text_vision_configs(self.text_config, self.vision_config, projection_dim=64)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_loss": False,
        }
        return config, inputs_dict


class OwlViTForObjectDetectionTester:
    def __init__(self, parent=None, is_training=True):
        self.parent = parent
        self.text_model_tester = OwlViTTextModelTester(parent)
        self.vision_model_tester = OwlViTVisionModelTester(parent)
        self.is_training = is_training
        self.text_config = self.text_model_tester.get_config().to_dict()
        self.vision_config = self.vision_model_tester.get_config().to_dict()
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, pixel_values, input_ids, attention_mask

    def get_config(self):
        return OwlViTConfig.from_text_vision_configs(self.text_config, self.vision_config, projection_dim=64)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, input_ids, attention_mask = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


_CASES = [
    [
        "OwlViTVisionModel",
        "transformers.OwlViTVisionModel",
        "mindone.transformers.OwlViTVisionModel",
        OwlViTVisionModelTester().prepare_config_and_inputs_for_common(),
        {"last_hidden_state": "last_hidden_state"},
    ],
    [
        "OwlViTTextModel",
        "transformers.OwlViTTextModel",
        "mindone.transformers.OwlViTTextModel",
        OwlViTTextModelTester().prepare_config_and_inputs_for_common(),
        {"last_hidden_state": "last_hidden_state"},
    ],
    [
        "OwlViTModel",
        "transformers.OwlViTModel",
        "mindone.transformers.OwlViTModel",
        OwlViTModelTester().prepare_config_and_inputs_for_common(),
        {"logits_per_image": "logits_per_image", "logits_per_text": "logits_per_text"},
    ],
    [
        "OwlViTForObjectDetection",
        "transformers.OwlViTForObjectDetection",
        "mindone.transformers.OwlViTForObjectDetection",
        OwlViTForObjectDetectionTester().prepare_config_and_inputs_for_common(),
        {"logits": "logits"},
    ],
]

_CASES = [
    [module, pt_module, ms_module, (config,), {}, (), inputs_dict, outputs]
    for module, pt_module, ms_module, (config, inputs_dict), outputs in _CASES
]


@pytest.mark.parametrize("dtype", DTYPE_AND_THRESHOLDS.keys())
@pytest.mark.parametrize("name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map", _CASES)
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
    return Image.open(requests.get(url, stream=True).raw)


@slow
def test_inference():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    model_name = "google/owlvit-base-patch32"
    model = OwlViTModel.from_pretrained(model_name)
    processor = OwlViTProcessor.from_pretrained(model_name)

    image = prepare_img()
    inputs = processor(
        text=[["a photo of a cat", "a photo of a dog"]],
        images=image,
        max_length=16,
        padding="max_length",
        return_tensors="np",
    )

    # forward pass
    outputs = model(**{k: ms.tensor(v) for k, v in inputs.items()})

    # verify the logits
    assert outputs.logits_per_image.shape == (inputs.pixel_values.shape[0], inputs.input_ids.shape[0])
    assert outputs.logits_per_text.shape == (inputs.input_ids.shape[0], inputs.pixel_values.shape[0])

    expected_logits = np.array([[3.4613, 0.9403]])
    diffs = compute_diffs(expected_logits, outputs.logits_per_image)
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"


@slow
def test_inference_interpolate_pos_encoding():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    model_name = "google/owlvit-base-patch32"
    model = OwlViTModel.from_pretrained(model_name)
    processor = OwlViTProcessor.from_pretrained(model_name)
    processor.image_processor.size = {"height": 800, "width": 800}

    image = prepare_img()
    inputs = processor(
        text=[["a photo of a cat", "a photo of a dog"]],
        images=image,
        max_length=16,
        padding="max_length",
        return_tensors="np",
    )

    # forward pass
    outputs = model(**{k: ms.tensor(v) for k, v in inputs.items()}, interpolate_pos_encoding=True)

    # verify the logits
    assert outputs.logits_per_image.shape == (inputs.pixel_values.shape[0], inputs.input_ids.shape[0])
    assert outputs.logits_per_text.shape == (inputs.input_ids.shape[0], inputs.pixel_values.shape[0])

    expected_logits = np.array([[3.6278, 0.8861]])
    diffs = compute_diffs(expected_logits, outputs.logits_per_image)
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"

    expected_shape = (1, 626, 768)
    assert outputs.vision_model_output.last_hidden_state.shape == expected_shape

    # OwlViTForObjectDetection part.
    model = OwlViTForObjectDetection.from_pretrained(model_name)

    outputs = model(**{k: ms.tensor(v) for k, v in inputs.items()}, interpolate_pos_encoding=True)

    num_queries = int((inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size) ** 2)
    assert outputs.pred_boxes.shape == (1, num_queries, 4)

    expected_slice_boxes = np.array([[0.0680, 0.0422, 0.1347], [0.2071, 0.0450, 0.4146], [0.2000, 0.0418, 0.3476]])
    diffs = compute_diffs(expected_slice_boxes, outputs.pred_boxes[0, :3, :3])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"

    model = OwlViTForObjectDetection.from_pretrained(model_name)
    query_image = prepare_img()
    inputs = processor(images=image, query_images=query_image, max_length=16, padding="max_length", return_tensors="np")

    outputs = model.image_guided_detection(
        **{k: ms.tensor(v) for k, v in inputs.items()}, interpolate_pos_encoding=True
    )

    # No need to check the logits, we just check inference runs fine.
    num_queries = int((inputs.pixel_values.shape[-1] / model.config.vision_config.patch_size) ** 2)
    assert outputs.target_pred_boxes.shape == (1, num_queries, 4)

    # Deactivate interpolate_pos_encoding on same model, and use default image size.
    # Verify the dynamic change caused by the activation/deactivation of interpolate_pos_encoding of variables:
    # (self.sqrt_num_patch_h, self.sqrt_num_patch_w), self.box_bias from (OwlViTForObjectDetection).
    processor = OwlViTProcessor.from_pretrained(model_name)

    image = prepare_img()
    inputs = processor(
        text=[["a photo of a cat", "a photo of a dog"]],
        images=image,
        max_length=16,
        padding="max_length",
        return_tensors="np",
    )

    outputs = model(**{k: ms.tensor(v) for k, v in inputs.items()}, interpolate_pos_encoding=False)

    num_queries = int((inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size) ** 2)
    assert outputs.pred_boxes.shape == (1, num_queries, 4)

    expected_default_box_bias = np.array(
        [
            [-3.1332, -3.1332, -3.1332, -3.1332],
            [-2.3968, -3.1332, -3.1332, -3.1332],
            [-1.9452, -3.1332, -3.1332, -3.1332],
        ]
    )
    diffs = compute_diffs(expected_default_box_bias, model.box_bias[:3, :4])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"

    # Interpolate with any resolution size.
    processor.image_processor.size = {"height": 1264, "width": 1024}

    image = prepare_img()
    inputs = processor(
        text=[["a photo of a cat", "a photo of a dog"]],
        images=image,
        max_length=16,
        padding="max_length",
        return_tensors="np",
    )

    outputs = model(**{k: ms.tensor(v) for k, v in inputs.items()}, interpolate_pos_encoding=True)

    num_queries = int(
        (inputs.pixel_values.shape[-2] // model.config.vision_config.patch_size)
        * (inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size)
    )
    assert outputs.pred_boxes.shape == (1, num_queries, 4)
    expected_slice_boxes = np.array([[0.0499, 0.0301, 0.0983], [0.2244, 0.0365, 0.4663], [0.1387, 0.0314, 0.1859]])
    diffs = compute_diffs(expected_slice_boxes, outputs.pred_boxes[0, :3, :3])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"

    query_image = prepare_img()
    inputs = processor(images=image, query_images=query_image, max_length=16, padding="max_length", return_tensors="np")

    outputs = model.image_guided_detection(
        **{k: ms.tensor(v) for k, v in inputs.items()}, interpolate_pos_encoding=True
    )

    # No need to check the logits, we just check inference runs fine.
    num_queries = int(
        (inputs.pixel_values.shape[-2] // model.config.vision_config.patch_size)
        * (inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size)
    )
    assert outputs.target_pred_boxes.shape == (1, num_queries, 4)


@slow
def test_inference_object_detection():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    model_name = "google/owlvit-base-patch32"
    model = OwlViTForObjectDetection.from_pretrained(model_name)

    processor = OwlViTProcessor.from_pretrained(model_name)

    image = prepare_img()
    text_labels = [["a photo of a cat", "a photo of a dog"]]
    inputs = processor(text=text_labels, images=image, max_length=16, padding="max_length", return_tensors="np")
    inputs = {k: ms.tensor(v) for k, v in inputs.items()}

    outputs = model(**inputs)

    num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
    assert outputs.pred_boxes.shape == (1, num_queries, 4)

    expected_slice_boxes = np.array([[0.0691, 0.0445, 0.1373], [0.1592, 0.0456, 0.3192], [0.1632, 0.0423, 0.2478]])
    diffs = compute_diffs(expected_slice_boxes, outputs.pred_boxes[0, :3, :3])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"

    # test post-processing
    post_processed_output = processor.post_process_grounded_object_detection(outputs)
    assert post_processed_output[0]["text_labels"] is None

    post_processed_output_with_text_labels = processor.post_process_grounded_object_detection(
        outputs, text_labels=text_labels
    )

    objects_labels = post_processed_output_with_text_labels[0]["labels"].tolist()
    assert objects_labels == [0, 0]

    objects_text_labels = post_processed_output_with_text_labels[0]["text_labels"]
    assert objects_text_labels is not None
    assert objects_text_labels == ["a photo of a cat", "a photo of a cat"]


@slow
def test_inference_one_shot_object_detection():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    model_name = "google/owlvit-base-patch32"
    model = OwlViTForObjectDetection.from_pretrained(model_name)

    processor = OwlViTProcessor.from_pretrained(model_name)

    image = prepare_img()
    query_image = prepare_img()
    inputs = processor(images=image, query_images=query_image, max_length=16, padding="max_length", return_tensors="np")
    inputs = {k: ms.tensor(v) for k, v in inputs.items()}

    outputs = model.image_guided_detection(**inputs)

    num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
    assert outputs.target_pred_boxes.shape == (1, num_queries, 4)

    expected_slice_boxes = np.array([[0.0691, 0.0445, 0.1373], [0.1592, 0.0456, 0.3192], [0.1632, 0.0423, 0.2478]])
    diffs = compute_diffs(expected_slice_boxes, outputs.target_pred_boxes[0, :3, :3])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"


@slow
def test_inference_one_shot_object_detection_fp16():
    model_name = "google/owlvit-base-patch32"
    model = OwlViTForObjectDetection.from_pretrained(model_name, mindspore_dtype=ms.float16)

    processor = OwlViTProcessor.from_pretrained(model_name)

    image = prepare_img()
    query_image = prepare_img()
    inputs = processor(images=image, query_images=query_image, max_length=16, padding="max_length", return_tensors="np")
    inputs = {k: ms.tensor(v) for k, v in inputs.items()}

    outputs = model.image_guided_detection(**inputs)

    # No need to check the logits, we just check inference runs fine.
    num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
    assert outputs.target_pred_boxes.shape == (1, num_queries, 4)
