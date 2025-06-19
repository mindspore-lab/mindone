# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import unittest

import numpy as np
import pytest
from parameterized import parameterized
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPProcessor

import mindspore as ms
from transformers.testing_utils import slow

from mindone.transformers import CLIPModel
from tests.modeling_test_utils import forward_compare, prepare_img
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [0, 1]


class CLIPTextModelTester:
    def __init__(
        self,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        scope=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
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
        return CLIPTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )


class CLIPVisionModelTester:
    def __init__(
        self,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
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
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return CLIPVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )


class CLIPModelTester:
    def __init__(self, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.text_model_tester = CLIPTextModelTester(**text_kwargs)
        self.vision_model_tester = CLIPVisionModelTester(**vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return CLIPConfig.from_text_vision_configs(
            self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64
        )


class CLIPModelTest(unittest.TestCase):
    # 初始化用例参数
    clip_text_tester = CLIPTextModelTester()
    clip_text_config, input_ids, input_mask = clip_text_tester.prepare_config_and_inputs()
    clip_vision_tester = CLIPVisionModelTester()
    clip_vision_config, pixel_values = clip_vision_tester.prepare_config_and_inputs()
    clip_tester = CLIPModelTester()
    clip_config = clip_tester.prepare_config_and_inputs()[0]

    CLIP_CASES = [
        [
            "CLIPTextModel",
            "transformers.CLIPTextModel",
            "mindone.transformers.CLIPTextModel",
            (clip_text_config,),
            {},
            (input_ids,),
            {
                "attention_mask": input_mask,
            },
            {
                "last_hidden_state": 0,
                "pooler_output": 1,
            },
        ],
        [
            "CLIPTextModelWithProjection",
            "transformers.CLIPTextModelWithProjection",
            "mindone.transformers.CLIPTextModelWithProjection",
            (clip_text_config,),
            {},
            (input_ids,),
            {
                "attention_mask": input_mask,
            },
            {
                "text_embeds": 0,
                "last_hidden_state": 1,
            },
        ],
        [
            "CLIPVisionModel",
            "transformers.CLIPVisionModel",
            "mindone.transformers.CLIPVisionModel",
            (clip_vision_config,),
            {},
            (pixel_values,),
            {},
            {
                "last_hidden_state": 0,
                "pooler_output": 1,
            },
        ],
        [
            "CLIPVisionModelWithProjection",
            "transformers.CLIPVisionModelWithProjection",
            "mindone.transformers.CLIPVisionModelWithProjection",
            (clip_vision_config,),
            {},
            (pixel_values,),
            {},
            {
                "image_embeds": 0,
                "last_hidden_state": 1,
            },
        ],
        [
            "CLIPModel",
            "transformers.CLIPModel",
            "mindone.transformers.CLIPModel",
            (clip_config,),
            {},
            (input_ids, pixel_values, input_mask),
            {},
            {
                "logits_per_image": 0,
                "logits_per_text": 1,
                "text_embeds": 2,
                "image_embeds": 3,
            },
        ],
    ]

    @parameterized.expand(
        [
            case
            + [
                dtype,
            ]
            + [
                mode,
            ]
            for case in CLIP_CASES
            for dtype in DTYPE_AND_THRESHOLDS
            for mode in MODES
        ],
    )
    def test_model_forward(
            self,
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

        diffs, pt_dtype, ms_dtype = forward_compare(
            pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
        )

        THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
        self.assertTrue(
            (np.array(diffs) < THRESHOLD).all(),
            f"For {name} forward test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}")


class CLIPModelIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_inference_logits(self, mode):
        ms.set_context(mode=mode)
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = prepare_img(image_url)
        text = ["a photo of a cat", "a photo of a dog"]
        inputs = processor(text=text, images=image, padding=True, return_tensors="np")

        input_ids = ms.Tensor(inputs.input_ids)
        pixel_values = ms.Tensor(inputs.pixel_values)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values)

        # check logits
        self.assertEqual(
            outputs[0].shape,
            (pixel_values.shape[0], input_ids.shape[0])
        )
        self.assertEqual(
            outputs[1].shape,
            (input_ids.shape[0], pixel_values.shape[0])
        )

        EXPECTED_LOGITS = ms.Tensor([[24.5701, 19.3049]])
        np.testing.assert_allclose(outputs[0], EXPECTED_LOGITS, rtol=1e-3, atol=1e-3)
