import logging
import unittest

import numpy as np
import pytest
from parameterized import parameterized
from transformers import LevitConfig

import mindspore as ms
from transformers.testing_utils import slow

from mindone.transformers import LevitForImageClassificationWithTeacher
from mindone.transformers.models.levit import LevitImageProcessor
from tests.modeling_test_utils import forward_compare, prepare_img

from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

# fp16 NaN
DTYPE_AND_THRESHOLDS = {"fp32": 1e-3, "bf16": 5e-2}
MODES = [1]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LevitModelTester:
    def __init__(
        self,
        batch_size=13,
        image_size=64,
        num_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        patch_size=16,
        hidden_sizes=[16, 32, 48],
        num_attention_heads=[1, 2, 3],
        depths=[2, 3, 4],
        key_dim=[8, 8, 8],
        drop_path_rate=0,
        mlp_ratio=[2, 2, 2],
        attention_ratio=[2, 2, 2],
        initializer_range=0.02,
        is_training=True,
        use_labels=True,
        num_labels=2,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_sizes = hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.depths = depths
        self.key_dim = key_dim
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.attention_ratio = attention_ratio
        self.mlp_ratio = mlp_ratio
        self.initializer_range = initializer_range
        self.down_ops = [
            ["Subsample", key_dim[0], hidden_sizes[0] // key_dim[0], 4, 2, 2],
            ["Subsample", key_dim[0], hidden_sizes[1] // key_dim[0], 4, 2, 2],
        ]
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.initializer_range = initializer_range

    def prepare_config_and_inputs(self):
        # Generate pixel values (B, C, H, W) as numpy float arrays
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            # Generate labels if needed for classification task
            labels = ids_numpy([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        # Create LevitConfig using parameters from __init__
        return LevitConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            patch_size=self.patch_size,
            hidden_sizes=self.hidden_sizes,
            num_attention_heads=self.num_attention_heads,
            depths=self.depths,
            key_dim=self.key_dim,
            drop_path_rate=self.drop_path_rate,
            mlp_ratio=self.mlp_ratio,
            attention_ratio=self.attention_ratio,
            initializer_range=self.initializer_range,
            down_ops=self.down_ops,
        )


class LevitModelTest(unittest.TestCase):
    # 初始化用例参数
    model_tester = LevitModelTester()
    config, pixel_values, labels = model_tester.prepare_config_and_inputs()

    LEVIT_CASES = [
        [
            "LevitModel",
            "transformers.LevitModel",
            "mindone.transformers.LevitModel",
            (config,),
            {},
            (pixel_values,),
            {},
            {
                "last_hidden_state": "last_hidden_state",
            },
        ],
        [
            "LevitForImageClassification",
            "transformers.LevitForImageClassification",
            "mindone.transformers.LevitForImageClassification",
            (config,),
            {},
            (pixel_values,),
            {},
            {
                "logits": "logits",
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
            for case in LEVIT_CASES
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


class LevitModelIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_inference_image_classification_head_logits(self, mode):
        ms.set_context(mode=mode)
        model_name = "facebook/levit-128S"
        model = LevitForImageClassificationWithTeacher.from_pretrained(model_name)
        image_processor = LevitImageProcessor.from_pretrained(model_name)

        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = prepare_img(image_url)
        inputs = image_processor(images=image, return_tensors="np")
        for k, v in inputs.items():
            inputs[k] = ms.Tensor(v)

        output_logits = model(**inputs).logits

        # check the logits
        EXPECTED_SHAPE = (1, 1000)
        self.assertEqual(output_logits.shape, EXPECTED_SHAPE)

        EXPECTED_SLICE = ms.Tensor([1.047667, -0.374364, -1.831444], ms.float32)
        np.testing.assert_allclose(output_logits[0, :3], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)
