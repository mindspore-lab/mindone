#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.

import unittest

import numpy as np
import pytest
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import VisualClozePipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

from ..pipeline_test_utils import THRESHOLD_PIXEL

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@slow
@ddt
class VisualClozePipelineIntegrationTests(unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32" or dtype == "float16":
            pytest.skip(
                "Skipping this case since this pipeline has precision issue in float16 and will oom in float32."
            )

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        image_paths = [
            # in-context examples
            [
                load_downloaded_image_from_hf_hub(
                    "huggingface/documentation-images",
                    "visualcloze_mask2image_incontext-example-1_mask.jpg",
                    subfolder="diffusers/visualcloze",
                ),
                load_downloaded_image_from_hf_hub(
                    "huggingface/documentation-images",
                    "visualcloze_mask2image_incontext-example-1_image.jpg",
                    subfolder="diffusers/visualcloze",
                ),
            ],
            # query with the target image
            [
                load_downloaded_image_from_hf_hub(
                    "huggingface/documentation-images",
                    "visualcloze_mask2image_query_mask.jpg",
                    subfolder="diffusers/visualcloze",
                ),
                None,  # No image needed for the target image
            ],
        ]
        task_prompt = "In each row, a logical task is demonstrated to achieve [IMAGE2] an aesthetically pleasing photograph based on [IMAGE1] sam 2-generated masks with rich color coding."  # noqa: E501
        content_prompt = "Majestic photo of a golden eagle perched on a rocky outcrop in a mountainous landscape. The eagle is positioned in the right foreground, facing left, with its sharp beak and keen eyes prominently visible. Its plumage is a mix of dark brown and golden hues, with intricate feather details. The background features a soft-focus view of snow-capped mountains under a cloudy sky, creating a serene and grandiose atmosphere. The foreground includes rugged rocks and patches of green moss. Photorealistic, medium depth of field, soft natural lighting, cool color palette, high contrast, sharp focus on the eagle, blurred background, tranquil, majestic, wildlife photography."  # noqa: E501
        pipe = VisualClozePipeline.from_pretrained(
            "VisualCloze/VisualClozePipeline-384", resolution=384, mindspore_dtype=ms_dtype
        )

        image = pipe(
            task_prompt=task_prompt,
            content_prompt=content_prompt,
            image=image_paths,
            upsampling_width=1344,
            upsampling_height=768,
            upsampling_strength=0.4,
            guidance_scale=30,
            num_inference_steps=30,
            max_sequence_length=512,
            generator=np.random.Generator(np.random.PCG64(0)),
        )[0][0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"visualcloze_combined_{dtype}.npy",
            subfolder="visualcloze",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
