# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import inspect
import json
import os
import pathlib
import tempfile
import warnings

import numpy as np
import requests
from transformers.testing_utils import check_json_file_has_correct_format, require_vision

from mindone.transformers import AutoImageProcessor, BatchFeature
from mindone.transformers.image_utils import AnnotationFormat, AnnotionFormat
from mindone.transformers.testing_utils import require_mindspore
from mindone.transformers.utils import is_mindspore_available, is_vision_available

if is_mindspore_available():
    import mindspore as ms
    from mindspore import ops

if is_vision_available():
    from PIL import Image


def prepare_image_inputs(
    batch_size,
    min_resolution,
    max_resolution,
    num_channels,
    size_divisor=None,
    equal_resolution=False,
    numpify=False,
    mindsporeify=False,
):
    """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
    or a list of PyTorch tensors if one specifies mindsporeify=True.

    One can specify whether the images are of the same resolution or not.
    """

    assert not (numpify and mindsporeify), "You cannot specify both numpy and PyTorch tensors at the same time"

    image_inputs = []
    for i in range(batch_size):
        if equal_resolution:
            width = height = max_resolution
        else:
            # To avoid getting image width/height 0
            if size_divisor is not None:
                # If `size_divisor` is defined, the image needs to have width/size >= `size_divisor`
                min_resolution = max(size_divisor, min_resolution)
            width, height = np.random.choice(np.arange(min_resolution, max_resolution), 2)
        image_inputs.append(np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8))

    if not numpify and not mindsporeify:
        # PIL expects the channel dimension as last dimension
        image_inputs = [Image.fromarray(np.moveaxis(image, 0, -1)) for image in image_inputs]

    if mindsporeify:
        image_inputs = [ms.tensor(image) for image in image_inputs]

    if numpify:
        # Numpy images are typically in channels last format
        image_inputs = [image.transpose(1, 2, 0) for image in image_inputs]

    return image_inputs


def prepare_video(num_frames, num_channels, width=10, height=10, numpify=False, mindsporeify=False):
    """This function prepares a video as a list of PIL images/NumPy arrays/PyTorch tensors."""

    video = []
    for i in range(num_frames):
        video.append(np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8))

    if not numpify and not mindsporeify:
        # PIL expects the channel dimension as last dimension
        video = [Image.fromarray(np.moveaxis(frame, 0, -1)) for frame in video]

    if mindsporeify:
        video = [ms.tensor(frame) for frame in video]

    return video


def prepare_video_inputs(
    batch_size,
    num_frames,
    num_channels,
    min_resolution,
    max_resolution,
    equal_resolution=False,
    numpify=False,
    mindsporeify=False,
):
    """This function prepares a batch of videos: a list of list of PIL images, or a list of list of numpy arrays if
    one specifies numpify=True, or a list of list of PyTorch tensors if one specifies mindsporeify=True.

    One can specify whether the videos are of the same resolution or not.
    """

    assert not (numpify and mindsporeify), "You cannot specify both numpy and PyTorch tensors at the same time"

    video_inputs = []
    for i in range(batch_size):
        if equal_resolution:
            width = height = max_resolution
        else:
            width, height = np.random.choice(np.arange(min_resolution, max_resolution), 2)
            video = prepare_video(
                num_frames=num_frames,
                num_channels=num_channels,
                width=width,
                height=height,
                numpify=numpify,
                mindsporeify=mindsporeify,
            )
        video_inputs.append(video)

    return video_inputs


class ImageProcessingTestMixin:
    test_cast_dtype = None
    image_processing_class = None
    fast_image_processing_class = None
    image_processors_list = None
    test_slow_image_processor = True
    test_fast_image_processor = True

    def setUp(self):
        image_processor_list = []

        if self.test_slow_image_processor and self.image_processing_class:
            image_processor_list.append(self.image_processing_class)

        if self.test_fast_image_processor and self.fast_image_processing_class:
            image_processor_list.append(self.fast_image_processing_class)

        self.image_processor_list = image_processor_list

    @require_vision
    @require_mindspore
    def test_slow_fast_equivalence(self):
        dummy_image = Image.open(
            requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
        )

        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="np")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="np")

        self.assertTrue(np.allclose(encoding_slow.pixel_values, encoding_fast.pixel_values, atol=1e-2))

    def test_image_processor_to_json_string(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            obj = json.loads(image_processor.to_json_string())
            for key, value in self.image_processor_dict.items():
                self.assertEqual(obj[key], value)

    def test_image_processor_to_json_file(self):
        for image_processing_class in self.image_processor_list:
            image_processor_first = image_processing_class(**self.image_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                json_file_path = os.path.join(tmpdirname, "image_processor.json")
                image_processor_first.to_json_file(json_file_path)
                image_processor_second = image_processing_class.from_json_file(json_file_path)

            self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_image_processor_from_and_save_pretrained(self):
        for image_processing_class in self.image_processor_list:
            image_processor_first = image_processing_class(**self.image_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                saved_file = image_processor_first.save_pretrained(tmpdirname)[0]
                check_json_file_has_correct_format(saved_file)
                image_processor_second = image_processing_class.from_pretrained(tmpdirname)

            self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_image_processor_save_load_with_autoimageprocessor(self):
        for image_processing_class in self.image_processor_list:
            image_processor_first = image_processing_class(**self.image_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                saved_file = image_processor_first.save_pretrained(tmpdirname)[0]
                check_json_file_has_correct_format(saved_file)

                image_processor_second = AutoImageProcessor.from_pretrained(tmpdirname)

            self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_save_load_fast_slow(self):
        "Test that we can load a fast image processor from a slow one and vice-versa."
        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest("Skipping slow/fast save/load test as one of the image processors is not defined")

        image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        image_processor_slow_0 = self.image_processing_class(**image_processor_dict)

        # Load fast image processor from slow one
        with tempfile.TemporaryDirectory() as tmpdirname:
            image_processor_slow_0.save_pretrained(tmpdirname)
            image_processor_fast_0 = self.fast_image_processing_class.from_pretrained(tmpdirname)

        image_processor_fast_1 = self.fast_image_processing_class(**image_processor_dict)

        # Load slow image processor from fast one
        with tempfile.TemporaryDirectory() as tmpdirname:
            image_processor_fast_1.save_pretrained(tmpdirname)
            image_processor_slow_1 = self.image_processing_class.from_pretrained(tmpdirname)

        self.assertEqual(image_processor_slow_0.to_dict(), image_processor_slow_1.to_dict())
        self.assertEqual(image_processor_fast_0.to_dict(), image_processor_fast_1.to_dict())

    def test_save_load_fast_slow_auto(self):
        "Test that we can load a fast image processor from a slow one and vice-versa using AutoImageProcessor."
        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest("Skipping slow/fast save/load test as one of the image processors is not defined")

        image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        image_processor_slow_0 = self.image_processing_class(**image_processor_dict)

        # Load fast image processor from slow one
        with tempfile.TemporaryDirectory() as tmpdirname:
            image_processor_slow_0.save_pretrained(tmpdirname)
            image_processor_fast_0 = AutoImageProcessor.from_pretrained(tmpdirname, use_fast=True)

        image_processor_fast_1 = self.fast_image_processing_class(**image_processor_dict)

        # Load slow image processor from fast one
        with tempfile.TemporaryDirectory() as tmpdirname:
            image_processor_fast_1.save_pretrained(tmpdirname)
            image_processor_slow_1 = AutoImageProcessor.from_pretrained(tmpdirname, use_fast=False)

        self.assertEqual(image_processor_slow_0.to_dict(), image_processor_slow_1.to_dict())
        self.assertEqual(image_processor_fast_0.to_dict(), image_processor_fast_1.to_dict())

    def test_init_without_params(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class()
            self.assertIsNotNone(image_processor)

    @require_mindspore
    @require_vision
    def test_cast_dtype_device(self):
        for image_processing_class in self.image_processor_list:
            if self.test_cast_dtype is not None:
                # Initialize image_processor
                image_processor = image_processing_class(**self.image_processor_dict)

                # create random PyTorch tensors
                image_inputs = self.image_processor_tester.prepare_image_inputs(
                    equal_resolution=False, mindsporeify=True
                )

                encoding = image_processor(image_inputs, return_tensors="ms")
                # for layoutLM compatiblity
                self.assertEqual(encoding.pixel_values.dtype, ms.float32)

                encoding = image_processor(image_inputs, return_tensors="ms").to(ms.float16)
                self.assertEqual(encoding.pixel_values.dtype, ms.float16)

                # Try with text + image feature
                encoding = image_processor(image_inputs, return_tensors="ms")
                encoding.update({"input_ids": ms.tensor([[1, 2, 3], [4, 5, 6]]).to(ms.int64)})
                encoding = encoding.to(ms.float16)

                self.assertEqual(encoding.pixel_values.dtype, ms.float16)
                self.assertEqual(encoding.input_ids.dtype, ms.int64)

    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="ms").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="ms").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="ms").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="ms").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, mindsporeify=True)

            for image in image_inputs:
                self.assertIsInstance(image, ms.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="ms").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            encoded_images = image_processing(image_inputs, return_tensors="ms").pixel_values
            self.assertEqual(
                tuple(encoded_images.shape),
                (self.image_processor_tester.batch_size, *expected_output_image_shape),
            )

    def test_call_numpy_4_channels(self):
        for image_processing_class in self.image_processor_list:
            # Test that can process images which have an arbitrary number of channels
            # Initialize image_processing
            image_processor = image_processing_class(**self.image_processor_dict)

            # create random numpy tensors
            self.image_processor_tester.num_channels = 4
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

            # Test not batched input
            encoded_images = image_processor(
                image_inputs[0],
                return_tensors="ms",
                input_data_format="channels_last",
                image_mean=0,
                image_std=1,
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processor(
                image_inputs,
                return_tensors="ms",
                input_data_format="channels_last",
                image_mean=0,
                image_std=1,
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_image_processor_preprocess_arguments(self):
        is_tested = False

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)

            # validation done by _valid_processor_keys attribute
            if hasattr(image_processor, "_valid_processor_keys") and hasattr(image_processor, "preprocess"):
                preprocess_parameter_names = inspect.getfullargspec(image_processor.preprocess).args
                preprocess_parameter_names.remove("self")
                preprocess_parameter_names.sort()
                valid_processor_keys = image_processor._valid_processor_keys
                valid_processor_keys.sort()
                self.assertEqual(preprocess_parameter_names, valid_processor_keys)
                is_tested = True

            # validation done by @filter_out_non_signature_kwargs decorator
            if hasattr(image_processor.preprocess, "_filter_out_non_signature_kwargs"):
                if hasattr(self.image_processor_tester, "prepare_image_inputs"):
                    inputs = self.image_processor_tester.prepare_image_inputs()
                elif hasattr(self.image_processor_tester, "prepare_video_inputs"):
                    inputs = self.image_processor_tester.prepare_video_inputs()
                else:
                    self.skipTest(reason="No valid input preparation method found")

                with warnings.catch_warnings(record=True) as raised_warnings:
                    warnings.simplefilter("always")
                    image_processor(inputs, extra_argument=True)

                messages = " ".join([str(w.message) for w in raised_warnings])
                self.assertGreaterEqual(len(raised_warnings), 1)
                self.assertIn("extra_argument", messages)
                is_tested = True

        if not is_tested:
            self.skipTest(reason="No validation found for `preprocess` method")


class AnnotationFormatTestMixin:
    # this mixin adds a test to assert that usages of the
    # to-be-deprecated `AnnotionFormat` continue to be
    # supported for the time being

    def test_processor_can_use_legacy_annotation_format(self):
        image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        fixtures_path = pathlib.Path(__file__).parent / "fixtures" / "tests_samples" / "COCO"

        with open(fixtures_path / "coco_annotations.txt", "r") as f:
            detection_target = json.loads(f.read())

        detection_annotations = {"image_id": 39769, "annotations": detection_target}

        detection_params = {
            "images": Image.open(fixtures_path / "000000039769.png"),
            "annotations": detection_annotations,
            "return_tensors": "ms",
        }

        with open(fixtures_path / "coco_panoptic_annotations.txt", "r") as f:
            panoptic_target = json.loads(f.read())

        panoptic_annotations = {"file_name": "000000039769.png", "image_id": 39769, "segments_info": panoptic_target}

        masks_path = pathlib.Path(fixtures_path / "coco_panoptic")

        panoptic_params = {
            "images": Image.open(fixtures_path / "000000039769.png"),
            "annotations": panoptic_annotations,
            "return_tensors": "pt",
            "masks_path": masks_path,
        }

        test_cases = [
            ("coco_detection", detection_params),
            ("coco_panoptic", panoptic_params),
            (AnnotionFormat.COCO_DETECTION, detection_params),
            (AnnotionFormat.COCO_PANOPTIC, panoptic_params),
            (AnnotationFormat.COCO_DETECTION, detection_params),
            (AnnotationFormat.COCO_PANOPTIC, panoptic_params),
        ]

        def _compare(a, b) -> None:
            if isinstance(a, (dict, BatchFeature)):
                self.assertEqual(a.keys(), b.keys())
                for k, v in a.items():
                    _compare(v, b[k])
            elif isinstance(a, list):
                self.assertEqual(len(a), len(b))
                for idx in range(len(a)):
                    _compare(a[idx], b[idx])
            elif isinstance(a, ms.Tensor):
                self.assertTrue(ops.allclose(a, b, atol=1e-3))
            elif isinstance(a, str):
                self.assertEqual(a, b)

        for annotation_format, params in test_cases:
            with self.subTest(annotation_format):
                image_processor_params = {**image_processor_dict, **{"format": annotation_format}}
                image_processor_first = self.image_processing_class(**image_processor_params)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    image_processor_first.save_pretrained(tmpdirname)
                    image_processor_second = self.image_processing_class.from_pretrained(tmpdirname)

                # check the 'format' key exists and that the dicts of the
                # first and second processors are equal
                self.assertIn("format", image_processor_first.to_dict().keys())
                self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

                # perform encoding using both processors and compare
                # the resulting BatchFeatures
                first_encoding = image_processor_first(**params)
                second_encoding = image_processor_second(**params)
                _compare(first_encoding, second_encoding)
