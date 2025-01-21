import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class ConsistencyModelPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "scheduler",
            "diffusers.schedulers.scheduling_consistency_models.CMStochasticIterativeScheduler",
            "mindone.diffusers.schedulers.scheduling_consistency_models.CMStochasticIterativeScheduler",
            dict(
                num_train_timesteps=40,
                sigma_min=0.002,
                sigma_max=80.0,
            ),
        ],
    ]

    def get_unet_config(self, class_cond=False):
        return [
            "unet",
            "diffusers.models.unets.unet_2d.UNet2DModel",
            "mindone.diffusers.models.unets.unet_2d.UNet2DModel",
            dict(
                pretrained_model_name_or_path="diffusers/consistency-models-test",
                subfolder="test_unet_class_cond" if class_cond else "test_unet",
            ),
        ]

    def get_dummy_components(self, class_cond=False):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
            ]
        }
        unet = self.get_unet_config(class_cond)
        pipeline_config = [unet] + self.pipeline_config

        return get_pipeline_components(components, pipeline_config)

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "batch_size": 1,
            "num_inference_steps": None,
            "timesteps": [22, 0],
            "output_type": "np",
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_consistency_model_pipeline_multistep(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.consistency_models.ConsistencyModelPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.consistency_models.ConsistencyModelPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold

    @data(*test_cases)
    @unpack
    def test_consistency_model_pipeline_multistep_class_cond(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components(class_cond=True)
        pt_pipe_cls = get_module("diffusers.pipelines.consistency_models.ConsistencyModelPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.consistency_models.ConsistencyModelPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()
        inputs["class_labels"] = 0

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold

    @data(*test_cases)
    @unpack
    def test_consistency_model_pipeline_onestep(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.consistency_models.ConsistencyModelPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.consistency_models.ConsistencyModelPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold

    @data(*test_cases)
    @unpack
    def test_consistency_model_pipeline_onestep_class_cond(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components(class_cond=True)
        pt_pipe_cls = get_module("diffusers.pipelines.consistency_models.ConsistencyModelPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.consistency_models.ConsistencyModelPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        inputs["class_labels"] = 0

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class ConsistencyModelPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    def get_inputs(self):
        inputs = {
            "num_inference_steps": None,
            "timesteps": [22, 0],
            "class_labels": 0,
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_consistency_model_cd_multistep(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.consistency_models.ConsistencyModelPipeline")
        pipe = pipe_cls.from_pretrained(
            "openai/diffusers-cd_imagenet64_l2", use_safetensors=True, mindspore_dtype=ms_dtype
        )
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()

        torch.manual_seed(0)
        image = pipe(**inputs)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"multistep_{dtype}.npy",
            subfolder="consistency_models",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

    @data(*test_cases)
    @unpack
    def test_consistency_model_cd_onestep(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.consistency_models.ConsistencyModelPipeline")
        pipe = pipe_cls.from_pretrained(
            "openai/diffusers-cd_imagenet64_l2", use_safetensors=True, mindspore_dtype=ms_dtype
        )
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None

        torch.manual_seed(0)
        image = pipe(**inputs)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"onestep_{dtype}.npy",
            subfolder="consistency_models",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
