import random
import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers import CLIPTextConfig, CLIPVisionConfig

import mindspore as ms

from mindone.diffusers import DiffusionPipeline, StableUnCLIPImg2ImgPipeline
from mindone.diffusers.utils.testing_utils import (
    load_downloaded_image_from_hf_hub,
    load_downloaded_numpy_from_hf_hub,
    slow,
)

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    floats_tensor,
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
class StableUnCLIPImg2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "feature_extractor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            dict(
                crop_size=32,
                size=32,
            ),
        ],
        [
            "image_encoder",
            "transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            dict(
                config=CLIPVisionConfig(
                    hidden_size=32,
                    projection_dim=32,
                    num_hidden_layers=5,
                    num_attention_heads=4,
                    image_size=32,
                    intermediate_size=37,
                    patch_size=1,
                ),
            ),
        ],
        [
            "image_normalizer",
            "diffusers.pipelines.stable_diffusion.StableUnCLIPImageNormalizer",
            "mindone.diffusers.pipelines.stable_diffusion.StableUnCLIPImageNormalizer",
            dict(
                embedding_dim=32,
            ),
        ],
        [
            "image_noising_scheduler",
            "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            dict(
                beta_schedule="squaredcos_cap_v2",
            ),
        ],
        [
            "tokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
            ),
        ],
        [
            "text_encoder",
            "transformers.models.clip.modeling_clip.CLIPTextModel",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModel",
            dict(
                config=CLIPTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=32,
                    projection_dim=32,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=5,
                    pad_token_id=1,
                    vocab_size=1000,
                ),
            ),
        ],
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                block_out_channels=(32, 64),
                attention_head_dim=(2, 4),
                class_embed_type="projection",
                # The class embeddings are the noise augmented image embeddings.
                # I.e. the image embeddings concated with the noised embeddings of the same dimension
                projection_class_embeddings_input_dim=64,
                cross_attention_dim=32,
                layers_per_block=1,
                upcast_attention=True,
                use_linear_projection=True,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(
                beta_schedule="scaled_linear",
                beta_start=0.00085,
                beta_end=0.012,
                prediction_type="v_prediction",
                set_alpha_to_one=False,
                steps_offset=1,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                # image encoding components
                "feature_extractor",
                "image_encoder",
                # image noising components
                "image_normalizer",
                "image_noising_scheduler",
                # regular denoising components
                "tokenizer",
                "text_encoder",
                "unet",
                "scheduler",
                "vae",
            ]
        }
        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        eval_components = ["image_encoder", "image_normalizer", "text_encoder", "unet", "vae"]

        for component in eval_components:
            pt_components[component] = pt_components[component].eval()
            ms_components[component] = ms_components[component].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0, pil_image=True):
        input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))

        if pil_image:
            input_image = input_image * 0.5 + 0.5
            input_image = input_image.clamp(0, 1)
            input_image = input_image.permute(0, 2, 3, 1).float().numpy()
            input_image = DiffusionPipeline.numpy_to_pil(input_image)[0]
            pt_input_image = ms_input_image = input_image
        else:
            pt_input_image = input_image
            ms_input_image = ms.Tensor(pt_input_image.numpy())

        pt_inputs = {
            "prompt": "An anime racoon running a marathon",
            "image": pt_input_image,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        ms_inputs = {
            "prompt": "An anime racoon running a marathon",
            "image": ms_input_image,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.stable_diffusion.StableUnCLIPImg2ImgPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.stable_diffusion.StableUnCLIPImg2ImgPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class StableUnCLIPImg2ImgPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip", mindspore_dtype=ms_dtype, variant="fp16"
        )

        prompt = "A fantasy landscape, trending on artstation"
        init_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "tarsila_do_amaral.png",
            subfolder="stable_unclip",
        )

        torch.manual_seed(0)
        image = pipe(init_image, prompt=prompt)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"stable_unclip_i2i_{dtype}.npy",
            subfolder="stable_unclip",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
