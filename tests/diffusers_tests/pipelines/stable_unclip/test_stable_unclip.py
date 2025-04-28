import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers import CLIPTextConfig, CLIPTokenizer

import mindspore as ms

from mindone.diffusers import DDPMScheduler, PriorTransformer, StableUnCLIPPipeline, UnCLIPScheduler
from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow
from mindone.transformers import CLIPTextModelWithProjection

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
class StableUnCLIPPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "prior_tokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
            ),
        ],
        [
            "prior_text_encoder",
            "transformers.models.clip.modeling_clip.CLIPTextModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModelWithProjection",
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
            "prior",
            "diffusers.models.transformers.prior_transformer.PriorTransformer",
            "mindone.diffusers.models.transformers.prior_transformer.PriorTransformer",
            dict(
                num_attention_heads=2,
                attention_head_dim=12,
                embedding_dim=32,
                num_layers=1,
            ),
        ],
        [
            "prior_scheduler",
            "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            dict(
                variance_type="fixed_small_log",
                prediction_type="sample",
                num_train_timesteps=1000,
                clip_sample=True,
                clip_sample_range=5.0,
                beta_schedule="squaredcos_cap_v2",
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
                # prior components
                "prior_tokenizer",
                "prior_text_encoder",
                "prior",
                "prior_scheduler",
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

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "prior_num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.stable_diffusion.StableUnCLIPPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.stable_diffusion.StableUnCLIPPipeline")

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


@slow
@ddt
class StableUnCLIPPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_stable_unclip(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        prior_model_id = "kakaobrain/karlo-v1-alpha"
        prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", mindspore_dtype=ms_dtype)

        prior_text_model_id = "openai/clip-vit-large-patch14"
        prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
        prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, mindspore_dtype=ms_dtype)
        prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
        prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)

        stable_unclip_model_id = "stabilityai/stable-diffusion-2-1-unclip-small"

        pipe = StableUnCLIPPipeline.from_pretrained(
            stable_unclip_model_id,
            mindspore_dtype=ms_dtype,
            prior_tokenizer=prior_tokenizer,
            prior_text_encoder=prior_text_model,
            prior=prior,
            prior_scheduler=prior_scheduler,
        )

        wave_prompt = (
            "dramatic wave, the Oceans roar, Strong wave spiral across the oceans as the waves unfurl into "
            "roaring crests; perfect wave form; perfect wave shape; dramatic wave shape; wave shape "
            "unbelievable; wave; wave shape spectacular"
        )

        torch.manual_seed(0)
        image = pipe(prompt=wave_prompt)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"stable_unclip_t2i_{dtype}.npy",
            subfolder="stable_unclip",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
