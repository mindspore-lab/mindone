import os
import sys

import numpy as np

import mindspore as ms

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3PipelineBoost
from test_utils import generate_pipeline, mock_fn

SD3_PIPELINE_COMPONENTS = {
    "transformer": [
        "mindone.diffusers.models.SD3Transformer2DModel",
        "fp16",
        (),
        {
            "sample_size": 128,
            "patch_size": 2,
            "in_channels": 16,
            "num_layers": 24,
            "attention_head_dim": 64,
            "num_attention_heads": 24,
            "caption_projection_dim": 1536,
            "joint_attention_dim": 4096,
            "pooled_projection_dim": 2048,
            "out_channels": 16,
        },
    ],
    "text_encoder": [None, "", (), {}],
    "text_encoder_2": [None, "", (), {}],
    "text_encoder_3": [None, "", (), {}],
    "tokenizer": [None, "", (), {}],
    "tokenizer_2": [None, "", (), {}],
    "tokenizer_3": [None, "", (), {}],
    "scheduler": ["mindone.diffusers.schedulers.FlowMatchEulerDiscreteScheduler", "fp16", (), {}],
    "vae": [None, "", (), {}],
}

MOCK_INFO = {
    "encode_prompt": (
        ms.Tensor.from_numpy(np.random.randn(1, 154, 4096).astype(np.float16)),
        ms.Tensor.from_numpy(np.random.randn(1, 154, 4096).astype(np.float16)),
        ms.Tensor.from_numpy(np.random.randn(1, 2048).astype(np.float16)),
        ms.Tensor.from_numpy(np.random.randn(1, 2048).astype(np.float16)),
    ),
    "prepare_latents": ms.Tensor.from_numpy(np.random.randn(1, 16, 128, 128).astype(np.float16)),
}


def test_stable_diffusion_3(mocker):
    ms.set_context(mode=0, jit_config={"jit_level": "O2"})
    pipe = generate_pipeline(StableDiffusion3PipelineBoost, SD3_PIPELINE_COMPONENTS)

    mock_fn(pipe, MOCK_INFO, mocker)

    generator = np.random.Generator(np.random.PCG64(0))
    img = pipe(
        prompt="A cat holding a sign that says hello world",
        generator=generator,
        use_cache_and_gate=True,
        output_type="latent",
    )
