import os
import sys

import numpy as np

import mindspore as ms

ms.set_context(deterministic="ON")
current_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):
    current_path = os.path.dirname(current_path)
sys.path.append(os.path.join(current_path, "examples/dit_infer_acceleration"))

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

np.random.seed(42)

MOCK_INFO = {
    "encode_prompt": (
        ms.Tensor.from_numpy(np.random.randn(1, 154, 4096).astype(np.float16)),
        ms.Tensor.from_numpy(np.random.randn(1, 154, 4096).astype(np.float16)),
        ms.Tensor.from_numpy(np.random.randn(1, 2048).astype(np.float16)),
        ms.Tensor.from_numpy(np.random.randn(1, 2048).astype(np.float16)),
    ),
    "prepare_latents": ms.Tensor.from_numpy(np.random.randn(1, 16, 128, 128).astype(np.float16)),
}

EXPECT_RESULT = np.array([[[-1.358, -0.08307, 0.9316, -1.998, -0.9067, -0.3474, -0.98, 0.364, 0.3218, -0.907]]])


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
    )[0][0]

    assert np.allclose(img.asnumpy()[:1, :1, :10], EXPECT_RESULT, atol=1e-3, rtol=1e-3)
