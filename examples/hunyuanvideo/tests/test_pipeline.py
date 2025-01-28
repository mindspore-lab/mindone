# flake8: noqa
import os
import sys
import time
from pathlib import Path

import numpy as np
from easydict import EasyDict as edict
from PIL import Image

import mindspore as ms
from mindspore import amp

sys.path.insert(0, ".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
# from mindone.visualize.videos import save_videos

from hyvideo.config import parse_args
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.modules import load_model

# from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.utils.helpers import set_model_param_dtype


def test_infer():
    args = edict()
    args.model_base = "ckpts"
    args.dit_weight = "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"

    # models_root_path = Path(model_base)
    # hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)


def test_pipeline():
    args = parse_args()

    # create dit model
    args.model = "HYVideo-T/2-depth1"  # HYVideo-T/2-cfgdistill
    args.dit_weight = "ckpts/transformer_depth1.pt"
    args.precision = "fp16"
    args.latent_channels = 16
    dtype = ms.float16
    factor_kwargs = {"dtype": dtype, "attn_mode": "vanilla"}
    print("creating model")
    model = load_model(
        args,
        in_channels=args.latent_channels,
        out_channels=args.latent_channels,
        factor_kwargs=factor_kwargs,
    )
    if dtype != ms.float32:
        set_model_param_dtype(model, dtype=dtype)
        amp.auto_mixed_precision(model, amp_level="auto", dtype=dtype)
        # amp.auto_mixed_precision(model, amp_level='O2', dtype=dtype)
    if args.dit_weight:
        print("loading model weights")
        model.load_from_checkpoint(args.dit_weight)

    # vae
    args.vae = edict()
    args.vae.config = edict()
    args.vae.config.block_out_channels = [128, 256, 512, 512]
    vae_ver = "884-16c-hy"

    # create schedule and pipeline
    print(args.flow_shift, args.flow_reverse, args.flow_solver)
    scheduler = FlowMatchDiscreteScheduler(
        shift=args.flow_shift,
        reverse=args.flow_reverse,
        solver=args.flow_solver,
    )

    print(args.latent_channels)
    pipeline = HunyuanVideoPipeline(
        vae=args.vae,
        text_encoder=None,
        text_encoder_2=None,
        transformer=model,
        scheduler=scheduler,
        progress_bar_config=None,
        args=args,
    )

    # prepare text emb cache if text encoder is none
    prompt = "A cat walks on the grass, realistic style."
    NEGATIVE_PROMPT = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"

    t1_emb_shape = (BS, S_t1, D_t1) = (1, 256, 4096)
    D_t2 = 768
    t2_emb_shape = (BS, D_t2)
    prompt_embeds = np.random.normal(size=t1_emb_shape).astype(np.float32)
    prompt_mask = np.zeros(shape=(BS, S_t1), dtype=np.int32)
    prompt_mask[0 : len(prompt) // 2] = 1  # TODO: this is just a rough estimation, not real num of tokens
    prompt_embeds_2 = np.random.normal(size=t2_emb_shape).astype(np.float32)

    negative_prompt_embeds = np.random.normal(size=t1_emb_shape).astype(np.float32)
    negative_prompt_mask = np.zeros(shape=(BS, S_t1), dtype=np.int32)
    negative_prompt_mask[
        0 : len(NEGATIVE_PROMPT) // 2
    ] = 1  # TODO: this is just a rough estimation, not real num of tokens
    negative_prompt_embeds_2 = np.random.normal(size=t2_emb_shape).astype(np.float32)
    np.savez(
        "text_embed/test.npz",
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_mask=negative_prompt_mask,
        prompt_embeds_2=prompt_embeds_2,
        negative_prompt_embeds_2=negative_prompt_embeds_2,
    )

    prompt_embeds = ms.Tensor(prompt_embeds)
    prompt_mask = ms.Tensor(prompt_mask)
    prompt_embeds_2 = ms.Tensor(prompt_embeds_2)

    negative_prompt_embeds = ms.Tensor(negative_prompt_embeds)
    negative_prompt_mask = ms.Tensor(negative_prompt_mask)
    negative_prompt_embeds_2 = ms.Tensor(negative_prompt_embeds_2)

    # run
    infer_steps = 2
    generator = [np.random.Generator(np.random.PCG64(seed=seed)) for seed in [42]]
    rope_dim_list = [16, 56, 56]

    target_height, target_width, target_video_length = 128, 128, 17
    latents_size = [(target_video_length - 1) // 4 + 1, target_height // 8, target_width // 8]
    patch_size = [1, 2, 2]
    rope_sizes = [s // patch_size[idx] for idx, s in enumerate(latents_size)]
    # freqs_cos
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=1,
        use_real=True,
        theta_rescale_factor=1,
    )
    n_tokens = freqs_cos.shape[0]
    samples = pipeline(
        prompt=prompt,
        height=target_height,
        width=target_width,
        video_length=target_video_length,
        num_inference_steps=infer_steps,
        guidance_scale=6,
        negative_prompt=NEGATIVE_PROMPT,
        num_videos_per_prompt=1,
        generator=generator,
        # output_type="pil",
        output_type="latent",
        freqs_cis=(freqs_cos, freqs_sin),
        n_tokens=n_tokens,
        embedded_guidance_scale=6,
        data_type="video" if target_video_length > 1 else "image",
        is_progress_bar=True,
        vae_ver=vae_ver,  # TODO
        enable_tiling=args.vae_tiling,  # TODO
        prompt_embeds=prompt_embeds,  # text emb cache
        prompt_mask=prompt_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_mask=negative_prompt_mask,
        prompt_embeds_2=prompt_embeds_2,
        negative_prompt_embeds_2=negative_prompt_embeds_2,
    )[0]

    print(samples.shape)


if __name__ == "__main__":
    ms.set_context(mode=1)
    # ms.set_context(mode=0, jit_syntax_level=ms.STRICT)

    test_pipeline()
    # test_infer()
