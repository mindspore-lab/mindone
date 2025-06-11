from time import time

import numpy as np
from cogvideox.acceleration import create_parallel_group
from cogvideox.models.autoencoder_kl_cogvideox_sp import AutoencoderKLCogVideoX_SP

import mindspore as ms
import mindspore.ops as mint
from mindspore.communication import get_group_size, init

from mindone.diffusers import AutoencoderKLCogVideoX
from mindone.diffusers.training_utils import pynative_no_grad
from mindone.diffusers.utils import pynative_context
from mindone.utils.seed import set_random_seed

THRESHOLD_FP16 = 1e-4
THRESHOLD_BF16 = 1e-4


def get_sample_data(encoder=True, dtype=ms.bfloat16):
    batch_size = 1
    frame = 80
    w = 1360
    h = 768
    if encoder:
        x = mint.rand([batch_size, 3, frame, h, w], dtype=dtype)  # [B, C, F, H, W]
        return dict(x=x)
    z = mint.rand([batch_size, 16, (frame - 1) // 4 + 1, h // 8, w // 8], dtype=dtype)  # [B, C, F, H, W]
    return dict(z=z)


def get_model_config():
    config = {
        "act_fn": "silu",
        "block_out_channels": [128, 256, 256, 512],
        "down_block_types": [
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ],
        "force_upcast": True,
        "in_channels": 3,
        "latent_channels": 16,
        "latents_mean": None,
        "latents_std": None,
        "layers_per_block": 3,
        "norm_eps": 1e-06,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_height": 480,
        "sample_width": 720,
        "scaling_factor": 0.7,
        "shift_factor": None,
        "temporal_compression_ratio": 4,
        "up_block_types": ["CogVideoXUpBlock3D", "CogVideoXUpBlock3D", "CogVideoXUpBlock3D", "CogVideoXUpBlock3D"],
        "use_post_quant_conv": False,
        "use_quant_conv": False,
        "invert_scale_latents": True,
    }
    return config


def run_performence(
    mode: int = 0,
    jit_level="O0",
    dtype=ms.bfloat16,
    encoder=True,
    dist_model=True,
    enable_slicing=True,
    enable_tiling=True,
    run_perf_step=10,
    get_profile=False,
    output_path="./data",
):
    ms.set_context(mode=mode)
    ms.set_context(jit_config={"jit_level": jit_level})

    # prepare data
    set_random_seed(1024)
    model_cfg = get_model_config()
    print(f"model_cfg:\n {model_cfg}")
    if dist_model:
        init()
        ms.context.set_auto_parallel_context(parallel_mode="data_parallel")
        create_parallel_group(get_group_size())
        vae = AutoencoderKLCogVideoX_SP(**model_cfg).to(dtype)
    else:
        vae = AutoencoderKLCogVideoX(**model_cfg).to(dtype)
    if enable_slicing:
        vae.enable_slicing()
    if enable_tiling:
        vae.enable_tiling()
    vae_parameters = vae.get_parameters()
    num_trainable_parameters = sum(param.numel() for param in vae_parameters)
    data = get_sample_data(encoder=encoder, dtype=dtype)
    vae_forward = vae.encode if encoder else vae.decode
    s = time()
    with pynative_context(), pynative_no_grad():
        videos = vae_forward(**data)
    print(f"==first step: {time() - s} s", videos[0].shape)
    print(f"=== Num trainable parameters = {num_trainable_parameters/1e9:.2f} B")
    print("=" * 10, "run_pref", "=" * 10, flush=True)
    if get_profile:
        profiler = ms.Profiler(
            start_profile=True, output_path=output_path, profiler_level=ms.profiler.ProfilerLevel.Level1
        )
    start = time()
    for i in range(run_perf_step):
        with pynative_context(), pynative_no_grad():
            s = time()
            videos = vae_forward(**data)
            print(f"==step {i}: {time() - s} s", videos[0].shape)
    print(f"=== dist_out forward cost: {(time() - start) / run_perf_step} s")
    if get_profile:
        profiler.stop()
        profiler.analyse()


def run_model(
    mode: int = 0,
    jit_level="O0",
    dtype=ms.bfloat16,
    encoder=True,
    run_pref=False,
    enable_slicing=True,
    enable_tiling=True,
    run_perf_step=10,
):
    ms.set_context(mode=mode)
    ms.set_context(jit_config={"jit_level": jit_level})
    init()
    ms.context.set_auto_parallel_context(parallel_mode="data_parallel")
    create_parallel_group(get_group_size())

    # prepare data
    set_random_seed(1024)
    threshold = THRESHOLD_BF16 if dtype == ms.bfloat16 else THRESHOLD_FP16
    model_cfg = get_model_config()
    print(f"model_cfg:\n {model_cfg}")
    non_dist_model = AutoencoderKLCogVideoX(**model_cfg).to(dtype)
    dist_model = AutoencoderKLCogVideoX_SP(**model_cfg).to(dtype)
    if enable_slicing:
        non_dist_model.enable_slicing()
        dist_model.enable_slicing()
    if enable_tiling:
        non_dist_model.enable_tiling()
        dist_model.enable_tiling()

    for (_, w0), (_, w1) in zip(non_dist_model.parameters_and_names(), dist_model.parameters_and_names()):
        w1.set_data(w0)  # FIXME: seed does not work
        np.testing.assert_allclose(w0.value().to(ms.float32).asnumpy(), w1.value().to(ms.float32).asnumpy())
    data = get_sample_data(encoder=encoder, dtype=dtype)
    non_dist_forward = non_dist_model.encode if encoder else non_dist_model.decode
    dist_forward = dist_model.encode if encoder else dist_model.decode
    # test forward
    with pynative_context(), pynative_no_grad():
        non_dist_out = non_dist_forward(**data)[0]
        dist_out = dist_forward(**data)[0]
    if isinstance(non_dist_out, ms.Tensor):
        non_dist_out = (non_dist_out,)
    if isinstance(dist_out, ms.Tensor):
        dist_out = (dist_out,)
    for non_dist_o, dist_o in zip(non_dist_out, dist_out):
        np.testing.assert_allclose(non_dist_o.to(ms.float32).asnumpy(), dist_o.to(ms.float32).asnumpy(), atol=threshold)
    if run_pref:
        with pynative_context(), pynative_no_grad():
            print("=" * 10, "run_pref", "=" * 10)
            start = time()
            for i in range(run_perf_step):
                non_dist_out = non_dist_forward(**data)
            print(f"=== non_dist_out forward cost: {(time() - start)/run_perf_step} s")
            start = time()
            for i in range(run_perf_step):
                dist_out = dist_forward(**data)
            print(f"=== dist_out forward cost: {(time() - start) / run_perf_step} s")
    print("Test 1 (Forward): Passed.")


if __name__ == "__main__":
    mode = ms.PYNATIVE_MODE
    jit_level = "O1"
    dtype = ms.bfloat16
    encoder = True
    enable_slicing = True
    enable_tiling = True

    run_acc = True
    # get performence when run_acc or run standalone
    run_perf = True
    dist_model = True
    run_perf_step = 3
    get_profile = False
    output_path = "vae_data"
    ms.set_context(pynative_synchronize=False)
    if run_acc:
        run_model(
            mode=mode,
            jit_level=jit_level,
            dtype=dtype,
            encoder=encoder,
            enable_slicing=enable_slicing,
            enable_tiling=enable_tiling,
            run_pref=run_perf,
            run_perf_step=run_perf_step,
        )
    else:
        run_performence(
            mode=mode,
            jit_level=jit_level,
            dtype=dtype,
            encoder=encoder,
            dist_model=dist_model,
            enable_slicing=enable_slicing,
            enable_tiling=enable_tiling,
            run_perf_step=run_perf_step,
            get_profile=get_profile,
            output_path=output_path,
        )
