# reference to https://github.com/Stability-AI/generative-models

import os
import time

import streamlit as st
from demo.streamlit_helpers import (
    create_model_with_streamlit,
    init_embedder_options,
    init_sampling_with_streamlit,
    init_save_locally,
    load_img_with_streamlit,
)
from gm.helpers import (
    SD_XL_BASE_RATIOS,
    embed_watermark,
    get_unique_embedder_keys_from_conditioner,
    perform_save_locally,
)
from gm.util import seed_everything

import mindspore as ms
from mindspore import Tensor, ops

SAVE_PATH = "outputs/demo/"
DEVICE_TARGET = "Ascend"

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0_ms.ckpt",
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_1.0_ms.ckpt",
    },
}


def run_txt2img(
    model,
    version,
    version_dict,
    is_legacy=False,
    return_latents=False,
    filter=None,
    stage2strength=None,
    amp_level="O0",
):
    W, H = st.selectbox("Resolution:", list(SD_XL_BASE_RATIOS.values()), 10)
    C = version_dict["C"]
    F = version_dict["f"]

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    sampler, num_rows, num_cols = init_sampling_with_streamlit(stage2strength=stage2strength)
    num_samples = num_rows * num_cols

    if st.button("Sample"):
        st.write(f"**Model I:** {version}")
        st.text("Txt2Img Sampling")
        outputs = st.empty()
        s_time = time.time()

        out = model.do_sample(
            sampler,
            value_dict,
            num_samples,
            H,
            W,
            C,
            F,
            force_uc_zero_embeddings=["txt"] if not is_legacy else [],
            return_latents=return_latents,
            filter=filter,
            amp_level=amp_level,
        )

        # draw image
        samples = out[0] if isinstance(out, (tuple, list)) else out
        grid = samples[None, ...]
        grid = embed_watermark(grid)
        _n, _b, _c, _h, _w = grid.shape
        grid = grid.transpose(0, 3, 1, 4, 2).reshape((_n * _h, _b * _w, _c))  # n b c h w -> (n h) (b w) c
        outputs.image(grid)

        print("Output Image Done.")
        print(f"Txt2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")
        st.text(f"Txt2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")

        return out


def run_img2img(model, is_legacy=False, return_latents=False, filter=None, stage2strength=None, amp_level="O0"):
    dtype = ms.float32 if amp_level not in ("O2", "O3") else ms.float16

    img = load_img_with_streamlit()
    if img is None:
        return None
    H, W = img.shape[2], img.shape[3]
    print(f"Input Image shape: ({H}, {W})")

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    strength = st.number_input("**Img2Img Strength**", value=0.75, min_value=0.0, max_value=1.0)
    sampler, num_rows, num_cols = init_sampling_with_streamlit(
        img2img_strength=strength,
        stage2strength=stage2strength,
    )
    num_samples = num_rows * num_cols

    if st.button("Sample"):
        st.write(f"**Model I:** {version}")
        st.text("Img2Img Sampling")
        outputs = st.empty()
        s_time = time.time()

        out = model.do_img2img(
            ops.repeat_elements(Tensor(img, dtype), num_samples, axis=0),
            sampler,
            value_dict,
            num_samples,
            force_uc_zero_embeddings=["txt"] if not is_legacy else [],
            return_latents=return_latents,
            filter=filter,
            amp_level=amp_level,
        )

        # draw image
        samples = out[0] if isinstance(out, (tuple, list)) else out
        grid = samples[None, ...]
        grid = embed_watermark(grid)
        _n, _b, _c, _h, _w = grid.shape
        grid = grid.transpose(0, 3, 1, 4, 2).reshape((_n * _h, _b * _w, _c))  # n b c h w -> (n h) (b w) c
        outputs.image(grid)

        print("Output Image Done.")
        print(f"Img2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")
        st.text(f"Img2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")

        return out


def apply_refiner(
    input, model, sampler, num_samples, prompt, negative_prompt, filter=None, finish_denoising=False, amp_level="O0"
):
    latent_h, latent_w = input.shape[2:]
    value_dict = {
        "orig_width": latent_w * 8,
        "orig_height": latent_h * 8,
        "target_width": latent_w * 8,
        "target_height": latent_h * 8,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "crop_coords_top": 0,
        "crop_coords_left": 0,
        "aesthetic_score": 6.0,
        "negative_aesthetic_score": 2.5,
    }

    st.text("Img2Img Sampling")
    st.warning(f"refiner input shape: {input.shape}")
    outputs = st.empty()
    s_time = time.time()
    samples = model.do_img2img(
        input,
        sampler,
        value_dict,
        num_samples,
        skip_encode=True,
        filter=filter,
        add_noise=not finish_denoising,
        amp_level=amp_level,
    )

    # draw image
    samples = samples[0] if isinstance(samples, (tuple, list)) else samples
    grid = samples[None, ...]
    grid = embed_watermark(grid)
    _n, _b, _c, _h, _w = grid.shape
    grid = grid.transpose(0, 3, 1, 4, 2).reshape((_n * _h, _b * _w, _c))  # n b c h w -> (n h) (b w) c
    outputs.image(grid)
    print("Output Image Done.")
    print(f"PipeLine(Refiner) sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")
    st.text(f"PileLine(Refiner) Refiner sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")

    return samples


if __name__ == "__main__":
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target=DEVICE_TARGET)

    st.title("Stable Diffusion XL")
    version = st.selectbox("Model Version", list(VERSION2SPECS.keys()), 0)
    version_dict = VERSION2SPECS[version]
    amp_level = st.selectbox("Mix Precision", ["O2", "O0"], 0)
    if version.startswith("SDXL-base"):
        mode = st.radio("Mode", ("txt2img",), 0)
        st.write("__________________________")
        add_pipeline = st.checkbox("Load SDXL-refiner?", False)
        st.write("__________________________")
    elif version.startswith("SDXL-refiner"):
        mode = st.radio("Mode", ("img2img",), 0)
        st.write("__________________________")
        add_pipeline = False
    else:
        raise NotImplementedError

    seed = st.sidebar.number_input("seed", value=42, min_value=0, max_value=int(1e9))
    seed_everything(seed)

    # Init Model
    model, filter = create_model_with_streamlit(
        version_dict["config"],
        checkpoints=version_dict["ckpt"].split(","),
        freeze=True,
        load_filter=False,
        amp_level=amp_level,
    )

    # Get prompt
    prompt = st.text_input(
        "prompt",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    )

    save_locally, save_path = init_save_locally(os.path.join(SAVE_PATH, mode, version))
    is_legacy = version_dict["is_legacy"]
    if is_legacy:
        negative_prompt = st.text_input("negative prompt", "")
    else:
        negative_prompt = ""  # which is unused

    stage2strength = None
    finish_denoising = False

    if add_pipeline:
        st.write("__________________________")
        version2 = st.selectbox(
            "Refiner:",
            [
                "SDXL-refiner-1.0",
            ],
        )
        st.warning(f"Warning: Running with {version2} as the second stage model. Make sure to provide (V)RAM :) ")
        if DEVICE_TARGET == "Ascend":
            st.warning(
                "Warning: Using the 'add_pipeline' function on device Ascend 910A may cause OOM. "
                "It is recommended to use txt2img and img2img tasks respectively. "
                "Alternatively, choose a smaller generation sizes, such as (768, 768)."
            )

        st.write("**Refiner Options:**")
        version_dict2 = VERSION2SPECS[version2]

        # Init Model
        model2, filter2 = create_model_with_streamlit(
            version_dict2["config"],
            checkpoints=version_dict2["ckpt"].split(","),
            freeze=True,
            load_filter=False,
            amp_level=amp_level,
        )

        stage2strength = st.number_input("**Refinement strength**", value=0.15, min_value=0.0, max_value=1.0)

        sampler2, *_ = init_sampling_with_streamlit(
            key=2,
            img2img_strength=stage2strength,
            specify_num_samples=False,
        )
        st.write("__________________________")
        finish_denoising = st.checkbox("Finish denoising with refiner.", True)
        if not finish_denoising:
            stage2strength = None

    if mode == "txt2img":
        out = run_txt2img(
            model,
            version,
            version_dict,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=filter,
            stage2strength=stage2strength,
            amp_level=amp_level,
        )
    elif mode == "img2img":
        out = run_img2img(
            model,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=filter,
            stage2strength=stage2strength,
            amp_level=amp_level,
        )
    else:
        raise ValueError(f"unknown mode {mode}")

    out = out if isinstance(out, (tuple, list)) else [out, None]
    (samples, samples_z) = out

    if save_locally and samples is not None:
        perform_save_locally(save_path, samples)

    if add_pipeline and samples_z is not None:
        st.write("**Running Refinement Stage**")
        samples = apply_refiner(
            samples_z,
            model=model2,
            sampler=sampler2,
            num_samples=samples_z.shape[0],
            prompt=prompt,
            negative_prompt=negative_prompt if is_legacy else "",
            filter=filter2,
            finish_denoising=finish_denoising,
        )

        if save_locally and samples is not None:
            perform_save_locally(os.path.join(save_path, "pipeline"), samples)
