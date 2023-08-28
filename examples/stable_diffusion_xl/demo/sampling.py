# reference to https://github.com/Stability-AI/generative-models

import os

import streamlit as st
from demo.streamlit_helpers import create_model_with_streamlit, init_embedder_options, init_sampling, init_save_locally
from gm.helpers import SD_XL_BASE_RATIOS, VERSION2SPECS, get_unique_embedder_keys_from_conditioner, perform_save_locally
from gm.util import seed_everything

import mindspore as ms

SAVE_PATH = "outputs/demo/txt2img/"
WEIGHT = "checkpoints/sd_xl_base_1.0_ms.ckpt"
CONFIG = "configs/inference/sd_xl_base.yaml"
DEVICE_TARGET = "Ascend"


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
    sampler, num_rows, num_cols = init_sampling(stage2strength=stage2strength)
    num_samples = num_rows * num_cols

    if st.button("Sample"):
        st.write(f"**Model I:** {version}")
        st.text("Sampling")
        outputs = st.empty()

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
        _n, _b, _c, _h, _w = grid.shape
        grid = grid.transpose(0, 3, 1, 4, 2).reshape((_n * _h, _b * _w, _c))  # n b c h w -> (n h) (b w) c
        outputs.image(grid)

        print("Output Image Done.")

        return out


if __name__ == "__main__":
    ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target=DEVICE_TARGET)

    st.title("Stable Diffusion")
    version = st.selectbox("Model Version", list(VERSION2SPECS.keys()), 0)
    version_dict = VERSION2SPECS[version]
    amp_level = st.selectbox("Mix Precision", ["O2", "O0"], 0)
    mode = st.radio("Mode", ("txt2img",), 0)  # ("txt2img", "img2img")
    st.write("__________________________")

    add_pipeline = False
    # TODO: Add Refiner
    # if version.startswith("SDXL-base"):
    #     add_pipeline = st.checkbox("Load SDXL-refiner?", False)
    #     st.write("__________________________")

    seed = st.sidebar.number_input("seed", value=42, min_value=0, max_value=int(1e9))
    seed_everything(seed)

    # Init Model
    model, filter = create_model_with_streamlit(
        CONFIG, checkpoints=WEIGHT.split(","), freeze=True, load_filter=False, amp_level=amp_level
    )

    # Get prompt
    prompt = st.text_input(
        "prompt",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    )

    save_locally, save_path = init_save_locally(os.path.join(SAVE_PATH, version))
    is_legacy = version_dict["is_legacy"]
    if is_legacy:
        negative_prompt = st.text_input("negative prompt", "")
    else:
        negative_prompt = ""  # which is unused

    if add_pipeline:
        # TODO: Add Refiner
        raise NotImplementedError

    if mode == "txt2img":
        out = run_txt2img(
            model,
            version,
            version_dict,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=filter,
            stage2strength=None,
            amp_level=amp_level,
        )
    elif mode == "img2img":
        raise NotImplementedError
    else:
        raise ValueError(f"unknown mode {mode}")

    out = out if isinstance(out, (tuple, list)) else [out, None]
    (samples, samples_z) = out

    if add_pipeline and samples_z is not None:
        raise NotImplementedError

    if save_locally and samples is not None:
        perform_save_locally(save_path, samples)
