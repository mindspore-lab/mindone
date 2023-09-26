# reference to https://github.com/Stability-AI/generative-models

import os

import streamlit as st
from gm.helpers import create_model, get_interactive_image, load_img
from gm.modules.diffusionmodules.discretizer import Img2ImgDiscretizationWrapper, Txt2NoisyDiscretizationWrapper
from gm.modules.diffusionmodules.sampler import EulerEDMSampler
from omegaconf import OmegaConf


@st.cache_resource()
def create_model_with_streamlit(config, **kwargs):
    config = OmegaConf.load(config)
    model, filter = create_model(config, **kwargs)
    return model, filter


def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None):
    # Hardcoded demo settings; might undergo some changes in the future

    value_dict = {}
    for key in keys:
        if key == "txt":
            if prompt is None:
                prompt = st.text_input("Prompt", "A professional photograph of an astronaut riding a pig")
            if negative_prompt is None:
                negative_prompt = st.text_input("Negative prompt", "")

            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

        if key == "original_size_as_tuple":
            orig_width = st.number_input(
                "orig_width",
                value=init_dict["orig_width"],
                min_value=16,
            )
            orig_height = st.number_input(
                "orig_height",
                value=init_dict["orig_height"],
                min_value=16,
            )

            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            crop_coord_top = st.number_input("crop_coords_top", value=0, min_value=0)
            crop_coord_left = st.number_input("crop_coords_left", value=0, min_value=0)

            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

    return value_dict


def init_save_locally(_dir, init_value: bool = False):
    save_locally = st.sidebar.checkbox("Save images locally", value=init_value)
    if save_locally:
        save_path = st.text_input("Save path", value=os.path.join(_dir, "samples"))
    else:
        save_path = None

    return save_locally, save_path


def init_sampling_with_streamlit(
    key=1,
    img2img_strength=1.0,
    specify_num_samples=True,
    stage2strength=None,
):
    num_rows, num_cols = 1, 1
    if specify_num_samples:
        num_cols = st.number_input(f"num cols #{key}", value=1, min_value=1, max_value=10)

    steps = st.sidebar.number_input(f"steps #{key}", value=40, min_value=1, max_value=1000)
    sampler = st.sidebar.selectbox(
        f"Sampler #{key}",
        [
            "EulerEDMSampler",
            "HeunEDMSampler",
            "EulerAncestralSampler",
            "DPMPP2SAncestralSampler",
            "DPMPP2MSampler",
            "LinearMultistepSampler",
        ],
        0,
    )
    discretization = st.sidebar.selectbox(
        f"Discretization #{key}",
        [
            "LegacyDDPMDiscretization",
            "EDMDiscretization",
        ],
    )

    discretization_config = get_discretization(discretization, key=key)

    guider_config = get_guider(key=key)

    sampler = get_sampler(sampler, steps, discretization_config, guider_config, key=key)

    if img2img_strength < 1.0:
        st.warning(f"Wrapping {sampler.__class__.__name__} with Img2ImgDiscretizationWrapper")
        sampler.discretization = Img2ImgDiscretizationWrapper(sampler.discretization, strength=img2img_strength)
    if stage2strength is not None:
        sampler.discretization = Txt2NoisyDiscretizationWrapper(
            sampler.discretization, strength=stage2strength, original_steps=steps
        )

    return sampler, num_rows, num_cols


def get_guider(key):
    guider = st.sidebar.selectbox(
        f"Discretization #{key}",
        [
            "VanillaCFG",
            "IdentityGuider",
        ],
    )

    if guider == "IdentityGuider":
        guider_config = {"target": "gm.modules.diffusionmodules.guiders.IdentityGuider"}
    elif guider == "VanillaCFG":
        scale = st.number_input(f"cfg-scale #{key}", value=5.0, min_value=0.0, max_value=100.0)

        thresholder = st.sidebar.selectbox(
            f"Thresholder #{key}",
            [
                "None",
            ],
        )

        if thresholder == "None":
            dyn_thresh_config = {"target": "gm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"}
        else:
            raise NotImplementedError

        guider_config = {
            "target": "gm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": scale, "dyn_thresh_config": dyn_thresh_config},
        }
    else:
        raise NotImplementedError
    return guider_config


def get_discretization(discretization, key=1):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "gm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif discretization == "EDMDiscretization":
        sigma_min = st.number_input(f"sigma_min #{key}", value=0.03)  # 0.0292
        sigma_max = st.number_input(f"sigma_max #{key}", value=14.61)  # 14.6146
        rho = st.number_input(f"rho #{key}", value=3.0)
        discretization_config = {
            "target": "gm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }
    else:
        raise NotImplementedError

    return discretization_config


def get_sampler(sampler_name, steps, discretization_config, guider_config, key=1):
    if sampler_name in ("EulerEDMSampler", "HeunEDMSampler"):
        s_churn = st.sidebar.number_input(f"s_churn #{key}", value=0.0, min_value=0.0)
        s_tmin = st.sidebar.number_input(f"s_tmin #{key}", value=0.0, min_value=0.0)
        s_tmax = st.sidebar.number_input(f"s_tmax #{key}", value=999.0, min_value=0.0)
        s_noise = st.sidebar.number_input(f"s_noise #{key}", value=1.0, min_value=0.0)

        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "HeunEDMSampler":
            raise NotImplementedError
        else:
            raise ValueError

    elif sampler_name in ("EulerAncestralSampler", "DPMPP2SAncestralSampler"):
        raise NotImplementedError
    elif sampler_name in ("DPMPP2MSampler",):
        raise NotImplementedError
    elif sampler_name in ("LinearMultistepSampler",):
        raise NotImplementedError
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler


def load_img_with_streamlit(display=True, key=None):
    image = st.file_uploader("Input", type=["jpg", "JPEG", "png"], key=key)
    image = get_interactive_image(image)
    if image is None:
        return None
    if display:
        st.image(image)
    return load_img(image)
