import os
import sys

import yaml

import mindspore as ms

from .pipeline import PhotoMakerStableDiffusionXLPipeline

# TODO: remove in future when mindone is ready for install
mindone_lib_path = os.path.abspath("../..")
sys.path.insert(0, mindone_lib_path)


from mindone.diffusers import StableDiffusionXLPipeline


def get_models_dict():
    with open("config/models.yaml", "r") as stream:
        try:
            data = yaml.safe_load(stream)

            print(data)
            return data

        except yaml.YAMLError as exc:
            print(exc)


def load_models(model_info, photomaker_path):
    path = model_info["path"]
    single_files = model_info["single_files"]
    use_safetensors = model_info["use_safetensors"]
    model_type = model_info["model_type"]

    if model_type == "original":
        if single_files:
            pipe = StableDiffusionXLPipeline.from_single_file(path, mindspore_dtype=ms.float16)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                path, mindspore_dtype=ms.float16, use_safetensors=use_safetensors
            )

    elif model_type == "Photomaker":
        if single_files:
            print("loading from a single_files")
            pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(path, mindspore_dtype=ms.float16)
        else:
            pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                path, mindspore_dtype=ms.float16, use_safetensors=use_safetensors
            )

        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img",  # define the trigger word
        )
        pipe.fuse_lora()
    else:
        raise NotImplementedError("You should choice between original and Photomaker!", f"But you choice {model_type}")
    return pipe
