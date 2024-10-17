import argparse
import json
import os
from io import BytesIO

import requests
from PIL import Image
from share4v.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from share4v.conversation import conv_templates
from share4v.mm_utils import tokenizer_image_token
from share4v.model import Share4VLlamaForCausalLM
from share4v.pipeline import TextGenerator
from transformers import AutoTokenizer

import mindspore as ms

# if the project path is not in the system path, add it manually
# os.sys.path.append(os.getcwd())


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def eval_model(args):
    model_path = args.model_path
    config_path = os.path.join(os.getcwd(), "configs")
    with open(os.path.join(config_path, "config.json"), "r") as f:
        config = json.load(f)
    dtype = ms.float16 if config.get("dtype") == "float16" else ms.float32
    image_file = args.image_file

    msmodel = Share4VLlamaForCausalLM(config)
    msmodel.set_train(False)

    # load weight
    msmodel.load_model(os.path.join(model_path, "mindspore_llama_model.ckpt"))

    # load llama
    msmodel.get_model()

    # load vision_tower
    vision_tower = msmodel.get_vision_tower()
    vision_tower.load_model()

    # image_processor
    image_processor = vision_tower.image_processor
    image_processor

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer

    msmodel.set_dtype(dtype)
    print(f"The model dtype is {msmodel.dtype}")

    # load image and preprocess
    image = load_image(image_file)

    image_tensor = ms.Tensor(image_processor.preprocess(image, return_tensors="np")["pixel_values"], dtype=dtype)

    # preparing query and prompt

    qs = args.query
    if config.get("mm_use_im_start_end"):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # mpt, share4v_v1, share4v_llama_2
    conv_mode = "share4v_v0"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # prompt: "A chat between a curious human and an artificial intelligence assistant.
    # The assistant gives helpful, detailed, and polite answers to the human's questions.###Human:
    # <image>\nDescribe this image ###Assistant:"

    print("The conversation mode is {}".format(conv_mode))

    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="ms").unsqueeze(0)

    # text generator
    pipeline = TextGenerator(msmodel, max_new_tokens=1024, use_kv_cache=False)

    inputs = {}

    inputs["input_ids"] = ids
    inputs["images"] = image_tensor
    # actually this won't affect, the value would re-assigned in text_generation.line258
    inputs["return_key_value_cache"] = False
    output_ids = pipeline.generate(**inputs)

    # decode to text
    input_ids = ids
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")

    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

    outputs = outputs.strip()
    stop_str = conv.sep
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]

    outputs = outputs.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/path/to/model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, default="Describe this image")
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    outputs = eval_model(args)
    print(outputs)
