#!/usr/bin/env python
import argparse
import copy
import json
import logging
import os
import time
from typing import Any, Dict

from llava.model.llava_next import LlavaNextForConditionalGeneration
from llava.pipeline import TextGenerator
from PIL import Image
from transformers import LlavaNextProcessor

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLaVa-Next prediction", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", default="models/llava-v1.6-mistral-7b-hf", help="Path of the model root")
    parser.add_argument("--input_image", default="assets/llava_v1_5_radar.jpg", help="Input Image")
    parser.add_argument("--prompt", default="What is shown in this image?", help="Input Prompt")
    parser.add_argument("--benchmark", action="store_true", help="Do performance benchmark")
    args = parser.parse_args()
    return args


def load_network(config: Dict[str, Any], ckpt_path: str) -> nn.Cell:
    config_ = copy.copy(config)
    config_["vision_config"]["hidden_size"] = 1024
    config_["text_config"]["hidden_size"] = 4096

    vision_config = config_.pop("vision_config")
    text_config = config_.pop("text_config")
    network = LlavaNextForConditionalGeneration(
        vision_config,
        text_config,
        dtype=ms.float16,
        attn_implementation="flash_attention",
        language_model_input_method="padding",  # dynamic
        **config_,
    )
    ms.load_checkpoint(ckpt_path, net=network, strict_load=True)
    return network


def main():
    args = parse_args()

    ms.set_context(jit_config=dict(jit_level="O1"))

    with open(os.path.join(args.model_path, "config.json"), "r") as f:
        config = json.load(f)

    model_path = os.path.join(args.model_path, "model.ckpt")
    logging.info(f"Loading the network from {model_path}")
    network = load_network(config, model_path)

    # prepare image and text prompt, using the appropriate prompt template
    logging.info(f"Loading the processer from {args.model_path}")
    processor = LlavaNextProcessor.from_pretrained(args.model_path)

    image = Image.open(args.input_image)
    logging.info(f"Input Image: {args.input_image}")

    input_prompt = f"[INST] <image>\n{args.prompt} [/INST]"
    logging.info(f"Input Prompt: {input_prompt}")

    inputs = processor(input_prompt, image, return_tensors="np")
    inputs = {k: Tensor(v) for k, v in inputs.items()}

    # autoregressively complete prompt
    trials = 2 if args.benchmark else 1
    logging.info("Starting inference...")
    for trial in range(trials):
        logging.info(f"Trial: {trial}")
        pipeline = TextGenerator(network, max_new_tokens=100, use_kv_cache=True)
        start = time.time()
        output = pipeline.generate(**inputs)
        end = time.time()
        logging.info(f"Time Taken: {end-start:.3f}, Tokens/Second: {len(output[0]) / (end - start):.1f}")

    print(processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s: %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    main()
