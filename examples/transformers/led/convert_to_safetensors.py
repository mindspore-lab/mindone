#!/usr/bin/env python
# -*- coding: utf-8 -*-

# download `convert.py` script from https://huggingface.co/spaces/safetensors/convert/blob/main/convert.py

# example usage:
# python convert_to_safetensors.py \
# --model_path ~/.cache/modelscope/hub/models/allenai/led-base-16384/pytorch_model.bin \
# --output_path ~/.cache/modelscope/hub/models/allenai/led-base-16384/model.safetensors

try:
    from convert import convert_file
except ImportError:
    print(
        "Please download `convert.py` script from https://huggingface.co/spaces/safetensors/convert/blob/main/convert.py"
    )
    exit(1)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="path/to/pytorch_model.bin")
    parser.add_argument("--output_path", type=str, default="path/to/model.safetensors")
    args = parser.parse_args()

    convert_file(
        args.model_path,
        args.output_path,
        discard_names=[
            "decoder.embed_tokens.weight",
            "encoder.embed_tokens.weight",
        ],  # remove the two embedding tables, because they are shared with `led.shared.weightns`
    )
