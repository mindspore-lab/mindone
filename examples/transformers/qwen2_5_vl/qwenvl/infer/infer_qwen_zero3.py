import argparse
import os
import sys
from pathlib import Path

from PIL import Image

import mindspore as ms
import mindspore.mint.distributed as dist
import mindspore.nn as nn
from mindspore.communication import GlobalComm

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from mindone.trainers.zero import prepare_network
from mindone.transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    dist.init_process_group()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)

    parser = argparse.ArgumentParser(description="inference the checkpoint from zero-3")
    parser.add_argument("--model_name_or_path", default="Qwen2.5-VL-3B-Instruct", help="model name")
    parser.add_argument("--ckpt_root", help="trained checkpoint root path")
    parser.add_argument("--image_path", required=True, help="image path")
    parser.add_argument("--prompt", required=True, nargs="+", help="prompt input")
    parser.add_argument("--max_pixels", default=578 * 28 * 28, type=int, help="max pixel numbers")
    parser.add_argument("--min_pixels", default=16 * 28 * 28, type=int, help="min pixel numbers")
    args = parser.parse_args()
    args.prompt = " ".join(args.prompt)

    local_rank = dist.get_rank()

    if local_rank == 0:
        print("USER:", args.prompt)

    with nn.no_init_parameters():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path, mindspore_dtype=ms.bfloat16, attn_implementation="flash_attention_2"
        )
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )

    # shard across devices
    model = prepare_network(model, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
    dist.barrier()

    # loaded the sharded checkpoint
    if args.ckpt_root:
        checkpoint_root_rank = os.path.join(args.ckpt_root, f"rank_{local_rank}")
        latest_checkpoint_dir = sorted(os.listdir(checkpoint_root_rank))[-1]
        latest_checkpoint_dir = os.path.join(checkpoint_root_rank, latest_checkpoint_dir)
        latest_checkpoint_path = os.path.join(latest_checkpoint_dir, "model.safetensors")

        trained_param_dict = ms.load_checkpoint(latest_checkpoint_path, format="safetensors")
        # FIXME: fix the name inconsistency
        fixed_trained_param_dict = dict()
        for k, v in trained_param_dict.items():
            k = k.replace(".net.weight", ".weight").replace(".net.bias", ".bias")
            fixed_trained_param_dict[k] = v
        _, ckpt_not_load = ms.load_param_into_net(model, fixed_trained_param_dict, strict_load=True)
        assert len(ckpt_not_load) == 0, ckpt_not_load

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    image = Image.open(args.image_path)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="ms")
    dist.barrier()

    generate_ids = model.generate(**inputs, max_new_tokens=128)
    result = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False, do_sample=False
    )[0]
    if local_rank == 0:
        print(result)


if __name__ == "__main__":
    main()
