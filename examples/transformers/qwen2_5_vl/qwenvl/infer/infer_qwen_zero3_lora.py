import argparse
import os
import sys
from pathlib import Path

from PIL import Image

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist
import mindspore.nn as nn
from mindspore.communication import GlobalComm

project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from mindone.peft import PeftConfig, PeftModel, load_peft_weights, set_peft_model_state_dict
from mindone.trainers.zero import prepare_network
from mindone.transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    dist.init_process_group()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)

    parser = argparse.ArgumentParser(description="inference the checkpoint from zero-3")
    parser.add_argument("--model_name_or_path", default="Qwen2.5-VL-72B-Instruct", help="model name")
    parser.add_argument("--lora_root", help="LoRA checkpoint root path")
    parser.add_argument("--image_path", required=True, help="image path")
    parser.add_argument("--prompt", required=True, nargs="+", help="prompt input")
    parser.add_argument("--max_pixels", default=578 * 28 * 28, type=int, help="max pixel numbers")
    parser.add_argument("--min_pixels", default=16 * 28 * 28, type=int, help="min pixel numbers")
    args = parser.parse_args()
    args.prompt = " ".join(args.prompt)

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    if local_rank == 0:
        print("USER:", args.prompt)

    with nn.no_init_parameters():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path, mindspore_dtype=ms.bfloat16, attn_implementation="flash_attention_2"
        )
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )

    # add lora
    if args.lora_root:
        lora_root_rank = os.path.join(args.lora_root, f"rank_{local_rank}")
        latest_checkpoint_dir = sorted(os.listdir(lora_root_rank))[-1]
        latest_checkpoint_dir = os.path.join(lora_root_rank, latest_checkpoint_dir)
        sharded_peft_weight = load_peft_weights(latest_checkpoint_dir)

        # combine the sharded lora weight
        # FIXME: we simply assume the lora weight is fully sharded. May not be true in case lora_rank < word size during training.
        full_peft_weight = dict()
        peft_weight_names = sorted(sharded_peft_weight.keys())
        for name in peft_weight_names:
            sharded_weight = sharded_peft_weight[name]
            full_tensor = mint.zeros(
                (sharded_weight.shape[0] * world_size, *sharded_weight.shape[1:]), dtype=sharded_weight.dtype
            )
            dist.all_gather_into_tensor(full_tensor, sharded_weight, group=GlobalComm.WORLD_COMM_GROUP)
            # FIXME: fix the name inconsistency
            name = name.replace(".net.weight", ".weight")
            full_peft_weight[name] = ms.Parameter(full_tensor)

        # load lora weight into net
        peft_config = PeftConfig.from_pretrained(latest_checkpoint_dir)
        model.is_gradient_checkpointing = False
        model = PeftModel(model, peft_config)
        result = set_peft_model_state_dict(model, full_peft_weight)
        assert len(result["unexpected_keys"]) == 0, result

    # shard across devices
    model = prepare_network(model, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
    dist.barrier()

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
