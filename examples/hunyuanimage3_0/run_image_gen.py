# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Optional

from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM

import mindspore as ms
import mindspore.mint.distributed as dist
import mindspore.nn as nn
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool

# import logging
# import sys
# import types


# from mindspore.nn.utils import no_init_parameters


# from hunyuan_image_3.distributed.util import init_distributed_group  # for sp
# from hunyuan_image_3.distributed.zero import free_model, shard_model  # free_model -> offload model


def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 locally")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to run")
    parser.add_argument("--model-id", type=str, default="./HunyuanImage-3", help="Path to the model")
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="flash_attention_2",
        choices=["sdpa", "flash_attention_2"],
        help="Attention implementation. 'sdpa' is not supported yet.",
    )
    parser.add_argument(
        "--moe-impl",
        type=str,
        default="eager",
        choices=["eager", "flashinfer"],
        help="MoE implementation. 'flashinfer' requires FlashInfer to be installed.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Use None for random seed.")
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument(
        "--image-size",
        type=str,
        default="auto",
        help="'auto' means image size is determined by the model. Alternatively, it can be in the "
        "format of 'HxW' or 'H:W', which will be aligned to the set of preset sizes.",
    )
    parser.add_argument(
        "--use-system-prompt",
        type=str,
        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"],
        help="Use system prompt. 'None' means no system prompt; 'dynamic' means the system prompt is "
        "determined by --bot-task; 'en_vanilla', 'en_recaption', 'en_think_recaption' are "
        "three predefined system prompts; 'custom' means using the custom system prompt. When "
        "using 'custom', --system-prompt must be provided. Default to load from the model "
        "generation config.",
    )
    parser.add_argument(
        "--system-prompt", type=str, help="Custom system prompt. Used when --use-system-prompt " "is 'custom'."
    )
    parser.add_argument(
        "--bot-task",
        type=str,
        choices=["image", "auto", "think", "recaption"],
        help="Type of task for the model. 'image' for direct image generation; 'auto' for text "
        "generation; 'think' for think->re-write->image; 'recaption' for re-write->image."
        "Default to load from the model generation config.",
    )
    parser.add_argument("--save", type=str, default="image.png", help="Path to save the generated image")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose level")
    parser.add_argument("--rewrite", type=int, default=0, help="Whether to rewrite the prompt with DeepSeek")
    parser.add_argument(
        "--sys-deepseek-prompt",
        type=str,
        choices=["universal", "text_rendering"],
        default="universal",
        help="System prompt for rewriting the prompt",
    )
    parser.add_argument("--ulysses_size", type=int, default=4, help="The size of the ulysses parallelism in the model.")
    parser.add_argument("--use_zero3", action="store_true", default=True, help="Whether to use ZeRO3 for the model")
    parser.add_argument("--reproduce", action="store_true", default=True, help="Whether to reproduce the results")
    # mindspore args
    parser.add_argument("--ms-mode", type=int, default=1, help="0 graph, 1 pynative")
    parser.add_argument(
        "--jit-level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2"],
        help="determine graph optimizatio/fusion level. only effective when in graph mode",
    )
    parser.add_argument(
        "--enable-ms-amp",
        type=str2bool,
        default=False,
        help="enable mindspore auto mixed precision. if False, use mixed precision set in the network definition",
    )
    parser.add_argument(
        "--amp-level",
        type=str,
        choices=["O0", "O1", "O2", "auto"],
        default="auto",
        help="determine auto mixed precision level. only effective when enable_ms_amp is True",
    )
    parser.add_argument(
        "--jit-syntax-level", default="lax", choices=["strict", "lax"], help="Set jit syntax level: strict or lax"
    )
    parser.add_argument(
        "--max-device-memory",
        type=str,
        default="59GB",
        help="e.g. `30GB` for 910, `59GB` for Ascend Atlas 800T A2 machines",
    )
    return parser.parse_args()


def init_distributed_group() -> None:
    """r initialize sequence parallel group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")


def shard_model(
    model: nn.Cell,
    device_id: Optional[int] = None,
    param_dtype: ms.dtype = ms.bfloat16,
    reduce_dtype: ms.dtype = ms.float32,
    buffer_dtype: ms.dtype = ms.float32,
    process_group: Optional[Any] = None,
    sharding_strategy: Optional[Any] = None,
    sync_module_states: bool = True,
) -> nn.Cell:
    model = prepare_network(model, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
    return model


def configure_model(
    self,
    model: HunyuanImage3ForCausalMM,
    use_sp: bool,
    dit_zero3: bool,
    shard_fn: Callable[[nn.Cell], nn.Cell],
    convert_model_dtype: bool,
) -> nn.Cell:
    """
    Configures a model object. This includes setting evaluation modes,
    applying distributed parallel strategy, and handling device placement.

    Args:
        model (mindspore.nn.Cell):
            The model instance to configure.
        use_sp (`bool`):
            Enable distribution strategy of sequence parallel.
        dit_zero3 (`bool`):
            Enable ZeRO3 sharding for DiT model.
        shard_fn (callable):
            The function to apply ZeRO3 sharding.
        convert_model_dtype (`bool`):
            Convert DiT model parameters dtype to 'config.param_dtype'.

    Returns:
        mindspore.nn.Cell:
            The configured model.
    """
    model.set_train(False)
    for param in model.trainable_params():
        param.requires_grad = False

    if use_sp:
        pass
        # for block in model.blocks:
        #     block.self_attn.construct = types.MethodType(sp_attn_forward, block.self_attn)
        # model.construct = types.MethodType(sp_dit_forward, model)

    if dist.is_initialized():
        dist.barrier()

    if dit_zero3:
        model = shard_fn(model)

    if convert_model_dtype:
        model.to(self.param_dtype)

    return model


def set_reproducibility(enable, global_seed=None):
    if enable:
        # Configure the seed for reproducibility
        import random

        import numpy as np

        if global_seed is not None:
            # Seed the RNG for Python
            random.seed(global_seed)
            # Seed the RNG for Numpy
            np.random.seed(global_seed)
            # Seed the RNG for all devices (both CPU and NPU)
            ms.manual_seed(global_seed)

    # Set following debug environment variable
    if enable:
        os.environ["HCCL_DETERMINSTIC"] = "True"

    # Use deterministic algorithms in MindSpore
    ms.set_deterministic(True)


def main(args):
    dist.init_process_group()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    ms.launch_blocking()

    local_rank = dist.get_rank()
    # world_size = dist.get_world_size()

    # if args.ulysses_size > 1:
    #     assert args.ulysses_size == world_size, "The number of ulysses_size should be equal to the world size."
    #     # assert args.ulysses_size % 4 == 0?
    #     init_distributed_group()

    if args.reproduce:
        set_reproducibility(args.reproduce, global_seed=args.seed)

    if not args.prompt:
        raise ValueError("Prompt is required")
    if not Path(args.model_id).exists():
        raise ValueError(f"Model path {args.model_id} does not exist")

    dtype = ms.bfloat16
    kwargs = dict(
        attn_implementation=args.attn_impl,
        mindspore_dtype=dtype,
        # device_map="auto",
        moe_impl=args.moe_impl,
    )
    with nn.no_init_parameters():
        model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **kwargs)
    model.load_tokenizer(args.model_id)

    # shard across devices
    # model = configure_model(
    #         model=model,
    #         use_sp=(args.ulysses_size > 1),
    #         use_zero3=args.use_zero3,
    #         shard_fn=shard_model,
    #         convert_model_dtype=False,
    #     )

    # shard across devices
    if args.use_zero3:
        print("Use zero3")
        model = prepare_network(model, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
        dist.barrier()

    if args.enable_ms_amp and dtype != ms.float32:
        print(f"Use MS auto mixed precision for model, amp_level: {args.amp_level}")
        if hasattr(model, "vae") and model.vae is not None:
            model.vae = auto_mixed_precision(
                model.vae, amp_level="O2", dtype=dtype, custom_fp32_cells=[ms.mint.nn.GroupNorm]
            )
            print("Use MS auto mixed precision for model.vae, amp_level: O2")

        if args.amp_level == "auto":
            ms.amp.auto_mixed_precision(model, amp_level=args.amp_level, dtype=dtype)
        else:
            # OOM risk
            whitelist_ops = [ms.mint.nn.GroupNorm]
            print("custom fp32 cell for vae: ", whitelist_ops)
            model = auto_mixed_precision(model, amp_level=args.amp_level, dtype=dtype, custom_fp32_cells=whitelist_ops)

    # Rewrite prompt with DeepSeek
    if args.rewrite:
        raise NotImplementedError("Prompt rewrite is not supported yet.")

    image = model.generate_image(
        prompt=args.prompt,
        seed=args.seed,
        image_size=args.image_size,
        use_system_prompt=args.use_system_prompt,
        system_prompt=args.system_prompt,
        bot_task=args.bot_task,
        diff_infer_steps=args.diff_infer_steps,
        verbose=args.verbose,
        stream=True,
    )

    # save picture
    if local_rank == 0:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.save)
        print(f"Image saved to {args.save}")

    # clean workspace
    ms.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    print("Finished.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
