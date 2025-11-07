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
from pathlib import Path

from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM

from mindspore.nn.utils import no_init_parameters


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

    parser.add_argument("--reproduce", action="store_true", help="Whether to reproduce the results")
    return parser.parse_args()


def set_reproducibility(enable, global_seed=None, benchmark=None):
    import mindspore as ms

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
        import os

        os.environ["HCCL_DETERMINSTIC"] = "True"

    # Use deterministic algorithms in MindSpore
    ms.set_deterministic(True)


def main(args):
    if args.reproduce:
        set_reproducibility(args.reproduce, global_seed=args.seed)

    if not args.prompt:
        raise ValueError("Prompt is required")
    if not Path(args.model_id).exists():
        raise ValueError(f"Model path {args.model_id} does not exist")

    kwargs = dict(
        attn_implementation=args.attn_impl,
        mindspore_dtype="auto",
        # device_map="auto",
        moe_impl=args.moe_impl,
    )
    with no_init_parameters():
        model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **kwargs)
        model.load_tokenizer(args.model_id)

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

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.save)
    print(f"Image saved to {args.save}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
