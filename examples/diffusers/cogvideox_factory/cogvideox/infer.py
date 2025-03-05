import argparse
import pathlib
import shutil

import numpy as np

import mindspore as ms
from mindspore.communication import get_rank

from mindone import diffusers
from mindone.diffusers.training_utils import init_distributed_device, set_seed
from mindone.diffusers.utils.mindspore_utils import randn_tensor
from mindone.models.modules.parallel import PARALLEL_MODULES
from mindone.trainers.zero import _prepare_network


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible inference.")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training.")
    parser.add_argument(
        "--mindspore_mode",
        type=int,
        default=1,
        choices=[0, 1],
        help="Forms of MindSpore programming execution, 0 means static graph mode and 1 means dynamic graph mode.",
    )
    parser.add_argument(
        "--jit_level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2"],
        help=(
            "Used to control the compilation optimization level, supports [O0, O1, O2]. The framework automatically "
            "selects the execution method. O0: All optimizations except those necessary for functionality are "
            "disabled, using an operator-by-operator execution method. O1: Enables common optimizations and automatic "
            "operator fusion optimizations, using an operator-by-operator execution method. This is an experimental "
            "optimization level, which is continuously being improved. O2: Enables extreme performance optimization, "
            "using a sinking execution method. Only effective when args.mindspore_mode is 0"
        ),
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="ZeRO-Stage in data parallel.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU. "
            "Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this "
            "argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_sequence_parallelism",
        type=str2bool,
        default=False,
        help="whether to enable sequence parallelism. Default is False",
    )
    parser.add_argument(
        "--sequence_parallel_shards",
        default=1,
        type=int,
        help="The number of shards in sequence parallel. Default is 1.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="THUDM/CogVideoX1.5-5b",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="prompt, if None, will set default prompt.",
    )
    parser.add_argument(
        "--transformer_ckpt_path",
        type=str,
        default=None,
        help="Path to the transformer checkpoint.",
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        default=None,
        help="Path to the transformer lora checkpoint.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="The height of the output video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1360,
        help="The width of the output video.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=77,
        help="The frame of the output video.",
    )
    parser.add_argument(
        "--max_sequence_length", type=int, default=224, help="Max sequence length of prompt embeddings."
    )
    parser.add_argument(
        "--npy_output_path",
        type=str,
        default=None,
        help="Path to save the inferred numpy array.",
    )
    parser.add_argument(
        "--video_output_path",
        type=str,
        default=None,
        help="Path to save the inferred video.",
    )
    return parser.parse_args()


def infer(args: argparse.Namespace) -> None:
    ms.set_context(mode=args.mindspore_mode, jit_config={"jit_level": args.jit_level})
    init_distributed_device(args)
    set_seed(args.seed)
    # enable_sequence_parallelism check
    enable_sequence_parallelism = getattr(args, "enable_sequence_parallelism", False)
    is_main_device = True
    if enable_sequence_parallelism:
        if args.world_size <= 1 or args.sequence_parallel_shards <= 1:
            print(
                f"world_size :{args.world_size}, "
                f"sequence_parallel_shards: {args.sequence_parallel_shards} "
                f"can not enable enable_sequence_parallelism=True."
            )
            enable_sequence_parallelism = False
        else:
            from acceleration import create_parallel_group, get_sequence_parallel_group

            create_parallel_group(sequence_parallel_shards=args.sequence_parallel_shards)
            ms.set_auto_parallel_context(enable_alltoall=True)
            sp_group = get_sequence_parallel_group()
            sp_rank = get_rank(sp_group)
            is_main_device = sp_rank == True
    dtype = (
        ms.float16 if args.mixed_precision == "fp16" else ms.bfloat16 if args.mixed_precision == "bf16" else ms.float32
    )
    if enable_sequence_parallelism:
        from training.models import AutoencoderKLCogVideoX_SP, CogVideoXTransformer3DModel_SP
        from transformers import AutoTokenizer

        from mindone.diffusers import CogVideoXDDIMScheduler
        from mindone.transformers import T5EncoderModel

        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            mindspore_dtype=dtype,
            revision=args.revision,
        )
        if args.zero_stage == 3:
            text_encoder = _prepare_network(text_encoder, "hccl_world_group", PARALLEL_MODULES)
        vae = AutoencoderKLCogVideoX_SP.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            mindspore_dtype=dtype,
            revision=args.revision,
            variant=args.variant,
        )
        transformer = CogVideoXTransformer3DModel_SP.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            mindspore_dtype=dtype,
            revision=args.revision,
            variant=args.variant,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        scheduler = CogVideoXDDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
            revision=args.revision,
        )

        pipe = diffusers.CogVideoXPipeline(tokenizer, text_encoder, vae, transformer, scheduler)
    else:
        pipe = diffusers.CogVideoXPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            mindspore_dtype=dtype,
            use_safetensors=True,
            revision=args.revision,
            variant=args.variant,
        )

    if args.transformer_ckpt_path:
        ckpt = ms.load_checkpoint(args.transformer_ckpt_path)
        processed_ckpt = {name[12:]: value for name, value in ckpt.items()}  # remove "transformer." prefix
        param_not_load, ckpt_not_load = ms.load_param_into_net(pipe.transformer, processed_ckpt)
        if param_not_load:
            raise RuntimeError(f"{param_not_load} was not loaded into net.")
        if ckpt_not_load:
            raise RuntimeError(f"{ckpt_not_load} was not loaded from checkpoint.")
        print("Successfully loaded transformer checkpoint.")
    if args.lora_ckpt_path:
        pipe.load_lora_weights(args.lora_ckpt_path, adapter_name=["cogvideox-lora"])
        pipe.set_adapters(["cogvideox-lora"], [1.0])
    if args.zero_stage == 3:
        pipe.transformer = _prepare_network(pipe.transformer, "hccl_world_group", PARALLEL_MODULES)

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
    prompt = prompt if args.prompt is None else args.prompt
    latents = None
    if enable_sequence_parallelism:
        shape = (
            1,
            (args.frame - 1) // pipe.vae_scale_factor_temporal + 1,
            pipe.transformer.config.in_channels,
            args.height // pipe.vae_scale_factor_spatial,
            args.width // pipe.vae_scale_factor_spatial,
        )
        latents = randn_tensor(shape, dtype=dtype)
        ms.mint.distributed.broadcast(latents, src=0, group=sp_group)
    video = pipe(
        prompt=prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frame,
        max_sequence_length=args.max_sequence_length,
        latents=latents,
    )[0][0]

    if is_main_device:
        if args.npy_output_path is not None:
            path = pathlib.Path(args.npy_output_path)
            if path.exists():
                shutil.rmtree(path)
            path.mkdir()
            index_max_length = len(str(len(video)))
            for index, image in enumerate(video):
                np.save(path / f"image_{str(index).zfill(index_max_length)}", np.array(image))
            print("Successfully saved the inferred numpy array.")

        if args.video_output_path is not None:
            diffusers.utils.export_to_video(video, args.video_output_path, fps=8)
            print("Successfully saved the inferred video.")


if __name__ == "__main__":
    args = parse_args()
    infer(args)
