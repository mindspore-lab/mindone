import argparse
import re

from mindone.utils.config import str2bool

from .constants import PRECISIONS, PROMPT_TEMPLATE, TEXT_ENCODER_PATH, TOKENIZER_PATH, VAE_PATH
from .modules.models import HUNYUAN_VIDEO_CONFIG


def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_inference_args(parser)
    parser = add_parallel_args(parser)

    args = parser.parse_args(namespace=namespace)
    args = sanity_check_args(args)

    return args


def add_network_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="HunyuanVideo network args")

    # Main model
    group.add_argument(
        "--model",
        type=str,
        choices=list(HUNYUAN_VIDEO_CONFIG.keys()),
        default="HYVideo-T/2-cfgdistill",
    )
    group.add_argument(
        "--latent-channels",
        type=str,
        default=16,
        help="Number of latent channels of DiT. If None, it will be determined by `vae`. If provided, "
        "it still needs to match the latent channels of the VAE model.",
    )
    group.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=PRECISIONS,
        help="Precision mode. Options: fp32, fp16, bf16. Applied to the backbone model and optimizer.",
    )

    # RoPE
    group.add_argument("--rope-theta", type=int, default=256, help="Theta used in RoPE.")
    return parser


def add_extra_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Extra models args, including vae, text encoders and tokenizers)")

    # - VAE
    group.add_argument(
        "--vae",
        type=str,
        default="884-16c-hy",
        choices=list(VAE_PATH),
        help="Name of the VAE model.",
    )
    group.add_argument(
        "--vae-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the VAE model.",
    )
    group.add_argument(
        "--vae-tiling",
        default=True,
        type=str2bool,
        help="Enable tiling for the VAE model to save GPU memory.",
    )

    group.add_argument(
        "--text-encoder",
        type=str,
        default="llm",
        choices=list(TEXT_ENCODER_PATH),
        help="Name of the text encoder model.",
    )
    group.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the text encoder model.",
    )
    group.add_argument(
        "--text-states-dim",
        type=int,
        default=4096,
        help="Dimension of the text encoder hidden states.",
    )
    group.add_argument("--text-len", type=int, default=256, help="Maximum length of the text input.")
    group.add_argument(
        "--tokenizer",
        type=str,
        default="llm",
        choices=list(TOKENIZER_PATH),
        help="Name of the tokenizer model.",
    )
    group.add_argument(
        "--prompt-template",
        type=str,
        default="dit-llm-encode",
        choices=PROMPT_TEMPLATE,
        help="Image prompt template for the decoder-only text encoder model.",
    )
    group.add_argument(
        "--prompt-template-video",
        type=str,
        default="dit-llm-encode-video",
        choices=PROMPT_TEMPLATE,
        help="Video prompt template for the decoder-only text encoder model.",
    )
    group.add_argument(
        "--hidden-state-skip-layer",
        type=int,
        default=2,
        help="Skip layer for hidden states.",
    )
    group.add_argument(
        "--apply-final-norm",
        action="store_true",
        help="Apply final normalization to the used text encoder hidden states.",
    )

    # - CLIP
    group.add_argument(
        "--text-encoder-2",
        type=str,
        default="clipL",
        choices=list(TEXT_ENCODER_PATH),
        help="Name of the second text encoder model.",
    )
    group.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the second text encoder model.",
    )
    group.add_argument(
        "--text-states-dim-2",
        type=int,
        default=768,
        help="Dimension of the second text encoder hidden states.",
    )
    group.add_argument(
        "--tokenizer-2",
        type=str,
        default="clipL",
        choices=list(TOKENIZER_PATH),
        help="Name of the second tokenizer model.",
    )
    group.add_argument(
        "--text-len-2",
        type=int,
        default=77,
        help="Maximum length of the second text input.",
    )

    return parser


def add_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule args")

    group.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )

    # Flow Matching
    group.add_argument(
        "--flow-shift",
        type=float,
        default=7.0,
        help="Shift factor for flow matching schedulers.",
    )
    group.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    group.add_argument(
        "--flow-solver",
        type=str,
        default="euler",
        help="Solver for flow matching.",
    )
    group.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help="Use linear quadratic schedule for flow matching."
        "Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    group.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    return parser


def add_inference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference args")

    # ======================== Model loads ========================
    group.add_argument(
        "--model-base",
        type=str,
        default="ckpts",
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--dit-weight",
        type=str,
        default="ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        help="Path to the HunyuanVideo model. If None, search the model in the args.model_root."
        "1. If it is a file, load the model directly. It supports torch checkpoint (.pt) and mindspore checkpoint (.ckpt)."
        "2. If it is a directory, search the model in the directory. Support two types of models: "
        "1) named `pytorch_model_*.pt`"
        "2) named `*_model_states.pt`, where * can be `mp_rank_00`.",
    )
    group.add_argument(
        "--zero-stage",
        type=int,
        default=None,
        help="If specified and zero-stage is 3, will load each checkpoint separately in each rank."
        "If specified and zero-stage is 3, expect dit-weight is a directory containing sub-folders like rank_x, "
        "In each sub-folder, there is a checkpoint file.",
    )
    group.add_argument(
        "--model-resolution",
        type=str,
        default="540p",
        choices=["540p", "720p"],
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    group.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )

    # ======================== Inference general setting ========================
    group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference and evaluation.",
    )
    group.add_argument(
        "--infer-steps",
        type=int,
        default=50,
        help="Number of denoising steps for inference.",
    )
    group.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )
    group.add_argument(
        "--save-path",
        type=str,
        default="./results",
        help="Path to save the generated samples.",
    )
    group.add_argument(
        "--save-path-suffix",
        type=str,
        default="",
        help="Suffix for the directory of saved samples.",
    )
    group.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Suffix for the names of saved samples.",
    )
    group.add_argument(
        "--num-videos",
        type=int,
        default=1,
        help="Number of videos to generate for each prompt.",
    )
    # ---sample size---
    group.add_argument(
        "--video-size",
        type=int,
        nargs="+",
        default=(720, 1280),
        help="Video size for training. If a single value is provided, it will be used for both height "
        "and width. If two values are provided, they will be used for height and width "
        "respectively.",
    )
    group.add_argument(
        "--video-length",
        type=int,
        default=129,
        help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1",
    )
    # --- prompt ---
    group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="A single prompt string or a path to a .txt file containing multiple prompts. "
        "If a .txt file is provided, each line should contain one prompt.",
    )
    group.add_argument(
        "--seed-type",
        type=str,
        default="auto",
        choices=["file", "random", "fixed", "auto"],
        help="Seed type for evaluation. If file, use the seed from the CSV file. If random, generate a "
        "random seed. If fixed, use the fixed seed given by `--seed`. If auto, `csv` will use the "
        "seed column if available, otherwise use the fixed `seed` value. `prompt` will use the "
        "fixed `seed` value.",
    )
    group.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

    # Classifier-Free Guidance
    group.add_argument("--neg-prompt", type=str, default=None, help="Negative prompt for sampling.")
    group.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier free guidance scale.")
    group.add_argument(
        "--embedded-cfg-scale",
        type=float,
        default=6.0,
        help="Embeded classifier free guidance scale.",
    )
    group.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )

    group.add_argument("--use-fp8", action="store_true", help="Enable use fp8 for inference acceleration.")
    group.add_argument("--attn-mode", type=str, default="flash", help="vanilla or flash")
    group.add_argument(
        "--use-conv2d-patchify",
        type=str2bool,
        default=True,
        help="use conv2d equivalence in PatchEmbed instead of conv3d",
    )
    group.add_argument("--output-type", type=str, default="pil", help="pil or latent")
    group.add_argument("--latent-noise-path", type=str, help="path to npy containing latent noise")
    group.add_argument(
        "--text-embed-path",
        type=str,
        default=None,
        help="A single .npz file path or a path to a .txt file containing multiple .npz file paths. "
        "If a .txt file is provided, each line should contain one .npz file path. "
        "This argument is required if `prompt` is a .txt file.",
    )
    # mindspore args
    group.add_argument("--ms-mode", type=int, default=0, help="0 graph, 1 pynative")
    group.add_argument(
        "--jit-level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2"],
        help="determine graph optimizatio/fusion level. only effective when in graph mode",
    )
    group.add_argument(
        "--enable-ms-amp",
        type=str2bool,
        default=False,
        help="enable mindspore auto mixed precision. if False, use mixed precision set in the network definition",
    )
    group.add_argument(
        "--amp-level",
        type=str,
        choices=["O0", "O1", "O2"],
        default="O2",
        help="determine auto mixed precision level. only effective when enable_ms_amp is True",
    )
    group.add_argument(
        "--jit-syntax-level", default="lax", choices=["strict", "lax"], help="Set jit syntax level: strict or lax"
    )
    group.add_argument("--max-device-memory", type=str, default="59GB", help="e.g. `30GB` for 910a, `59GB` for 910b")
    return parser


def add_parallel_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Parallel args")

    group.add_argument(
        "--use-parallel",
        type=str2bool,
        default=False,
        help="enable mindspore parallel training.",
    )
    group.add_argument(
        "--parallel-mode",
        type=str,
        default="data",
        choices=["data", "optim", "semi", "zero"],
        help="parallel mode: data, optim, zero",
    )
    group.add_argument("--sp-size", type=int, default=1, help="For sequence parallel")
    # ======================== Model loads ========================
    group.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Ulysses degree.",
    )
    group.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Ulysses degree.",
    )

    return parser


def sanity_check_args(args):
    # VAE channels
    vae_pattern = r"\d{2,3}-\d{1,2}c-\w+"
    if not re.match(vae_pattern, args.vae):
        raise ValueError(f"Invalid VAE model: {args.vae}. Must be in the format of '{vae_pattern}'.")
    vae_channels = int(args.vae.split("-")[1][:-1])
    if args.latent_channels is None:
        args.latent_channels = vae_channels
    if vae_channels != args.latent_channels:
        raise ValueError(f"Latent channels ({args.latent_channels}) must match the VAE channels ({vae_channels}).")
    return args
