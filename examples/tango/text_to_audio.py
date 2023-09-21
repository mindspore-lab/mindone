import argparse
import os

import IPython
from audioio import write
from audioldm.latent_diffusion.dpm_solver import DPMSolverSampler
from tango import Tango

import mindspore as ms


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--prompts",
        type=str,
        default="A dog is barking",
        nargs="+",
    )
    parser.add_argument("--config_path", type=str, default="configs", help="Path containing config.json")
    parser.add_argument(
        "--ckpt", type=str, default="../../../tango_ms_full.ckpt", help="Path for saved model bin file."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="How many denoising steps for generation.",
    )
    parser.add_argument("--guidance", type=float, default=3, help="Guidance scale for classifier free guidance.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="How many samples per prompt.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(
        mode=ms.context.PYNATIVE_MODE, device_target="Ascend", device_id=device_id, max_device_memory="30GB"
    )

    # set_random_seed(args.seed)

    tango = Tango(args.config_path)
    ckpt = ms.load_checkpoint(args.ckpt)
    ms.load_param_into_net(tango, ckpt)

    sampler = DPMSolverSampler(tango.model, "dpmsolver", prediction_type="v")

    if type(args.prompts) is str:
        args.prompts = [args.prompts]
    for prompt in args.prompts:
        print("prompt:", prompt)
        audio = tango.generate(
            prompt, sampler=sampler, steps=args.num_steps, guidance=args.guidance, samples=args.num_samples
        )
        print("audio:", audio.shape)
        write(f"{prompt}.wav", audio, sr=16000)
        IPython.display.Audio(data=audio, rate=16000)


if __name__ == "__main__":
    main()
