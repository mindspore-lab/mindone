import argparse
import json
import os
import time

from audioio import write
from audioldm.latent_diffusion.dpm_solver import DPMSolverSampler
from tango import Tango
from tqdm import tqdm

import mindspore as ms


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument("--config_path", type=str, default="configs", help="Path containing config.json")
    parser.add_argument(
        "--ckpt", type=str, default="../../../tango_ms_full.ckpt", help="Path for saved model bin file."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/test_audiocaps_subset.json",
        help="json file containing the test prompts for generation.",
    )
    parser.add_argument("--text_key", type=str, default="captions", help="Key containing the text in the json file.")
    parser.add_argument(
        "--test_references",
        type=str,
        default="data/audiocaps_test_references/subset",
        help="Folder containing the test reference wav files.",
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
        default=8,
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

    # Load Models #
    tango = Tango(args.config_path)

    # Load Trained Weight #
    ckpt = ms.load_checkpoint(args.ckpt)
    ms.load_param_into_net(tango, ckpt)

    sampler = DPMSolverSampler(tango.model, "dpmsolver", prediction_type="v")

    # text_prompts = [json.loads(line)[args.text_key] for line in open(args.test_file).readlines()]
    text_prompts = [
        "A machine is making clicking sound as people talk in the background",
        "A machine is making clicking sound",
    ]

    # Generate #
    num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
    all_outputs = []

    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k : k + batch_size]
        print("text:", text)
        latents = tango.model.inference(
            text,
            num_steps,
            guidance,
            num_samples,
            disable_progress=True,
            sampler=sampler,
            padding=True,
            truncation=True,
        )
        print("latents:", latents.shape)
        mel = tango.vae.decode_first_stage(latents)
        print("mel:", mel.shape)
        wave = tango.vae.decode_to_waveform(mel)
        print("wave:", wave.shape)
        all_outputs += [item for item in wave]
        break

    # Save #
    exp_id = str(int(time.time()))
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    if num_samples == 1:
        output_dir = "outputs/{}_{}_steps_{}_guidance_{}".format(
            exp_id, "_".join(args.ckpt.split("/")[1:-1]), num_steps, guidance
        )
        os.makedirs(output_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            write("{}/output_{}.wav".format(output_dir, j), wav, sr=16000)

        result = dict()
        result["Steps"] = num_steps
        result["Guidance Scale"] = guidance
        result["Test Instances"] = len(text_prompts)

        # result["scheduler_config"] = dict(scheduler.config)
        result["args"] = dict(vars(args))
        result["output_dir"] = output_dir

        with open("outputs/summary.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n\n")


if __name__ == "__main__":
    main()
