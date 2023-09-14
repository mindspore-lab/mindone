import os
import json
import time
import mindspore as ms
import argparse
import soundfile as sf
from tqdm import tqdm

from models import build_pretrained_models, AudioDiffusion

from tango import Tango


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--original_args", type=str, default=None,
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model", type=str, default="../../../tango_ms_full.ckpt",
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="json file containing the test prompts for generation."
    )
    parser.add_argument(
        "--text_key", type=str, default="captions",
        help="Key containing the text in the json file."
    )
    parser.add_argument(
        "--test_references", type=str, default="data/audiocaps_test_references/subset",
        help="Folder containing the test reference wav files."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3,
        help="Guidance scale for classifier free guidance."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="How many samples per prompt.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend", device_id=device_id, max_device_memory="30GB")

    # set_random_seed(args.seed)

    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))

    vae_config = dotdict(json.load(open(train_args.pop("vae_model_config_path"))))
    stft_config = dotdict(json.load(open(train_args.pop("stft_model_config_path"))))
    main_config = train_args

    # Load Models #
    tango = Tango(
        vae_config,
        stft_config,
        main_config,
    )
    vae, stft, model = tango.vae, tango.stft, tango.model

    # Load Trained Weight #
    ckpt = ms.load_checkpoint(args.model)
    ms.load_param_into_net(tango, ckpt)

    # scheduler = DDPMScheduler.from_pretrained(train_args.scheduler_name, subfolder="scheduler")
    # evaluator = EvaluationHelper(16000, "cuda:0")
    # sampler = DPMSolverSampler(model, "dpmsolver", prediction_type="noise")

    # Load Data #
    if train_args.prefix:
        prefix = train_args.prefix
    else:
        prefix = ""

    text_prompts = [json.loads(line)[args.text_key] for line in open(args.test_file).readlines()]
    text_prompts = [prefix + inp for inp in text_prompts]

    # Generate #
    num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
    all_outputs = []

    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k: k+batch_size]
        print('text:', text)

        latents = model.inference(text, num_steps, guidance, num_samples, disable_progress=True)        
        print('latents:', latents.shape)
        mel = vae.decode_first_stage(latents)
        print('mel:', mel.shape)
        wave = vae.decode_to_waveform(mel)
        print('wave:', wave.shape)
        all_outputs += [item for item in wave]
        exit()

    # Save #
    exp_id = str(int(time.time()))
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    if num_samples == 1:
        output_dir = "outputs/{}_{}_steps_{}_guidance_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance)
        os.makedirs(output_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            sf.write("{}/output_{}.wav".format(output_dir, j), wav, samplerate=16000)

        result = dict()
        result["Steps"] = num_steps
        result["Guidance Scale"] = guidance
        result["Test Instances"] = len(text_prompts)

        result["scheduler_config"] = dict(scheduler.config)
        result["args"] = dict(vars(args))
        result["output_dir"] = output_dir

        with open("outputs/summary.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n\n")


if __name__ == "__main__":
    main()
