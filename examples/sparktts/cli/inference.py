import argparse
import logging
import os
from datetime import datetime
from time import time

import soundfile as sf
from cli.SparkTTS import SparkTTS

from mindone.utils import init_env
from mindone.utils.config import str2bool


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TTS inference.")

    parser.add_argument("--mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)")
    parser.add_argument("--debug", type=str2bool, default=False, help="Execute inference in debug mode.")
    parser.add_argument("--seed", type=int, default=42, help="Inference seed")
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports [“O0”, “O1”, “O2”]."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example/results",
        help="Directory to save generated audio files",
    )
    parser.add_argument("--text", type=str, required=True, help="Text for TTS generation")
    parser.add_argument("--prompt_text", type=str, help="Transcript of prompt audio")
    parser.add_argument(
        "--prompt_speech_path",
        type=str,
        help="Path to the prompt audio file",
    )
    parser.add_argument("--gender", choices=["male", "female"])
    parser.add_argument("--pitch", choices=["very_low", "low", "moderate", "high", "very_high"])
    parser.add_argument("--speed", choices=["very_low", "low", "moderate", "high", "very_high"])
    return parser.parse_args()


def run_tts(args):
    device_id, rank_id, device_num = init_env(
        args.mode,
        debug=args.debug,
        seed=args.seed,
        jit_level=args.jit_level,
    )

    """Perform TTS inference and save the generated audio."""
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Saving audio to: {args.save_dir}")

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize the model
    model = SparkTTS(args.model_dir)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(args.save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Perform inference and save the output audio
    wav = model.inference(
        args.text,
        args.prompt_speech_path,
        prompt_text=args.prompt_text,
        gender=args.gender,
        pitch=args.pitch,
        speed=args.speed,
    )
    sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio saved at: {save_path}")


if __name__ == "__main__":
    start = time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parse_args()
    run_tts(args)
    print(f"Time cost: {time() - start:.2f}s")
