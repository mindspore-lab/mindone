# pllavarun.py
import time
from argparse import ArgumentParser

from tasks.eval.eval_utils import load_video
from tasks.eval.model_utils import load_pllava, pllava_answer

import mindspore as ms

ms.set_context(jit_config=dict(jit_level="O1"))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="./models/pllava-7b")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--video", type=str, default="./examples/-0og5HrzhpY_0.mp4", help="Path to the video file")
    parser.add_argument("--question", type=str, default="What is shown in this video?")
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading videos...")
    frames = load_video(args.video, args.num_frames)  # returns a list of PIL images

    print("Initializing PLLaVA model...")
    model, processor = load_pllava(
        args.pretrained_model_name_or_path,
        args.num_frames,
        (args.num_frames, 12, 12),
    )

    SYSTEM = """You are a powerful Video Magic ChatBot, a large vision-language assistant.
    You are able to understand the video content that the user provides and assist the user in a video-language related task.
    The user might provide you with the video and maybe some extra noisy information to help you out or ask you a question.
    Make use of the information in a proper way to be competent for the job.
    ### INSTRUCTIONS:
    1. Follow the user's instruction.
    2. Be critical yet believe in yourself.
    """

    prompt = SYSTEM + "USER: " + args.question + " </s> USER:<image> ASSISTANT:"

    output_token, output_text = pllava_answer(
        model,
        processor,
        [frames],
        prompt,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        num_beams=1,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        temperature=1.0,
    )

    if args.benchmark:
        # run again for benchmark
        start_time = time.time()
        output_token, output_text = pllava_answer(
            model,
            processor,
            [frames],
            prompt,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
        )
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Tokens length: {output_token.shape[1]}")
        print(f"Time elapsed: {time_elapsed:.4f}")
        print(f"tokens per second: {(output_token.shape[1] / time_elapsed):.4f}")

    cleaned_output = output_text.split("ASSISTANT: ", 1)[1]
    print(f"Response: {cleaned_output}")  # cleaned response


if __name__ == "__main__":
    main()
