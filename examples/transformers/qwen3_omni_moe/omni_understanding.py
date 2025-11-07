import argparse
from functools import partial

import numpy as np
import soundfile as sf
from qwen_omni_utils import process_mm_info

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network
from mindone.transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


def generate(args):
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_name,
        mindspore_dtype=ms.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # use zero3 parallel
    shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
    model.thinker = shard_fn(model.thinker)
    model.talker = shard_fn(model.talker)

    min_pixels = 128 * 8 * 8
    max_pixels = 768 * 8 * 8
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "audio", "audio": args.audio},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]

    # Set whether to use audio in video
    USE_AUDIO_IN_VIDEO = True

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="np",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )

    for key, value in inputs.items():
        if isinstance(value, np.ndarray):
            inputs[key] = ms.tensor(value)
        if inputs[key].dtype == ms.int64:
            inputs[key] = inputs[key].to(ms.int32)
        elif inputs[key].dtype != ms.int32:
            inputs[key] = inputs[key].to(model.dtype)

    # Inference: Generation of the output text and audio
    text_ids, audio = model.generate(
        **inputs,
        speaker="Ethan",
        thinker_return_dict_in_generate=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        talker_do_sample=False,
    )

    text = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(text)
    if audio is not None:
        sf.write(
            "output.wav",
            audio.reshape(-1).asnumpy(),
            samplerate=24000,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3OmniMoE demo.")

    parser.add_argument("--prompt", type=str, default="What can you see and hear? Answer in one short sentence.")
    parser.add_argument(
        "--image",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Path to the pre-trained model."
    )

    # Parse the arguments
    args = parser.parse_args()

    # set up card communication
    dist.init_process_group(backend="hccl")
    ms.set_auto_parallel_context(parallel_mode="data_parallel")

    generate(args)
