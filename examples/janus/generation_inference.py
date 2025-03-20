import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))  # for mindone
import argparse
import datetime
from time import time

import numpy as np
import PIL.Image
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.compat import get_multinomial_op
from janus.models.modeling_vlm import MultiModalityConfig
from janus.utils.io import set_model_param_dtype
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, mint, ops
from mindspore.nn.utils import no_init_parameters

from mindone.utils.config import str2bool
from mindone.utils.seed import set_random_seed


def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    use_cache: bool = False,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = Tensor(input_ids, ms.int64)

    tokens = mint.zeros((parallel_size * 2, len(input_ids)), dtype=ms.int64)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens).to(mmgpt.dtype)

    generated_tokens = mint.zeros((parallel_size, image_token_num_per_image), dtype=ms.int32)

    if use_cache:
        init_kv = ms.mutable(mmgpt.language_model.model.prepare_static_cache(inputs_embeds, args.max_new_tokens))
        # pad input emb for aligning the shape, meets graph mode
        emb_length = inputs_embeds.shape[-1] if inputs_embeds is not None else 0
        padded_inputs_embeds = ops.zeros(
            (inputs_embeds.shape[0], args.max_new_tokens, emb_length),
            inputs_embeds.dtype if inputs_embeds is not None else None,
        )
        for batch_idx in range(inputs_embeds.shape[0]):
            padded_inputs_embeds[batch_idx, : inputs_embeds.shape[1]] = inputs_embeds[batch_idx][:]
        inputs_embeds = padded_inputs_embeds
    else:
        init_kv = None
    outputs = []
    # FIXME: use mint multinomial after ms2.5 adaptation
    multinomial = get_multinomial_op()

    st = time()
    for i in tqdm(range(image_token_num_per_image)):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=ms.mutable(outputs[1]) if (i != 0 and use_cache) else init_kv,
            return_dict=False,
        )
        hidden_states = outputs[0]

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        if temperature > 0:
            probs = mint.nn.functional.softmax(logits / temperature, dim=-1)
            next_token = multinomial(probs, num_samples=1)
        else:
            next_token = mint.argmax(logits, dim=-1, keepdim=True)

        generated_tokens[:, i] = next_token.squeeze(axis=-1)

        next_token = mint.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)

        if use_cache:
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        else:
            inputs_embeds = ops.concat((inputs_embeds, img_embeds.unsqueeze(dim=1)), axis=1)

    time_cost = time() - st
    print(
        "Time cost (s): {:.4f}, step time (s): {:.4f}\nEst. throughput (tokens/s): {:4f}\n".format(
            time_cost,
            time_cost / image_token_num_per_image,
            generated_tokens.shape[-1] / time_cost,
        )
    )

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=ms.int32),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(ms.float32).transpose(0, 2, 3, 1).asnumpy()

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs("generated_samples", exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    for i in range(parallel_size):
        save_path = os.path.join("generated_samples", "img_{}-{}.jpg".format(i, time_str))
        PIL.Image.fromarray(visual_img[i]).save(save_path)
        print("Image saved in", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_mode", type=int, default=1, help="mindspore mode, 0: graph, 1: pynative")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
        help="prompt for image content. the more detailed, the better",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Temperature value for controlling randomness in sampling. 0 - no randomness in sampling. default 1.0",
    )
    parser.add_argument(
        "--parallel_size",
        type=int,
        default=1,
        help="number of images to generate in parallel, i.e. number of images in a batch",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckpts/Janus-Pro-1B",
        help="path to model weight folder",
    )
    parser.add_argument("--use_cache", type=str2bool, default=True, help="use kv cache or not")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to model checkpoint in .ckpt format, if None, will use the pretrained weight in mode_path",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    # ms context
    ms.set_context(mode=args.ms_mode)
    if args.ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O0"}, enable_compile_cache=True)
    set_random_seed(args.seed)

    # specify the path to the model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    config = MultiModalityConfig.from_pretrained(args.model_path)
    if args.ckpt_path is not None:
        with no_init_parameters():
            vl_gpt = MultiModalityCausalLM(config=config)
        dtype = ms.bfloat16
        vl_gpt = set_model_param_dtype(vl_gpt, dtype)

        parameter_dict = ms.load_checkpoint(args.ckpt_path)
        param_not_load, ckpt_not_load = ms.load_param_into_net(vl_gpt, parameter_dict, strict_load=True)
        print("net param not load: {}".format(param_not_load))
        print("ckpt param not load: {}".format(ckpt_not_load))
    else:
        with no_init_parameters():
            vl_gpt = MultiModalityCausalLM.from_pretrained(args.model_path, config=config)
        dtype = ms.bfloat16
        vl_gpt = set_model_param_dtype(vl_gpt, dtype)
    vl_gpt.set_train(False)

    if args.ms_mode == 0 and not args.use_cache:
        bs = args.parallel_size * 2
        hidden_size = vl_gpt.language_model.model.layers[0].hidden_size
        input_dyn = Tensor(shape=[bs, None, hidden_size], dtype=dtype)
        vl_gpt.language_model.model.set_inputs(inputs_embeds=input_dyn)

    conversation = [
        {
            "role": "<|User|>",
            "content": args.prompt,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    generate(
        vl_gpt,
        vl_chat_processor,
        prompt,
        temperature=args.temperature,
        parallel_size=args.parallel_size,
        use_cache=args.use_cache,
    )
