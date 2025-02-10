import argparse
import mindspore as ms
from mindspore import mint, ops, Tensor
from transformers import AutoModelForCausalLM
import numpy as np
import os
import PIL.Image
from tqdm import tqdm

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))  # for mindone

from mindone.utils.config import str2bool
from mindone.utils.seed import set_random_seed
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import set_model_param_dtype
import numpy as np
import os
import PIL.Image
from tqdm import tqdm


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
    use_cache: bool= False,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = Tensor(input_ids, ms.int32)

    tokens = mint.zeros((parallel_size*2, len(input_ids)), dtype=ms.int64)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens).to(mmgpt.dtype)

    generated_tokens = mint.zeros((parallel_size, image_token_num_per_image), dtype=ms.int32)

    outputs = []
    for i in tqdm(range(image_token_num_per_image)):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=use_cache, # TODO support kv cache
            past_key_values=outputs.past_key_values if (i != 0 and use_cache)else None,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        if temperature > 0:
            probs = mint.nn.functional.softmax(logits / temperature, dim=-1)
            next_token = mint.multinomial(probs, num_samples=1)
        else:
            next_token = mint.argmax(logits, dim=-1, keepdim=True)

        generated_tokens[:, i] = next_token.squeeze(axis=-1)

        next_token = mint.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)


        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)

        if use_cache:
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        else:
            inputs_embeds = ops.concat((inputs_embeds, img_embeds.unsqueeze(dim=1)), axis=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=ms.int32), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(ms.float32).transpose(0, 2, 3, 1).asnumpy()

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)
        print('images saved in', save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cache", type=str2bool, default=False, help="use kv cache or not")
    parser.add_argument("--ms_mode", type=str, default=1, help="0: graph, 1: pynative")
    parser.add_argument("--temperature", type=float, default=1, help="Temperature value for controlling randomness in sampling. 0 - no randomness in sampling. default 1.0")
    parser.add_argument("--parallel_size", type=int, default=1, help="number of images to generate in parallel, i.e. number of images in a batch")
    parser.add_argument("--model_path", type=str, default="ckpts/Janus-Pro-1B", help="path to model weight folder")
    # parser.add_argument("--jit_level", type=str, default="O0", choices=["O0", "O1", "O2"], help="graph optimization level")
    args = parser.parse_args()

    # ms context
    ms.set_context(mode=args.ms_mode)
    if args.ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O0"})
    set_random_seed(42)

    # specify the path to the model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(args.model_path)
    vl_gpt = set_model_param_dtype(vl_gpt, ms.bfloat16)
    vl_gpt.set_train(False)

    # conversation = [
    #     {
    #         "role": "User",
    #         "content": "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, "
    #         "under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
    #     },
    #     {"role": "Assistant", "content": ""},
    # ]

    conversation = [
        {
            "role": "<|User|>",
            "content": "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
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
