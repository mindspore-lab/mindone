import mindspore as ms
from mindspore import mint, Tensor
from transformers import AutoModelForCausalLM

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))  # for mindone

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image


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
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = Tensor(input_ids, ms.int64)

    tokens = mint.zeros((parallel_size*2, len(input_ids)), dtype=ms.int32)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = mint.zeros((parallel_size, image_token_num_per_image), dtype=ms.int32)

    outputs = []
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = mint.nn.functional.softmax(logits / temperature, dim=-1)

        next_token = mint.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = mint.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=ms.int32), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(ms.float32).transpose(0, 2, 3, 1).asnumpy()

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)


if __name__ == "__main__":
    ms.set_context(device_id=6, mode=1, pynative_synchronize=True)
    
    # specify the path to the model
    model_path = "/mnt/disk2/fredhong/hf_ckpts/Janus-Pro-1B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path)
    # vl_gpt = vl_gpt.to(ms.bfloat16)

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
    )
