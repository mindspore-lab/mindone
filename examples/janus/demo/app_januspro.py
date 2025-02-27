import argparse
import os
import sys

import gradio as gr
from transformers import AutoConfig, AutoModelForCausalLM

import mindspore as ms
from mindspore import mint, ops

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

import os

import numpy as np
from janus.models import VLChatProcessor
from janus.utils.io import set_model_param_dtype
from PIL import Image

from mindone.utils.config import str2bool
from mindone.utils.seed import set_random_seed

# args
parser = argparse.ArgumentParser()
parser.add_argument("--ms_mode", type=int, default=1, help="mindspore mode, 0: graph, 1: pynative")
parser.add_argument(
    "--model_path",
    type=str,
    default="ckpts/Janus-Pro-7B",
    help="path to model weight folder",
)
parser.add_argument("--share", type=str2bool, default=False, help="private or share demo (public)")
args = parser.parse_args()

# ms init
ms.set_context(mode=args.ms_mode)  # TODO: t2v support graph mode
if args.ms_mode == 0:
    ms.set_context(jit_config={"jit_level": "O0"})

# Load model and processor
# model_path = "deepseek-ai/Janus-Pro-7B"
config = AutoConfig.from_pretrained(args.model_path)
language_config = config.language_config
language_config._attn_implementation = "eager"
vl_gpt = AutoModelForCausalLM.from_pretrained(args.model_path, language_config=language_config, trust_remote_code=True)

vl_gpt = set_model_param_dtype(vl_gpt, ms.bfloat16)
vl_gpt.set_train(False)

vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
tokenizer = vl_chat_processor.tokenizer


# Multimodal Understanding function
def multimodal_understanding(image, question, seed, top_p, temperature):
    # Clear cache before generating
    # ms.hal.empty_cache()

    # set seed
    set_random_seed(seed)

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = [Image.fromarray(image)]

    prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(
        ms.bfloat16
    )

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
    )

    answer = tokenizer.decode(outputs[0].asnumpy().tolist(), skip_special_tokens=True)

    return answer


def generate(
    input_ids,
    width,
    height,
    temperature: float = 1,
    parallel_size: int = 5,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    patch_size: int = 16,
):
    # Clear cache before generating
    # ms.hal.empty_cache()

    tokens = mint.zeros((parallel_size * 2, len(input_ids)), dtype=ms.int64)

    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens).to(vl_gpt.dtype)
    generated_tokens = mint.zeros((parallel_size, image_token_num_per_image), dtype=ms.int32)

    use_cache = False
    outputs = None
    for i in range(image_token_num_per_image):
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=outputs.past_key_values if (i != 0 and use_cache) else None,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        if temperature > 0:
            probs = mint.nn.functional.softmax(logits / temperature, dim=-1)
            next_token = mint.multinomial(probs, num_samples=1)
        else:
            next_token = mint.argmax(logits, dim=-1, keepdim=True)

        generated_tokens[:, i] = next_token.squeeze(axis=-1)
        next_token = mint.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

        img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)

        if use_cache:
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        else:
            inputs_embeds = ops.concat((inputs_embeds, img_embeds.unsqueeze(dim=1)), axis=1)

    patches = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=ms.int32),
        shape=[parallel_size, 8, width // patch_size, height // patch_size],
    )

    return generated_tokens.to(dtype=ms.int32), patches


def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(ms.float32).transpose(0, 2, 3, 1).asnumpy()
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img


# @spaces.GPU(duration=120)  # Specify a duration to avoid timeout
def generate_image(prompt, seed=None, guidance=5, t2i_temperature=1.0):
    # Clear cache and avoid tracking gradients
    # ms.hal.empty_cache()

    # Set the seed for reproducible results
    if seed is not None:
        set_random_seed(seed)

    width = 384
    height = 384
    parallel_size = 2

    messages = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=messages,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    text = text + vl_chat_processor.image_start_tag

    input_ids = ms.Tensor(tokenizer.encode(text), ms.int64)
    output, patches = generate(
        input_ids,
        width // 16 * 16,
        height // 16 * 16,
        cfg_weight=guidance,
        parallel_size=parallel_size,
        temperature=t2i_temperature,
    )
    images = unpack(patches, width // 16 * 16, height // 16 * 16, parallel_size=parallel_size)

    # return [Image.fromarray(images[i]).resize((768, 768), Image.LANCZOS) for i in range(parallel_size)]
    return [Image.fromarray(images[i]) for i in range(parallel_size)]


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
    with gr.Row():
        image_input = gr.Image()
        with gr.Column():
            question_input = gr.Textbox(label="Question")
            und_seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")

    understanding_button = gr.Button("Chat")
    understanding_output = gr.Textbox(label="Response")

    examples_inpainting = gr.Examples(
        label="Multimodal Understanding examples",
        examples=[
            [
                "explain this meme",
                "images/doge.png",
            ],
            # [
            #    "Convert the formula into latex code.",
            #    "images/equation.png",
            # ],
        ],
        inputs=[question_input, image_input],
    )

    gr.Markdown(value="# Text-to-Image Generation")

    with gr.Row():
        cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight")
        t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="temperature")

    prompt_input = gr.Textbox(label="Prompt. (Prompt in more detail can help produce better images!)")
    seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    generation_button = gr.Button("Generate Images")

    image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)

    examples_t2i = gr.Examples(
        label="Text to image generation examples.",
        examples=[
            "Master shifu racoon wearing drip attire as a street gangster.",
            "The face of a beautiful girl",
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A glass of red wine on a reflective surface.",
            "A cute and adorable baby fox with big brown eyes,"
            "autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal"
            "engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
            "The image features an intricately designed eye set against a circular backdrop adorned with ornate"
            "swirl patterns that evoke both realism and surrealism. At the center of attention is a strikingly vivid blue iris surrounded by"
            "delicate veins radiating outward from the pupil to create depth and intensity. The eyelashes are long and dark, casting subtle"
            "shadows on the skin around them which appears smooth yet slightly textured as if aged or weathered over time.\n\nAbove the eye,"
            "there's a stone-like structure resembling part of classical architecture, adding layers of mystery and timeless elegance to the"
            "composition. This architectural element contrasts sharply but harmoniously with the organic curves surrounding it. Below the eye"
            "lies another decorative motif reminiscent of baroque artistry, further enhancing the overall sense of eternity encapsulated within"
            "each meticulously crafted detail. \n\nOverall, the atmosphere exudes a mysterious aura intertwined seamlessly with elements suggesting"
            " timelessness, achieved through the juxtaposition of realistic textures and surreal artistic flourishes. Each component\u2014from the intricate "
            "designs framing the eye to the ancient-looking stone piece above\u2014contributes uniquely towards creating a visually captivating tableau imbued "
            "with enigmatic allure.",
        ],
        inputs=prompt_input,
    )

    understanding_button.click(
        multimodal_understanding,
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output,
    )

    generation_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input, t2i_temperature],
        outputs=image_output,
    )

if args.share:
    demo.launch(share=True)
else:
    demo.queue(concurrency_count=1, max_size=10).launch(server_name="127.0.0.1", server_port=37906, root_path="/path")
