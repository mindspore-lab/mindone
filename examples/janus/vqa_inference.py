import argparse
import os, sys
import mindspore as ms
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

# TODO: mindone support AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images, set_model_param_dtype
from mindone.utils.seed import set_random_seed


def multimodal_understanding(image: str, question: str, seed: int, top_p: float, temperature: float,
            vl_gpt: MultiModalityCausalLM,
            vl_chat_processor: VLChatProcessor,
            ):
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
    tokenizer = vl_chat_processor.tokenizer

    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(ms.bfloat16)  # NOTE: no device, all inputs are bf16

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

    return answer, prepare_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_mode", type=int, default=1, help="mindspore mode, 0: graph, 1: pynative")
    parser.add_argument("--image", type=str, default="images/doge.png", help="path to input image")
    parser.add_argument("--question", type=str, default="explain this meme", help="path to input image")
    parser.add_argument("--model_path", type=str, default="ckpts/Janus-Pro-1B", help="path to model weight folder. e.g. deepseek-ai/Janus-Pro-7B, deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--temperature", type=float, default=0., help="Temperature value for controlling randomness in sampling. 0 - no randomness in sampling. default 1.0")
    parser.add_argument("--top_p", type=float, default=0.95, help="only effective when temperature > 0. do sample on the tokens with top_p probaility mass")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # parser.add_argument("--jit_level", type=str, default="O0", choices=["O0", "O1", "O2"], help="graph optimization level")
    args = parser.parse_args()

    # ms context
    ms.set_context(mode=args.ms_mode)
    if args.ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O0"})

    # specify the path to the model
    # model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)

    # TODO: support setting FA. currently can set in modeling_vlm.py
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        # use_flash_attention_2=True,
    )
    vl_gpt = set_model_param_dtype(vl_gpt, ms.bfloat16)
    vl_gpt.set_train(False)

    # infer
    answer, prepare_inputs = multimodal_understanding(args.image, args.question, args.seed, args.top_p, args.temperature,
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
        )

    print(f"{prepare_inputs['sft_format'][0]}", answer)

