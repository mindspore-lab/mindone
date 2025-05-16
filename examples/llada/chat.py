from src import LLaDAModelLM
from transformers import AutoConfig, AutoTokenizer

import mindspore as ms
from mindspore import Tensor, mint

from examples.llada.generate import generate

ms.set_context(mode=ms.PYNATIVE_MODE)


def chat():
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    config = AutoConfig.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    # config.flash_attention = True
    model = LLaDAModelLM.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", mindspore_dtype=ms.bfloat16, config=config)

    gen_length = 128
    steps = 128
    print("*" * 66)
    print(f"**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **")
    print("*" * 66)

    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")

        m = [{"role": "user", "content": user_input}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)["input_ids"]
        input_ids = (
            Tensor(input_ids)
            if (len(input_ids.shape) == 2 and input_ids.shape[0] == 1)
            else Tensor(input_ids).unsqueeze(0)
        )  # (1, L)

        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = mint.cat([prompt, input_ids[:, 1:]], dim=1)

        out = generate(
            model,
            prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=32,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
        )

        answer = tokenizer.batch_decode(out[:, prompt.shape[1] :], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")

        # remove the <EOS>
        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1
        print("-----------------------------------------------------------------------")


if __name__ == "__main__":
    chat()
