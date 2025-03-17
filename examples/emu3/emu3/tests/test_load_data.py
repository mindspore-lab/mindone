import numpy as np
from emu3.mllm import Emu3Tokenizer

import mindspore as ms

bos_token = "<|extra_203|>"
eos_token = "<|extra_204|>"
pad_token = "<|endoftext|>"
img_token = "<|image token|>"
boi_token = "<|image start|>"
eoi_token = "<|image end|>"
eol_token = "<|extra_200|>"
eof_token = "<|extra_201|>"
# task = "img_gen"
task = "vqa"
null_prompt_prob = 0.5
apply_loss_on_only_vision = False
apply_loss_on_only_text = True
ignore_index = -100
chat_template = "You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:"
visual_token_pattern = "<|visual token {token_id:0>6d}|>"
tokenizer = Emu3Tokenizer.from_pretrained(
    "/home/susan/workspace/checkpoints/BAAI/Emu3-Stage1", model_max_length=2000, padding_side="right", use_fast=False
)
codebook_size = 32768
bov = tokenizer.encode(visual_token_pattern.format(token_id=0))[0]
eov = tokenizer.encode(visual_token_pattern.format(token_id=codebook_size - 1))[0]


def getitem(path):
    data = ms.load_checkpoint(path)  # {"name": name, "images": token_ids, "texts": prompt}
    image_prompt = ""
    if data["images"].dtype == ms.int32:
        image_tokens = data["images"].asnumpy()
        image_prompt = format_image_prompt(image_tokens)

    # structure:
    # [BOS] {caption text} [SOV] {meta text} [SOT] {vision tokens} [EOV] [EOS].
    if task == "img_gen":
        # p_prob = random.random()
        # if p_prob < null_prompt_prob:
        #     prompt = ""
        # else:
        prompt = data["texts"]

        # image generation template
        input = tokenizer.bos_token + prompt + image_prompt
    else:  # vqa
        prompt = data["texts"]
        response = data["response"]
        vt_prompts = chat_template.format(
            image_prompt=image_prompt, text_prompt=prompt
        )  # instruction + input vision & text prompts
        input = vt_prompts + response  # instruction + input vision & text prompts + response

    if task == "img_gen":
        sample = tokenizer(
            input,
            padding="max_length",
            return_token_type_ids=False,
            return_tensors="np",
        )  # keys: "input_ids", "attention_mask"
    else:
        sample = tokenizer(
            text=vt_prompts,
            text_pair=response,
            padding="max_length",
            return_token_type_ids=False,
            return_tensors="np",
        )  # keys: "input_ids", "attention_mask"

    labels = sample["input_ids"]
    # mask labels
    if apply_loss_on_only_vision:  # image generation
        labels = np.where(np.logical_and(labels >= bov, labels <= eov), labels, ignore_index)
    elif apply_loss_on_only_text:  # vqa
        prompt_ids = tokenizer.encode(vt_prompts)
        response_ids = tokenizer.encode(response)
        labels[..., : len(prompt_ids)] = ignore_index  # maks input text and vision prompts
        if (len(prompt_ids) + len(response_ids)) < labels.shape[-1]:  # mask remaining padding tokens
            labels[..., len(prompt_ids) + len(response_ids) :] = ignore_index

    sample["labels"] = labels
    for k, v in sample.items():
        if k != "attention_mask":
            sample[k] = np.squeeze(sample[k], axis=0).astype(np.int32)
        else:
            sample[k] = np.squeeze(sample[k], axis=0).astype(np.bool_)

    print("input_ids", sample["input_ids"][:10], "...", sample["input_ids"][-10:])
    print("attention_mask", sample["attention_mask"][:10], "...", sample["attention_mask"][-10:])
    print("labels", sample["labels"][:10], "...", sample["labels"][-10:])


def format_image_prompt(image_tokens):
    h, w = image_tokens.shape
    imgstr = to_imgstr(image_tokens)

    image_prompt = (
        tokenizer.boi_token
        + f"{h}*{w}"
        + tokenizer.img_token
        + imgstr
        + tokenizer.eol_token
        + tokenizer.eof_token
        + tokenizer.eoi_token
    )

    return image_prompt


def to_imgstr(image_tokens):
    image_token_str = [
        [visual_token_pattern.format(token_id=token_id) for token_id in token_row] for token_row in image_tokens
    ]
    image_row_str = ["".join(token_row) for token_row in image_token_str]
    imgstr = tokenizer.eol_token.join(image_row_str)
    return imgstr


if __name__ == "__main__":
    path = "/home/susan/workspace/datasets/emu3/NuminaMath-TIR/feature/test_00000.ckpt"
    getitem(path)
