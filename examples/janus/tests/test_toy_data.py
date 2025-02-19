import sys, os
import numpy as np
sys.path.append(".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import mindspore as ms
from mindspore.dataset.vision import Inter

def gen_t2i_train_sample(model_path='ckpts/Janus-Pro-1B', max_length=1088):  # 512+576
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    prompt = "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair" 
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    #                padding="max_length", 
    #                max_length=max_length,
    #                trucation=True,  # FIXME
    #                )
    input_ids = input_ids + [vl_chat_processor.image_id] * vl_chat_processor.num_image_tokens
    input_ids = input_ids + [vl_chat_processor.image_end_id + vl_chat_processor.tokenizer.eos_token_id]  # TODO need EOS token?

    assert len(input_ids) <= max_length
    valid_seq_len = len(input_ids) 
    # attention mask
    attention_masks = np.zeros(shape=[1, max_length])
    attention_masks[0, :len(input_ids)] =  1

    # pad and truncate
    num_pad = max_length - len(input_ids)
    if num_pad > 0:
        input_ids = input_ids + [vl_chat_processor.pad_id] * num_pad

    input_ids = input_ids[:max_length]
    input_ids = np.array(input_ids, np.int64)[None, ...]
    print(input_ids)

    # label, only train on vision seq
    ignore_index = -100  # TODO: read from config? but CE Loss didn't accept setting ignore_index
    labels = input_ids
    labels = np.where((np.array(input_ids) == vl_chat_processor.image_id),
                labels,
                ignore_index,
                )
    labels = np.array(labels, np.int64)

    # image pixels
    # TODO: need to align to the image precoessing protocal for VQ16: resize to 384x384 (interp ?), norm to [-1, 1] 
    image_path = 'images/doge.png'
    size = (384, 384)
    image = Image.open(image_path).convert("RGB")
    image = ms.dataset.vision.Resize(size, interpolation=Inter.ANTIALIAS)(image)
    image = np.array(image)
    image = (image / 255.0) * 2  - 1
    image = np.transpose(image, (2, 0, 1))
    image = image[None, None, ...]  # add bs, n_images dimension 

    # image seq mask 
    image_seq_masks = (input_ids ==  vl_chat_processor.image_id)
    assert image_seq_masks.sum() ==  vl_chat_processor.num_image_tokens
    
    return input_ids, labels, attention_masks, image, image_seq_masks

def gen_vqa_train_sample():
    pass

def gen_captioning_train_sample():
    pass


if __name__ == "__main__":
    gen_t2i_train_sample()
