from typing import List, Union

import mindspore as ms

from text_encoder import TextEncoder
from constants import PROMPT_TEMPLATE


# prompt_template
prompt_template_name = "dit-llm-encode"
prompt_template_video_name = "dit-llm-encode-video"

prompt_template = (
    PROMPT_TEMPLATE[prompt_template_name]
    if prompt_template_name is not None
    else None
)

# prompt_template_video
prompt_template_video = (
    PROMPT_TEMPLATE[prompt_template_video_name]
    if prompt_template_video_name is not None
    else None
)

# max_length
if prompt_template_video_name is not None:
    crop_start = PROMPT_TEMPLATE[prompt_template_video_name].get(
        "crop_start", 0
    )
elif prompt_template_name is not None:
    crop_start = PROMPT_TEMPLATE[prompt_template_name].get("crop_start", 0)
else:
    crop_start = 0
text_len = 256
max_length = text_len + crop_start


# args
text_encoder_name = "llm"
tokenizer = "llm"
text_encoder_precision = "fp32"    # fp16 may get bad results

text_encoder_2_name = "clipL"
tokenizer_2 = "clipL"
text_len_2 = 77
text_encoder_precision_2 = "fp32"

text_encoder_path = ""
tokenizer_path = ""
text_encoder_2_path = ""
tokenizer_2_path = ""

hidden_state_skip_layer = 2
apply_final_norm = False
reproduce = False


def build_model(text_encoder_choices: List[str]):
    text_encoder, text_encoder_2 = None, None

    # llm
    if text_encoder_name in text_encoder_choices:
        text_encoder = TextEncoder(
            text_encoder_type=text_encoder_name,
            max_length=max_length,
            text_encoder_precision=text_encoder_precision,
            text_encoder_path = text_encoder_path,
            tokenizer_type=tokenizer,
            tokenizer_path=tokenizer_path,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=hidden_state_skip_layer,
            apply_final_norm=apply_final_norm,
            reproduce=reproduce,
        )

    # clipL
    if text_encoder_2_name in text_encoder_choices:
        text_encoder_2 = TextEncoder(
            text_encoder_type=text_encoder_2_name,
            max_length=text_len_2,
            text_encoder_precision=text_encoder_precision_2,
            text_encoder_path = text_encoder_2_path,
            tokenizer_type=tokenizer_2,
            tokenizer_path=tokenizer_2_path,
            reproduce=reproduce,
        )

    return text_encoder, text_encoder_2


def encode(text: Union[str, List[str]], text_encoder_choices: List[str]):
    text_encoder, text_encoder_2 = build_model(text_encoder_choices)
    output, output_2 = None, None

    # llm
    if text_encoder is not None:
        output = text_encoder(
            text,
            use_attention_mask=None,
            output_hidden_states=False,
            do_sample=False,
            hidden_state_skip_layer=None,
            return_texts=False,
            data_type="video",
        )

    # clipL
    if text_encoder_2 is not None:
        output_2 = text_encoder_2(        
            text,
            use_attention_mask=None,
            output_hidden_states=False,
            do_sample=False,
            hidden_state_skip_layer=None,
            return_texts=False,
            data_type="video",
        )

    return output, output_2


if __name__ == "__main__":
    mode = 1
    jit_level = "O0"
    ms.set_context(mode=mode)
    if mode == 0:
        ms.set_context(jit_config={"jit_level": jit_level})

    text_encoder_choices = ["llm", "clipL"]
    text = ["hello world"]
    output, output_2 = encode(text, text_encoder_choices)
    print(output)
    print(output_2)
