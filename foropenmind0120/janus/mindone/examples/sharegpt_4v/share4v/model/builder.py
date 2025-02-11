from share4v.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from share4v.model import Share4VLlamaForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

import mindspore as ms


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device="Ascend"):
    # device: 'Ascend', 'GPU', 'CPU'
    kwargs = {}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=ms.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["dtype"] = ms.float16

    if "sharegpt4v" in model_name.lower():
        # Load ShareGPT4V model
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = Share4VLlamaForCausalLM.from_pretrained(model_path, **kwargs)

    else:
        raise NotImplementedError("Please make sure the model is ShareGPT4V model")

    image_processor = None

    if "sharegpt4v" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            print("trying load vision tower")
            vision_tower.load_model()
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    # set llama model, clip vision tower, mm_projector to inference mode
    model.set_train(False)
    model.set_dtype(kwargs["dtype"])

    return tokenizer, model, image_processor, context_len
