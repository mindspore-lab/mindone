import os
import sys

import numpy as np
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import ops

# TODO: remove in future when mindone is ready for install
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
from mindone.transformers import MT5EncoderModel
from mindone.transformers.models.mt5.modeling_mt5 import MT5LayerNorm
from mindone.utils.amp import auto_mixed_precision


def load_text_encoder(
    text_encoder_name, text_encoder_dtype, cache_dir, use_amp=False, amp_level="O2", custom_fp32_cells=[]
):
    text_encoder, loading_info = MT5EncoderModel.from_pretrained(
        text_encoder_name,
        cache_dir=cache_dir,
        output_loading_info=True,
        mindspore_dtype=text_encoder_dtype if not use_amp else ms.float32,
        use_safetensors=True,
    )
    loading_info.pop("unexpected_keys")  # decoder weights are ignored
    print(loading_info)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, cache_dir=cache_dir)
    if use_amp:
        text_encoder = auto_mixed_precision(
            text_encoder, amp_level=amp_level, dtype=text_encoder_dtype, custom_fp32_cells=custom_fp32_cells
        )

    return text_encoder, tokenizer


def run_prompts_save_numpy(prompts_list, text_encoder, tokenizer, max_length=512, text_preprocessing=None):
    if text_preprocessing is not None:
        prompts_list = [text_preprocessing(p) for p in prompts_list]

    text_inputs = tokenizer(
        prompts_list,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors=None,
    )
    text_input_ids = ms.Tensor(text_inputs.input_ids)
    prompt_attention_mask = ms.Tensor(text_inputs.attention_mask)
    text_embedding = (
        ops.stop_gradient(text_encoder(text_input_ids, attention_mask=prompt_attention_mask))[0].float().asnumpy()
    )

    return text_embedding, text_input_ids.to(ms.int32).asnumpy(), prompt_attention_mask.to(ms.int32).asnumpy()


if __name__ == "__main__":
    text_encoder_name = "google/mt5-xxl"
    cache_dir = "./"
    prompts_list = [
        "Studies have been shown that owning a dog is good for you",
        "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien.",
        "This movie is awesome",
        "This movie is awful",
        "Hello World.",
    ]
    mode = 0
    use_amp = False
    ms.set_context(mode=mode)
    mode_dict = {0: "graph", 1: "pynative"}
    output_dir = f"ms_mt5-xxl_embeddings_{mode_dict[mode]}"
    print(f"save to {output_dir}")
    # original diffusers amp
    if not use_amp:
        for text_encoder_dtype in [ms.float16, ms.bfloat16]:
            print("loading model...")
            text_encoder, tokenizer = load_text_encoder(text_encoder_name, text_encoder_dtype, cache_dir, use_amp=False)
            print(f"run inference with {text_encoder_dtype}")
            text_embedding, text_input_ids, attention_mask = run_prompts_save_numpy(
                prompts_list, text_encoder, tokenizer
            )
            save_dir = os.path.join(output_dir, f"diffusers_{text_encoder_dtype}/")
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "text_embedding.npy"), text_embedding)
            np.save(os.path.join(save_dir, "text_input_ids.npy"), text_input_ids)
            np.save(os.path.join(save_dir, "attention_mask.npy"), attention_mask)
    else:
        # ms auto_mixed_precision,
        for text_encoder_dtype in [ms.float16, ms.bfloat16]:
            print("loading model...")
            text_encoder, tokenizer = load_text_encoder(
                text_encoder_name, text_encoder_dtype, cache_dir, use_amp=True, custom_fp32_cells=[MT5LayerNorm]
            )
            print(f"run inference with {text_encoder_dtype}")
            text_embedding, text_input_ids, attention_mask = run_prompts_save_numpy(
                prompts_list, text_encoder, tokenizer
            )
            save_dir = os.path.join(output_dir, f"ms_amp_{text_encoder_dtype}/")
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "text_embedding.npy"), text_embedding)
            np.save(os.path.join(save_dir, "text_input_ids.npy"), text_input_ids)
            np.save(os.path.join(save_dir, "attention_mask.npy"), attention_mask)
