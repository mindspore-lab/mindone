from typing import List, Optional, Tuple, Union

import mindspore as ms

from hyvideo.text_encoder import TextEncoder
from constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT


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


def build_model():
    text_encoder = TextEncoder(
            text_encoder_type="llm",
            max_length=351,
            text_encoder_precision="fp16",
            tokenizer_type="llm",
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            # logger=logger,
            # device=device if not args.use_cpu_offload else "cpu",
        )

    text_encoder_2 = TextEncoder(
            text_encoder_type="clipL",
            max_length=77,
            text_encoder_precision="fp16",
            tokenizer_type="clipL",
            reproduce=False,
            # logger=logger,
            # device=device if not args.use_cpu_offload else "cpu",
        )

    return text_encoder, text_encoder_2


def encode_prompt(
    # self,
    prompt,
    # device,
    num_videos_per_prompt,
    do_classifier_free_guidance,
    negative_prompt=None,
    prompt_embeds: Optional[ms.Tensor] = None,
    attention_mask: Optional[ms.Tensor] = None,
    negative_prompt_embeds: Optional[ms.Tensor] = None,
    negative_attention_mask: Optional[ms.Tensor] = None,
    lora_scale: Optional[float] = None,
    clip_skip: Optional[int] = None,
    text_encoder: Optional[TextEncoder] = None,
    data_type: Optional[str] = "image",
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_videos_per_prompt (`int`):
            number of videos that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the video generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        attention_mask (`torch.Tensor`, *optional*):
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        negative_attention_mask (`torch.Tensor`, *optional*):
        lora_scale (`float`, *optional*):
            A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
        text_encoder (TextEncoder, *optional*):
        data_type (`str`, *optional*):
    """
    # if text_encoder is None:
    #     text_encoder = self.text_encoder

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    # if lora_scale is not None and isinstance(self, LoraLoaderMixin):    # lora_scale is None
    #     self._lora_scale = lora_scale

    #     # dynamically adjust the LoRA scale
    #     if not USE_PEFT_BACKEND:
    #         adjust_lora_scale_text_encoder(text_encoder.model, lora_scale)
    #     else:
    #         scale_lora_layers(text_encoder.model, lora_scale)

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        # textual inversion: process multi-vector tokens if necessary
        # if isinstance(self, TextualInversionLoaderMixin):    # False
        #     prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

        text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

        if clip_skip is None:
            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
            prompt_embeds = prompt_outputs.hidden_state
        else:
            prompt_outputs = text_encoder.encode(
                text_inputs,
                output_hidden_states=True,
                data_type=data_type,
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                prompt_embeds
            )

        attention_mask = prompt_outputs.attention_mask
        if attention_mask is not None:
            # attention_mask = attention_mask.to(device)
            bs_embed, seq_len = attention_mask.shape
            attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
            attention_mask = attention_mask.view(
                bs_embed * num_videos_per_prompt, seq_len
            )

    if text_encoder is not None:
        prompt_embeds_dtype = text_encoder.dtype
    # elif self.transformer is not None:    # ms.bfloat16
    #     prompt_embeds_dtype = self.transformer.dtype
    # else:
    #     prompt_embeds_dtype = prompt_embeds.dtype
    prompt_embeds_dtype = ms.bfloat16
    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype)

    if prompt_embeds.ndim == 2:
        bs_embed, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
    else:
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

    # get unconditional embeddings for classifier free guidance
    # if do_classifier_free_guidance and negative_prompt_embeds is None:    # False
    #     uncond_tokens: List[str]
    #     if negative_prompt is None:
    #         uncond_tokens = [""] * batch_size
    #     elif prompt is not None and type(prompt) is not type(negative_prompt):
    #         raise TypeError(
    #             f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
    #             f" {type(prompt)}."
    #         )
    #     elif isinstance(negative_prompt, str):
    #         uncond_tokens = [negative_prompt]
    #     elif batch_size != len(negative_prompt):
    #         raise ValueError(
    #             f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
    #             f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
    #             " the batch size of `prompt`."
    #         )
    #     else:
    #         uncond_tokens = negative_prompt

    #     # textual inversion: process multi-vector tokens if necessary
    #     # if isinstance(self, TextualInversionLoaderMixin):    # False
    #     #     uncond_tokens = self.maybe_convert_prompt(
    #     #         uncond_tokens, text_encoder.tokenizer
    #     #     )

    #     # max_length = prompt_embeds.shape[1]
    #     uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

    #     negative_prompt_outputs = text_encoder.encode(
    #         uncond_input, data_type=data_type
    #     )
    #     negative_prompt_embeds = negative_prompt_outputs.hidden_state

    #     negative_attention_mask = negative_prompt_outputs.attention_mask
    #     if negative_attention_mask is not None:
    #         # negative_attention_mask = negative_attention_mask.to(device)
    #         _, seq_len = negative_attention_mask.shape
    #         negative_attention_mask = negative_attention_mask.repeat(
    #             1, num_videos_per_prompt
    #         )
    #         negative_attention_mask = negative_attention_mask.view(
    #             batch_size * num_videos_per_prompt, seq_len
    #         )

    # if do_classifier_free_guidance:    # False
    #     # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
    #     seq_len = negative_prompt_embeds.shape[1]

    #     negative_prompt_embeds = negative_prompt_embeds.to(
    #         dtype=prompt_embeds_dtype
    #     )

    #     if negative_prompt_embeds.ndim == 2:
    #         negative_prompt_embeds = negative_prompt_embeds.repeat(
    #             1, num_videos_per_prompt
    #         )
    #         negative_prompt_embeds = negative_prompt_embeds.view(
    #             batch_size * num_videos_per_prompt, -1
    #         )
    #     else:
    #         negative_prompt_embeds = negative_prompt_embeds.repeat(
    #             1, num_videos_per_prompt, 1
    #         )
    #         negative_prompt_embeds = negative_prompt_embeds.view(
    #             batch_size * num_videos_per_prompt, seq_len, -1
    #         )

    # if text_encoder is not None:
    #     if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:    # False
    #         # Retrieve the original scale by scaling back the LoRA layers
    #         unscale_lora_layers(text_encoder.model, lora_scale)

    return (
        prompt_embeds,
        negative_prompt_embeds,
        attention_mask,
        negative_attention_mask,
    )


def main(prompt, text_encoder, text_encoder_2):
    (
        prompt_embeds,
        negative_prompt_embeds,
        prompt_mask,
        negative_prompt_mask,
    ) = encode_prompt(
        prompt,
        # device,
        1,
        False,
        [NEGATIVE_PROMPT],
        prompt_embeds=None,
        attention_mask=None,
        negative_prompt_embeds=None,
        negative_attention_mask=None,
        lora_scale=None,
        clip_skip=None,
        text_encoder=text_encoder,
        data_type="video",
    )
    if text_encoder_2 is not None:
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_mask_2,
            negative_prompt_mask_2,
        ) = encode_prompt(
            prompt,
            # device,
            1,
            False,
            [NEGATIVE_PROMPT],
            prompt_embeds=None,
            attention_mask=None,
            negative_prompt_embeds=None,
            negative_attention_mask=None,
            lora_scale=None,
            clip_skip=None,
            text_encoder=text_encoder_2,
            data_type="video",
        )
    else:
        prompt_embeds_2 = None
        negative_prompt_embeds_2 = None
        prompt_mask_2 = None
        negative_prompt_mask_2 = None

    return prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask, \
           prompt_embeds_2, negative_prompt_embeds_2, prompt_mask_2, negative_prompt_mask_2


if __name__ == "__main__":
    mode = 1
    jit_level = "O0"
    ms.set_context(mode=mode)
    if mode == 0:
        ms.set_context(jit_config={"jit_level": jit_level})

    # text_encoder_choices = ["llm"]  # ["clipL", "llm"]
    text = ["hello world"]
    text_encoder, text_encoder_2 = build_model()

    all_output = main(text, text_encoder, text_encoder_2)
