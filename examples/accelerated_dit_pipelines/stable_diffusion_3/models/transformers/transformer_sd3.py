from typing import Any, Dict, List, Optional, Tuple, Union

import mindspore as ms

from mindone.diffusers.models.transformers.transformer_2d import Transformer2DModelOutput


def sd3_transformer2d_construct(
    self,
    hidden_states: ms.Tensor,
    encoder_hidden_states: ms.Tensor = None,
    pooled_projections: ms.Tensor = None,
    timestep: ms.Tensor = None,
    block_controlnet_hidden_states: List = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = False,
    cache_params: Tuple = None,
    if_skip: bool = False,
    delta_cache: ms.Tensor = None,
    delta_cache_hidden: ms.Tensor = None,
    use_cache: bool = False,
) -> Union[ms.Tensor, Transformer2DModelOutput, Tuple]:
    """
    The [`SD3Transformer2DModel`] forward method.
    Args:
        hidden_states (`ms.Tensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`ms.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`ms.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `ms.Tensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `mindspore.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.
        cache_params (`Tuple`):
            A tuple of cache parameters which contains start cache layer id, step_stride, use cache layer nums, start use cache step
        if_skip (`bool`):
            if skip the dit bolck calculation.
        delta_cache (`ms.Tensor`):
            the tensor for caching hidden state of skipped bolcks.
        delta_cache (`ms.Tensor`):
            the tensor for caching encoder hidden state of skipped bolcks.
        use_cache (`bool`):
            if `using_cache` is True, results of some layers according to `cache_params` are cached.
    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is not None and "scale" in joint_attention_kwargs:
        # weight the lora layers by setting `lora_scale` for each PEFT layer here
        # and remove `lora_scale` from each PEFT layer at the end.
        # scale_lora_layers & unscale_lora_layers maybe contains some operation forbidden in graph mode
        raise RuntimeError(
            f"You are trying to set scaling of lora layer by passing {joint_attention_kwargs['scale']=}. "
            f"However it's not allowed in on-the-fly model forwarding. "
            f"Please manually call `scale_lora_layers(model, lora_scale)` before model forwarding and "
            f"`unscale_lora_layers(model, lora_scale)` after model forwarding. "
            f"For example, it can be done in a pipeline call like `StableDiffusionPipeline.__call__`."
        )

    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    ((encoder_hidden_states, hidden_states), delta_cache, delta_cache_hidden) = self.forward_blocks(
        hidden_states,
        encoder_hidden_states,
        block_controlnet_hidden_states,
        temb,
        use_cache,
        if_skip,
        cache_params,
        delta_cache,
        delta_cache_hidden,
    )

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    # unpatchify
    patch_size = self.config["patch_size"]
    height = height // patch_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        hidden_states.shape[0],
        height,
        width,
        patch_size,
        patch_size,
        self.out_channels,
    )
    # hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    hidden_states = hidden_states.transpose(0, 5, 1, 3, 2, 4)
    output = hidden_states.reshape(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)

    if not return_dict:
        return (output, delta_cache, delta_cache_hidden) if use_cache else (output,)

    return Transformer2DModelOutput(sample=output)


def forward_blocks_range(
    self,
    hidden_states,
    encoder_hidden_states,
    block_controlnet_hidden_states,
    temb,
    start_idx,
    end_idx,
):
    for index_block, block in enumerate(self.transformer_blocks[start_idx:end_idx]):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )
        # controlnet residual
        if block_controlnet_hidden_states is not None and block.context_pre_only is False:
            interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
            hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

    return hidden_states, encoder_hidden_states


def forward_blocks(
    self,
    hidden_states,
    encoder_hidden_states,
    block_controlnet_hidden_states,
    temb,
    use_cache,
    if_skip,
    cache_params,
    delta_cache,
    delta_cache_hidden,
):
    if not use_cache:
        hidden_states, encoder_hidden_states = self.forward_blocks_range(
            hidden_states,
            encoder_hidden_states,
            block_controlnet_hidden_states,
            temb,
            start_idx=0,
            end_idx=len(self.transformer_blocks),
        )
    else:
        # infer [0, cache_start)
        hidden_states, encoder_hidden_states = self.forward_blocks_range(
            hidden_states,
            encoder_hidden_states,
            block_controlnet_hidden_states,
            temb,
            start_idx=0,
            end_idx=cache_params[0],
        )
        # infer [cache_start, cache_end)
        cache_end = cache_params[0] + cache_params[2]
        hidden_states_before_cache = hidden_states.copy()
        encoder_hidden_states_before_cache = encoder_hidden_states.copy()
        if not if_skip:
            hidden_states, encoder_hidden_states = self.forward_blocks_range(
                hidden_states,
                encoder_hidden_states,
                block_controlnet_hidden_states,
                temb,
                start_idx=cache_params[0],
                end_idx=cache_end,
            )
            delta_cache = hidden_states - hidden_states_before_cache
            delta_cache_hidden = encoder_hidden_states - encoder_hidden_states_before_cache
        else:
            hidden_states = hidden_states_before_cache + delta_cache
            encoder_hidden_states = encoder_hidden_states_before_cache + delta_cache_hidden

        # infer [cache_end, len(self.blocks))
        hidden_states, encoder_hidden_states = self.forward_blocks_range(
            hidden_states,
            encoder_hidden_states,
            block_controlnet_hidden_states,
            temb,
            start_idx=cache_end,
            end_idx=len(self.transformer_blocks),
        )
    return (encoder_hidden_states, hidden_states), delta_cache, delta_cache_hidden
