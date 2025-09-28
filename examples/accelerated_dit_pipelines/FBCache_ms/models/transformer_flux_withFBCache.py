import mindspore as ms
from mindspore import mint
from typing import Any, Callable, Dict, List, Optional, Union     
from mindone.diffusers.models.modeling_outputs import Transformer2DModelOutput
from mindone.diffusers.utils import logging
from .utils import are_two_tensors_similar
logger = logging.get_logger(__name__)

def FBCache_transformer_construct(
    self,
    hidden_states: ms.Tensor,
    encoder_hidden_states: ms.Tensor = None,
    pooled_projections: ms.Tensor = None,
    timestep: ms.Tensor = None,
    img_ids: ms.Tensor = None,
    txt_ids: ms.Tensor = None,
    guidance: ms.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = False,
    controlnet_blocks_repeat: bool = False,
    return_hidden_states_first: bool = False,
    residual_diff_threshold: float = 0.12,
) -> Union[ms.Tensor, Transformer2DModelOutput]:
    """
    Modified transformer forward pass with Forward Block Caching (FBCache) for accelerated inference.
    
    Args:
        hidden_states (ms.Tensor): Input hidden states of shape [batch_size, seq_len, in_channels]
        encoder_hidden_states (ms.Tensor): Encoder hidden states of shape [batch_size, text_seq_len, joint_attention_dim]
        pooled_projections (ms.Tensor): Pooled text projections of shape [batch_size, projection_dim]
        timestep (ms.Tensor): Timestep tensor for diffusion process
        img_ids (ms.Tensor): Image token IDs for positional embeddings
        txt_ids (ms.Tensor): Text token IDs for positional embeddings
        guidance (ms.Tensor): Guidance scale tensor
        joint_attention_kwargs (Optional[Dict[str, Any]]): Additional kwargs for joint attention
        controlnet_block_samples: ControlNet block samples for residual connections
        controlnet_single_block_samples: ControlNet single block samples
        return_dict (bool): Whether to return a Transformer2DModelOutput
        controlnet_blocks_repeat (bool): Whether to repeat ControlNet blocks
        return_hidden_states_first (bool): Whether to return hidden states first
        residual_diff_threshold (float): Threshold for residual difference to determine cache reuse
        
    Returns:
        Union[ms.Tensor, Transformer2DModelOutput]: Either the output tensor or a Transformer2DModelOutput object
        containing the sample tensor.
    """

    if joint_attention_kwargs is not None and "scale" in joint_attention_kwargs:
        raise RuntimeError(
            f"You are trying to set scaling of lora layer by passing {joint_attention_kwargs['scale']=}. "
            f"However it's not allowed in on-the-fly model forwarding. "
            f"Please manually call `scale_lora_layers(model, lora_scale)` before model forwarding and "
            f"`unscale_lora_layers(model, lora_scale)` after model forwarding."
        )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d ms.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d mindspore Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d ms.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d mindspore Tensor"
        )
        img_ids = img_ids[0]

    ids = mint.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
    ############FB Cache starts here###############
    if self.cache_context.enable_taylor:
        self.cache_context.step()
    original_hidden_states = hidden_states
    first_block = self.transformer_blocks[0]
    hidden_states = first_block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        joint_attention_kwargs=joint_attention_kwargs,
    )
    if not isinstance(hidden_states, ms.Tensor): # distinguish transformer blocks and single
        hidden_states, encoder_hidden_states = hidden_states
        if not return_hidden_states_first:
            hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
    first_hidden_states_residual = hidden_states - original_hidden_states
    can_use_cache = are_two_tensors_similar(
        self.cache_context.first_residual,
        first_hidden_states_residual,
        threshold=residual_diff_threshold,
    )
    if can_use_cache:
        hidden_states = hidden_states + self.cache_context.get_residual()
    else:
        self.cache_context.update_first_residual(first_hidden_states_residual)
        original_hidden_states = hidden_states
        for index_block, block in enumerate(self.transformer_blocks[1:]):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = (len(self.transformer_blocks) + len(controlnet_block_samples) - 1) // len(
                    controlnet_block_samples
                )  # not supporting numpy
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = (
                    len(self.single_transformer_blocks) + len(controlnet_single_block_samples) - 1
                ) // len(
                    controlnet_single_block_samples
                )  # not supporting numpy
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        cache_context.update_residual(hidden_states - original_hidden_states)
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output)

    return Transformer2DModelOutput(sample=output)
