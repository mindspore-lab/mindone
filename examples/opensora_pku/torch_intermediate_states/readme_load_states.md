### Updated files
- opensora/utils/sample.utils.py (NEW)
- opensora/sample/sample.py <- opensora/sample/sample_t2v.py
- opensora/sample/pipeline_opensora.py

- opensora/models/diffusion/common.py <- opensora/models/diffusion/opensora/rope.py
- opensora/models/diffusion/opensora/modeling_opensora.py
- opensora/models/diffusion/opensora/modules.py

### Debugging script
scripts/text_condition/single-device/sample_debug.sh

### Intermediate dicts to load
Details of saving intermediate states in Pytorch Version opensora/models/diffusion/modeling_opensora.py forward():

<b>Note: I only save them in first step of denoising.</b>

```python
def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs, 
    ):
        #####################################
        ## !!!SAVE `input parameters
        np.save("./hidden_states_input.npy", hidden_states.float().cpu().numpy())
        np.save("./timestep_input.npy", timestep.float().cpu().numpy())
        np.save("./encoder_hidden_states_input.npy", encoder_hidden_states.float().cpu().numpy())
        np.save("./attention_mask_input.npy", attention_mask.float().cpu().numpy())
        np.save("./encoder_attention_mask_input.npy", encoder_attention_mask.float().cpu().numpy())
        #####################################

        batch_size, c, frame, h, w = hidden_states.shape
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 4:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #   (keep = +0,     discard = -10000.0)
            # b, frame, h, w -> a video
            # b, 1, h, w -> only images
            attention_mask = attention_mask.to(self.dtype)

            attention_mask = attention_mask.unsqueeze(1)  # b 1 t h w
            attention_mask = F.max_pool3d(
                attention_mask, 
                kernel_size=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size), 
                stride=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size)
                )
            attention_mask = rearrange(attention_mask, 'b 1 t h w -> (b 1) 1 (t h w)') 
            attention_mask = (1 - attention_mask.bool().to(self.dtype)) * -10000.0


        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            # b, 1, l
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0

        #####################################
        ## !!! SAVE `masks` after conversion
        # Note: they used "0" as True, "-10000" as False, and masks have transposed dimension
        # I do not suggest to load these masks in Mindspore version
        np.save("./attention_mask_converted.npy", attention_mask.float().cpu().numpy())
        np.save("./encoder_attention_mask_converted.npy", encoder_attention_mask.float().cpu().numpy())
        #####################################

        # 1. Input
        frame = ((frame - 1) // self.config.patch_size_t + 1) if frame % 2 == 1 else frame // self.config.patch_size_t  # patchfy
        height, width = hidden_states.shape[-2] // self.config.patch_size, hidden_states.shape[-1] // self.config.patch_size

        hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, batch_size, frame
        )

        #####################################
        ## !!! SAVE states after `_operate_on_patched_inputs` 
        np.save("./hidden_states_operate_on_patched_inputs.npy", hidden_states.float().cpu().numpy())
        np.save("./encoder_hidden_states_operate_on_patched_inputs.npy", encoder_hidden_states.float().cpu().numpy())
        np.save("./timestep_operate_on_patched_inputs.npy", timestep.float().cpu().numpy())
        np.save("./embedded_timestep_operate_on_patched_inputs.npy", embedded_timestep.float().cpu().numpy())
        #####################################

        # To
        # x            (t*h*w b d) or (t//sp*h*w b d)
        # cond_1       (l b d) or (l//sp b d)
        hidden_states = rearrange(hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        timestep = timestep.view(batch_size, 6, -1).transpose(0, 1).contiguous()

        sparse_mask = {}
        if npu_config is None:
            if get_sequence_parallel_state():
                head_num = self.config.num_attention_heads // nccl_info.world_size
            else:
                head_num = self.config.num_attention_heads
        else:
            head_num = None
        for sparse_n in [1, 4]:
            sparse_mask[sparse_n] = Attention.prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n, head_num)
        #####################################
        ## !!! SAVE sparse masks
        # Note: they used "0" as True, "-10000" as False, and masks have transposed dimension
        # I do not suggest to load these masks in Mindspore version
        attention_mask_sparse_1_False, encoder_attention_mask_sparse_1_False = sparse_mask[1][False] # mask_sparse_1d
        attention_mask_sparse_1_True, encoder_attention_mask_sparse_1_True = sparse_mask[1][True] # mask_sparse_1d_group
        attention_mask_sparse_4_False, encoder_attention_mask_sparse_4_False = sparse_mask[4][False] # mask_sparse_1d
        attention_mask_sparse_4_True, encoder_attention_mask_sparse_4_True = sparse_mask[4][True] # sparse_1d_group
        np.save("./attention_mask_sparse_1_False.npy", attention_mask_sparse_1_False.float().cpu().numpy())
        np.save("./encoder_attention_mask_sparse_1_False.npy", encoder_attention_mask_sparse_1_False.float().cpu().numpy())
        np.save("./attention_mask_sparse_1_True.npy", attention_mask_sparse_1_True.float().cpu().numpy())
        np.save("./encoder_attention_mask_sparse_1_True.npy", encoder_attention_mask_sparse_1_True.float().cpu().numpy())
        np.save("./attention_mask_sparse_4_False.npy", attention_mask_sparse_4_False.float().cpu().numpy())
        np.save("./encoder_attention_mask_sparse_4_False.npy", encoder_attention_mask_sparse_4_False.float().cpu().numpy())
        np.save("./attention_mask_sparse_4_True.npy", attention_mask_sparse_4_True.float().cpu().numpy())
        np.save("./encoder_attention_mask_sparse_4_True.npy", encoder_attention_mask_sparse_4_True.float().cpu().numpy())
        #####################################


        # 2. Blocks
        #####################################
        # !!! SAVE initial input states
        np.save(f"./hidden_states_before_block.npy", hidden_states.float().cpu().numpy())
        #####################################
        for i, block in enumerate(self.transformer_blocks):
            if i > 1 and i < 30:
                attention_mask, encoder_attention_mask = sparse_mask[block.attn1.processor.sparse_n][block.attn1.processor.sparse_group]
            else:
                attention_mask, encoder_attention_mask = sparse_mask[1][block.attn1.processor.sparse_group]


            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    frame, 
                    height, 
                    width, 
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    frame=frame, 
                    height=height, 
                    width=width, 
                )
                #####################################
                # !!! SAVE updated states
                np.save(f"./hidden_states_{i}_block.npy", hidden_states.float().cpu().numpy())
                #####################################
            

        # To (b, t*h*w, h) or (b, t//sp*h*w, h)
        hidden_states = rearrange(hidden_states, 's b h -> b s h', b=batch_size).contiguous()

        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states=hidden_states,
            timestep=timestep,
            embedded_timestep=embedded_timestep,
            num_frames=frame, 
            height=height,
            width=width,
        )  # b c t h w

        #####################################
        #!!! SAVE output hidden states
        np.save("./hidden_states_output.npy", output.float().cpu().numpy())
        #####################################

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

```