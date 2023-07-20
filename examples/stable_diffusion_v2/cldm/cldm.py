# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import logging

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

_logger = logging.getLogger(__name__)

class ControlledUnetModel(UNetModel):
    
    def construct(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []

        self.set_train(False)

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x
        for celllist in self.input_blocks:
            for cell in celllist:
                h = cell(h, emb, context)
            hs.append(h)
        for module in self.middle_block:
            h = module(h, emb, context)
        
        # TODO: only upper part was in torch.no_grad(), not sure if set_train(True) needed here
        hs_index = -1
        if control is not None:
            h += control[hs_index]
        
        for celllist in self.output_blocks:
            if only_mid_control or control is None:
                h = self.cat((h, hs[hs_index]))
            else:
                h = self.cat([h, hs[hs_index] + control[hs_index-1]], axis=1)

            for cell in celllist:
                h = cell(h, emb, context)
            hs_index -= 1

        return self.out(h)

class ControlNet(nn.Cell):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = 1.0 - dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.SequentialCell(
            linear(model_channels, time_embed_dim, dtype=self.dtype),
            nn.SiLU().to_float(self.dtype),
            linear(time_embed_dim, time_embed_dim, dtype=self.dtype),
        )

        self.input_blocks = nn.CellList(
            [
                nn.CellList([
                        conv_nd(
                            dims, in_channels, model_channels, 3, padding=1, has_bias=True, pad_mode="pad"
                        ).to_float(self.dtype)
                ])
            ]
        )

        self.zero_convs = nn.CellList([self.make_zero_conv(model_channels)])

        self.input_hint_block = nn.CellList([
            conv_nd(dims, hint_channels, 16, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
            nn.SiLU().to_float(self.dtype),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU().to_float(self.dtype),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2, has_bias=True, pad_mode="pad").to_float(self.dtype),
            nn.SiLU().to_float(self.dtype),
            conv_nd(dims, 32, 32, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
            nn.SiLU().to_float(self.dtype),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2, has_bias=True, pad_mode="pad").to_float(self.dtype),
            nn.SiLU().to_float(self.dtype),
            conv_nd(dims, 96, 96, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
            nn.SiLU().to_float(self.dtype),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2, has_bias=True, pad_mode="pad").to_float(self.dtype),
            nn.SiLU().to_float(self.dtype),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype))
        ])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = nn.CellList(
                    [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            dtype=self.dtype,
                        )
                    ]
                )
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            dtype=self.dtype,
                            dropout=self.dropout,
                            use_linear=use_linear_in_transformer,
                        )
                    )
                self.input_blocks.append(layers)
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.CellList([
                            ResBlock(
                                ch,
                                time_embed_dim,
                                self.dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                                dtype=self.dtype,
                            )
                    ])
                    if resblock_updown
                    else nn.CellList([Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)])
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        _logger.debug(
            "Attention: output_channels={}, num_heads={}, num_head_channels:{}, dim_head={}".format(
                ch, num_heads, num_head_channels, dim_head
            )
        )

        self.middle_block = nn.CellList(
            [
                ResBlock(
                    ch,
                    time_embed_dim,
                    self.dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                    dtype=self.dtype,
                    dropout=self.dropout,
                    use_linear=use_linear_in_transformer,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    self.dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                ),
            ]
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return nn.CellList([zero_module(conv_nd(self.dims, channels, channels, 1, padding=0, has_bias=True, pad_mode="pad").to_float(self.dtype))])

    def construct(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x

        for celllist, zero_conv in zip(self.input_blocks, self.zero_convs):
            for cell in celllist:
                if guided_hint is not None:
                    h = cell(h, emb, context)
                    h += guided_hint
                    guided_hint = None
                else:
                    h = cell(h, emb, context)
                outs.append(zero_conv(h, emb, context))

        for module in self.middle_block:
            h = module(h, emb, context)  

        outs.append(self.middle_block_out(h, emb, context))

        return outs

class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.randn_like = ops.StandardNormal()

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        self.set_train(False)
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = ops.transpose(control, (0, 3, 1, 2)) # 'b h w c -> b c h w'
        control.to_float(self.dtype)
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = ops.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=ops.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        from typing import Union, List
        import math

        def _make_grid(tensor: Union[ms.Tensor, List[ms.Tensor]],
            nrow: int = 8,
            padding: int = 2,
            value_range: Optional[Tuple[int, int]] = None,
            scale_each: bool = False,
            pad_value: float = 0.0,
        ) -> ms.Tensor:
            # if list of tensors, convert to a 4D mini-batch Tensor
            if isinstance(tensor, list):
                tensor = ops.stack(tensor, dim=0)

            if len(tensor.shape) == 2:  # single image H x W
                tensor = tensor.unsqueeze(0)
            if len(tensor.shape)  == 3:  # single image
                if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
                    tensor = ops.cat((tensor, tensor, tensor), 0)
                tensor = tensor.unsqueeze(0)

            if len(tensor.shape) == 4 and tensor.shape[1] == 1:  # single-channel images
                tensor = ops.cat((tensor, tensor, tensor), 1)


            if not isinstance(tensor, ms.Tensor):
                raise TypeError("tensor should be of type ms.Tensor")
            if tensor.shape[0]  == 1:
                return tensor.squeeze(0)

            # make the mini-batch of images into a grid
            nmaps = tensor.size(0)
            xmaps = min(nrow, nmaps)
            ymaps = int(math.ceil(float(nmaps) / xmaps))
            height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
            num_channels = tensor.shape[1]
            # TODO: still under development
            grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
            k = 0
            for y in range(ymaps):
                for x in range(xmaps):
                    if k >= nmaps:
                        break
                    # Tensor.copy_() is a valid method but seems to be missing from the stubs
                    # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
                    grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                        2, x * width + padding, width - padding
                    ).copy_(tensor[k])
                    k = k + 1
            return grid

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(ms.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = self.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = ops.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = ops.transpose(diffusion_row, (1, 0, 2, 3, 4))  # 'n b c h w -> b n c h w'
            b, n, c, h, w = diffusion_grid.shape
            diffusion_grid = ops.reshape(diffusion_grid, (b * n, c, h, w))
            # TODO: still under development
            # diffusion_grid = _make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        # if sample:
        #     # get denoise row
        #     samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
        #                                              batch_size=N, ddim=use_ddim,
        #                                              ddim_steps=ddim_steps, eta=ddim_eta)
        #     x_samples = self.decode_first_stage(samples)
        #     log["samples"] = x_samples
        #     if plot_denoise_rows:
        #         denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
        #         log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.trainable_params())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = nn.optim.adam.AdamWeightDecay(params, learning_rate=lr)
        return opt

    # def low_vram_shift(self, is_diffusing):
    #     if is_diffusing:
    #         self.model = self.model.cuda()
    #         self.control_model = self.control_model.cuda()
    #         self.first_stage_model = self.first_stage_model.cpu()
    #         self.cond_stage_model = self.cond_stage_model.cpu()
    #     else:
    #         self.model = self.model.cpu()
    #         self.control_model = self.control_model.cpu()
    #         self.first_stage_model = self.first_stage_model.cuda()
    #         self.cond_stage_model = self.cond_stage_model.cuda()
