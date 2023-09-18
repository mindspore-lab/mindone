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

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, Downsample, ResBlock, UNetModel
from ldm.modules.diffusionmodules.util import conv_nd, linear, timestep_embedding, zero_module
from ldm.util import exists, instantiate_from_config

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

_logger = logging.getLogger(__name__)


class ControlledUnetModel(UNetModel):
    def construct(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []

        # self.set_train(False)

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x
        for celllist in self.input_blocks:
            for cell in celllist:
                h = cell(h, emb, context)
            hs.append(h)
        for module in self.middle_block:
            h = module(h, emb, context)

        # TODO: only upper part was in do not need update gradients, not sure if set_train(True) needed here
        if control is not None:
            h += control.pop()

        for celllist in self.output_blocks:
            if only_mid_control or control is None:
                h = self.cat((h, hs.pop()))
            else:
                h = self.cat([h, hs.pop() + control.pop()])

            for cell in celllist:
                h = cell(h, emb, context)

        # # to run graph mode:
        # if control is not None:
        #     item = control[-1]
        #     h = h + item
        
        # hs_len = len(hs)
        # control_len = len(control)

        # for i, celllist in enumerate(self.output_blocks):
        #     hs_item = hs[hs_len-1-i]
        #     if only_mid_control or control is None:
        #         h = self.cat((h, hs_item))
        #     else:
        #         item = control[control_len-2-i]
        #         h = self.cat([h, hs_item + item])
            
        #     for cell in celllist:
        #         h = cell(h, emb, context)
        
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
        # self.time_embed = nn.SequentialCell(
        #     linear(model_channels, time_embed_dim, dtype=self.dtype),
        #     nn.SiLU().to_float(self.dtype),
        #     # nn.Sigmoid().to_float(self.dtype),
        #     linear(time_embed_dim, time_embed_dim, dtype=self.dtype),
        # )
        self.time_embed = nn.CellList([
            linear(model_channels, time_embed_dim, dtype=self.dtype),
            nn.Sigmoid().to_float(self.dtype),
            linear(time_embed_dim, time_embed_dim, dtype=self.dtype),
        ])

        self.input_blocks = nn.CellList(
            [
                nn.CellList(
                    [
                        conv_nd(
                            dims, in_channels, model_channels, 3, padding=1, has_bias=True, pad_mode="pad"
                        ).to_float(self.dtype)
                    ]
                )
            ]
        )

        self.zero_convs = nn.CellList([self.make_zero_conv(model_channels)])

        self.input_hint_block = nn.CellList(
            [
                conv_nd(dims, hint_channels, 16, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
                # nn.SiLU().to_float(self.dtype),
                nn.Sigmoid().to_float(self.dtype),
                conv_nd(dims, 16, 16, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
                # nn.SiLU().to_float(self.dtype),
                nn.Sigmoid().to_float(self.dtype),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2, has_bias=True, pad_mode="pad").to_float(self.dtype),
                # nn.SiLU().to_float(self.dtype),
                nn.Sigmoid().to_float(self.dtype),
                conv_nd(dims, 32, 32, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
                # nn.SiLU().to_float(self.dtype),
                nn.Sigmoid().to_float(self.dtype),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2, has_bias=True, pad_mode="pad").to_float(self.dtype),
                # nn.SiLU().to_float(self.dtype),
                nn.Sigmoid().to_float(self.dtype),
                conv_nd(dims, 96, 96, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
                # nn.SiLU().to_float(self.dtype),
                nn.Sigmoid().to_float(self.dtype),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2, has_bias=True, pad_mode="pad").to_float(self.dtype),
                # nn.SiLU().to_float(self.dtype),
                nn.Sigmoid().to_float(self.dtype),
                zero_module(
                    conv_nd(dims, 256, model_channels, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype)
                ),
            ]
        )

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

                    if not exists(num_attention_blocks) or _ < num_attention_blocks[level]:
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
                    nn.CellList(
                        [
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
                            if resblock_updown
                            else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)
                        ]
                    )
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
        return zero_module(
            conv_nd(self.dims, channels, channels, 1, padding=0, has_bias=True, pad_mode="pad").to_float(self.dtype)
        )

    def construct(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        # emb = self.time_embed(t_emb)
        emb = t_emb
        for cell in self.time_embed:
            if type(cell) is not nn.Sigmoid:
                emb = cell(emb)
            else:
                emb = emb * cell(emb)

        guided_hint = hint
        for cell in self.input_hint_block:
            if type(cell) is not nn.Sigmoid:
                guided_hint = cell(guided_hint)
            else:
                guided_hint = guided_hint * cell(guided_hint)

        outs = []

        h = x
        for celllist, zero_conv in zip(self.input_blocks, self.zero_convs):
            for cell in celllist:
                h = cell(h, emb, context)
            if guided_hint is not None:
                h += guided_hint
                guided_hint = None
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

    def construct(self, x, c, control):
        t = self.uniform_int(
            (x.shape[0],), ms.Tensor(0, dtype=ms.dtype.int32), ms.Tensor(self.num_timesteps, dtype=ms.dtype.int32)
        )
        x, c_crossattn, c_concat = self.get_input(x, c, control)
        c_concat, c_crossattn = [c_concat], [c_crossattn]
        return self.p_losses(x, c_concat, c_crossattn, t)
    
    def get_input(self, x, c, control, bs=None, *args, **kwargs):
        # batch: images, captions, controls
        if len(x.shape) == 3:
            x = x[..., None]
        x = self.transpose(x, (0, 3, 1, 2)) # RGB -> BGR ?
        z = ops.stop_gradient(self.get_first_stage_encoding(self.encode_first_stage(x)))        
        
        c = ops.stop_gradient(self.cond_stage_model.encode(c))

        if bs is not None:
            control = control[:bs]
        control = ops.transpose(control, (0, 3, 1, 2))  # 'b h w c -> b c h w'
        # control.to_float(self.dtype)
        return z, c, control

    def apply_model(self, x_noisy, t, c_concat=None, c_crossattn=None, *args, **kwargs):
        diffusion_model = self.model.diffusion_model
        cond_txt = ops.concat(c_crossattn, 1)
        if c_concat is None:
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control
            )
        else:
            hint = ops.concat(c_concat, 1)
            control = self.control_model(x=x_noisy, hint=hint, timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control
            )
        return eps

    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

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
            params += list(self.model.diffusion_model.output_blocks.get_parameters())
            params += list(self.model.diffusion_model.out.get_parameters())
        opt = nn.optim.adam.AdamWeightDecay(params, learning_rate=lr)
        return opt

    def p_losses(self, x_start, c_concat, c_crossattn, t, noise=None):
        noise = ms.numpy.randn(x_start.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(
            x_noisy,
            t,
            c_concat=c_concat, 
            c_crossattn=c_crossattn
        )

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            # target = sqrt_alpha_cum * noise - sqrt_one_minus_alpha_prod * x_start
            target = self.get_velocity(x_start, noise, t)  # TODO: parse train step from randint
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        logvar_t = self.logvar[t]
        loss = loss_simple / ops.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()

        # NOTE: original_elbo_weight is never set larger than 0. Diffuser remove it too. Let's remove it to save mem.
        # loss_vlb = self.get_loss(model_output, target, mean=False).mean((1, 2, 3))
        # loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # loss += (self.original_elbo_weight * loss_vlb)

        return loss
