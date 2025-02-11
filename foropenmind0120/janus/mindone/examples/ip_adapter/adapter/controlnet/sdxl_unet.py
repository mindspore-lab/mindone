from adapter.sdxl.modules.diffusionmodules.openaimodel import IPAdatperUNetModel
from gm.modules.diffusionmodules.util import timestep_embedding
from gm.util import instantiate_from_config

import mindspore.ops as ops


class IPAdapterControlNetUnetModel(IPAdatperUNetModel):
    def __init__(self, control_stage_config, guess_mode=False, strength=1.0, sd_locked=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if sd_locked:
            for param in self.get_parameters():
                param.requires_grad = False

        self.controlnet = instantiate_from_config(control_stage_config)
        self.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        )

    def construct(self, x, timesteps=None, context=None, y=None, control=None, only_mid_control=False, **kwargs):
        hs = []

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        emb_c = self.controlnet.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)
            emb_c = emb_c + self.controlnet.label_emb(y)

        if control is not None:
            guided_hint = control
            for cell in self.controlnet.input_hint_block:
                guided_hint = cell(guided_hint)
        else:
            guided_hint = None

        control_list = []

        h_c = x
        h = x

        for c_celllist, celllist, zero_convs in zip(
            self.controlnet.input_blocks, self.input_blocks, self.controlnet.zero_convs
        ):
            if control is not None:
                h_c = c_celllist(h_c, emb_c, context)
                if guided_hint is not None:
                    h_c += guided_hint
                    guided_hint = None
                control_list.append(zero_convs(h_c, emb_c, context))

            h = celllist(h, emb, context)
            hs.append(h)

        if control is not None:
            h_c = self.controlnet.middle_block(h_c, emb_c, context)

        h = self.middle_block(h, emb, context)

        if control is not None:
            control_list.append(self.controlnet.middle_block_out(h_c, emb_c, context))
            control_list = [c * scale for c, scale in zip(control_list, self.control_scales)]

        control_index = -1
        if control_list:
            h = h + control_list[control_index]
            control_index -= 1

        hs_index = -1
        for celllist in self.output_blocks:
            if only_mid_control or len(control_list) == 0:
                h = ops.concat([h, hs[hs_index]], axis=1)
            else:
                h = ops.concat([h, hs[hs_index] + control_list[control_index]], axis=1)
            hs_index -= 1
            control_index -= 1
            h = celllist(h, emb, context)

        return self.out(h)
