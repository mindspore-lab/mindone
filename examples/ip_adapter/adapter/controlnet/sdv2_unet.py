from adapter.sdv2.modules.diffusionmodules.openaimodel import IPAdapterUNetModel
from ldm.modules.diffusionmodules.util import timestep_embedding
from ldm.util import instantiate_from_config


class IPAdapterControlNetUnetModel(IPAdapterUNetModel):
    def __init__(
        self,
        control_stage_config,
        context_length=77,
        context_mode="all",
        guess_mode=False,
        strength=1.0,
        sd_locked=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.context_length = context_length
        self.context_mode = context_mode
        if self.context_mode not in ["text", "image", "all"]:
            raise ValueError(f"unsupported context condition: `{self.context_mode}`")

        if sd_locked:
            for param in self.get_parameters():
                param.requires_grad = False

        # add controlnet init
        self.controlnet = instantiate_from_config(control_stage_config)
        self.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        )

    def construct(self, x, timesteps=None, context=None, y=None, control=None, only_mid_control=False, **kwargs):
        """
        x: latent image in shape [bs, z, H//4, W//4]
        timesteps: in shape [bs]
        context: text embedding [bs, seq_len, f] f=768 for sd1.5, 1024 for sd 2.x
        control: control signal [bs, 3, H, W]
        """
        assert control is not None

        hs = []

        if context is not None:
            if self.context_mode == "text":
                context_c = context[:, : self.context_length]
            elif self.context_mode == "image":
                context_c = context[:, self.context_length :]
            else:
                # use whole context as controlnet condition
                context_c = context
        else:
            context_c = None

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        emb_c = self.controlnet.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)
            emb_c = emb_c + self.controlnet.label_emb(y)

        guided_hint = control
        # hint: [bs 3 H W] -> [bs Z H//4 W//4]
        for cell in self.controlnet.input_hint_block:
            guided_hint = cell(guided_hint)

        control_list = []

        h_c = x
        h = x

        for c_celllist, celllist, zero_convs in zip(
            self.controlnet.input_blocks, self.input_blocks, self.controlnet.zero_convs
        ):
            for cell in c_celllist:
                h_c = cell(h_c, emb_c, context_c)

            # add encoded hint with latent image encoded projected with conv2d
            if guided_hint is not None:
                h_c += guided_hint
                guided_hint = None

            control_list.append(zero_convs(h_c, emb_c, context_c))

            for cell in celllist:
                h = cell(h, emb, context)
            hs.append(h)

        for c_module, module in zip(self.controlnet.middle_block, self.middle_block):
            h_c = c_module(h_c, emb_c, context_c)
            h = module(h, emb, context)

        control_list.append(self.controlnet.middle_block_out(h_c, emb_c, context_c))
        control_list = [c * scale for c, scale in zip(control_list, self.control_scales)]

        control_index = -1
        if control_list:
            h = h + control_list[control_index]
            control_index -= 1

        hs_index = -1
        for celllist in self.output_blocks:
            if only_mid_control or len(control_list) == 0:
                h = self.cat((h, hs[hs_index]))
            else:
                h = self.cat([h, hs[hs_index] + control_list[control_index]])
            hs_index -= 1
            control_index -= 1
            for cell in celllist:
                h = cell(h, emb, context)

        return self.out(h)
