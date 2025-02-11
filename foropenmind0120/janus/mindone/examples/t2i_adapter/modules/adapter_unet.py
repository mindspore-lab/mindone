import sys
from typing import List, Optional

from mindspore import Tensor, ops

sys.path.append("../../stable_diffusion_xl/")
from gm.modules.diffusionmodules.openaimodel import UNetModel as UNetXL
from gm.modules.diffusionmodules.util import timestep_embedding


class T2IAdapterUNetXL(UNetXL):
    def construct(
        self, x, timesteps=None, context=None, y=None, adapter_states: Optional[List[Tensor]] = None, **kwargs
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :param adapter_states: (optional) a list of adapter states for each down block.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"
        hs, hs_idx = (), -1
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x
        adapter_idx = 0
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)

            if i in [5, 8] and adapter_states:  # include adapter features in hidden states for 2nd and 3rd blocks
                h += adapter_states[adapter_idx]
                adapter_idx += 1

            hs += (h,)

            if i == 3 and adapter_states:  # do not include adapter features in hidden states after the first block
                h += adapter_states[adapter_idx]
                adapter_idx += 1

            hs_idx += 1

        h = self.middle_block(h, emb, context)
        if adapter_states:
            h += adapter_states[adapter_idx]

        for module in self.output_blocks:
            h = ops.concat([h, hs[hs_idx]], axis=1)
            hs_idx -= 1
            h = module(h, emb, context)

        return self.out(h)
