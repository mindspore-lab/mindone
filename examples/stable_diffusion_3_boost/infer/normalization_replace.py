from typing import Optional, Tuple

import mindspore as ms
from mindspore import ops, mint

from mindone.diffusers.models.layers_compat import group_norm

def ada_layernorm_construct(self, x: ms.Tensor, timestep: ms.Tensor) -> ms.Tensor:
    # Argument 'timestep' is a 0-dim tensor, we will unsqueezed it firstly
    # because inputs tensor of nn.Dense should has more than 1 dim.
    emb = self.linear(self.silu(self.emb(timestep[None])))
    scale, shift = mint.chunk(emb, 2, dim=1)
    x = self.norm(x) * (1 + scale.expand_dims(1)) + shift.expand_dims(1)
    return x

def ada_layernormzero_construct(
        self,
        x: ms.Tensor,
        timestep: Optional[ms.Tensor] = None,
        class_labels: Optional[ms.Tensor] = None,
        hidden_dtype=None,
        emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mint.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa.expand_dims(1)) + shift_msa.expand_dims(1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

def ada_groupnorm_construct(self, x: ms.Tensor, emb: ms.Tensor) -> ms.Tensor:
    if self.act:
        emb = self.act(emb)
    emb = self.linear(emb)
    emb = emb.expand_dims(2).expand_dims(2)

    scale, shift = mint.chunk(emb, 2, dim=1)
    x = group_norm(x, self.num_groups, None, None, self.eps)
    x = x * (1 + scale) + shift
    return x

def ada_layernorm_continuous_construct(self, x: ms.Tensor, conditioning_embedding: ms.Tensor) -> ms.Tensor:
    # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
    emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
    scale, shift = ops.chunk(emb, 2, axis=1)
    x = self.norm(x) * (1 + scale).expand_dims(1) + shift.expand_dims(1)
    return x
