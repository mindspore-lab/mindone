from typing import Literal, Optional

from llama3.llama import llama3_1B, llama3_5B, llama3_30B

import mindspore.nn as nn
from mindspore import Tensor


class STDiTLlama3Wrapper(nn.Cell):
    def __init__(self, model_size: Literal["1B", "5B", "30B"] = "1B", **kwargs):
        super().__init__(auto_prefix=False)

        attn_implementation = "flash_attention" if kwargs.get("enable_flashattn", False) else "eager"
        gradient_checkpointing = kwargs.get("use_recompute", False)

        model_kwargs = dict(
            in_channels=4,
            out_channels=8,
            attn_implementatio=attn_implementation,
            gradient_checkpointing=gradient_checkpointing,
        )

        if model_size == "1B":
            self.llama = llama3_1B(**model_kwargs)
        elif model_size == "5B":
            self.llama = llama3_5B(**model_kwargs)
        else:
            self.llama = llama3_30B(**model_kwargs)

        self.patch_size = self.llama.patch_size
        self.hidden_size = self.llama.hidden_size
        self.num_heads = self.llama.num_attention_heads
        self.input_sq_size = None
        self.in_channels = self.llama.in_channels

    def construct(
        self,
        x: Tensor,
        timestep: Tensor,
        y: Tensor,
        mask: Optional[Tensor] = None,
        frames_mask: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        height: Optional[Tensor] = None,
        width: Optional[Tensor] = None,
        **kwargs,
    ):
        latent_embedding = x
        text_embedding = y
        return self.llama(latent_embedding, timestep, text_embedding)
