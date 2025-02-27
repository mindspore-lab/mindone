from sgm.modules.embedders.open_clip import create_model as openclip_create_model
from sgm.modules.embedders.open_clip import tokenize as openclip_tokenize

from mindspore import Tensor, nn


class AbstractEncoder(nn.Cell):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(self, arch="ViT-H-14", version="", device="cuda", max_length=77, freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model = openclip_create_model(arch, pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        for param in self.get_parameters():
            param.requires_grad = False

    def construct(self, text):
        tokens, _ = openclip_tokenize(text)
        z = self.encode_with_transformer(Tensor(tokens))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
