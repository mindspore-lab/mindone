import numpy as np

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import Uniform

from .modules import Decoder, Encoder

__all__ = ["AutoencoderVQ"]


class VectorQuantizer(nn.Cell):
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim, embedding_table=Uniform(scale=1.0 / self.n_e))

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", Tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape
        self.cast = ops.Cast()

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = ops.randint(0, self.re_embed, size=new[unknown].shape)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def construct(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"
        z = ops.transpose(z, (0, 2, 3, 1))
        z_flattened = z.view(-1, self.e_dim)
        d = (
            ops.sum(z_flattened**2, dim=1, keepdim=True)
            + ops.sum(self.embedding.embedding_table**2, dim=1)
            - 2 * ops.matmul(z_flattened, ops.transpose(self.embedding.embedding_table, (1, 0)))
        )
        min_encoding_indices = ops.argmin(d, axis=1)
        min_encoding_indices = self.cast(min_encoding_indices, ms.int64)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * ops.mean((ops.stop_gradient(z_q) - z) ** 2) + ops.mean((z_q - ops.stop_gradient(z)) ** 2)
        else:
            loss = ops.mean((ops.stop_gradient(z_q) - z) ** 2) + self.beta * ops.mean((z_q - ops.stop_gradient(z)) ** 2)

        # preserve gradients
        z_q = z + ops.stop_gradient(z_q - z)

        # reshape back to match original input shape
        z_q = ops.transpose(z_q, (0, 3, 1, 2))  # (b h w c) -> (b c h w)
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


class AutoencoderVQ(nn.Cell):
    def __init__(
        self,
        ddconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        monitor=None,
        use_fp16=False,
        upcast_sigmoid=False,
        batch_resize_range=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    ):
        super().__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(dtype=self.dtype, upcast_sigmoid=upcast_sigmoid, **ddconfig)
        self.decoder = Decoder(dtype=self.dtype, upcast_sigmoid=upcast_sigmoid, **ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1, pad_mode="valid", has_bias=True).to_float(
            self.dtype
        )
        self.post_quant_conv = nn.Conv2d(
            embed_dim, ddconfig["z_channels"], 1, pad_mode="valid", has_bias=True
        ).to_float(self.dtype)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

    def init_from_ckpt(self, path, ignore_keys=list(), remove_prefix=["first_stage_model.", "autoencoder."]):
        sd = ms.load_checkpoint(path)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        vae_prefix = ["encoder.", "decoder.", "quant_conv.", "post_quant_conv.", "quantize."]
        for pname in keys:
            is_vae_param = False
            for pf in remove_prefix:
                if pname.startswith(pf):
                    sd[pname.replace(pf, "")] = sd.pop(pname)
                    is_vae_param = True
            for pf in vae_prefix:
                if pname.startswith(pf):
                    is_vae_param = True
            if not is_vae_param:
                sd.pop(pname)
        ms.load_param_into_net(self, sd, strict_load=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def construct(self, input):
        quant, emb_loss, info = self.encode(input)
        dec = self.decode(quant)
        # if return_pred_indices:
        #     return dec, diff, ind

        return dec, quant, emb_loss
