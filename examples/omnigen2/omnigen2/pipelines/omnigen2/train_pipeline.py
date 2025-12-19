from mindspore import _no_grad, dtype, nn, ops, tensor
from mindspore.communication import GlobalComm, get_group_size, get_rank


class OmniGen2TrainPipeline(nn.Cell):
    def __init__(self, text_encoder, vae, model, freqs_cis, transport):
        super().__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.model = model
        self.freqs_cis = freqs_cis
        self.transport = transport

        # Initialize for distributed training
        self.reduce, self.rank, self.rank_size = None, 0, 1
        if GlobalComm.INITED:
            self.reduce = ops.AllReduce(op=ops.ReduceOp.SUM)
            self.rank, self.rank_size = get_rank(), get_group_size()

    def set_train(self, mode=True):
        # Set the diffusion model only to train or eval mode
        self.model.set_train(mode)

    @_no_grad()
    def encode_vae(self, img):
        z0 = self.vae.diag_gauss_dist.sample(self.vae.encode(img.to(dtype=self.vae.dtype))[0])
        if self.vae.config.shift_factor is not None:
            z0 = z0 - self.vae.config.shift_factor
        if self.vae.config.scaling_factor is not None:
            z0 = z0 * self.vae.config.scaling_factor
        z0 = z0.to(dtype=self.model.dtype)
        return z0

    def construct(self, input_images, output_image, text_input_ids, text_mask):
        with _no_grad():
            text_feats = self.text_encoder(
                input_ids=text_input_ids, attention_mask=text_mask, output_hidden_states=False
            ).last_hidden_state

        input_latents = []
        for i, img in enumerate(input_images):
            if img is not None and len(img) > 0:
                input_latents.append([])
                for j, img_j in enumerate(img):
                    input_latents[i].append(self.encode_vae(img_j).squeeze(0))
            else:
                input_latents.append(None)

        output_latents = [self.encode_vae(img).squeeze(0) for img in output_image]

        model_kwargs = dict(
            text_hidden_states=text_feats,
            text_attention_mask=text_mask,
            ref_image_hidden_states=input_latents,
            freqs_cis=self.freqs_cis,
        )

        num_tokens_in_batch = tensor(sum(latent.numel() for latent in output_latents), dtype=dtype.int32)
        if self.reduce is not None:
            num_tokens_in_batch = self.reduce(num_tokens_in_batch)

        loss = self.transport.training_losses(self.model, output_latents, model_kwargs, reduction="sum").sum()
        return (loss * self.rank_size) / num_tokens_in_batch
