from typing import Optional

from diffusion import SpacedDiffusion
from diffusion.diffusion_utils import _extract_into_tensor, discretized_gaussian_log_likelihood, mean_flat, normal_kl

import mindspore as ms
from mindspore import ParameterTuple, Tensor, context, nn, ops
from mindspore.amp import DynamicLossScaler, LossScaler, StaticLossScaler, all_finite

_ema_op = ops.MultitypeFuncGraph("ema_op")


@_ema_op.register("Tensor", "Tensor", "Tensor")
def ema_op(factor, ema_weight, weight):
    return ops.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


class TrainStep(nn.Cell):
    """Training step with loss scale.

    The steps of model optimization are performed in the following order:
        1. calculate grad
        2. allreduce grad
        3. clip grad [optional]
        4. call optimizer
        5. ema weights [optional]
    """

    def __init__(
        self,
        network: nn.Cell,
        vae: nn.Cell,
        diffusion: SpacedDiffusion,
        optimizer: nn.Optimizer,
        scaler: LossScaler,
        ema_decay: Optional[float] = 0.9999,
        grad_clip_norm: Optional[float] = None,
        scale_factor=0.18215,
    ):
        super().__init__()
        self.network = network.set_grad()
        self.vae = vae
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.scaler = scaler
        if isinstance(self.scaler, StaticLossScaler):
            self.drop_overflow = False
        elif isinstance(self.scaler, DynamicLossScaler):
            self.drop_overflow = True
        else:
            raise NotImplementedError(f"Unsupported scaler: {type(self.scaler)}")
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode == context.ParallelMode.STAND_ALONE:
            self.grad_reducer = lambda x: x
        elif self.parallel_mode in (context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL):
            self.grad_reducer = nn.DistributedGradReducer(self.weights)
        else:
            raise NotImplementedError(f"When creating reducer, Got Unsupported parallel mode: {self.parallel_mode}")
        self._jit_config_dict = network.jit_config_dict

        self.ema_decay = ema_decay
        self.state_dict = ParameterTuple(network.get_parameters())  # same as model.state_dict in torch
        if self.ema_decay:
            self.ema_state_dict = self.state_dict.clone(prefix="ema", init="same")
        else:
            self.ema_state_dict = ParameterTuple(())  # placeholder
        self.grad_clip_norm = grad_clip_norm
        self.hyper_map = ops.HyperMap()
        self.scale_factor = scale_factor

        def forward_fn(x, y, text_embed):
            t = ops.randint(0, diffusion.num_timesteps, (x.shape[0],))
            # model_kwargs = dict(y=y)
            # loss = diffusion.training_losses(network, x, t, model_kwargs)
            noise = ops.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise=noise)
            model_output = network(x_t, t, y=y, text_embed=text_embed)
            B, F, C = x_t.shape[:3]
            assert model_output.shape == (B, F, C * 2) + x_t.shape[3:]
            model_output, model_var_values = ops.split(model_output, C, axis=2)

            # Learn the variance using the variational bound, but don't let it affect our mean prediction.
            # _vb_terms_bpd(model=lambda *_: frozen_out, x_start=x, x_t=x_t, t=t, clip_denoised=False) begin
            true_mean, _, true_log_variance_clipped = diffusion.q_posterior_mean_variance(x_start=x, x_t=x_t, t=t)
            # p_mean_variance(model=lambda *_: frozen_out, x_t, t, clip_denoised=False) begin
            min_log = _extract_into_tensor(diffusion.posterior_log_variance_clipped, t, x_t.shape)
            max_log = _extract_into_tensor(ops.log(diffusion.betas), t, x_t.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            pred_xstart = diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=ops.stop_gradient(model_output))
            model_mean, _, _ = diffusion.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
            # assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
            # p_mean_variance end
            kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
            kl = mean_flat(kl) / 0.693147  # np.log(2.0)
            decoder_nll = -discretized_gaussian_log_likelihood(x, means=model_mean, log_scales=0.5 * model_log_variance)
            decoder_nll = mean_flat(decoder_nll) / 0.693147  # np.log(2.0)
            # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            vb = ops.where((t == 0), decoder_nll.to(kl.dtype), kl)
            # _vb_terms_bpd end

            loss = mean_flat((noise - model_output) ** 2) + vb
            loss = loss.mean()
            loss = scaler.scale(loss)
            return loss

        self.grad_fn = ops.value_and_grad(forward_fn, grad_position=None, weights=self.weights, has_aux=False)

    def get_condition_embeddings(self, text_tokens, control=None):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        if self.cond_stage_trainable:
            text_emb = self.text_encoder(text_tokens)
        else:
            text_emb = ops.stop_gradient(self.text_encoder(text_tokens))

        return text_emb

    def update(self, loss, grads):
        if self.grad_clip_norm:
            loss = ops.depend(loss, self.optimizer(ops.clip_by_global_norm(grads, clip_norm=self.grad_clip_norm)))
        else:
            loss = ops.depend(loss, self.optimizer(grads))

        if self.ema_decay:
            loss = ops.depend(
                loss,
                self.hyper_map(
                    ops.partial(_ema_op, Tensor(self.ema_decay, ms.float32)), self.ema_state_dict, self.state_dict
                ),
            )
        return loss

    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        b, c, h, w = x.shape

        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def vae_decode_video(self, x):
        """
        Args:
            x: (b f c h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """
        b, f, c, h, w = x.shape
        x = x.reshape((b * f, c, h, w))

        y = self.vae_decode(x)
        _, h, w, c = y.shape
        y = y.reshape((b, f, h, w, c))

        return y

    def get_latents(self, x):
        # "b f c h w -> (b f) c h w"
        B, F, C, H, W = x.shape
        if C != 3:
            raise ValueError("Expect input shape (b f 3 h w), but get {}".format(x.shape))
        x = ops.reshape(x, (-1, C, H, W))

        z = ops.stop_gradient(self.vae_encode(x))

        # (b*f c h w) -> (b f c h w)
        z = ops.reshape(z, (B, F, z.shape[1], z.shape[2], z.shape[3]))

        return z

    def construct(self, x: ms.Tensor, text_tokens: ms.Tensor = None, labels: ms.Tensor = None, control=None, **kwargs):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape [bs, F, 3, 256, 256]
            text: text tokens padded to fixed shape [bs, 77]
            control: other conditions for future extension

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume unet3d input/output shape: (b c f h w)
                unet2d input/output shape: (b c h w)
        """
        # 1. get image/video latents z using vae
        x = self.get_latents(x)
        # 2. get conditions
        if text_tokens is not None:
            # assert self.condition == "text", "When text tokens inputs are not None, expect the condition type is `text`"
            # f"but got {self.condition}!"
            text_embed = self.get_condition_embeddings(text_tokens, control)
        else:
            text_embed = None

        if labels is not None:
            # assert self.condition == "class", "When labels inputs are not None, expect the condition type is `class`"
            # f"but got {self.condition}!"
            y = labels
        else:
            y = None
        loss, grads = self.grad_fn(x, y, text_embed)
        grads = self.grad_reducer(grads)
        loss = self.scaler.unscale(loss)
        grads = self.scaler.unscale(grads)

        if self.drop_overflow:
            status = all_finite(grads)
            if status:
                loss = self.update(loss, grads)
            loss = ops.depend(loss, self.scaler.adjust(status))
        else:
            loss = self.update(loss, grads)
        return loss
