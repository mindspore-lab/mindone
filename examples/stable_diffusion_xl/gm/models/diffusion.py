# reference to https://github.com/Stability-AI/generative-models

from contextlib import contextmanager
from typing import Dict, List, Union

import numpy as np
from gm.modules import UNCONDITIONAL_CONFIG
from gm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from gm.util import append_dims, default, get_obj_from_str, instantiate_from_config
from omegaconf import ListConfig, OmegaConf

import mindspore as ms
from mindspore import Tensor, nn, ops


class DiffusionEngine(nn.Cell):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sigma_sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_amp=False,
        input_key: str = "image",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.no_cond_log = no_cond_log
        self.scale_factor = scale_factor
        self.disable_first_stage_amp = disable_first_stage_amp

        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(model)

        self.denoiser = instantiate_from_config(denoiser_config)
        self.conditioner = instantiate_from_config(default(conditioner_config, UNCONDITIONAL_CONFIG))
        self.first_stage_model = self.init_freeze_first_stage(first_stage_config)

        # for train
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config) if sigma_sampler_config else None
        self.loss_fn = instantiate_from_config(loss_fn_config) if loss_fn_config else None

        if ckpt_path is not None:
            self.load_pretrained(ckpt_path)

    def load_pretrained(self, path: str) -> None:
        if path.endswith("ckpt"):
            sd = ms.load_checkpoint(path)
        else:
            raise NotImplementedError

        missing, unexpected = ms.load_param_into_net(self, sd, strict_load=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            missing = [k for k in missing if (not k.startswith("optimizer.") and not k.startswith("ema."))]
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def init_freeze_first_stage(self, config):
        model = instantiate_from_config(config)
        model.set_train(False)
        model.set_grad(False)
        for _, param in model.parameters_and_names():
            param.requires_grad = False
        return model

    def decode_first_stage(self, z):
        if self.disable_first_stage_amp:
            self.first_stage_model.to_float(ms.float32)
            z = z.astype(ms.float32)

        z = 1.0 / self.scale_factor * z
        out = self.first_stage_model.decode(z)
        return out

    def encode_first_stage(self, x):
        if self.disable_first_stage_amp:
            self.first_stage_model.to_float(ms.float32)
            x = x.astype(ms.float32)

        z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    def openai_input_warpper(self, cond: dict) -> Dict:
        return {
            "concat": cond.get("concat", None),
            "context": cond.get("crossattn", None),
            "y": cond.get("vector", None),
        }

    # TODO: Delete it
    def _denoise(self, sigmas, noised_input, **kwargs):
        c_skip, c_out, c_in, c_noise = self.denoiser(sigmas, noised_input.ndim)
        model_output = self.model(ops.cast(noised_input * c_in, ms.float32), ops.cast(c_noise, ms.int32), **kwargs)
        model_output = model_output * c_out + noised_input * c_skip
        return model_output

    def get_grad_func(self, optimizer, reducer, scaler, jit=True):
        from mindspore.amp import all_finite

        loss_fn = self.loss_fn
        denoiser = self.denoiser
        model = self.model

        def _forward_func(x, noised_input, sigmas, w, concat, context, y):
            c_skip, c_out, c_in, c_noise = denoiser(sigmas, noised_input.ndim)
            model_output = model(
                ops.cast(noised_input * c_in, ms.float32),
                ops.cast(c_noise, ms.int32),
                concat=concat,
                context=context,
                y=y,
            )
            model_output = model_output * c_out + noised_input * c_skip
            loss = loss_fn(model_output, x, w)
            loss = loss.mean()
            return scaler.scale(loss)

        grad_fn = ops.value_and_grad(_forward_func, grad_position=None, weights=optimizer.parameters)

        def grad_and_update_func(x, noised_input, sigmas, w, concat, context, y):
            loss, grads = grad_fn(x, noised_input, sigmas, w, concat, context, y)
            grads = reducer(grads)
            unscaled_grads = scaler.unscale(grads)
            grads_finite = all_finite(unscaled_grads)
            loss = ops.depend(loss, optimizer(unscaled_grads))
            return scaler.unscale(loss), unscaled_grads, grads_finite

        @ms.jit
        def jit_warpper(*args, **kwargs):
            return grad_and_update_func(*args, **kwargs)

        return grad_and_update_func if not jit else jit_warpper

    def train_step(self, batch, grad_func):
        # get latent and condition
        x = batch[self.input_key]
        x = self.encode_first_stage(x)
        cond = self.conditioner(batch)
        cond = self.openai_input_warpper(cond)
        sigmas = self.sigma_sampler(x.shape[0])
        noise = ops.randn_like(x)
        noised_input = self.loss_fn.get_noise_input(x, noise, sigmas)
        w = append_dims(self.denoiser.w(sigmas), x.ndim)

        # get loss
        print("Compute Loss Starting...")
        loss, _, _ = grad_func(x, noised_input, sigmas, w, **cond)
        # noised_input = Tensor(np.random.randn(1, 4, 128, 128), dtype=ms.float16)
        # loss, _, _ = grad_func(noised_input, cond['concat'], cond['context'], cond['y'])
        print("Compute Loss Done...")

        return loss

    def on_train_start(self, *args, **kwargs):
        if self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema:
            # TODO: Add ema store and copy
            # self.model_ema.store(self.model.parameters())
            # self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.ema:
                # TODO: Add ema restore
                # self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, learning_rate, cfg):
        return get_obj_from_str(cfg["target"])(params, learning_rate=learning_rate, **cfg.get("params", dict()))

    def configure_optimizers(self):
        # TODO: Add Optimizer lr scheduler
        pass

    def do_sample(
        self,
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings: List = None,
        batch2model_input: List = None,
        return_latents=False,
        filter=None,
        amp_level="O0",
    ):
        from gm.helpers import get_batch, get_unique_embedder_keys_from_conditioner

        print("Sampling")

        dtype = ms.float32 if amp_level not in ("O2", "O3") else ms.float16

        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        if batch2model_input is None:
            batch2model_input = []

        num_samples = [num_samples]
        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(self.conditioner), value_dict, num_samples, dtype=dtype
        )
        for key in batch:
            if isinstance(batch[key], Tensor):
                print(key, batch[key].shape)
            elif isinstance(batch[key], list):
                print(key, [len(i) for i in batch[key]])
            else:
                print(key, batch[key])
        print("Get Condition Done.")

        print("Embedding Starting...")
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=force_uc_zero_embeddings,
        )
        print("Embedding Done.")

        for k in c:
            if not k == "crossattn":
                c[k], uc[k] = map(
                    lambda y: y[k][: int(np.prod(num_samples))],
                    (c, uc)
                    # lambda y: y[k][: math.prod(num_samples)], (c, uc)
                )

        additional_model_inputs = {}
        for k in batch2model_input:
            additional_model_inputs[k] = batch[k]

        shape = (np.prod(num_samples), C, H // F, W // F)
        randn = Tensor(np.random.randn(*shape), ms.float32)

        print("Sample latent Starting...")
        samples_z = sampler(self, randn, cond=c, uc=uc)
        print("Sample latent Done.")

        print("Decode latent Starting...")
        samples_x = self.decode_first_stage(samples_z)
        samples_x = samples_x.asnumpy()
        print("Decode latent Done.")

        samples = np.clip((samples_x + 1.0) / 2.0, a_min=0.0, a_max=1.0)

        if filter is not None:
            print("Filter Starting...")
            samples = filter(samples)
            print("Filter Done.")

        if return_latents:
            return samples, samples_z
        return samples

    def log_conditionings(self):
        # TODO
        raise NotImplementedError

    def log_images(self):
        # TODO
        raise NotImplementedError
