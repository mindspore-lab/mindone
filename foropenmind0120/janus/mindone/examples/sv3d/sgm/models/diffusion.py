# reference to https://github.com/Stability-AI/generative-models

import copy
from typing import Dict, List, Optional, Union

import numpy as np
from omegaconf import ListConfig, OmegaConf
from sgm.helpers import get_batch, get_unique_embedder_keys_from_conditioner
from sgm.modules.diffusionmodules.wrappers import OpenAIWrapper
from sgm.util import append_dims, get_obj_from_str, instantiate_from_config

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
        scale_factor: float = 1.0,
        latents_mean: Union[List, ListConfig, None] = None,
        latents_std: Union[List, ListConfig, None] = None,
        disable_first_stage_amp=False,
        input_key: str = "image",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        load_first_stage_model: bool = True,
        load_conditioner: bool = True,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.no_cond_log = no_cond_log
        self.scale_factor = scale_factor
        self.latents_mean = list(latents_mean) if latents_mean else latents_mean
        self.latents_std = list(latents_std) if latents_std else latents_std
        self.disable_first_stage_amp = disable_first_stage_amp

        # 1. unet
        if network_config is not None:
            model = instantiate_from_config(network_config)
            self.model = (get_obj_from_str(network_wrapper) if network_wrapper is not None else OpenAIWrapper)(model)
        else:
            self.model = None

        # 2. vae
        self.first_stage_model = self.init_freeze_first_stage(first_stage_config) if load_first_stage_model else None

        # 3. text-encoders and vector conditioner
        # ? should not wrapped it with tuple? SHOULD NOT USE default in graph mode...
        self.conditioner = instantiate_from_config(conditioner_config) if load_conditioner else None

        self.denoiser = instantiate_from_config(denoiser_config)

        # # comment out sampler when using new sampler.py, as the new one init a model within sampler's instance method
        # self.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None

        # for train
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config) if sigma_sampler_config else None
        self.loss_fn = instantiate_from_config(loss_fn_config) if loss_fn_config else None

        if ckpt_path is not None:
            self.load_pretrained(ckpt_path)

    def load_pretrained(self, ckpts, verbose=True):
        load_first_stage_model = self.first_stage_model is not None
        load_conditioner = self.conditioner is not None

        if ckpts:
            print(f"Loading model from {ckpts} within diffusion model")
            if isinstance(ckpts, str):
                ckpts = [ckpts]

            sd_dict = {}
            for ckpt in ckpts:
                assert ckpt.endswith(".ckpt")
                _sd_dict = ms.load_checkpoint(ckpt)
                sd_dict.update(_sd_dict)

                if "global_step" in sd_dict:
                    global_step = sd_dict["global_step"]
                    print(f"loaded ckpt from global step {global_step}")
                    print(f"Global Step: {sd_dict['global_step']}")

            # FIXME: parameter auto-prefix name bug on mindspore 2.2.10
            sd_dict = {k.replace("._backbone", ""): v for k, v in sd_dict.items()}

            # filter first_stage_model and conditioner
            _keys = copy.deepcopy(list(sd_dict.keys()))
            for _k in _keys:
                if not load_first_stage_model and _k.startswith("first_stage_model."):
                    sd_dict.pop(_k)
                if not load_conditioner and _k.startswith("conditioner."):
                    sd_dict.pop(_k)

            m, u = ms.load_param_into_net(self, sd_dict, strict_load=False)

            if len(m) > 0 and verbose:
                ignore_lora_key = len(ckpts) == 1
                m = m if not ignore_lora_key else [k for k in m if "lora_" not in k]
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)
        else:
            print("WARNING: No checkpoints were provided.")

    def init_freeze_first_stage(self, config):
        model = instantiate_from_config(config)
        model.set_train(False)
        model.set_grad(False)
        for _, param in model.parameters_and_names():
            param.requires_grad = False
        return model

    def decode_first_stage(self, z):
        if self.disable_first_stage_autocast:
            self.first_stage_model.to_float(ms.float32)
            z = z.astype(ms.float32)

        if self.latents_mean and self.latents_std:
            latents_mean = Tensor(self.latents_mean, dtype=ms.float32).reshape(1, 4, 1, 1)
            latents_std = Tensor(self.latents_std, dtype=ms.float32).reshape(1, 4, 1, 1)
            z = z * latents_std / self.scale_factor + latents_mean
        else:
            z = 1.0 / self.scale_factor * z
        out = self.first_stage_model.decode(z)
        return out

    def encode_first_stage(self, x):
        if self.disable_first_stage_autocast:
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

    def get_grad_func(self, optimizer, reducer, scaler, jit=True, overflow_still_update=True):
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
            if overflow_still_update:
                loss = ops.depend(loss, optimizer(unscaled_grads))
            else:
                if grads_finite:
                    loss = ops.depend(loss, optimizer(unscaled_grads))
            overflow_tag = not grads_finite
            return scaler.unscale(loss), unscaled_grads, overflow_tag

        @ms.jit
        def jit_warpper(*args, **kwargs):
            return grad_and_update_func(*args, **kwargs)

        return grad_and_update_func if not jit else jit_warpper

    def train_step_pynative(self, x, *tokens, grad_func=None):
        # get latent
        x = self.encode_first_stage(x)

        # get condition
        vector, crossattn, concat = self.conditioner.embedding(*tokens)
        cond = {"context": crossattn, "y": vector, "concat": concat}

        # get noise and sigma
        sigmas = self.sigma_sampler(x.shape[0])
        noise = ops.randn_like(x)
        noised_input = self.loss_fn.get_noise_input(x, noise, sigmas)
        w = append_dims(self.denoiser.w(sigmas), x.ndim)

        # compute loss
        print("Compute Loss Starting...")
        loss, _, overflow = grad_func(x, noised_input, sigmas, w, **cond)
        print("Compute Loss Done...")

        return loss, overflow

    def instantiate_optimizer_from_config(self, params, learning_rate, cfg):
        return get_obj_from_str(cfg["target"])(params, learning_rate=learning_rate, **cfg.get("params", dict()))

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
        adapter_states: Optional[List[Tensor]] = None,
        amp_level="O0",
        init_latent_path=None,  # '/path/to/sdxl_init_latent.npy'
        init_noise_scheduler_path=None,  # '/path/to/euler_a_noise_scheduler.npy'
        control: Optional[Tensor] = None,
        lpw=False,
        max_embeddings_multiples=4,
    ):
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
            lpw=lpw,
            max_embeddings_multiples=max_embeddings_multiples,
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
        if init_latent_path is not None:
            print("Loading latent noise from ", init_latent_path)
            randn = Tensor(np.load(init_latent_path), ms.float32)
            # assert randn.shape==shape, 'unmatch shape due to loaded noise'
        else:
            randn = Tensor(np.random.randn(*shape), ms.float32)

        if init_noise_scheduler_path is not None:
            init_noise_scheduler = Tensor(np.load(init_noise_scheduler_path), ms.float32)
        else:
            init_noise_scheduler = None

        print("Sample latent Starting...")
        samples_z = sampler(
            self,
            randn,
            cond=c,
            uc=uc,
            adapter_states=adapter_states,
            control=control,
            init_noise_scheduler=init_noise_scheduler,
        )
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
