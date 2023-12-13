# reference to https://github.com/Stability-AI/generative-models

from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import numpy as np
from gm.helpers import get_batch, get_unique_embedder_keys_from_conditioner
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

        if network_config is not None:
            model = instantiate_from_config(network_config)
            self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(model)
        else:
            self.model = None

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

        print("Sample latent Starting...")
        samples_z = sampler(self, randn, cond=c, uc=uc, adapter_states=adapter_states)
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

    def do_img2img(
        self,
        img,
        sampler,
        value_dict,
        num_samples,
        force_uc_zero_embeddings=[],
        additional_kwargs={},
        offset_noise_level: int = 0.0,
        return_latents=False,
        skip_encode=False,
        filter=None,
        add_noise=True,
        amp_level="O0",
    ):
        dtype = ms.float32 if amp_level not in ("O2", "O3") else ms.float16

        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(self.conditioner), value_dict, [num_samples], dtype=dtype
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
            c[k], uc[k] = map(lambda y: y[k][:num_samples], (c, uc))
        for k in additional_kwargs:
            c[k] = uc[k] = additional_kwargs[k]

        z = img if skip_encode else self.encode_first_stage(img)
        noise = ops.randn_like(z)

        sigmas = sampler.discretization(sampler.num_steps)
        sigma = Tensor(sigmas[0], z.dtype)
        print(f"all sigmas: {sigmas}")
        print(f"noising sigma: {sigmas[0]}")

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(ops.randn(z.shape[0], dtype=z.dtype), z.ndim)
        if add_noise:
            noised_z = z + noise * append_dims(sigma, z.ndim)
            noised_z = noised_z / ops.sqrt(
                1.0 + sigma**2.0
            )  # Note: hardcoded to DDPM-like scaling. need to generalize later.
        else:
            noised_z = z / ops.sqrt(1.0 + sigma**2.0)

        print("Sample latent Starting...")
        samples_z = sampler(self, noised_z, cond=c, uc=uc)
        print("Sample latent Done.")

        print("Decode latent Starting...")
        samples_x = self.decode_first_stage(samples_z)
        samples_x = samples_x.asnumpy()
        print("Decode latent Done.")

        samples = np.clip((samples_x + 1.0) / 2.0, a_min=0.0, a_max=1.0)

        if filter is not None:
            samples = filter(samples)

        if return_latents:
            return samples, samples_z

        return samples


class DiffusionEngineDreamBooth(DiffusionEngine):
    def __init__(self, prior_loss_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_loss_weight = prior_loss_weight

    def get_grad_func(self, optimizer, reducer, scaler, jit=True, overflow_still_update=True):
        from mindspore.amp import all_finite

        loss_fn = self.loss_fn
        denoiser = self.denoiser
        model = self.model

        def _shared_step(x, noised_input, sigmas, w, concat, context, y):
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
            return loss

        def _forward_func(
            x,
            noised_input,
            sigmas,
            w,
            concat,
            context,
            y,
            reg_x,
            reg_noised_input,
            reg_sigmas,
            reg_w,
            reg_concat,
            reg_context,
            reg_y,
        ):
            loss_train = _shared_step(x, noised_input, sigmas, w, concat, context, y)
            loss_reg = _shared_step(reg_x, reg_noised_input, reg_sigmas, reg_w, reg_concat, reg_context, reg_y)
            loss = loss_train + self.prior_loss_weight * loss_reg
            return scaler.scale(loss)

        grad_fn = ops.value_and_grad(_forward_func, grad_position=None, weights=optimizer.parameters)

        def grad_and_update_func(
            x,
            noised_input,
            sigmas,
            w,
            concat,
            context,
            y,
            reg_x,
            reg_noised_input,
            reg_sigmas,
            reg_w,
            reg_concat,
            reg_context,
            reg_y,
        ):
            loss, grads = grad_fn(
                x,
                noised_input,
                sigmas,
                w,
                concat,
                context,
                y,
                reg_x,
                reg_noised_input,
                reg_sigmas,
                reg_w,
                reg_concat,
                reg_context,
                reg_y,
            )
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

    def _get_inputs(self, x, *tokens):
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
        return x, noised_input, sigmas, w, cond

    def train_step_pynative(self, instance_image, class_image, *all_tokens, grad_func=None):
        assert len(all_tokens) % 2 == 0
        position = len(all_tokens) // 2
        instance_tokens, class_tokens = all_tokens[:position], all_tokens[position:]

        # get latent and condition
        x, noised_input, sigmas, w, cond = self._get_inputs(instance_image, *instance_tokens)
        reg_x, reg_noised_input, reg_sigmas, reg_w, reg_cond = self._get_inputs(class_image, *class_tokens)

        concat, context, y = cond["concat"], cond["context"], cond["y"]
        reg_concat, reg_context, reg_y = reg_cond["concat"], reg_cond["context"], reg_cond["y"]

        # get loss
        print("Compute Loss Starting...")
        loss, _, overflow = grad_func(
            x,
            noised_input,
            sigmas,
            w,
            concat,
            context,
            y,
            reg_x,
            reg_noised_input,
            reg_sigmas,
            reg_w,
            reg_concat,
            reg_context,
            reg_y,
        )
        print("Compute Loss Done...")

        return loss, overflow


class DiffusionEngineMultiGraph(DiffusionEngine):
    def __init__(self, **kwargs):
        network_config = kwargs.pop("network_config", None)
        if not network_config["target"] == "gm.modules.diffusionmodules.openaimodel.UNetModel":
            raise NotImplementedError
        kwargs["network_config"] = None

        super(DiffusionEngineMultiGraph, self).__init__(**kwargs)

        from gm.modules.diffusionmodules.openaimodel import UNetModelStage1, UNetModelStage2
        from gm.modules.diffusionmodules.wrappers import IdentityWrapper

        params = network_config["params"]
        self.stage1 = IdentityWrapper(UNetModelStage1(**params))
        self.stage2 = IdentityWrapper(UNetModelStage2(**params))
        self.model = None

    def load_pretrained(self, ckpts, verbose=True):
        if ckpts:
            print(f"Loading model from {ckpts}")
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

            # filter for multi-stage model
            new_stage_dict = {}
            for k in sd_dict:
                if k.startswith("model.diffusion_model."):
                    if (
                        k.startswith("model.diffusion_model.output_blocks")
                        or k.startswith("model.diffusion_model.out")
                        or k.startswith("model.diffusion_model.id_predictor")
                    ):
                        new_k = "stage2" + k[len("model") :]
                    else:
                        new_k = "stage1" + k[len("model") :]
                else:
                    new_k = k

                new_stage_dict[new_k] = sd_dict[k]

            m, u = ms.load_param_into_net(self, new_stage_dict, strict_load=False)

            if len(m) > 0 and verbose:
                ignore_lora_key = len(ckpts) == 1
                m = m if not ignore_lora_key else [k for k in m if "lora_" not in k]
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)
        else:
            print(f"Warning: DiffusionEngineMultiGraph, Loading checkpoint from {ckpts} fail")

    def save_checkpoint(self, save_ckpt_dir):
        ckpt = []
        for n, p in self.parameters_and_names():
            new_n = n[:]

            # FIXME: save checkpoint bug on mindspore 2.2.0
            if "._backbone" in new_n:
                _index = new_n.find("._backbone")
                new_n = new_n[:_index] + new_n[_index + len("._backbone") :]

            if new_n.startswith("stage1."):
                new_n = "model." + new_n[len("stage1.") :]
            elif new_n.startswith("stage2."):
                new_n = "model." + new_n[len("stage2.") :]

            ckpt.append({"name": new_n, "data": Tensor(p.asnumpy())})

        ms.save_checkpoint(ckpt, save_ckpt_dir)
        print(f"Save checkpoint to {save_ckpt_dir}")

    def do_sample(self, *args, **kwargs):
        raise NotImplementedError

    def do_img2img(self, *args, **kwargs):
        raise NotImplementedError

    def get_grad_func(self, *args, **kwargs):
        raise NotImplementedError

    def train_step_pynative(self, *args, **kwargs):
        raise NotImplementedError
