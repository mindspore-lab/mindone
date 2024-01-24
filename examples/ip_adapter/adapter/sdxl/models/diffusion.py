from typing import List, Optional

import numpy as np
from gm.models.diffusion import DiffusionEngine, get_batch, get_unique_embedder_keys_from_conditioner

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor


class ControlNetDiffusionEngine(DiffusionEngine):
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
        control: Optional[Tensor] = None,
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
        c, uc = self.conditioner.get_controlnet_unconditional_conditioning(
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

        # support non-guess mode only
        control = ops.concat((control, control), axis=0)

        print("Sample latent Starting...")
        samples_z = sampler(self, randn, cond=c, uc=uc, adapter_states=adapter_states, control=control)
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
