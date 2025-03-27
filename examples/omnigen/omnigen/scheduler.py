from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm

import mindspore as ms
import mindspore.ops as ops

from mindone.transformers.cache_utils import DynamicCache


class OmniGenDynamicCache(DynamicCache):
    def __init__(
        self,
        num_tokens_for_img: int,
    ) -> None:
        super().__init__()
        self.num_tokens_for_img = num_tokens_for_img

    def update(
        self,
        key_states: ms.Tensor,
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.key_cache) == layer_idx:
            # only cache the states for condition tokens
            key_states = key_states[..., : -(self.num_tokens_for_img + 1), :]
            value_states = value_states[..., : -(self.num_tokens_for_img + 1), :]
            # Update the number of seen tokens
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        elif len(self.key_cache[layer_idx]) == 0:
            key_states = key_states[..., : -(self.num_tokens_for_img + 1), :]
            value_states = value_states[..., : -(self.num_tokens_for_img + 1), :]
            return key_states, value_states
        else:
            # only cache the states for condition tokens
            key_tensor, value_tensor = self[layer_idx]
            k = ops.cat([key_tensor, key_states], axis=-2)
            v = ops.cat([value_tensor, value_states], axis=-2)
            return k, v


class OmniGenScheduler:
    """Scheduler for the diffusion process"""

    def __init__(self, num_steps: int = 50, time_shifting_factor: int = 1):
        self.num_steps = num_steps
        self.time_shift = time_shifting_factor

        t = ms.numpy.linspace(0, 1, num_steps + 1)
        t = t / (t + time_shifting_factor - time_shifting_factor * t)
        self.sigma = t

    def crop_position_ids_for_cache(self, position_ids, num_tokens_for_img):
        """Crop position IDs for cache"""
        if isinstance(position_ids, list):
            for i in range(len(position_ids)):
                position_ids[i] = position_ids[i][:, -(num_tokens_for_img + 1) :]
        else:
            position_ids = position_ids[:, -(num_tokens_for_img + 1) :]
        return position_ids

    def crop_attention_mask_for_cache(self, attention_mask, num_tokens_for_img):
        """Crop attention mask for cache"""
        if isinstance(attention_mask, list):
            return [x[..., -(num_tokens_for_img + 1) :, :] for x in attention_mask]
        return attention_mask[..., -(num_tokens_for_img + 1) :, :]

    def crop_cache(self, cache, num_tokens_for_img):
        """Crop cache to remove unneeded tokens"""
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][..., : -(num_tokens_for_img + 1), :]
            cache.value_cache[i] = cache.value_cache[i][..., : -(num_tokens_for_img + 1), :]
        return cache

    def __call__(self, z, func, model_kwargs, use_kv_cache: bool = True):
        """Run the diffusion process"""
        num_tokens_for_img = z.shape[-1] * z.shape[-2] // 4

        # Initialize cache
        if isinstance(model_kwargs["input_ids"], list):
            cache = (
                [OmniGenDynamicCache(num_tokens_for_img) for _ in range(len(model_kwargs["input_ids"]))]
                if use_kv_cache
                else None
            )
        else:
            cache = OmniGenDynamicCache(num_tokens_for_img) if use_kv_cache else None

        # cache = None
        # Run diffusion steps
        for i in tqdm(range(self.num_steps)):
            # import pdb
            # pdb.set_trace()
            timesteps = ops.zeros(len(z)) + self.sigma[i]
            timesteps.to(z.dtype)
            pred, cache = func(z, timesteps, past_key_values=cache, **model_kwargs)
            # print(pred, cache)
            sigma_next = self.sigma[i + 1]
            sigma = self.sigma[i]
            z = z + (sigma_next - sigma) * pred

            # Update model kwargs for caching after first step
            if i == 0 and use_kv_cache:
                num_tokens_for_img = z.shape[-1] * z.shape[-2] // 4
                if isinstance(cache, list):
                    model_kwargs["input_ids"] = [None] * len(cache)
                else:
                    model_kwargs["input_ids"] = None

                model_kwargs["position_ids"] = self.crop_position_ids_for_cache(
                    model_kwargs["position_ids"], num_tokens_for_img
                )
                model_kwargs["attention_mask"] = self.crop_attention_mask_for_cache(
                    model_kwargs["attention_mask"], num_tokens_for_img
                )

        del cache
        return z
