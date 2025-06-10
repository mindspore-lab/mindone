import numpy as np
from PIL import Image
from transformers import PretrainedConfig
from transformers.models.auto import AutoConfig

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint, ops

from mindone.transformers.mindspore_adapter.utils import _DTYPE_2_MAX, _DTYPE_2_MIN
from mindone.transformers.models.auto import AutoModel, AutoModelForCausalLM

from .modeling_llada import LLaDAModelLM
from .sampling import cosine_schedule, mask_by_random_topk


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    MindSpore uses float32 for better inference speed
    """
    if temperature == 0:
        return logits
    logits = logits.to(ms.float32)
    noise = mint.rand_like(logits, dtype=ms.float32)
    gumbel_noise = (-mint.log(noise)) ** temperature
    return mint.exp(logits) / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mint.sum(mask_index, dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = mint.zeros((mask_num.shape[0], steps), dtype=ms.int64) + base

    for i in range(mask_num.shape[0]):
        num_transfer_tokens[i, : remainder[0, i]] += 1

    return num_transfer_tokens


class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "codebook_size",
            "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])


class MMadaModelLM(LLaDAModelLM):
    config_class = MMadaConfig
    base_model_prefix = "model"

    def __init__(self, config: MMadaConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)

    def t2i_generate(
        self,
        input_ids: ms.Tensor = None,
        uncond_input_ids: ms.Tensor = None,
        attention_mask=None,
        uncond_attention_mask=None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        generator: ms.Generator = None,
        config=None,
        seq_len=1024,
        mask_token_id=126336,
        resolution=512,
        codebook_size=8192,
        **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1) : -1].clone()
        input_ids_minus_lm_vocab_size = mint.where(
            input_ids_minus_lm_vocab_size == mask_token_id,
            mask_token_id,
            input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens,
        )

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, : resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = mint.cat([uncond_prefix, input_ids[:, resolution + 1 :]], dim=1)
                model_input = mint.cat([input_ids, uncond_input_ids])
                attention_mask = mint.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] * attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self.construct(model_input, attention_bias=attention_bias, return_dict=False)[0]
                cond_logits, uncond_logits = mint.chunk(logits, 2, dim=0)
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[
                    :,
                    -(num_vq_tokens + 1) : -1,
                    len(uni_prompting.text_tokenizer)
                    + num_new_special_tokens : len(uni_prompting.text_tokenizer)
                    + num_new_special_tokens
                    + codebook_size,
                ]
            else:
                attention_bias = (attention_mask[:, :, None] * attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self.construct(input_ids, attention_bias=attention_bias, return_dict=False)[0]
                logits = logits[
                    :,
                    -(num_vq_tokens + 1) : -1,
                    len(uni_prompting.text_tokenizer)
                    + num_new_special_tokens : len(uni_prompting.text_tokenizer)
                    + num_new_special_tokens
                    + codebook_size,
                ]

            probs = mint.softmax(logits, dim=-1)
            sampled = probs.reshape(-1, logits.shape[-1])
            sampled_ids = mint.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = mint.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)

            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(ms.Tensor(ratio))
            selected_probs = mint.gather(probs, -1, sampled_ids.to(ms.int64)[..., None])
            selected_probs = selected_probs.squeeze(-1)

            selected_probs = mint.where(unknown_map, selected_probs, _DTYPE_2_MAX[selected_probs.dtype])
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0)
            mask_len = mint.max(
                ms.Tensor([1], dtype=ms.int32), mint.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )

            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            input_ids[:, -(num_vq_tokens + 1) : -1] = mint.where(
                masking, mask_token_id, sampled_ids + len(uni_prompting.text_tokenizer) + num_new_special_tokens
            )
            input_ids_minus_lm_vocab_size = mint.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    def construct_process(
        self,
        input_ids,
        labels,
        batch_size_t2i=0,
        batch_size_lm=0,
        batch_size_mmu=0,
        max_seq_length=128,
        p_mask_lm=None,
        p_mask_mmu=None,
        answer_lengths=None,
        t2i_masks=None,
        answer_lengths_lm=None,
    ):
        attention_bias = mint.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] * t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self.construct(input_ids, attention_bias=attention_bias, return_dict=False)[0]
        self.output_size = logits.shape[-1]

        if batch_size_t2i == 0:
            loss_t2i = ms.Tensor(0.0, dtype=ms.float32)
        else:
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1 :].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1 :].contiguous().view(-1),
                ignore_index=-100,
            )

        masked_indices = input_ids == self.config.mask_token_id
        masked_indices_lm = masked_indices[batch_size_t2i : batch_size_t2i + batch_size_lm]
        masked_indices_mmu = masked_indices[-batch_size_mmu:]

        loss_lm = (
            F.cross_entropy(
                logits[batch_size_t2i : batch_size_t2i + batch_size_lm][masked_indices_lm]
                .contiguous()
                .view(-1, self.output_size),
                labels[batch_size_t2i : batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1),
                ignore_index=-100,
                reduction="none",
            )
            / p_mask_lm[masked_indices_lm]
        )
        loss_lm = loss_lm.sum() / (
            logits[batch_size_t2i : batch_size_t2i + batch_size_lm].shape[0]
            * logits[batch_size_t2i : batch_size_t2i + batch_size_lm].shape[1]
        )

        loss_lm = mint.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (
            logits[batch_size_t2i : batch_size_t2i + batch_size_lm].shape[0]
        )

        loss_mmu = (
            F.cross_entropy(
                logits[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1, self.output_size),
                labels[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1),
                ignore_index=-100,
                reduction="none",
            )
            / p_mask_mmu[masked_indices_mmu]
        )
        loss_mmu = mint.sum(loss_mmu / answer_lengths[masked_indices_mmu]) / (logits[-batch_size_mmu:].shape[0])

        return logits, loss_t2i, loss_lm, loss_mmu

    def construct_process_with_r2i(
        self,
        input_ids,
        labels,
        t2i_masks=None,
        max_seq_length=128,
        batch_size_t2i=0,
        batch_size_lm=0,
        batch_size_mmu=0,
        batch_size_r2i=0,
        p_mask_lm=None,
        p_mask_mmu=None,
        p_mask_r2i=None,
        answer_lengths=None,
        answer_lengths_lm=None,
        answer_lengths_r2i=None,
    ):
        attention_bias = mint.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] * t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self.construct(input_ids, attention_bias=attention_bias, return_dict=False)[0]
        self.output_size = logits.shape[-1]

        if batch_size_t2i == 0:
            loss_t2i = ms.Tensor(0.0, dtype=ms.float32)
        else:
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1 :].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1 :].contiguous().view(-1),
                ignore_index=-100,
            )

        start_lm = batch_size_t2i
        end_lm = start_lm + batch_size_lm
        start_mmu = end_lm
        end_mmu = start_mmu + batch_size_mmu
        start_r2i = end_mmu
        end_r2i = start_r2i + batch_size_r2i

        masked_indices = input_ids == self.config.mask_token_id
        masked_indices_lm = masked_indices[start_lm:end_lm]
        masked_indices_mmu = masked_indices[start_mmu:end_mmu]
        masked_indices_r2i = masked_indices[start_r2i:end_r2i]

        loss_lm = (
            F.cross_entropy(
                logits[start_lm:end_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
                labels[start_lm:end_lm][masked_indices_lm].contiguous().view(-1),
                ignore_index=-100,
                reduction="none",
            )
            / p_mask_lm[masked_indices_lm]
        )
        loss_lm = loss_lm.sum() / (logits[start_lm:end_lm].shape[0] * logits[start_lm:end_lm].shape[1])
        loss_lm = mint.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[start_lm:end_lm].shape[0])

        loss_mmu = (
            F.cross_entropy(
                logits[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1, self.output_size),
                labels[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1),
                ignore_index=-100,
                reduction="none",
            )
            / p_mask_mmu[masked_indices_mmu]
        )
        loss_mmu = mint.sum(loss_mmu / answer_lengths[masked_indices_mmu]) / (logits[start_mmu:end_mmu].shape[0])

        loss_r2i = (
            F.cross_entropy(
                logits[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1, self.output_size),
                labels[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1),
                ignore_index=-100,
                reduction="none",
            )
            / p_mask_r2i[masked_indices_r2i]
        )
        loss_r2i = mint.sum(loss_r2i / answer_lengths_r2i[masked_indices_r2i]) / (logits[start_r2i:end_r2i].shape[0])

        return logits, loss_t2i, loss_lm, loss_mmu, loss_r2i

    def construct_t2i(self, input_ids, labels, batch_size_t2i=0, max_seq_length=128, t2i_masks=None):
        attention_bias = mint.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] * t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self.construct(input_ids, attention_bias=attention_bias, return_dict=False)[0]
        self.output_size = logits.shape[-1]

        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1 :].contiguous().view(-1, self.output_size),
            labels[:batch_size_t2i, max_seq_length + 1 :].contiguous().view(-1),
            ignore_index=-100,
        )

        return loss_t2i

    def mmu_generate(
        self,
        idx=None,
        input_embeddings=None,
        max_new_tokens=128,
        steps=128,
        block_length=128,
        temperature=0.0,
        top_k=None,
        eot_token=None,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        attention_mask=None,
    ):
        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] * attention_mask[:, None, :]).bool().unsqueeze(1)
        else:
            attention_bias = None

        # result = []
        batch_size = idx.shape[0]
        x = mint.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=ms.int64)
        x[:, : idx.shape[1]] = idx.clone()
        prompt_index = x != mask_id

        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (
                x[:, idx.shape[1] + num_block * block_length : idx.shape[1] + (num_block + 1) * block_length :]
                == mask_id
            )
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = mint.cat([x, un_x], dim=0)
                    logits = self.construct(x_, return_dict=False)[0]
                    logits, un_logits = mint.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.construct(x, attention_bias=attention_bias, return_dict=False)[0]

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = mint.argmax(logits_with_noise, dim=-1)
                if remasking == "low_confidence":
                    p = mint.softmax(logits.to(ms.float32), dim=-1)
                    x0_p = mint.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = mint.rand((x0.shape[0], x0.shape[1]))
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length :] = _DTYPE_2_MIN[x0_p.dtype]

                x0 = mint.where(mask_index, x0, x)
                confidence = mint.where(mask_index, x0_p, _DTYPE_2_MIN[x0_p.dtype])

                transfer_index = mint.zeros_like(x0, dtype=ms.bool_)
                for j in range(confidence.shape[0]):
                    _, select_index = ops.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x

    def mmu_generate_fast(
        self,
        idx=None,
        input_embeddings=None,
        max_new_tokens=128,
        steps=128,
        block_length=128,
        temperature=0.0,
        top_k=None,
        eot_token=None,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        attention_mask=None,
    ):
        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] * attention_mask[:, None, :]).bool().unsqueeze(1)
        else:
            attention_bias = None

        # result = []
        batch_size = idx.shape[0]
        x = mint.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=ms.int64)
        x[:, : idx.shape[1]] = idx.clone()
        prompt_index = x != mask_id

        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (
                x[:, idx.shape[1] + num_block * block_length : idx.shape[1] + (num_block + 1) * block_length :]
                == mask_id
            )
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = mint.cat([x, un_x], dim=0)
                    logits = self.construct(x_, return_dict=False)[0]
                    logits, un_logits = mint.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.construct(x, attention_bias=attention_bias, return_dict=False)[0]

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = mint.argmax(logits_with_noise, dim=-1)
                if remasking == "low_confidence":
                    p = mint.softmax(logits.to(ms.float32), dim=-1)
                    x0_p = mint.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = mint.rand((x0.shape[0], x0.shape[1]))
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length :] = _DTYPE_2_MIN[x0_p.dtype]

                x0 = mint.where(mask_index, x0, x)
                confidence = mint.where(mask_index, x0_p, _DTYPE_2_MIN[x0_p.dtype])

                transfer_index = mint.zeros_like(x0, dtype=ms.bool_)
                for j in range(confidence.shape[0]):
                    _, select_index = ops.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if mint.all(tokens_at_block_end == eot_token):
                        break
        return x

    def t2i_generate_decoding_stepwise(
        self,
        input_ids: ms.Tensor = None,
        uncond_input_ids: ms.Tensor = None,
        attention_mask=None,
        uncond_attention_mask=None,
        temperature=1.0,
        timesteps=18,
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        generator: ms.Generator = None,
        config=None,
        seq_len=1024,
        mask_token_id=126336,
        resolution=512,
        codebook_size=8192,
        vq_model=None,
        **kwargs,
    ):
        # mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1) : -1].clone()
        input_ids_minus_lm_vocab_size = mint.where(
            input_ids_minus_lm_vocab_size == mask_token_id,
            mask_token_id,
            input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens,
        )

        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, : resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = mint.cat([uncond_prefix, input_ids[:, resolution + 1 :]], dim=1)
                model_input = mint.cat([input_ids, uncond_input_ids])
                attention_mask = mint.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] * attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self.construct(model_input, attention_bias=attention_bias, return_dict=False)[0]
                cond_logits, uncond_logits = mint.chunk(logits, 2, dim=0)
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[
                    :,
                    -(num_vq_tokens + 1) : -1,
                    len(uni_prompting.text_tokenizer)
                    + num_new_special_tokens : len(uni_prompting.text_tokenizer)
                    + num_new_special_tokens
                    + codebook_size,
                ]
            else:
                attention_bias = (attention_mask[:, :, None] * attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self.construct(input_ids, attention_bias=attention_bias, return_dict=False)[0]
                logits = logits[
                    :,
                    -(num_vq_tokens + 1) : -1,
                    len(uni_prompting.text_tokenizer)
                    + num_new_special_tokens : len(uni_prompting.text_tokenizer)
                    + num_new_special_tokens
                    + codebook_size,
                ]

            probs = mint.softmax(logits, dim=-1)
            sampled = probs.reshape(-1, logits.shape[-1])
            sampled_ids = mint.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = mint.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            current_image_vq_indices = sampled_ids.clone()
            current_image_vq_indices = mint.clamp(current_image_vq_indices, 0, 8192 - 1)
            current_image = vq_model.decode_code(current_image_vq_indices)
            images = mint.clamp((current_image + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).asnumpy().astype(np.uint8)
            pil_images = Image.fromarray(images[0])
            yield pil_images, f"Step {step + 1}/{timesteps}"

            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(ms.Tensor(ratio))
            selected_probs = mint.gather(probs, -1, sampled_ids.to(ms.int32)[..., None])
            selected_probs = selected_probs.squeeze(-1)

            selected_probs = mint.where(unknown_map, selected_probs, _DTYPE_2_MAX[selected_probs.dtype])
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0)
            mask_len = mint.max(
                ms.Tensor([1], dtype=ms.int32), mint.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )

            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            input_ids[:, -(num_vq_tokens + 1) : -1] = mint.where(
                masking, mask_token_id, sampled_ids + len(uni_prompting.text_tokenizer) + num_new_special_tokens
            )
            input_ids_minus_lm_vocab_size = mint.where(masking, mask_token_id, sampled_ids)

        return sampled_ids


AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)
