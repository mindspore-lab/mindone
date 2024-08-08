from typing import Optional

import numpy as np
from modules.clip.clip_config import CLIPTextCfg, CLIPVisionCfg
from modules.encoders.image_encoder import ImageEncoder
from modules.encoders.text_encoder import MultimodalTransformer, TextEncoder
from modules.utils._common import (
    LogitsProcessorList,
    MaxLengthCriteria,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    StoppingCriteriaList,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from modules.utils.beam_search import BeamSearchScorer

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

GENERATION_TYPES = {"top_k": TopKLogitsWarper, "top_p": TopPLogitsWarper, "beam_search": "beam_search"}


class MultimodalCfg(CLIPTextCfg):
    def __init__(
        self,
        mlp_ratio: Optional[int] = 4,
        dim_head: Optional[int] = 64,
        heads: Optional[int] = 8,
        n_queries: Optional[int] = 256,
        attn_pooler_heads: Optional[int] = 8,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mlp_ratio = mlp_ratio
        self.dim_head = dim_head
        self.heads = heads
        self.n_queries = n_queries
        self.attn_pooler_heads = attn_pooler_heads


class CoCa(nn.Cell):
    """
    Coca Model
    """

    def __init__(
        self,
        embed_dim,
        multimodal_cfg: MultimodalCfg,
        text_cfg: CLIPTextCfg,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        use_fp16: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.text = TextEncoder(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            epsilon=1e-5,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            use_quick_gelu=quick_gelu,
            dtype=self.dtype,
        )
        self.visual = ImageEncoder(
            embed_dim=embed_dim,
            image_resolution=vision_cfg.image_size,
            vision_layers=vision_cfg.layers,
            vision_width=vision_cfg.width,
            vision_patch_size=vision_cfg.patch_size,
            vision_head_width=vision_cfg.head_width,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pool_type=vision_cfg.pool_type,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            output_tokens=vision_cfg.output_tokens,
            epsilon=1e-5,
            use_quick_gelu=quick_gelu,
            dtype=self.dtype,
        )

        self.text_decoder = MultimodalTransformer(
            width=multimodal_cfg.width,
            layers=multimodal_cfg.layers,
            heads=multimodal_cfg.layers,
            epsilon=1e-5,
            use_quick_gelu=quick_gelu,
            context_length=multimodal_cfg.context_length,
            output_dim=text_cfg.vocab_size,
            dtype=self.dtype,
        )
        self.logit_scale = Parameter(Tensor(np.ones([]) * init_logit_scale, dtype=self.dtype))
        self.pad_id = pad_id
        self.context_length = multimodal_cfg.context_length
        self.l2_normalize = ops.L2Normalize(axis=-1, epsilon=1e-12)
        self.cast = ops.Cast()

    def _encode_image(self, images, normalize: bool = True):
        image_latent, token_embs = self.visual(images)
        image_latent = self.l2_normalize(image_latent) if normalize else image_latent

        return image_latent, token_embs

    def _encode_text(self, text, normalize: bool = True):
        text_latent, tokens_emb = self.text(text)
        text_latent = self.l2_normalize(text_latent) if normalize else text_latent

        return text_latent, tokens_emb

    def encode_image(self, images, normalize: bool = True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize: bool = True):
        text_latent, _ = self._encode_text(text, normalize=normalize)
        return text_latent

    def construct(self, images, text=None, image_latent=None, image_embs=None):
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(images)

        if text is None:
            return {"image_features": image_latent, "image_embs": image_embs}

        text_latent, token_embs = self._encode_text(text)

        labels = text[:, -token_embs.shape[1] :]

        logits = self.text_decoder(image_embs, token_embs)

        out_dict = {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.logit_scale.exp(),
        }

        return out_dict

    def generate(
        self,
        image,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.0,
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False,  # if True output.shape == (batch_size, seq_len)
        is_video=False,
    ):
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        sot_token_id = 49406 if sot_token_id is None else sot_token_id
        eos_token_id = 49407 if eos_token_id is None else eos_token_id
        pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
        logit_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                RepetitionPenaltyLogitsProcessor(repetition_penalty),
            ]
        )
        if stopping_criteria is None:
            stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

        stopping_criteria = StoppingCriteriaList(stopping_criteria)

        if generation_type == "beam_search":
            output = self._generate_beamsearch(
                image_inputs=image,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                sot_token_id=sot_token_id,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                min_seq_len=min_seq_len,
                stopping_criteria=stopping_criteria,
                logit_processor=logit_processor,
                is_video=is_video,
            )
            if fixed_output_length and output.shape[1] < seq_len:
                return ops.cat(
                    (output, ops.ones(output.shape[0], seq_len - output.shape[1], dtype=output.dtype) * self.pad_id),
                    axis=1,
                )
            return output

        elif generation_type == "top_p":
            logit_warper = GENERATION_TYPES[generation_type](top_p)
        elif generation_type == "top_k":
            logit_warper = GENERATION_TYPES[generation_type](top_k)
        else:
            raise ValueError(
                f"generation_type has to be one of " f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
            )
        if is_video:
            image_latent, image_embs = 0, 0
            for data in image:
                data_latent, data_embs = self._encode_image(data)
                image_latent += data_latent
                image_embs += image_embs
            image_latent = image_latent / image.shape[0]
            image_embs = image_embs / image.shape[0]
            if text is None:
                text = ops.ones((1, 1), dtype=ms.int32) * sot_token_id
        else:
            image_latent, image_embs = self._encode_image(image)

            if text is None:
                text = ops.ones((image.shape[0], 1), dtype=ms.int32) * sot_token_id

        num_dims = len(text.shape)

        if num_dims == 1:
            text = text[None, :]

        cur_len = text.shape[1]
        out = text

        while True:
            x = out[:, -max_seq_len:]
            cur_len = x.shape[1]
            logits = self(image, x, image_latent=image_latent, image_embs=image_embs)["logits"][:, -1]
            mask = ms.Tensor(bool(out[:, -1] == eos_token_id) | bool(out[:, -1] == pad_token_id))
            sample = ops.ones((out.shape[0], 1), dtype=ms.int32) * pad_token_id

            if mask.all():
                if not fixed_output_length:
                    break
            else:
                logits = logits[~mask, :]
                filtered_logits = logit_processor(x[~mask, :], logits)
                filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                probs = ops.softmax(filtered_logits / temperature, axis=-1)

                if cur_len + 1 == seq_len:
                    sample[~mask, :] = ops.ones((sum(~mask), 1), dtype=ms.int32) * eos_token_id
                else:
                    sample[~mask, :] = ops.multinomial(probs, 1, replacement=False)

            out = ops.cat((out, sample), axis=-1)

            cur_len += 1

            if stopping_criteria(out, None):
                break

        if num_dims == 1:
            out = out.squeeze(0)
        return out

    def _generate_beamsearch(
        self,
        image_inputs,
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        logit_processor=None,
        is_video=False,
    ):
        if is_video:
            batch_size = 1
            image_latent, image_embs = 0, 0
            for data in image_inputs:
                data = ops.repeat_interleave(data.unsqueeze(0), num_beams, axis=0)
                data_latent, data_embs = self._encode_image(image_inputs)
                image_latent += data_latent
                image_embs += data_embs
            image_latent = image_latent / image_inputs.shape[0]
            image_embs = image_embs / image_inputs.shape[0]
        else:
            batch_size = image_inputs.shape[0]
            image_inputs = ops.repeat_interleave(image_inputs, num_beams, axis=0)
            image_latent, image_embs = self._encode_image(image_inputs)

        input_ids = ops.ones((batch_size * num_beams, 1), dtype=ms.int32)
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
        )
        # instantiate logits processors
        logits_processor = (
            LogitsProcessorList([MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)])
            if logit_processor is None
            else logit_processor
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = ops.full((batch_size, num_beams), -1e9, dtype=ms.float32)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            # predicted tokens in cur_len step
            current_tokens = ops.zeros(batch_size * num_beams, dtype=input_ids.dtype)

            # indices which will form the beams in the next time step
            reordering_indices = ops.zeros(batch_size * num_beams, dtype=ms.int32)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = prepare_inputs_for_generation(input_ids=input_ids, image_inputs=image_inputs)
            outputs = self(
                model_inputs["images"], model_inputs["text"], image_latent=image_latent, image_embs=image_embs
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs["logits"][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = ops.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = ops.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]

                group_input_ids = ops.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], axis=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * ops.div(beam_idx, group_size, rounding_mode="floor")
                    + group_start_idx
                    + (beam_idx % group_size)
                )

            input_ids = ops.cat([input_ids, current_tokens.unsqueeze(-1)], axis=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs["sequences"]


def prepare_inputs_for_generation(input_ids, image_inputs, past=None, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(axis=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        position_ids = None
    return {
        "text": input_ids,
        "images": image_inputs,
        "past_key_values": past,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
