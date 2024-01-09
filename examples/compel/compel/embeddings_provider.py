import math
from abc import ABC
from enum import Enum
from typing import List, Optional, Tuple, Union

import mindspore as ms
from mindspore import ops

__all__ = ["EmbeddingsProvider", "DownweightMode", "ReturnedEmbeddingsType"]


class DownweightMode(Enum):
    REMOVE = 0  # Remove downweighted tokens from the token sequence (shifts all subsequent tokens)
    MASK = 1  # Default: Leave tokens in-place but mask them out using attention masking


class BaseTextualInversionManager(ABC):
    def expand_textual_inversion_token_ids_if_necessary(self, token_ids: List[int]) -> List[int]:
        raise NotImplementedError()


class ReturnedEmbeddingsType(Enum):
    LAST_HIDDEN_STATES_NORMALIZED = 0  # SD1/2 regular
    PENULTIMATE_HIDDEN_STATES_NORMALIZED = 1  # SD1.5 with "clip skip"
    PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED = 2  # SDXL


class EmbeddingsProvider:
    def __init__(
        self,
        tokenizer,
        text_encoder,
        textual_inversion_manager: BaseTextualInversionManager = None,
        truncate: bool = True,
        padding_attention_mask_value: int = 1,
        downweight_mode: DownweightMode = DownweightMode.MASK,
        returned_embeddings_type: ReturnedEmbeddingsType = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
    ):
        """
        `tokenizer`: converts strings to lists of int token ids
        `text_encoder`: convert lists of token ids to embedding tensors
        `textual_inversion_manager`: manage token insertion for textual inversions with vector length >1
        `truncate`: if True, truncate inputs to the maximum length specified by the tokenizer. if False, returns
                    tensors that may be longer than the maximum length (but will always be an integer multiple of maximum length)
        `padding_attention_mask_value`: Value to write into the attention mask for padding tokens. Stable Diffusion needs 1.
        `downweight_mode`: if MASK, downweight by blending with a version of the prompt with the downweighted terms masked out.
                    if REMOVE, the blend is against a version of the prompt with the downweighted tokens removed
        `returned_embeddings_type`: controls how the embedding vectors are taken from the result of running the text
            encoder over the parsed prompt's text. For SD<=2.1, use LAST_HIDDEN_STATES_NORMALIZED, or
            PENULTIMATE_HIDDEN_STATES_NORMALIZED if you want to do "clip skip". For SDXL use PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED.
        """
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.textual_inversion_manager = textual_inversion_manager
        self.truncate_to_model_max_length = truncate
        self.padding_attention_mask_value = padding_attention_mask_value
        self.downweight_mode = downweight_mode
        self.returned_embeddings_type = returned_embeddings_type

    @property
    def max_token_count(self) -> int:
        return self.text_encoder.context_length

    @classmethod
    def apply_embedding_weights(
        cls, embeddings: ms.Tensor, per_embedding_weights: List[float], normalize: bool
    ) -> ms.Tensor:
        per_embedding_weights = ms.Tensor(per_embedding_weights, dtype=embeddings.dtype)
        if normalize:
            per_embedding_weights = per_embedding_weights / ops.sum(per_embedding_weights)

        reshaped_weights = per_embedding_weights.reshape(
            per_embedding_weights.shape
            + (
                1,
                1,
            )
        )
        blended_embeddings = ops.sum(embeddings * reshaped_weights, dim=1)
        # blended_embeddings now has shape (77, 768)
        return blended_embeddings

    def get_embeddings_for_weighted_prompt_fragments(
        self,
        text_batch: List[List[str]],
        fragment_weights_batch: List[List[float]],
        should_return_tokens: bool = False,
    ) -> Union[ms.Tensor, Tuple[ms.Tensor, ms.Tensor]]:
        """

        :param text_batch: A list of fragments of text to which different weights are to be applied.
        :param fragment_weights_batch: A list of weights, one for each entry in `fragments`.
        :param should_return_tokens: If True, return a tuple of (embeddings, tokens), otherwise just return embeddings.
        :return: A tensor of shape `[1, 77, token_dim]` containing weighted embeddings where token_dim is 768 for SD1
                    and 1280 for SD2
        """
        if len(text_batch) != len(fragment_weights_batch):
            raise ValueError(
                "lengths of text and fragment_weights lists are not the same "
                + f"({len(text_batch)} != {len(fragment_weights_batch)})"
            )

        batch_z = None
        batch_tokens = None
        for fragments, weights in zip(text_batch, fragment_weights_batch):
            # First, weight tokens in individual fragments by scaling the feature vectors as requested (effectively
            # applying a multiplier to the CFG scale on a per-token basis).
            # For tokens weighted<1, intuitively we want SD to become not merely *less* interested in the concept
            # captured by the fragment but actually *dis*interested in it (a 0.01 interest in "red" is still an active
            # interest, however small, in redness; what the user probably intends when they attach the number 0.01 to
            # "red" is to tell SD that it should almost completely *ignore* redness).
            # To do this, the embedding is lerped away from base_embedding in the direction of an embedding for a prompt
            # string from which the low-weighted fragment has been simply removed. The closer the weight is to zero, the
            # closer the resulting embedding is to an embedding for a prompt that simply lacks this fragment.

            # handle weights >=1
            tokens, per_token_weights, mask = self.get_token_ids_and_expand_weights(fragments, weights)
            base_embedding = self.build_weighted_embedding_tensor(tokens, per_token_weights, mask)

            # this is our starting point
            embeddings = base_embedding.unsqueeze(0)
            per_embedding_weights = [1.0]

            # now handle weights <1
            # Do this by building extra embeddings tensors that lack the words being <1 weighted. These will be lerped
            # with the embeddings tensors that have the words, such that if the weight of a word is 0.5, the resulting
            # embedding will be exactly half-way between the unweighted prompt and the prompt with the <1 weighted words
            # removed.
            # e.g. for "mountain:1 man:0.5", intuitively the "man" should be "half-gone". therefore, append an embedding
            # for "mountain" (i.e. without "man") to the already-produced embedding for "mountain man", and weight it
            # such that the resulting lerped embedding is exactly half-way between "mountain man" and "mountain".
            fragment_token_index_ranges = self._get_token_ranges_for_fragments(tokens.asnumpy().tolist(), fragments)

            for index in range(len(fragment_token_index_ranges)):
                fragment_weight = weights[index]
                if fragment_weight < 1:
                    if self.downweight_mode == DownweightMode.MASK:
                        fragment_start_token_id, fragment_end_token_id = fragment_token_index_ranges[index]
                        # mask out this fragment
                        mask_without_fragment = mask.astype(mask.dtype, copy=True)
                        mask_without_fragment[fragment_start_token_id : fragment_end_token_id + 1] = 0
                        if not self.truncate_to_model_max_length:
                            # but don't mask chunk-delimiting eos/bos markers
                            mask_without_fragment[0 :: self.max_token_count] = 1
                            mask_without_fragment[self.max_token_count - 1 :: self.max_token_count] = 1
                        embedding_without_this = self.build_weighted_embedding_tensor(
                            tokens, per_token_weights, mask_without_fragment
                        )
                    else:
                        fragments_without_this = fragments[0:index] + fragments[index + 1 :]
                        weights_without_this = weights[0:index] + weights[index + 1 :]
                        (
                            tokens_without_fragment,
                            per_token_weights_without_fragment,
                            mask_without_fragment,
                        ) = self.get_token_ids_and_expand_weights(fragments_without_this, weights_without_this)
                        embedding_without_this = self.build_weighted_embedding_tensor(
                            tokens_without_fragment, per_token_weights_without_fragment
                        )

                    embeddings = ops.cat((embeddings, embedding_without_this.unsqueeze(0)), axis=1)
                    # weight of the embedding *without* this fragment gets *stronger* as its weight approaches 0
                    # if fragment_weight = 0, basically we want embedding_without_this to completely overwhelm base_embedding
                    # therefore:
                    # fragment_weight = 1: we are at base_z => lerp weight 0
                    # fragment_weight = 0.5: we are halfway between base_z and here => lerp weight 1
                    # fragment_weight = 0: we're now entirely overriding base_z ==> lerp weight inf
                    # so let's use tan(), because:
                    # tan is 0.0 at 0,
                    #        1.0 at PI/4, and
                    #        inf at PI/2
                    # -> tan((1-weight)*PI/2) should give us ideal lerp weights
                    epsilon = 1e-5
                    fragment_weight = max(epsilon, fragment_weight)  # inf is bad
                    embedding_lerp_weight = math.tan((1.0 - fragment_weight) * math.pi / 2)

                    per_embedding_weights.append(embedding_lerp_weight)

            lerped_embeddings = self.apply_embedding_weights(embeddings, per_embedding_weights, normalize=True).squeeze(
                0
            )

            # print(f"assembled tokens for '{fragments}' into tensor of shape {lerped_embeddings.shape}")

            # append to batch
            batch_z = (
                lerped_embeddings.unsqueeze(0)
                if batch_z is None
                else ops.cat([batch_z, lerped_embeddings.unsqueeze(0)], dim=1)
            )
            batch_tokens = (
                tokens.unsqueeze(0) if batch_tokens is None else ops.cat([batch_tokens, tokens.unsqueeze(0)], dim=1)
            )

        # should have shape (B, 77, 768)
        # print(f"assembled all tokens into tensor of shape {batch_z.shape}")

        if should_return_tokens:
            return batch_z, batch_tokens
        else:
            return batch_z

    def get_token_ids(
        self,
        texts: List[str],
        include_start_and_end_markers: bool = True,
        padding: str = False,
        truncation_override: Optional[bool] = None,
    ) -> List[List[int]]:
        """
        Convert a list of strings like `["a cat", "a dog", "monkey riding a bicycle"]` into a list of lists of token
        ids like `[[bos, 0, 1, eos], [bos, 0, 2, eos], [bos, 3, 4, 0, 5, eos]]`. bos/eos markers are skipped if
        `include_start_and_end_markers` is `False`. Each list will be restricted to the maximum permitted length
        (typically 75 tokens + eos/bos markers).

        :param texts: The strings to convert.
        :param include_start_and_end_markers: If True (default), returned token id lists will start with the beginning
            of sequence marker and end with the end-of-sequence marker (`eos`).
        :padding: Padding argument passed through to the Tokenizer.
        :truncation_override: Optional, overrides the `truncate` argument passed to `__init__`.
        :return: A list of lists of token ids corresponding to the input strings.
        """
        # for args documentation of self.tokenizer() see ENCODE_KWARGS_DOCSTRING in tokenization_utils_base.py
        # (part of `transformers` lib)
        truncation = self.truncate_to_model_max_length if truncation_override is None else truncation_override
        if isinstance(texts, str):
            texts = [texts]

        token_ids_list = [
            self.tokenizer(
                x,
                truncation=truncation,
                padding=padding,
                return_tensors=None,  # just give me lists of ints
                add_special_tokens=True,
            )["input_ids"]
            for x in texts
        ]

        result = []
        for token_ids in token_ids_list:
            # trim eos/bos
            token_ids = token_ids[1:-1]
            # pad for textual inversions with vector length >1
            if self.textual_inversion_manager is not None:
                token_ids = self.textual_inversion_manager.expand_textual_inversion_token_ids_if_necessary(token_ids)

            # add back eos/bos if requested
            if include_start_and_end_markers:
                token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]

            result.append(token_ids)

        return result

    def get_pooled_embeddings(
        self, texts: List[str], attention_mask: Optional[ms.Tensor] = None
    ) -> Optional[ms.Tensor]:
        token_ids = self.get_token_ids(texts, padding="max_length", truncation_override=True)
        token_ids = ms.Tensor(token_ids, dtype=ms.int64)

        text_encoder_output = self.text_encoder(token_ids, attention_mask, return_dict=True)
        pooled = text_encoder_output.text_embeds

        return pooled

    def get_token_ids_and_expand_weights(
        self, fragments: List[str], weights: List[float]
    ) -> (ms.Tensor, ms.Tensor, ms.Tensor):
        """
        Given a list of text fragments and corresponding weights: tokenize each fragment, append the token sequences
        together and return a padded token sequence starting with the bos marker, ending with the eos marker, and padded
        or truncated as appropriate to `self.max_length`. Also return a list of weights expanded from the passed-in
        weights to match each token.

        :param fragments: Text fragments to tokenize and concatenate. May be empty.
        :param weights: Per-fragment weights (i.e. quasi-CFG scaling). Values from 0 to inf are permitted. In practise with SD1.5
                        values >1.6 tend to produce garbage output. Must have same length as `fragment`.
        :return: A tuple of tensors `(token_ids, weights, mask)`. `token_ids` is ints, `weights` is floats, `mask` is
                        ints, all have shape `[self.max_length]`.
        """
        if len(fragments) != len(weights):
            raise ValueError(
                f"lengths of text and fragment_weights lists are not the same ({len(fragments)} != {len(weights)})"
            )

        # empty is meaningful
        if len(fragments) == 0:
            fragments = [""]
            weights = [1.0]
        per_fragment_token_ids = self.get_token_ids(fragments, include_start_and_end_markers=False)
        all_token_ids: List[int] = []
        all_token_weights: List[float] = []
        # print("all fragments:", fragments, weights)
        for this_fragment_token_ids, weight in zip(per_fragment_token_ids, weights):
            # append
            all_token_ids += this_fragment_token_ids
            # fill out weights tensor with one float per token
            all_token_weights += [float(weight)] * len(this_fragment_token_ids)

        return self._chunk_and_pad_token_ids(all_token_ids, all_token_weights)

    def _chunk_and_pad_token_ids(
        self, token_ids: List[int], token_weights: List[float]
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        remaining_token_ids = token_ids
        remaining_token_weights = token_weights
        chunk_length_without_eos_bos_markers = self.max_token_count - 2

        all_token_ids = []
        all_token_weights = []
        all_masks = []
        while True:
            # each chunk must leave room for bos/eos
            chunk_token_ids = remaining_token_ids[0:chunk_length_without_eos_bos_markers]
            chunk_token_weights = remaining_token_weights[0:chunk_length_without_eos_bos_markers]
            # update remaining
            remaining_token_ids = remaining_token_ids[chunk_length_without_eos_bos_markers:]
            remaining_token_weights = remaining_token_weights[chunk_length_without_eos_bos_markers:]

            # pad out to a self.max_length-entry array: [eos_token, <prompt tokens>, eos_token[, pad_token, ...]]
            # (typically self.max_length == 77)
            chunk_token_ids = [self.tokenizer.bos_token_id] + chunk_token_ids + [self.tokenizer.eos_token_id]
            chunk_token_weights = [1.0] + chunk_token_weights + [1.0]
            chunk_mask = [1] * len(chunk_token_ids)

            pad_length = self.max_token_count - len(chunk_token_ids)
            chunk_token_ids += [self.tokenizer.pad_token_id] * pad_length
            chunk_token_weights += [1.0] * pad_length
            chunk_mask += [self.padding_attention_mask_value] * pad_length

            all_token_ids += chunk_token_ids
            all_token_weights += chunk_token_weights
            all_masks += chunk_mask

            if self.truncate_to_model_max_length or len(remaining_token_ids) == 0:
                break

        all_token_ids_tensor = ms.Tensor(all_token_ids, dtype=ms.int64)
        all_per_token_weights_tensor = ms.Tensor(all_token_weights, dtype=ms.float32)
        all_masks = ms.Tensor(all_masks, dtype=ms.int64)
        # print(f"assembled all_token_ids_tensor with shape {all_token_ids_tensor.shape}")
        return all_token_ids_tensor, all_per_token_weights_tensor, all_masks

    def build_weighted_embedding_tensor(
        self, token_ids: ms.Tensor, per_token_weights: ms.Tensor, attention_mask: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        """
        Build a tensor that embeds the passed-in token IDs and applies the given per_token weights

        :param token_ids: A tensor of shape `n*[self.max_length]` containing token IDs (ints) where n is some arbitrary
            integer (i.e. n==1 for shorter prompts, or it may be >1 if there are more than max_length tokens in the
            original prompt)
        :param per_token_weights: A tensor containing weights (floats), with the same shape as token_ids
        :param attention_mask: A tensor containing a mask (ints), with the same shape as token_ids, where 1 means use
            the corresponding token and 0 means ignore the corresponding token.

        :return: A tensor of shape `[1, token_ids.shape[0], token_dim]` representing the requested weighted embeddings
            where `token_dim` is 768 for SD1 and 1280 for SD2.
        """
        # print(f"building weighted embedding tensor for {tokens} with weights {token_weights}")
        if token_ids.shape[0] % self.max_token_count != 0:
            raise ValueError(f"token_ids has shape {token_ids.shape} - expected a multiple of {self.max_token_count}")

        chunk_start_index = 0
        empty_token_ids = ms.Tensor(
            [self.tokenizer.bos_token_id]
            + [self.tokenizer.eos_token_id]
            + [self.tokenizer.pad_token_id] * (self.max_token_count - 2),
            dtype=ms.int32,
        ).unsqueeze(0)
        empty_z = self._encode_token_ids_to_embeddings(empty_token_ids)
        weighted_z = None

        chunk_size = self.max_token_count
        while chunk_start_index < token_ids.shape[0]:
            next_chunk_start_index = chunk_start_index + chunk_size
            chunk_per_token_weights = per_token_weights[chunk_start_index:next_chunk_start_index]
            chunk_token_ids = token_ids[chunk_start_index:next_chunk_start_index].unsqueeze(0)
            chunk_attention_mask = (
                attention_mask[chunk_start_index:next_chunk_start_index].unsqueeze(0)
                if attention_mask is not None
                else None
            )

            z = self._encode_token_ids_to_embeddings(chunk_token_ids, chunk_attention_mask)
            batch_weights_expanded = chunk_per_token_weights[None, :, None].expand_as(z).to(z.dtype)

            z_delta_from_empty = z - empty_z
            this_weighted_z = empty_z + (z_delta_from_empty * batch_weights_expanded)
            weighted_z = this_weighted_z if weighted_z is None else ops.cat([weighted_z, this_weighted_z], axis=1)
            chunk_start_index += chunk_size

        return weighted_z

    def _encode_token_ids_to_embeddings(
        self, token_ids: ms.Tensor, attention_mask: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        # needs_hidden_states = (self.returned_embeddings_type == ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED or
        #                       self.returned_embeddings_type == ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED)
        assert (
            self.returned_embeddings_type == ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
        ), "Current FrozenCLIPEmbedder only supports returning the last layer hidden states after layer normalization!"
        text_encoder_output = self.text_encoder(token_ids, attention_mask)
        return text_encoder_output
        # if self.returned_embeddings_type is ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED:
        #     penultimate_hidden_state = text_encoder_output.hidden_states[-2]
        #     return penultimate_hidden_state
        # elif self.returned_embeddings_type is ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED:
        #     penultimate_hidden_state = text_encoder_output.hidden_states[-2]
        #     return self.text_encoder.text_model.final_layer_norm(penultimate_hidden_state)
        # elif self.returned_embeddings_type is ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED:
        #     # already normalized
        #     return text_encoder_output.last_hidden_state

        # assert False, f"unrecognized ReturnEmbeddingsType: {self.returned_embeddings_type}"

    def _get_token_ranges_for_fragments(
        self, chunked_and_padded_token_ids: List[int], fragments: List[str]
    ) -> List[Tuple[int, int]]:
        """
        Match token id sequences for the strings in `fragments` with token id sequences in `chunked_and_padded_token_ids`,
         taking into account any eos and bos markers that indicate `self.tokenizer.max_model_length`-sized chunks.

        :return: a list of tuples indicating start and end indices of each fragment's corresponding token id sequence in
         `chunked_and_padded_token_ids`.
        """
        per_fragment_token_ids = self.get_token_ids(fragments, include_start_and_end_markers=False)
        fragment_start = 0

        corresponding_indices = []
        for fragment_index, fragment_token_ids in enumerate(per_fragment_token_ids):
            if len(fragment_token_ids) == 0:
                corresponding_indices.append((None, None))
                continue
            if self.truncate_to_model_max_length and fragment_start >= self.max_token_count - 1:
                break
            # find the start
            while True:
                if fragment_start >= len(chunked_and_padded_token_ids) - 1:
                    if self.truncate_to_model_max_length:
                        fragment_start = len(chunked_and_padded_token_ids) - 1
                        break
                    else:
                        raise RuntimeError(
                            f"couldn't find start of token sequence for fragment at index {fragment_index} '{fragments[fragment_index]}'"
                        )
                if chunked_and_padded_token_ids[fragment_start] == fragment_token_ids[0]:
                    break
                fragment_start += 1
            # step through
            fragment_end = fragment_start
            fragment_relative_index = 0
            while True:
                if fragment_end >= len(chunked_and_padded_token_ids) - 1:
                    if self.truncate_to_model_max_length:
                        fragment_end = len(chunked_and_padded_token_ids) - 1
                        break
                    else:
                        raise RuntimeError(
                            f"couldn't find end of token sequence for fragment at index {fragment_index} '{fragments[fragment_index]}'"
                        )
                if not self.truncate_to_model_max_length and (
                    chunked_and_padded_token_ids[fragment_end] == self.tokenizer.eos_token_id
                    or chunked_and_padded_token_ids[fragment_end] == self.tokenizer.bos_token_id
                ):
                    # bos/eos: chunk boundaries
                    fragment_end += 1
                elif chunked_and_padded_token_ids[fragment_end] == fragment_token_ids[fragment_relative_index]:
                    # matching token
                    fragment_relative_index += 1
                    if fragment_relative_index == len(fragment_token_ids):
                        break
                    fragment_end += 1
                else:
                    raise RuntimeError(
                        f"token sequence mismatch for fragment at index {fragment_index} '{fragments[fragment_index]}':"
                        f"expected {fragment_token_ids}, found {chunked_and_padded_token_ids[fragment_start:fragment_end + 1]}"
                    )

            corresponding_indices.append((fragment_start, fragment_end))
            fragment_start = fragment_end + 1

        return corresponding_indices
