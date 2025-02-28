from typing import List, Optional, Tuple, Union

import mindspore as ms
from mindspore import ops

from .embeddings_provider import BaseTextualInversionManager, DownweightMode, EmbeddingsProvider, ReturnedEmbeddingsType
from .prompt_parser import Blend, Conjunction, FlattenedPrompt, PromptParser

__all__ = ["Compel", "DownweightMode"]


class Compel:
    def __init__(
        self,
        tokenizer,
        text_encoder,
        textual_inversion_manager: Optional[BaseTextualInversionManager] = None,
        truncate_long_prompts: bool = True,
        padding_attention_mask_value: int = 1,
        downweight_mode: DownweightMode = DownweightMode.MASK,
        returned_embeddings_type: ReturnedEmbeddingsType = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
        requires_pooled: Union[bool, List[bool]] = False,
    ):
        """
        Initialize Compel..
        `textual_inversion_manager`: Optional instance to handle expanding multi-vector textual inversion tokens.
        `truncate_long_prompts`: if True, truncate input prompts to 77 tokens long including beginning/end markers
            (default behaviour).
            If False, do not truncate, and instead assemble as many 77 token long chunks, each capped by beginning/end
            markers, as is necessary to encode the whole prompt. You will likely need to supply both positive and
            negative prompts in this case - use `pad_conditioning_tensors_to_same_length` to prevent having tensor
            length mismatch errors when passing the embeds on to your DiffusionPipeline for inference.
        `padding_attention_mask_value`: Value to write into the attention mask for padding tokens. Stable Diffusion needs 1.
        `downweight_mode`: Specifies whether downweighting should be applied by MASKing out the downweighted tokens
            (default) or REMOVEing them (legacy behaviour; messes up position embeddings of tokens following).
        `returned_embeddings_type`: controls how the embedding vectors are taken from the result of running the text
            encoder over the parsed prompt's text. For SD<=2.1, use LAST_HIDDEN_STATES_NORMALIZED, or
            PENULTIMATE_HIDDEN_STATES_NORMALIZED if you want to do "clip skip". For SDXL use PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED.
        `requires_pooled`: for SDXL, append the pooled embeddings when returning conditioning tensors

        """
        assert not isinstance(text_encoder, (tuple, list)) and not isinstance(
            tokenizer, (tuple, list)
        )  # support single text encoder along with single tokenizer

        self.conditioning_provider = EmbeddingsProvider(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            textual_inversion_manager=textual_inversion_manager,
            truncate=truncate_long_prompts,
            padding_attention_mask_value=padding_attention_mask_value,
            downweight_mode=downweight_mode,
            returned_embeddings_type=returned_embeddings_type,
        )
        self.requires_pooled = requires_pooled

    def build_conditioning_tensor(self, text: str) -> ms.Tensor:
        """
        Build a conditioning tensor by parsing the text for Compel syntax, constructing a Conjunction, and then
        building a conditioning tensor from that Conjunction.
        """
        conjunction = self.parse_prompt_string(text)
        conditioning, _ = self.build_conditioning_tensor_for_conjunction(conjunction)

        if self.requires_pooled:
            pooled = self.conditioning_provider.get_pooled_embeddings([text])
            return conditioning, pooled
        else:
            return conditioning

    def __call__(self, text: Union[str, List[str]]) -> ms.Tensor:
        """
        Take a string or a list of strings and build conditioning tensors to match.

        If multiple strings are passed, the resulting tensors will be padded until they have the same length.

        :return: A tensor consisting of conditioning tensors for each of the passed-in strings, concatenated along dim 0.
        """
        if not isinstance(text, list):
            text = [text]

        cond_tensor = []
        pooled = []
        for text_input in text:
            output = ops.stop_gradient(self.build_conditioning_tensor(text_input))  # stop gradient from text encoder

            if self.requires_pooled:
                cond_tensor.append(output[0])
                pooled.append(output[1])
            else:
                cond_tensor.append(output)
        cond_tensor = self.pad_conditioning_tensors_to_same_length(conditionings=cond_tensor)
        cond_tensor = ops.cat(cond_tensor)

        if self.requires_pooled:
            pooled = ops.cat(pooled)
            return cond_tensor, pooled
        else:
            return cond_tensor

    @classmethod
    def parse_prompt_string(cls, prompt_string: str) -> Conjunction:
        """
        Parse the given prompt string and return a structured Conjunction object that represents the prompt it contains.
        """
        pp = PromptParser()
        conjunction = pp.parse_conjunction(prompt_string)
        return conjunction

    def describe_tokenization(self, text: str) -> List[str]:
        """
        For the given text, return a list of strings showing how it will be tokenized.

        :param text: The text that is to be tokenized.
        :return: A list of strings representing the output of the tokenizer. It's expected that the output list may be
        longer than the number of words in `text` because the tokenizer may split words to multiple tokens. Because of
        this, word boundaries are indicated in the output with `</w>` strings.
        """
        return self.conditioning_provider.tokenizer.tokenize(text)

    def build_conditioning_tensor_for_conjunction(self, conjunction: Conjunction) -> Tuple[ms.Tensor, dict]:
        """
        Build a conditioning tensor for the given Conjunction object.
        :return: A tuple of (conditioning tensor, options dict). The contents of the options dict depends on the prompt,
        at the moment it is only used for returning cross-attention control conditioning data (`.swap()`).
        """
        if len(conjunction.prompts) > 1 and conjunction.type != "AND":
            raise ValueError("Only AND conjunctions are supported by build_conditioning_tensor()")
        # concatenate each prompt in the conjunction (typically there will only be 1)
        to_concat = []
        options = {}
        empty_conditioning = None
        for i, p in enumerate(conjunction.prompts):
            this_conditioning, this_options = self.build_conditioning_tensor_for_prompt_object(p)
            options.update(this_options)  # this is not a smart way to do this but 🤷‍
            weight = conjunction.weights[i]
            if weight != 1:
                # apply weight if we need to
                empty_conditioning = (
                    self.build_conditioning_tensor("") if empty_conditioning is None else empty_conditioning
                )
                [padded_empty_conditioning, _] = self.pad_conditioning_tensors_to_same_length(
                    [empty_conditioning, this_conditioning]
                )
                this_conditioning = padded_empty_conditioning + (this_conditioning - padded_empty_conditioning) * weight
            to_concat.append(this_conditioning)
        assert all(len(c.shape) == len(to_concat[0].shape) for c in to_concat)
        if len(to_concat[0].shape) == 2:
            token_dim = 0
        elif len(to_concat[0].shape) == 3:
            token_dim = 1
        else:
            assert False, f"unhandled conditioning shape length: {to_concat[0].shape}"
        return ops.cat(to_concat, axis=token_dim), options

    def build_conditioning_tensor_for_prompt_object(
        self,
        prompt: Union[Blend, FlattenedPrompt],
    ) -> Tuple[ms.Tensor, dict]:
        """
        Build a conditioning tensor for the given prompt object (either a Blend or a FlattenedPrompt).
        """
        if type(prompt) is Blend:
            return self._get_conditioning_for_blend(prompt), {}
        elif type(prompt) is FlattenedPrompt:
            assert not prompt.wants_cross_attention_control, "cross_attention_control not supported"

            return self._get_conditioning_for_flattened_prompt(prompt), {}

        raise ValueError(f"unsupported prompt type: {type(prompt).__name__}")

    @classmethod
    def _pad_conditioning_tensors_to_same_length(
        cls, conditionings: List[ms.Tensor], emptystring_conditioning: ms.Tensor
    ) -> List[ms.Tensor]:
        c0_shape = conditionings[0].shape
        if not all([len(c.shape) == len(c0_shape) for c in conditionings]):
            raise ValueError(
                "Conditioning tensors must all have either 2 dimensions (unbatched) or 3 dimensions (batched)"
            )

        if len(c0_shape) == 2:
            # need to be unsqueezed
            conditionings = [c.unsqueeze(0) for c in conditionings]
            c0_shape = conditionings[0].shape
        if len(c0_shape) != 3:
            raise ValueError("All conditioning tensors must have the same number of dimensions (2 or 3)")

        if not all([c.shape[0] == c0_shape[0] and c.shape[2] == c0_shape[2] for c in conditionings]):
            raise ValueError(
                "All conditioning tensors must have the same batch size ({c0_shape[0]}) and "
                f"number of embeddings per token ({c0_shape[1]})"
            )

        if len(emptystring_conditioning.shape) == 2:
            emptystring_conditioning = emptystring_conditioning.unsqueeze(0)
        empty_z = ops.cat([emptystring_conditioning] * c0_shape[0])
        max_token_count = max([c.shape[1] for c in conditionings])
        # if necessary, pad shorter tensors out with an emptystring tensor
        for i, c in enumerate(conditionings):
            while c.shape[1] < max_token_count:
                c = ops.cat([c, empty_z], axis=1)
                conditionings[i] = c
        return conditionings

    def pad_conditioning_tensors_to_same_length(
        self,
        conditionings: List[ms.Tensor],
    ) -> List[ms.Tensor]:
        """
        If `truncate_long_prompts` was set to False on initialization, or if your prompt includes a `.and()` operator,
        conditioning tensors do not have a fixed length. This is a problem when using a negative and a positive prompt
        to condition the diffusion process. This function pads any of the passed-in tensors, as necessary, to ensure
        they all have the same length, returning the padded tensors in the same order they are passed.

        Example:
            ``` python
            embeds = compel('("a cat playing in the forest", "an impressionist oil painting").and()')
            negative_embeds = compel("ugly, deformed, distorted")
            [embeds, negative_embeds] = compel.pad_conditioning_tensors_to_same_length([embeds, negative_embeds])
            ```
        """
        emptystring_conditioning = self.build_conditioning_tensor("")
        if type(emptystring_conditioning) is tuple:
            # discard pooled
            emptystring_conditioning = emptystring_conditioning[0]
        return type(self)._pad_conditioning_tensors_to_same_length(
            conditionings, emptystring_conditioning=emptystring_conditioning
        )

    def _get_conditioning_for_flattened_prompt(
        self, prompt: FlattenedPrompt, should_return_tokens: bool = False
    ) -> Union[ms.Tensor, Tuple[ms.Tensor, ms.Tensor]]:
        if type(prompt) is not FlattenedPrompt:
            raise ValueError(f"embeddings can only be made from FlattenedPrompts, got {type(prompt).__name__} instead")
        fragments = [x.text for x in prompt.children]
        weights = [x.weight for x in prompt.children]
        return self.conditioning_provider.get_embeddings_for_weighted_prompt_fragments(
            text_batch=[fragments], fragment_weights_batch=[weights], should_return_tokens=should_return_tokens
        )

    def _get_conditioning_for_blend(self, blend: Blend):
        conditionings_to_blend = []
        for i, flattened_prompt in enumerate(blend.prompts):
            this_conditioning = self._get_conditioning_for_flattened_prompt(flattened_prompt)
            conditionings_to_blend.append(this_conditioning)
        conditionings_to_blend = self.pad_conditioning_tensors_to_same_length(conditionings_to_blend)
        conditionings_to_blend_tensor = ops.cat(conditionings_to_blend).unsqueeze(0)
        conditioning = EmbeddingsProvider.apply_embedding_weights(
            conditionings_to_blend_tensor, blend.weights, normalize=blend.normalize_weights
        )
        return conditioning

    def _get_tokens_length(self, texts: [str]) -> int:
        tokens = self.conditioning_provider.get_token_ids(texts, include_start_and_end_markers=False)
        return sum([len(x) for x in tokens])

    def get_tokens(self, text: str) -> List[int]:
        return self.conditioning_provider.get_token_ids([text], include_start_and_end_markers=False)[0]
