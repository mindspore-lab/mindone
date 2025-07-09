import copy
import inspect
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from packaging import version
from transformers import logging
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.tokenization_utils import ExtensionsTrie
from transformers.utils.generic import ModelOutput

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import mint, ops

from mindone.transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    HybridCache,
    MambaCache,
    OffloadedStaticCache,
    SlidingWindowCache,
    StaticCache,
    get_seq_length,
    init_static_cache,
    reset,
)
from mindone.transformers.generation.candidate_generator import (
    AssistantVocabTranslatorCache,
    AssistedCandidateGenerator,
    AssistedCandidateGeneratorDifferentTokenizers,
    CandidateGenerator,
    EarlyExitCandidateGenerator,
    PromptLookupCandidateGenerator,
    UniversalSpeculativeDecodingGenerator,
)
from mindone.transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from mindone.transformers.generation.stopping_criteria import (
    ConfidenceCriteria,
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
)
from mindone.transformers.mindspore_adapter.paged_attention_block_tables import BlockTables
from mindone.transformers.mindspore_adapter.select_operator import get_multinomial_op

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from mindone.transformers.modeling_utils import MSPreTrainedModel as PreTrainedModel

logger = logging.get_logger(__name__)


NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
    "offloaded_static": OffloadedStaticCache,
    "sliding_window": SlidingWindowCache,
    "hybrid": HybridCache,
    "mamba": MambaCache,
}
QUANT_BACKEND_CLASSES_MAPPING = {}

# Variable names used to hold the cache at generation time
ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(ms.Tensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `ms.Tensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(ms.Tensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `ms.Tensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(ms.Tensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `ms.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(ms.Tensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `ms.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(mindspore.Tensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: ms.Tensor = None
    scores: Optional[Tuple[ms.Tensor]] = None
    logits: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[Tuple[ms.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[ms.Tensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[ms.Tensor]]]] = None


@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decoder generation models, when using non-beam methods.

    Args:
        sequences (`ms.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(ms.Tensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `ms.Tensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(ms.Tensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `ms.Tensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(ms.Tensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `ms.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(ms.Tensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `ms.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(ms.Tensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `ms.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(mindspore.Tensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: ms.Tensor = None
    scores: Optional[Tuple[ms.Tensor]] = None
    logits: Optional[Tuple[ms.Tensor]] = None
    encoder_attentions: Optional[Tuple[ms.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[ms.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[ms.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[ms.Tensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[ms.Tensor]]]] = None


@dataclass
class GenerateBeamDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using beam methods.

    Args:
        sequences (`ms.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`ms.Tensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(ms.Tensor)` *optional*, returned when `output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `ms.Tensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        logits (`tuple(ms.Tensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `ms.Tensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        beam_indices (`ms.Tensor`, *optional*, returned when `output_scores=True`):
            Beam indices of generated token id at each generation step. `ms.Tensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        attentions (`tuple(tuple(ms.Tensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `ms.Tensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(ms.Tensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `ms.Tensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(ms.Tensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: ms.Tensor = None
    sequences_scores: Optional[ms.Tensor] = None
    scores: Optional[Tuple[ms.Tensor]] = None
    logits: Optional[Tuple[ms.Tensor]] = None
    beam_indices: Optional[ms.Tensor] = None
    attentions: Optional[Tuple[Tuple[ms.Tensor]]] = None
    hidden_states: Optional[Tuple[Tuple[ms.Tensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[ms.Tensor]]]] = None


@dataclass
class GenerateBeamEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decoder generation models, when using beam methods.

    Args:
        sequences (`ms.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`ms.Tensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(ms.Tensor)` *optional*, returned when `output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `ms.Tensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        logits (`tuple(ms.Tensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `ms.Tensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        beam_indices (`ms.Tensor`, *optional*, returned when `output_scores=True`):
            Beam indices of generated token id at each generation step. `ms.Tensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        encoder_attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(ms.Tensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `ms.Tensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (`tuple(tuple(ms.Tensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `ms.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(ms.Tensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `ms.Tensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(ms.Tensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: ms.Tensor = None
    sequences_scores: Optional[ms.Tensor] = None
    scores: Optional[Tuple[ms.Tensor]] = None
    logits: Optional[Tuple[ms.Tensor]] = None
    beam_indices: Optional[ms.Tensor] = None
    encoder_attentions: Optional[Tuple[ms.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[ms.Tensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[ms.Tensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[ms.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[ms.Tensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[ms.Tensor]]]] = None


# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]


class GenerationMixin:
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in model classes.
    Inheriting from this class causes the model to have special generation-related behavior, such as loading a
    `GenerationConfig` at initialization time or ensuring `generate`-related tests are run in `transformers` CI.

    A model class should inherit from `GenerationMixin` to enable calling methods like `generate`, or when it
    has defined a custom `generate` method that relies on `GenerationMixin`, directly or indirectly, which
    approximately shares the same interface to public methods like `generate`. Three examples:
        - `LlamaForCausalLM` should inherit from `GenerationMixin` to enable calling `generate` and other public
            methods in the mixin;
        - `BlipForQuestionAnswering` has a custom `generate` method that approximately shares the same interface as
           `GenerationMixin.generate` (it has a few extra arguments, and the same output). That function also calls
           `GenerationMixin.generate` indirectly, through an inner model. As such, `BlipForQuestionAnswering` should
           inherit from `GenerationMixin` to benefit from all generation-related automation in our codebase;
        - `BarkModel` has a custom `generate` method and one of its inner models calls `GenerationMixin.generate`.
            However, its `generate` does not share the same interface as `GenerationMixin.generate`. In this case,
            `BarkModel` shoud NOT inherit from `GenerationMixin`, as it breaks the `generate` interface.

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *contrastive search* if `penalty_alpha>0` and `top_k>1`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`
        - *beam-search decoding* if `num_beams>1` and `do_sample=False`
        - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
        - *diverse beam-search decoding* if `num_beams>1` and `num_beam_groups>1`
        - *constrained beam-search decoding* if `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values: Union[Cache, Tuple] = None,
        attention_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        cache_position: Optional[ms.Tensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = (
                get_seq_length(past_key_values, dynamic=self._supports_default_dynamic_input())
                if past_key_values is not None
                else 0
            )
            cache_position = ops.arange(past_length, input_ids.shape[1], dtype=ms.int32)

        if kwargs["use_cache"]:
            model_inputs["cache_position"] = cache_position

        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        # Excpetion 4: If input_embeds are passed then slice it through `cache_position`, to keep only the unprocessed tokens and
        # generate the first token for each sequence. Later use the generated Input ids for continuation.
        if past_key_values is not None:
            # Make sure `past_key_values` is mutable (required for graph mode to work faster)
            model_inputs["past_key_values"] = (
                ms.mutable(past_key_values) if isinstance(past_key_values, tuple) else past_key_values
            )
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:  # Exception 1 or # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # For static shape, padding input_id to max_len when no cache, in prefill-stage
                if not self._supports_default_dynamic_input() and past_key_values is None:
                    pad_len = max(0, attention_mask.shape[1] - input_ids.shape[1])
                    input_ids = mint.nn.functional.pad(input_ids, (0, pad_len), value=0)
                model_inputs[input_ids_key] = input_ids
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone()

        # 4. Create missing `position_ids` on the fly
        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        )
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.construct).parameters.keys())
        ):
            position_ids = attention_mask.to(ms.int32).cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                _past_key_values = past_key_values
                if (
                    isinstance(past_key_values, (tuple, list))
                    and get_seq_length(past_key_values, dynamic=self._supports_default_dynamic_input()) == 0
                ):
                    _past_key_values = None

                if _past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    if self._supports_default_dynamic_input() or attention_mask is None:
                        model_input = model_input[:, -current_input_length:]
                    else:  # static shape input
                        cur_len = attention_mask.sum(-1).max()
                        model_input = model_input[:, cur_len - current_input_length : cur_len]
                    model_input = model_input.clone()
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                logger.warning_once(
                    f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                    "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                    "writing code, see Llama for an example implementation. If you're a user, please report this "
                    "issue on GitHub."
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    cache_position=cache_position,
                    batch_size=batch_size,
                )
        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)

        # Paged Attention
        if self.config._attn_implementation == "paged_attention":
            bs, seq_len = input_ids.shape
            step = kwargs["step"]
            if step == 0:
                self.enable_dynamic_shape()

                # init block tables
                self.block_mgr = BlockTables(1024, 32, self.config.max_position_embeddings)
                self.block_mgr.init_cache_engine(bs)

                # get slot mapping and block tables
                max_input_length = self.config.max_position_embeddings
                self.valid_length_each_example = ms.tensor(seq_len).reshape(bs)
                block_tables, slot_mapping = self.block_mgr.assemble_pa_full_inputs(
                    max_input_length, self.valid_length_each_example, [False]
                )
                slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))

                # set batch valid length
                self.batch_valid_length = ms.tensor(seq_len).to(ms.int32).reshape(bs)

                self.phase = "prefill"
                self._add_flags_custom(True)
            else:
                model_inputs.update({"input_ids": input_ids[:, -1].reshape(bs, 1)})

                # get slot mapping and block tables
                self.valid_length_each_example += 1
                block_tables, slot_mapping = self.block_mgr.assemble_pa_inc_inputs(
                    self.valid_length_each_example, [False]
                )

                # set batch valid length
                self.batch_valid_length += 1

                if step == 1:
                    self.phase = "increment"
                    self._add_flags_custom(False)
            slot_mapping = ms.tensor(slot_mapping)
            block_tables = ms.tensor(block_tables)
            model_inputs.update(
                {
                    "block_tables": block_tables,
                    "slot_mapping": slot_mapping,
                    "batch_valid_length": self.batch_valid_length,
                }
            )
            model_inputs.pop("step", None)

        return model_inputs

    def _add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.infer_attention.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.infer_attention.paged_attention_mgr.add_flags(is_first_iteration=is_first_iteration)

    def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool = False):
        past_key_values = None
        cache_name = "past_key_values"
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        elif "cache_params" in outputs:
            past_key_values = outputs.cache_params
            cache_name = "cache_params"

        # Bloom fix: standardizes the cache format when requested
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
        return cache_name, past_key_values

    def _prepare_model_inputs(
        self,
        inputs: Optional[ms.Tensor] = None,
        bos_token_id: Optional[ms.Tensor] = None,
        model_kwargs: Optional[Dict[str, ms.Tensor]] = None,
    ) -> Tuple[ms.Tensor, Optional[str], Dict[str, ms.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. "
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. In the presence of `inputs_embeds` for text models:
        # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
        # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
        # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
        # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
        # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                        "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                        "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                    )
                # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
                # the attention mask) can rely on the actual model input.
                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, model_kwargs=model_kwargs
                )
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[ms.Tensor] = None,
        bos_token_id: Optional[ms.Tensor] = None,
        model_kwargs: Optional[Dict[str, ms.Tensor]] = None,
    ) -> ms.Tensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs.last_hidden_state.shape[:-1]
            return ops.ones(shape, dtype=ms.int32) * -100

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, ms.Tensor):
                batch_size = value.shape[0]
                break

        if "inputs_embeds" in model_kwargs:
            return ops.ones((batch_size, 0), dtype=ms.int32)

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        return ops.ones((batch_size, 1), dtype=ms.int32) * bos_token_id

    def _prepare_attention_mask_for_generation(
        self,
        inputs_tensor: ms.Tensor,
        generation_config: GenerationConfig,
        model_kwargs: Dict[str, Any],
    ) -> ms.Tensor:
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor

        # `input_ids` may be present in the model kwargs, instead of being the main input (e.g. multimodal model)
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        # No information for attention mask inference -> return default attention mask
        default_attention_mask = ops.ones(inputs_tensor.shape[:2], dtype=ms.int32)
        if pad_token_id is None:
            return default_attention_mask

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [ms.int32, ms.int64]
        if not is_input_ids:
            return default_attention_mask

        is_pad_token_in_inputs = (pad_token_id is not None) and (
            mnp.isin(element=inputs_tensor, test_elements=pad_token_id).any()
        )
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or ~(
            mnp.isin(element=eos_token_id, test_elements=pad_token_id).any()
        )
        can_infer_attention_mask = is_pad_token_in_inputs * is_pad_token_not_equal_to_eos_token_id
        attention_mask_from_padding = inputs_tensor.ne(pad_token_id).to(ms.int32)

        attention_mask = (
            attention_mask_from_padding * can_infer_attention_mask + default_attention_mask * ~can_infer_attention_mask
        )
        return attention_mask

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: ms.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs and generation config.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.construct).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }
        encoder_kwargs["output_attentions"] = generation_config.output_attentions
        encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, ms.Tensor],
        decoder_start_token_id: ms.Tensor,
        **ignore_kwargs,
    ) -> Tuple[ms.Tensor, Dict[str, ms.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. `decoder_start_token_id` must have shape (batch_size, 1)
        if decoder_start_token_id.ndim == 1:
            if decoder_start_token_id.shape[0] != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expected to have length {batch_size} but got {decoder_start_token_id.shape[0]}"
                )
            decoder_start_token_id = decoder_start_token_id.view(-1, 1)
        else:
            decoder_start_token_id = ops.ones((batch_size, 1), dtype=ms.int32) * decoder_start_token_id

        # 3. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_start_token_id
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token. Note that the
        # original checkpoints can't be detected through `self.__class__.__name__.lower()`, needing custom logic.
        # See: https://github.com/huggingface/transformers/pull/31470
        elif "donut" in self.__class__.__name__.lower() or (
            self.config.model_type == "vision-encoder-decoder" and "donut" in self.config.encoder.model_type.lower()
        ):
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item():
            decoder_input_ids = ops.cat([decoder_start_token_id, decoder_input_ids], axis=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = ops.cat(
                    (ops.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    axis=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[ms.Tensor] = None,
        **model_kwargs,
    ) -> Tuple[ms.Tensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        if expand_size == 1:
            return input_ids, model_kwargs

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], ms.Tensor)
                ):
                    if dict_to_expand[key].dtype == ms.bool_:
                        dict_to_expand[key] = (
                            dict_to_expand[key].to(ms.int32).repeat_interleave(expand_size, dim=0).to(ms.bool_)
                        )
                    else:
                        dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = ops.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], axis=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]

                if self._supports_default_dynamic_input():
                    model_kwargs["attention_mask"] = ops.cat(
                        [attention_mask, ops.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)], axis=-1
                    )
                else:  # update static attention mask
                    cur_lens = attention_mask.sum(-1)
                    for batch_idx in range(attention_mask.shape[0]):
                        cur_len = int(cur_lens[batch_idx])
                        if cur_len < attention_mask.shape[-1]:
                            attention_mask[batch_idx, cur_len] = 1
                        else:
                            attention_mask[batch_idx, :-1] = attention_mask[batch_idx, 1:]
                            attention_mask[batch_idx, -1:] = 1
                    model_kwargs["attention_mask"] = attention_mask
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = ops.cat(
                    [
                        decoder_attention_mask,
                        ops.ones((decoder_attention_mask.shape[0], 1), dtype=decoder_attention_mask.dtype),
                    ],
                    axis=-1,
                )

        if model_kwargs.get("use_cache", True):
            # the first step for static shape
            if (
                not self._supports_default_dynamic_input()
                and model_kwargs.get("attention_mask", None) is not None
                and model_kwargs["attention_mask"].shape[-1] == model_kwargs["cache_position"].shape[0]
            ):
                cur_idx = int(model_kwargs["attention_mask"].sum(-1).max()) - 1
                past_idx = cur_idx - 1
                model_kwargs["cache_position"] = (
                    model_kwargs["cache_position"][past_idx : past_idx + 1] + num_new_tokens
                )
            else:
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            if self._supports_default_dynamic_input() or model_kwargs.get("attention_mask", None) is None:
                cur_idx = int(past_positions[-1]) + 1
                new_positions = mint.arange(cur_idx, cur_idx + num_new_tokens, dtype=past_positions.dtype)
                model_kwargs["cache_position"] = mint.cat((past_positions, new_positions))
            else:
                max_len = past_positions.shape[-1]
                cur_idx = int(model_kwargs["attention_mask"].sum(-1).max()) - 1
                new_positions = mint.arange(cur_idx, cur_idx + num_new_tokens, dtype=past_positions.dtype)
                cache_position = mint.cat((past_positions[:cur_idx], new_positions))
                if cache_position.shape[-1] < max_len:  # pad to max_len
                    cache_position = mint.cat(
                        (cache_position, ops.zeros((max_len - cache_position.shape[-1]), dtype=cache_position.dtype))
                    )
                model_kwargs["cache_position"] = cache_position

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError(
            f"Make sure that a `_reorder_cache` function is correctly implemented in {self.__class__.__module__} to"
            f" enable beam search for {self.__class__}"
        )

    def _get_candidate_generator(
        self,
        generation_config: GenerationConfig,
        input_ids: ms.Tensor,
        inputs_tensor: ms.Tensor,
        assistant_model: "PreTrainedModel",
        logits_processor: LogitsProcessorList,
        target_tokenizer: "PreTrainedTokenizerBase",
        assistant_tokenizer: "PreTrainedTokenizerBase",
        model_kwargs: Dict,
    ) -> CandidateGenerator:
        """
        Returns the candidate generator to be used in `assisted_generation`
        """
        different_tokenizers = all(v is not None for v in (assistant_model, target_tokenizer, assistant_tokenizer))

        if generation_config.assistant_early_exit is not None:
            candidate_generator = EarlyExitCandidateGenerator(
                input_ids=input_ids,
                assistant_model=self,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                inputs_tensor=inputs_tensor,
                logits_processor=logits_processor,
            )
        elif generation_config.prompt_lookup_num_tokens is not None:
            candidate_generator = PromptLookupCandidateGenerator(
                eos_token_id=generation_config._eos_token_tensor,
                num_output_tokens=generation_config.prompt_lookup_num_tokens,
                max_matching_ngram_size=generation_config.max_matching_ngram_size,
                max_length=generation_config.max_length,
            )
        elif different_tokenizers:
            if generation_config.do_sample is True:
                atm_translator = AssistantVocabTranslatorCache.get_translator(
                    target_tokenizer, assistant_tokenizer, self.config.vocab_size
                )
                candidate_generator = UniversalSpeculativeDecodingGenerator(
                    input_ids=input_ids,
                    assistant_model=assistant_model,
                    generation_config=generation_config,
                    model_kwargs=model_kwargs,
                    inputs_tensor=inputs_tensor,
                    logits_processor=logits_processor,
                    target_tokenizer=target_tokenizer,
                    assistant_tokenizer=assistant_tokenizer,
                    atm_translator=atm_translator,
                )
            elif generation_config.do_sample is False:
                candidate_generator = AssistedCandidateGeneratorDifferentTokenizers(
                    input_ids=input_ids,
                    assistant_model=assistant_model,
                    generation_config=generation_config,
                    model_kwargs=model_kwargs,
                    inputs_tensor=inputs_tensor,
                    logits_processor=logits_processor,
                    target_tokenizer=target_tokenizer,
                    assistant_tokenizer=assistant_tokenizer,
                )
            else:
                raise ValueError(
                    f"Invalid value for `do_sample`: expected a boolean, got {type(generation_config.do_sample).__name__}"
                )
        else:
            candidate_generator = AssistedCandidateGenerator(
                input_ids=input_ids,
                assistant_model=assistant_model,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                inputs_tensor=inputs_tensor,
                logits_processor=logits_processor,
            )
        return candidate_generator

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: ms.Tensor,
        prefix_allowed_tokens_fn: Callable[[int, ms.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_prompt_ids: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
            processors.append(
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    generation_config.guidance_scale,
                    self,
                    unconditional_ids=negative_prompt_ids,
                    unconditional_attention_mask=negative_prompt_attention_mask,
                    use_cache=generation_config.use_cache,
                )
            )
        if generation_config.sequence_bias is not None:
            processors.append(SequenceBiasLogitsProcessor(sequence_bias=generation_config.sequence_bias))

        if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=generation_config.diversity_penalty,
                    num_beams=generation_config.num_beams,
                    num_beam_groups=generation_config.num_beam_groups,
                )
            )
        if (
            generation_config.encoder_repetition_penalty is not None
            and generation_config.encoder_repetition_penalty != 1.0
        ):
            if len(encoder_input_ids.shape) == 2:
                processors.append(
                    EncoderRepetitionPenaltyLogitsProcessor(
                        penalty=generation_config.encoder_repetition_penalty,
                        encoder_input_ids=encoder_input_ids,
                    )
                )
            else:
                warnings.warn(
                    "Passing `encoder_repetition_penalty` requires some form of `input_ids` to be passed to "
                    "`generate`, ignoring the argument.",
                    UserWarning,
                )
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        if (
            generation_config.encoder_no_repeat_ngram_size is not None
            and generation_config.encoder_no_repeat_ngram_size > 0
        ):
            if len(encoder_input_ids.shape) == 2:
                processors.append(
                    EncoderNoRepeatNGramLogitsProcessor(
                        generation_config.encoder_no_repeat_ngram_size,
                        encoder_input_ids,
                    )
                )
            else:
                warnings.warn(
                    "Passing `encoder_no_repeat_ngram_size` requires some form of `input_ids` to be passed to "
                    "`generate`, ignoring the argument.",
                    UserWarning,
                )
        if generation_config.bad_words_ids is not None:
            processors.append(
                NoBadWordsLogitsProcessor(
                    generation_config.bad_words_ids,
                    generation_config._eos_token_tensor,
                )
            )
        if (
            generation_config.min_length is not None
            and generation_config._eos_token_tensor is not None
            and generation_config.min_length > 0
        ):
            processors.append(
                MinLengthLogitsProcessor(
                    generation_config.min_length,
                    generation_config._eos_token_tensor,
                )
            )
        if (
            generation_config.min_new_tokens is not None
            and generation_config._eos_token_tensor is not None
            and generation_config.min_new_tokens > 0
        ):
            processors.append(
                MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length,
                    generation_config.min_new_tokens,
                    generation_config._eos_token_tensor,
                )
            )
        if prefix_allowed_tokens_fn is not None:
            processors.append(
                PrefixConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn,
                    generation_config.num_beams // generation_config.num_beam_groups,
                )
            )
        if generation_config.forced_bos_token_id is not None:
            processors.append(
                ForcedBOSTokenLogitsProcessor(
                    generation_config.forced_bos_token_id,
                )
            )
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                ForcedEOSTokenLogitsProcessor(
                    generation_config.max_length,
                    generation_config.forced_eos_token_id,
                )
            )
        if generation_config.remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        if generation_config.exponential_decay_length_penalty is not None:
            processors.append(
                ExponentialDecayLengthPenalty(
                    generation_config.exponential_decay_length_penalty,
                    generation_config._eos_token_tensor,
                    input_ids_seq_length,
                )
            )
        if generation_config.suppress_tokens is not None:
            processors.append(
                SuppressTokensLogitsProcessor(
                    generation_config.suppress_tokens,
                )
            )
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            processors.append(
                SuppressTokensAtBeginLogitsProcessor(
                    generation_config.begin_suppress_tokens,
                    begin_index,
                )
            )

        # Fixme
        # if generation_config.forced_decoder_ids is not None:
        #     raise ValueError(
        #         "You have explicitly specified `forced_decoder_ids`. Please remove the `forced_decoder_ids` argument "
        #         "in favour of `input_ids` or `decoder_input_ids` respectively.",
        #     )

        processors = self._merge_criteria_processor_list(processors, logits_processor)

        # Processors previously known as `LogitsWarpers`, only applied with sampling strategies
        if generation_config.do_sample:
            # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
            # better score (i.e. keep len(list(generation_config._eos_token_tensor)) + 1)
            if generation_config.num_beams > 1:
                if isinstance(generation_config._eos_token_tensor, list):
                    min_tokens_to_keep = len(generation_config._eos_token_tensor) + 1
                elif isinstance(generation_config._eos_token_tensor, ms.Tensor):
                    min_tokens_to_keep = generation_config._eos_token_tensor.shape[0] + 1
                else:
                    min_tokens_to_keep = 2
            else:
                min_tokens_to_keep = 1

            # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
            # all samplers can be found in `generation_utils_samplers.py`
            if generation_config.temperature is not None and generation_config.temperature != 1.0:
                processors.append(TemperatureLogitsWarper(generation_config.temperature))
            if generation_config.top_k is not None and generation_config.top_k != 0:
                processors.append(
                    TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep)
                )
            if generation_config.top_p is not None and generation_config.top_p < 1.0:
                processors.append(
                    TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep)
                )
            if generation_config.min_p is not None:
                # Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
                processors.append(
                    MinPLogitsWarper(min_p=generation_config.min_p, min_tokens_to_keep=min_tokens_to_keep)
                )
            if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
                processors.append(
                    TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
                )
            if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
                processors.append(
                    EpsilonLogitsWarper(epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
                )
            if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
                processors.append(
                    EtaLogitsWarper(epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep)
                )

        # Watermarking should be after all logits processing is finished (see #34630)
        if generation_config.watermarking_config is not None:
            processors.append(generation_config.watermarking_config.construct_processor(self.config.vocab_size))

        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors

    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        if generation_config.stop_strings is not None:
            if tokenizer is None:
                raise ValueError(
                    "There are one or more stop strings, either in the arguments to `generate` or in the "
                    "model's generation config, but we could not locate a tokenizer. When generating with "
                    "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                )
            # criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))
            raise NotImplementedError
        if generation_config._eos_token_tensor is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))
        if (
            generation_config.is_assistant
            and generation_config.assistant_confidence_threshold is not None
            and generation_config.assistant_confidence_threshold > 0
        ):
            criteria.append(
                ConfidenceCriteria(assistant_confidence_threshold=generation_config.assistant_confidence_threshold)
            )
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _merge_criteria_processor_list(
        self,
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        """
        Merge user-defined processors/criteria with the ones instantiated inside `generate`. In case the same
        processor/criteria is present on both lists, use the user-defined one.

        (Note: up to v4.49.0, this funtion threw an exception is the same logit processor was found twice.)
        """
        if len(custom_list) == 0:
            return default_list

        final_list = type(default_list)()
        for default in default_list:
            using_custom = False
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    logger.warning_once(
                        f"A custom {object_type} of type {type(custom)} has been passed to `.generate()`, but it "
                        f"was also created in `.generate()`, given its parameterization. The custom {type(custom)} "
                        f"will take precedence. Please check the docstring of {type(custom)} to see related "
                        "`.generate()` flags."
                    )
                    final_list.append(custom)
                    using_custom = True
                    break
            if not using_custom:
                final_list.append(default)

        for custom in custom_list:
            if custom not in final_list:
                final_list.append(custom)
        return final_list

    def compute_transition_scores(
        self,
        sequences: ms.Tensor,
        scores: tuple[ms.Tensor],
        beam_indices: Optional[ms.Tensor] = None,
        normalize_logits: bool = False,
    ) -> ms.Tensor:
        """
        Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was
        used). This is a convenient method to quicky obtain the scores of the selected tokens at generation time.

        Parameters:
            sequences (`torch.LongTensor`):
                The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or
                shorter if all batches finished early due to the `eos_token_id`.
            scores (`tuple(torch.FloatTensor)`):
                Transition scores for each vocabulary token at each generation step. Beam transition scores consisting
                of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
                Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
                with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
            beam_indices (`torch.LongTensor`, *optional*):
                Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
                `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
                generate-time.
            normalize_logits (`bool`, *optional*, defaults to `False`):
                Whether to normalize the logits (which, for legacy reasons, may be unnormalized).

        Return:
            `ms.Tensor`: A `ms.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)` containing
                the transition scores (logits)

        Examples:

        ```python
        >>> from transformers import GPT2Tokenizer
        >>> from mindway.transformers import AutoModelForCausalLM
        >>> import numpy as np

        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer.pad_token_id = tokenizer.eos_token_id
        >>> inputs = tokenizer(["Today is"], return_tensors="np")

        >>> # Example 1: Print the scores for each token generated with Greedy Search
        >>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
        >>> transition_scores = model.compute_transition_scores(
        ...     outputs.sequences, outputs.scores, normalize_logits=True
        ... )
        >>> # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
        >>> # encoder-decoder models, like BART or T5.
        >>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        >>> generated_tokens = outputs.sequences[:, input_length:]
        >>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
        ...     # | token | token string | log probability | probability
        ...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
        |   262 |  the     | -1.414 | 24.33%
        |  1110 |  day     | -2.609 | 7.36%
        |   618 |  when    | -2.010 | 13.40%
        |   356 |  we      | -1.859 | 15.58%
        |   460 |  can     | -2.508 | 8.14%

        >>> # Example 2: Reconstruct the sequence scores from Beam Search
        >>> outputs = model.generate(
        ...     **inputs,
        ...     max_new_tokens=5,
        ...     num_beams=4,
        ...     num_return_sequences=4,
        ...     return_dict_in_generate=True,
        ...     output_scores=True,
        ... )
        >>> transition_scores = model.compute_transition_scores(
        ...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        ... )
        >>> # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
        >>> # Tip 1: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
        >>> # use case, you might want to recompute it with `normalize_logits=True`.
        >>> # Tip 2: the output length does NOT include the input length
        >>> output_length = np.sum(transition_scores.numpy() < 0, axis=1)
        >>> length_penalty = model.generation_config.length_penalty
        >>> reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
        >>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
        True
        ```"""
        # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach were the first (and only) beam is always selected
        if beam_indices is None:
            beam_indices = mint.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
            beam_indices = beam_indices.expand(-1, len(scores))

        # 2. reshape scores as [batch_size*vocab_size, # generation steps] with # generation steps being
        # seq_len - input_length
        scores = mint.stack(scores).reshape(len(scores), -1).swapaxes(0, 1)

        # 3. Optionally normalize the logits (across the vocab dimension)
        if normalize_logits:
            scores = scores.reshape(-1, self.config.vocab_size, scores.shape[-1])
            scores = mint.nn.functional.log_softmax(scores, dim=1)
            scores = scores.reshape(-1, scores.shape[-1])

        # 4. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 5. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards
        beam_indices[beam_indices_mask] = 0

        # 6. multiply beam_indices with vocab size to gather correctly from scores
        beam_sequence_indices = beam_indices * self.config.vocab_size

        # 7. Define which indices contributed to scores
        cut_idx = sequences.shape[-1] - max_beam_length
        indices = sequences[:, cut_idx:] + beam_sequence_indices

        # 8. Compute scores
        transition_scores = scores.gather(0, indices)

        # 9. Mask out transition_scores of beams that stopped early
        transition_scores[beam_indices_mask] = 0

        return transition_scores

    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        # TODO(joao): remove this function in v4.50, i.e. when we remove the inheritance of `GenerationMixin` from
        # `PreTrainedModel`. With that inheritance removed, all model classes inheriting from `GenerationMixin` can
        # safely call `GenerationMixin.generate`
        if not self.can_generate():
            terminations_with_generation_support = [
                "ForCausalLM",
                "ForConditionalGeneration",
                "ForSpeechSeq2Seq",
                "ForVision2Seq",
            ]
            raise TypeError(
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head. Classes that support generation often end in one of these "
                f"names: {terminations_with_generation_support}."
            )

    def _validate_assistant(self, assistant_model, tokenizer, assistant_tokenizer):
        if assistant_model is None:
            return

        if self.config.is_encoder_decoder and not assistant_model.config.is_encoder_decoder:
            attributes_to_check = ["encoder_attention_heads", "encoder_ffn_dim", "encoder_layers"]
            attributes_to_check = [attr for attr in dir(assistant_model.config) if attr in attributes_to_check]
            are_equal = all(
                getattr(self.config, attr) == getattr(assistant_model.config, attr) for attr in attributes_to_check
            )
            if not are_equal:
                raise ValueError(
                    "The main model and the assistant don't have compatible encoder-dependent input shapes. "
                    "Ensure you load the assistant with the correct encoder-decoder class, e.g. `AutoModelForSpeechSeq2Seq` for Whisper."
                )

        doc_reference = (
            "(see https://huggingface.co/docs/transformers/en/generation_strategies#universal-assisted-decoding)"
        )
        if self.config.get_text_config().vocab_size == assistant_model.config.get_text_config().vocab_size:
            if assistant_tokenizer is not None:
                raise ValueError(
                    f"`assistant_tokenizer` is not required when the main and assistant models use the same tokenizer. Please omit `assistant_tokenizer` "
                    f"from `generate()` {doc_reference}."
                )
        else:
            if tokenizer is None or assistant_tokenizer is None:
                raise ValueError(
                    f"The main and assistant moedels have different tokenizers. Please provide `tokenizer` and `assistant_tokenizer` "
                    f"to `generate()` {doc_reference}."
                )

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # If a `Cache` instance is passed, checks whether the model is compatible with it
        if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
            raise ValueError(
                f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
                "check the model documentation for supported cache formats."
            )

        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.construct).parameters)

        # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
        if self.config.is_encoder_decoder:
            base_model = getattr(self, self.base_model_prefix, None)

            # allow encoder kwargs
            encoder = getattr(self, "encoder", None)
            # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
            # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
            # TODO: A better way to handle this.
            if encoder is None and base_model is not None:
                encoder = getattr(base_model, "encoder", None)

            if encoder is not None:
                encoder_model_args = set(inspect.signature(encoder.construct).parameters)
                model_args |= encoder_model_args

            # allow decoder kwargs
            decoder = getattr(self, "decoder", None)
            if decoder is None and base_model is not None:
                decoder = getattr(base_model, "decoder", None)

            if decoder is not None:
                decoder_model_args = set(inspect.signature(decoder.construct).parameters)
                model_args |= {f"decoder_{x}" for x in decoder_model_args}

        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""
        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

        # 2. Min length warnings due to unfeasible parameter combinations
        min_length_error_suffix = (
            " Generation will stop at the defined maximum length. You should decrease the minimum length and/or "
            "increase the maximum length."
        )
        if has_default_max_length:
            min_length_error_suffix += (
                f" Note that `max_length` is set to {generation_config.max_length}, its default value."
            )
        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            warnings.warn(
                f"Unfeasible length constraints: `min_length` ({generation_config.min_length}) is larger than"
                f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
                UserWarning,
            )
        if generation_config.min_new_tokens is not None:
            min_length = generation_config.min_new_tokens + input_ids_length
            if min_length > generation_config.max_length:
                warnings.warn(
                    f"Unfeasible length constraints: `min_new_tokens` ({generation_config.min_new_tokens}), when "
                    f"added to the prompt length ({input_ids_length}), is larger than"
                    f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
                    UserWarning,
                )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        has_default_min_length,
        model_input_name,
        input_ids_length,
        inputs_tensor,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        # if both `inputs_embeds` and `input_ids` are passed, we do not correct the length
        # otherwise we need total length [inputs-embeds-len + new-tokens-len] to not go beyond indicated `max_length``
        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.max_length -= inputs_tensor.shape[1]
        elif has_default_max_length:  # by default let's always generate 20 new tokens
            if generation_config.max_length == GenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        # same for min length
        if generation_config.min_new_tokens is not None:
            if not has_default_min_length:
                logger.warning(
                    f"Both `min_new_tokens` (={generation_config.min_new_tokens}) and `min_length`(="
                    f"{generation_config.min_length}) seem to have been set. `min_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.min_length = generation_config.min_new_tokens + input_ids_length

        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.min_length = max(generation_config.min_length - inputs_tensor.shape[1], 0)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # parameterization priority:
        # kwargs > non-global default values in `generation_config` > `model.generation_config` > GenerationConfig()
        # TODO (joao): per-model generation config classes.

        using_model_generation_config = False
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # the following conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same);
            # 3) there are non-default generation parameters in the model config.
            # 4) the user must have set new generation parameters in the model config.
            if (
                self.generation_config._from_model_config  # 1)
                and self.generation_config._original_object_hash == hash(self.generation_config)  # 2)
                and len(self.config._get_non_default_generation_parameters()) > 0  # 3)
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:  # 4)
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed in v5."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )",
                        UserWarning,
                    )
                    self.generation_config = new_generation_config

            generation_config = self.generation_config
            using_model_generation_config = True

        generation_config = copy.deepcopy(generation_config)

        if not using_model_generation_config:
            # If `generation_config` is provided:
            # - `use_model_defaults`: let's fallback ALL default values to the model's generation config
            # - otherwise: legacy behavior, let's just make sure we have the tokens defined
            model_base_version = version.parse(version.parse(self.generation_config.transformers_version).base_version)
            if use_model_defaults is True or (
                use_model_defaults is None and model_base_version >= version.parse("4.50.0")
            ):
                modified_values = {}
                default_generation_config = GenerationConfig()
                for key, default_value in default_generation_config.__dict__.items():
                    if key.startswith("_") or key == "transformers_version":  # metadata
                        continue
                    custom_gen_config_value = getattr(generation_config, key)
                    model_gen_config_value = getattr(self.generation_config, key)
                    if custom_gen_config_value == default_value and model_gen_config_value != default_value:
                        modified_values[key] = model_gen_config_value
                        setattr(generation_config, key, model_gen_config_value)
                if len(modified_values) > 0:
                    logger.warning_once(
                        f"`generation_config` default values have been modified to match model-specific defaults: "
                        f"{modified_values}. If this is not desired, please set these values explicitly."
                    )
            else:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.decoder_start_token_id is None:
                    generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id

        # Finally, apply any passed kwargs
        model_kwargs = generation_config.update(**kwargs)

        return generation_config, model_kwargs

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # the lines below are equivalent to `mint.arange` [0,1,2,3, .., input_shape-1]
        if "inputs_embeds" in model_kwargs and not self.config.is_encoder_decoder:
            cache_position = mint.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=ms.int32).cumsum(0) - 1
        elif "decoder_inputs_embeds" in model_kwargs and self.config.is_encoder_decoder:
            cache_position = (
                mint.ones_like(model_kwargs["decoder_inputs_embeds"][0, :, 0], dtype=ms.int32).cumsum(0) - 1
            )
        else:
            cache_position = mint.ones_like(input_ids[0, :], dtype=ms.int32).cumsum(0) - 1

        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = get_seq_length(cache, dynamic=self._supports_default_dynamic_input())
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

            # [past_length, past_length+1, ..., input_shape-1]
            cache_position = cache_position[past_length:]

            # for static input, cache fallback to static shape
            if not self._supports_default_dynamic_input() and model_kwargs.get("attention_mask", None) is not None:
                attention_mask = model_kwargs["attention_mask"]
                cur_len = int(attention_mask.sum(-1).max())
                valid_len = cur_len - past_length
                max_len = cache_position.shape[0]
                if valid_len < max_len:
                    cache_position = cache_position[:valid_len]
                    cache_position = mint.cat([cache_position, mint.zeros(max_len - valid_len, dtype=ms.int32)])

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    def _get_cache(
        self, cache_implementation: str, batch_size: int, max_cache_len: int, model_kwargs
    ) -> tuple[tuple[ms.Tensor, ms.Tensor]]:
        """
        Sets a cache class for `generate`, that will persist across calls. A new cache will only be initialized a
        new `generate` call requires a larger cache or uses a different batch size.

        Returns the resulting cache object.
        """
        cache_cls: Cache = NEED_SETUP_CACHE_CLASSES_MAPPING[cache_implementation]
        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )

        if hasattr(self, "_cache"):
            cache_to_check = self._cache.self_attention_cache if requires_cross_attention_cache else self._cache

        if cache_implementation == "sliding_window":
            max_cache_len = min(self.config.sliding_window, max_cache_len)

        need_new_cache = (
            not hasattr(self, "_cache")
            or (not isinstance(cache_to_check, cache_cls))
            or cache_to_check.max_batch_size != batch_size
        )
        if cache_implementation != "mamba":
            need_new_cache = need_new_cache or cache_to_check.max_cache_len < max_cache_len

        if requires_cross_attention_cache and hasattr(self, "_cache"):
            need_new_cache = (
                need_new_cache
                or self._cache.cross_attention_cache.max_cache_len != model_kwargs["encoder_outputs"][0].shape[1]
            )

        if need_new_cache:
            if hasattr(self.config, "_pre_quantization_dtype"):
                cache_dtype = self.config._pre_quantization_dtype
            else:
                cache_dtype = self.dtype

            cache_kwargs = {
                "config": self.config.get_text_config(),
                "max_batch_size": batch_size,
                "max_cache_len": max_cache_len,
                "dtype": cache_dtype,
            }
            self._cache = cache_cls(**cache_kwargs)
            if requires_cross_attention_cache:
                encoder_kwargs = cache_kwargs.copy()
                encoder_kwargs["max_cache_len"] = model_kwargs["encoder_outputs"][0].shape[1]
                self._cache = EncoderDecoderCache(self._cache, cache_cls(**encoder_kwargs))
        else:
            self._cache.reset()
        return self._cache

    def _supports_default_dynamic_cache(self) -> bool:
        """
        Return `True` if current model can use a `DynamicCache` instance when initializing the `past_key_values`.
        This is mostly the same as `_supports_cache_class` attribute, but add exception for `Jamba` model which
        uses its own `HybridMambaAttentionDynamicCache` and do not need to initialize the Cache in advance in
        order to save memory (because no back and forth `to_legacy_cache` and `from_legacy_cache` will be performed
        for `HybridMambaAttentionDynamicCache`).
        """
        return (
            self._supports_cache_class
            and "jamba" not in self.__class__.__name__.lower()
            and "zamba" not in self.__class__.__name__.lower()
            and "bamba" not in self.__class__.__name__.lower()
        )

    def _supports_default_dynamic_input(self) -> bool:
        """
        Return `True` if current model use dynamic cache or _supports_dynamic_input is `True` , but add exception for `paged_attention` which use dynamic input
        shape.
        """
        return (
            self._supports_dynamic_input
            or self._supports_default_dynamic_cache()
            or self.config._attn_implementation == "paged_attention"
        )

    def _prepare_legacy_cache(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        cache_name: str,
        batch_size: int,
    ):
        """
        Prepares a static legacy cache (tuple of tuples) for `generate`.
        """
        if self._supports_default_dynamic_input():  # cache will be default None and will be processed further in model
            return

        past = model_kwargs.get(cache_name, None)
        max_batch_size, max_cache_len, cache_dtype = (
            getattr(generation_config, "num_beams", 1) * batch_size,
            generation_config.max_length,
            self.dtype,
        )
        need_new_cache = (
            past is None
            or (not isinstance(past, tuple))
            or (not isinstance(past[0][0], ms.Tensor))
            or past[0][0].shape[0] != max_batch_size
            or past[0][0].shape[2] < max_cache_len
        )

        if need_new_cache:
            model_kwargs[cache_name] = init_static_cache(
                config=self.config,
                max_batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                dtype=cache_dtype,
            )
        else:
            model_kwargs[cache_name] = reset(past)

        return

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
    ) -> bool:
        """
        Prepares the cache for generation (if applicable), given `generate`'s parameterization. If a cache is
        instantiated, writes it to `model_kwargs`, under the name expected by the model.
        """

        cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"
        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )

        # Quick escape route 1: if the user specifies a cache, we only need to:
        # a) check for conflicting `generate` arguments
        # b) convert to the new cache format (if the user passes a legacy cache and model supports it)
        user_defined_cache = model_kwargs.get(cache_name)
        if user_defined_cache is not None:
            if generation_config.cache_implementation is not None:
                raise ValueError(
                    f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                    "Cache object) is unsupported. Please use only one of the two."
                )
            if isinstance(user_defined_cache, tuple) and self._supports_default_dynamic_cache():
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(user_defined_cache)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(user_defined_cache)
                )
            return

        # Quick escape route 2: if the user specifies no cache is to be used. (conflicting arguments are handled in
        # `generation_config.validate()`)
        if generation_config.use_cache is False:
            return

        # Quick escape route 3: model that only supports legacy caches. Prepare a static legacy cache
        # This is the case for most models currently.
        if not self._supports_default_dynamic_cache():
            if generation_config.cache_implementation is not None:
                warnings.warn(
                    "This model does not support `Cache` instances, it only supports the legacy cache format (tuple "
                    f"of tuples). `cache_implementation` (set to {generation_config.cache_implementation}) will be "
                    "ignored.",
                    UserWarning,
                )

            self._prepare_legacy_cache(
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                cache_name=cache_name,
                batch_size=batch_size,
            )
            return

        # Otherwise we NEED to prepare a cache class, based on `generation_config.cache_implementation`

        # TODO(joao): support static caches in assisted generation. assisted generation needs to roll back caches,
        # which is only supported in dynamic caches atm
        if assistant_model is not None and generation_config.cache_implementation is not None:
            logger.warning_once(
                "An assistant model is provided, using a dynamic cache instead of a cache of type="
                f"'{generation_config.cache_implementation}'."
            )
            generation_config.cache_implementation = None

        if generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs[cache_name] = self._get_cache(
                    cache_implementation=generation_config.cache_implementation,
                    batch_size=max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size,
                    max_cache_len=max_cache_length,
                    model_kwargs=model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue and tag @zucchini-nlp."
                    )
                raise NotImplementedError
            elif generation_config.cache_implementation == "offloaded":
                raise NotImplementedError
            elif generation_config.cache_implementation == "dynamic":
                model_kwargs[cache_name] = DynamicCache()

        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        else:
            model_kwargs[cache_name] = (
                DynamicCache()
                if not requires_cross_attention_cache
                else EncoderDecoderCache(DynamicCache(), DynamicCache())
            )

    def _supports_logits_to_keep(self) -> bool:
        """
        Return True if the current model supports the keyword argument `logits_to_keep` in forward()
        to save memory. Checking it in this way allows to avoid using a new model attribute.
        """
        return "logits_to_keep" in set(inspect.signature(self.construct).parameters.keys())

    def _prepare_special_tokens(
        self, generation_config: GenerationConfig, kwargs_has_attention_mask: Optional[bool] = None, **ignore_kwargs
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token):
            if token is None or isinstance(token, ms.Tensor):
                return token
            return ms.Tensor(token, dtype=ms.int32)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id)
        decoder_start_token_tensor = _tensor_or_none(generation_config.decoder_start_token_id)

        # for BC we also try to get `decoder_start_token_id` or `bos_token_id` (#30892)
        if self.config.is_encoder_decoder:
            decoder_start_token_tensor = (
                decoder_start_token_tensor if decoder_start_token_tensor is not None else bos_token_tensor
            )

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Sanity checks/warnings
        if self.config.is_encoder_decoder and decoder_start_token_tensor is None:
            raise ValueError(
                "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
            )
        # we can't infer attn mask if pad token is set to be eos token in model's generation config
        if eos_token_tensor is not None and mnp.isin(element=eos_token_tensor, test_elements=pad_token_tensor).any():
            if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                logger.warning_once(
                    "The attention mask is not set and cannot be inferred from input because pad token is same as "
                    "eos token. As a consequence, you may observe unexpected behavior. Please pass your input's "
                    "`attention_mask` to obtain reliable results."
                )

        if eos_token_tensor is not None and (ops.is_floating_point(eos_token_tensor) or (eos_token_tensor < 0).any()):
            logger.warning(
                f"`eos_token_id` should consist of positive integers, but is {eos_token_tensor}. Your generation will not "
                "stop until the maximum length is reached. Depending on other flags, it may even crash."
            )

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._decoder_start_token_tensor = decoder_start_token_tensor

    def _padding_inputs(
        self,
        generation_config,
        input_ids: ms.Tensor,
        inputs_embeds: ms.Tensor = None,
        labels: ms.Tensor = None,
        position_ids: ms.Tensor = None,
        attention_mask: ms.Tensor = None,
    ):
        # init empty array
        bs, max_length = len(input_ids), generation_config.max_length
        emb_length = inputs_embeds.shape[-1] if inputs_embeds is not None else 0
        ignore_label_index = 0

        padded_input_ids = ops.zeros((bs, max_length), ms.int32)
        padded_labels = ops.full((bs, max_length), ignore_label_index, dtype=ms.int32)
        padded_position_ids = ops.zeros((bs, max_length), ms.int32)
        padded_attention_mask = ops.zeros((bs, max_length), ms.bool_)

        padded_inputs_embeds = (
            ops.zeros((bs, max_length, emb_length), inputs_embeds.dtype) if inputs_embeds is not None else None
        )

        _labels = labels
        _position_ids = position_ids

        if attention_mask is None:
            if inputs_embeds is not None:
                attention_mask = ops.ones(inputs_embeds.shape[:2], dtype=ms.bool_)
            else:
                attention_mask = ops.ones(input_ids.shape[:], dtype=ms.bool_)
        else:
            attention_mask = attention_mask.astype(ms.bool_)
        cur_len = int(attention_mask.sum(-1).max())

        if position_ids is None:
            position_ids = ops.arange(0, cur_len, dtype=ms.int32)
        if labels is None:
            labels = ops.full(
                (
                    bs,
                    cur_len,
                ),
                ignore_label_index,
                dtype=ms.int32,
            )

        for batch_idx, cur_attention_mask in enumerate(attention_mask):
            cur_len = cur_attention_mask.sum()

            padded_attention_mask[batch_idx, :cur_len] = attention_mask[batch_idx][:]
            padded_input_ids[batch_idx, : min(cur_len, input_ids[batch_idx].shape[0])] = input_ids[batch_idx][:]
            padded_labels[batch_idx, :cur_len] = labels[batch_idx][:]
            padded_position_ids[batch_idx, :cur_len] = ops.arange(0, cur_len, dtype=position_ids.dtype)

            if inputs_embeds is not None:
                padded_inputs_embeds[batch_idx, :cur_len] = inputs_embeds[batch_idx][:]

        new_input_ids = padded_input_ids
        new_attention_mask = padded_attention_mask
        new_labels = None if _labels is None else padded_labels
        new_position_ids = None if _position_ids is None else padded_position_ids

        new_inputs_embeds = None if inputs_embeds is None else padded_inputs_embeds

        return new_input_ids, new_inputs_embeds, new_labels, new_position_ids, new_attention_mask

    def generate(
        self,
        inputs: Optional[ms.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, ms.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[ms.Tensor] = None,
        negative_prompt_attention_mask: Optional[ms.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, ms.Tensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`ms.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, ms.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            use_model_defaults (`bool`, *optional*):
                When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
                generation configuration (`model.generation_config`), as opposed to the global defaults
                (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
                `True`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `ms.Tensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `ms.Tensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        synced_gpus = False  # Set to `True` when zero3

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.construct).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and ops.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.asnumpy())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # This lines will always select the last logits, which is not the right way for static shape
        # if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
        #     model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length
        )

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            raise NotImplementedError
        elif generation_mode == GenerationMode.DOLA_GENERATION:
            raise NotImplementedError

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            raise NotImplementedError

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam sample
            result = self._beam_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            raise NotImplementedError

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool) -> bool:
        """
        Returns whether there are still unfinished sequences in the device. The existence of unfinished sequences is
        fed through `this_peer_finished`. ZeRO stage 3-friendly.
        """
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = ms.Tensor(0.0 if this_peer_finished else 1.0)
            # send 0.0 if we finished, 1.0 otherwise
            this_peer_finished_flag = ops.AllReduce()(this_peer_finished_flag)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                return False
        elif this_peer_finished:
            return False
        return True

    def heal_tokens(self, input_ids: ms.Tensor, tokenizer: Optional["PreTrainedTokenizerBase"] = None) -> ms.Tensor:
        r"""
        Generates sequences of token ids for models with a language modeling head.
        Parameters:
            input_ids (`ms.Tensor`): The sequence used as a prompt for the generation.
            tokenizer (`PreTrainedTokenizerBase`, *optional*): The tokenizer used to decode the input ids.
        Return:
            `ms.Tensor` where each sequence has its tail token replaced with its appropriate extension.
        """
        if tokenizer is None:
            raise ValueError(
                " When generating with token healing, you must pass the model's tokenizer to the `tokenizer` "
                "argument of `generate`."
            )

        bos_token_id, pad_token_id = tokenizer.bos_token_id, tokenizer.pad_token_id
        vocab_trie = ExtensionsTrie(tokenizer.get_vocab())
        generation_config = GenerationConfig(max_new_tokens=1, pad_token_id=pad_token_id)

        # assumption: leading/trailing whitespace is not meaningful, so the prompts are
        # stripped before re-tokenizing to desensitize generation to whitespace artefacts
        prompts = [p.strip() for p in tokenizer.batch_decode(input_ids, skip_special_tokens=True)]
        input_ids = ms.Tensor(
            tokenizer(
                prompts,
                return_tensors="np",
                padding=True,
            ).input_ids
        )

        # replace bos with pad to not condition healing on it
        input_ids = ops.where(input_ids == bos_token_id, pad_token_id, input_ids)

        tail_ids = input_ids[:, -1].tolist()

        space_tok = tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(" "))[0]
        # tail tokens are used for a prefix search, thus, whitespaces are replaced with
        # their tokenization (e.g. 'Ġ') to enable search for tokens prefixed with a whitespace
        tail_toks = (tokenizer.decode(t).replace(" ", space_tok) for t in tail_ids)

        for batch_idx, (tail_id, tail_tok) in enumerate(zip(tail_ids, tail_toks)):
            batch_ids = input_ids[batch_idx]
            if ops.all(batch_ids == pad_token_id).item():
                continue  # skip empty sequences (all pad ids)

            # apply bias for alternatives (extensions) to the tail token
            seq_bias = {(alt_tok,): 10.0 for alt_tok in vocab_trie.values(prefix=tail_tok)}
            if len(seq_bias) == 1:
                continue  # skip if there are no token alternatives to heal with

            # slightly favor original token to limit aggressive healing e.g. 'http' -> 'https'
            seq_bias[(tail_id,)] += 1.0
            generation_config.update(sequence_bias=seq_bias)

            trimmed_ids = batch_ids[:-1]

            """
            the latter code assumes trimmed_ids is not empty
            so have to check the its element count
            """
            if trimmed_ids.numel() == 0:
                continue

            # if the prompt is a single (non-pad) token, regenerate from bos
            if len(batch_ids[batch_ids != pad_token_id]) == 1:
                trimmed_ids[-1] = bos_token_id

            input_ids[batch_idx] = self.generate(trimmed_ids.unsqueeze(0), generation_config=generation_config)

        return input_ids

    def _sample(
        self,
        input_ids: ms.Tensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, ms.Tensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `ms.Tensor`:
            A `ms.Tensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # Padding inputs to avoid dynamic shape
        if not self._supports_default_dynamic_input():
            (
                padded_input_ids,
                padded_inputs_embeds,
                padded_labels,
                padded_position_ids,
                padded_attention_mask,
            ) = self._padding_inputs(
                generation_config,
                input_ids,
                model_kwargs.get("inputs_embeds", None),
                model_kwargs.get("labels", None),
                model_kwargs.get("position_ids", None),
                model_kwargs.get("attention_mask", None),
            )
            input_ids = padded_input_ids
            model_kwargs["attention_mask"] = padded_attention_mask
            if model_kwargs.get("inputs_embeds", None) is not None:
                model_kwargs["inputs_embeds"] = padded_inputs_embeds
            if model_kwargs.get("labels", None) is not None:
                model_kwargs["labels"] = padded_labels
            if model_kwargs.get("position_ids", None) is not None:
                model_kwargs["position_ids"] = padded_position_ids

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = ops.ones(batch_size, dtype=ms.int32)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        multinomial = get_multinomial_op()
        step = 0
        s_time = time.time()
        graph_compiled_time_buffer = []

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus):
            # prepare model inputs

            if input_ids.dtype == ms.int64:
                input_ids = input_ids.to(ms.int32)

            if self.config._attn_implementation == "paged_attention":
                model_kwargs["step"] = step
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            # Note that this is slightly different from hf transformers, since page attention dose not accept
            # undeterminante None/bool inputs
            model_inputs.update({"output_attentions": output_attentions})
            model_inputs.update({"output_hidden_states": output_hidden_states})

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=False if ms.get_context("mode") == ms.GRAPH_MODE else True,
            )
            if not isinstance(outputs, ModelOutput):
                outputs = ModelOutput(
                    loss=None,
                    logits=outputs[0],
                    past_key_values=outputs[1] if model_inputs.get("use_cache", False) else None,
                )

            if self._supports_default_dynamic_input() or model_kwargs.get("attention_mask", None) is None:
                next_token_logits = outputs.logits[:, -1, :]
            else:  # Get the right logits from static input shape
                attention_mask = model_kwargs["attention_mask"]
                cur_idx = int(attention_mask.sum(-1).max()) - 1

                if outputs.logits.shape[1] == attention_mask.shape[-1]:
                    next_token_logits = outputs.logits[:, cur_idx, :]  # (bs, seq, dim)
                else:
                    next_token_logits = outputs.logits[:, -1, :]

                # `input_ids` obtain effective length after 1st step
                if input_ids.shape[1] == attention_mask.shape[1]:
                    input_ids = input_ids[:, : cur_idx + 1]

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            step_time = time.time() - s_time
            if step < 2:
                print(f"==> sampling, step: {step}, time cost: {step_time:.5f}s")
            else:
                graph_compiled_time_buffer.append(step_time)
                token_speed = len(graph_compiled_time_buffer) / sum(graph_compiled_time_buffer)
                print(
                    f"==> sampling, step: {step}, time cost: {step_time:.5f}s, running avg speed: {token_speed:.5f}token/s"
                )
            s_time = time.time()
            step += 1

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = ops.softmax(next_token_scores, axis=-1, dtype=ms.float32).to(next_token_scores.dtype)
                next_tokens = multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = ops.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            next_tokens = next_tokens.to(ms.int32)

            # update generated ids, model inputs, and length for next step
            input_ids = ops.cat([input_ids, next_tokens[:, None]], axis=-1)
            if streamer is not None:
                streamer.put(next_tokens.asnumpy())

            unfinished_sequences = unfinished_sequences & ~ms.Tensor(stopping_criteria(input_ids, scores), ms.bool_)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    # Auxiliary functions for beam search
    def _temporary_reorder_cache(self, past_key_values, beam_idx):
        """
        Temporary function to handle the different types of cache reordering processes while we roll out `Cache`.
        TODO: standardize cache formats and make all models compatible with `Cache`. It would remove the need
        for this function, with `Cache.reorder_cache` being the sole remaining code path
        """
        model_class = self.__class__.__name__.lower()
        # Exception 1: code path for models using the legacy cache format
        if isinstance(past_key_values, (tuple, list)):
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
        # Exception 2: models with different cache formats. These are limited to `DynamicCache` until their
        # cache format is standardized, to avoid adding complexity to the codebase.
        elif "gptbigcode" in model_class:
            if not isinstance(past_key_values, (DynamicCache, EncoderDecoderCache)):
                raise ValueError(
                    f"Using an unsupported cache format with {model_class}. Currently, it only supports the "
                    "legacy tuple format or `DynamicCache`"
                )
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        # Standard code path: use the `Cache.reorder_cache`
        else:
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    @staticmethod
    def _flatten_beam_dim(tensor: ms.Tensor) -> ms.Tensor:
        """[batch_size, num_beams, ...] -> [batch_size * num_beams, ...]"""
        shape = list(tensor.shape)
        return mint.reshape(tensor, [shape[0] * shape[1]] + shape[2:])

    @staticmethod
    def _unflatten_beam_dim(tensor: ms.Tensor, batch_size: int, num_beams: int) -> ms.Tensor:
        """[batch_size * num_beams, ...] -> [batch_size, num_beams, ...]"""
        shape = list(tensor.shape)
        return mint.reshape(tensor, [batch_size, num_beams] + shape[1:])

    @staticmethod
    def _gather_beams(tensor: ms.Tensor, beam_indices: ms.Tensor) -> ms.Tensor:
        """
        Gathers the beam slices indexed by beam_indices into new beam array.
        Args:
            tensor (`ms.Tensor`): A tensor containing data to be gathered. The tensor is a 2D or a 3D tensor
                with the two first dimensions depicting the batch and the beam dimensions.
            beam_indices (`ms.Tensor` of shape `(batch_size, num_beams_to_select)`): The indices of the beams to
                select .
        Returns:
            A tensor with the selected beams
        """
        # `take_along_dim` requires its indices arg to have the same number of dims as `input`
        while len(beam_indices.shape) < len(tensor.shape):
            beam_indices = beam_indices.unsqueeze(-1)
        gathered_tensor = ms.Tensor(ms.numpy.take_along_axis(arr=tensor, indices=beam_indices, axis=1))
        return gathered_tensor

    @staticmethod
    def _beam_search_has_unfinished_sequences(
        running_beam_scores: ms.Tensor,
        beam_scores: ms.Tensor,
        is_sent_finished: ms.Tensor,
        next_token_hits_stopping_criteria: ms.Tensor,
        cur_len: int,
        max_length: int,
        decoder_prompt_len: int,
        early_stopping: Union[bool, str],
        length_penalty: float,
    ):
        """
        Beam Search stopping condition -- halts the generation loop if any of these conditions becomes False
        """
        # a. Can the open beams improve the top completed scores?
        # early_stopping == False -> apply heuristic = always get the best score from
        #   `cur_len - decoder_prompt_len`. See the discussion below for more details.
        #   https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        # early_stopping == "never" -> compute the best score from `max_length` or `cur_len`, depending on the
        #   sign of `length_penalty`. Positive `length_penalty` favors longer sequences, thus we use
        #   `max_length` there.
        if early_stopping == "never" and length_penalty > 0.0:
            best_hypothetical_length = max_length - decoder_prompt_len
        else:
            best_hypothetical_length = cur_len - decoder_prompt_len
        best_possible_running_score = running_beam_scores[:, :1] / (best_hypothetical_length**length_penalty)
        worst_finished_score = mint.where(is_sent_finished, mint.min(beam_scores, dim=1, keepdim=True)[0], -1.0e9)
        improvement_possible = mint.any(best_possible_running_score > worst_finished_score)

        # b. Is there still a beam without fully completed sequences? This is only relevant if early_stopping is
        # enabled, where we want to finish as soon as all beams have a completed sequence.
        exists_open_beam = ms.Tensor(
            ~(mint.all(is_sent_finished) & ms.Tensor(early_stopping is True, ms.int32)), ms.int32
        )

        # c. Have we hit a stopping criteria with all running sequences and have no way to continue? e.g. we have
        # reached `max_length``
        valid_continuations = ~mint.all(next_token_hits_stopping_criteria)

        return improvement_possible & exists_open_beam & valid_continuations

    def _get_top_k_continuations(
        self,
        accumulated_log_probs: ms.Tensor,
        running_sequences: ms.Tensor,
        running_beam_indices: ms.Tensor,
        cur_len: int,
        decoder_prompt_len: int,
        do_sample: bool,
        beams_to_keep: int,
        num_beams: int,
        vocab_size: int,
        batch_size: int,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """
        Get top-K continuations given the accumulated log probs on the next token.
        A few notes to understand what's going on:
        1. Each item in batch has `num_beams` * `vocab_size` candidate continuations. For each item, get the
        top K [K = (number of EOS tokens + 1) * `num_beams`] candidates with the highest accumulated
        log-probabilities, or sample them without replacement using the accumulated scores
        2. We gather the top K (as opposed to `num_beams`, or any number lower than K) here so that we have at
        least `num_beams` sequences remaining to continue the live beam search.
        3. Note that other stopping criteria might result in impossible to continue beams, i.e. all continuations
        selected in this step hit the stopping criteria.
        """
        # TODO (joao): This function should take an optional beam scorer function, to manipulate the scores after
        # token selection. The function should be an argument exposed, so that custom scoring functions can be
        # defined.

        # Gather the top K scores from _all_ beams.
        if do_sample:
            topk_indices = mint.multinomial(
                mint.nn.functional.softmax(accumulated_log_probs, dim=-1), num_samples=beams_to_keep
            )
            topk_log_probs = mint.gather(input=accumulated_log_probs, dim=1, index=topk_indices)
        else:
            topk_log_probs, topk_indices = mint.topk(accumulated_log_probs, k=beams_to_keep)

        # Gather K top beams, recover the beam index by floor division and token id by modulo division
        topk_current_beam_indices = topk_indices // vocab_size
        topk_running_beam_indices = self._gather_beams(running_beam_indices, topk_current_beam_indices)
        topk_running_sequences = self._gather_beams(running_sequences, topk_current_beam_indices)
        topk_ids = topk_indices % vocab_size

        # Update sequences for the K top-k new sequences.
        topk_running_sequences[:, :, cur_len] = topk_ids

        # we want to store the beam indices with batch information -> real beam index = beam index % num beams
        batch_offset = mint.arange(batch_size).view(-1, 1) * num_beams
        batch_modified_indices = topk_current_beam_indices + batch_offset
        topk_running_beam_indices[:, :, cur_len - decoder_prompt_len] = batch_modified_indices

        return topk_log_probs, topk_running_sequences, topk_running_beam_indices

    def _get_running_beams_for_next_iteration(
        self,
        topk_log_probs: ms.Tensor,
        topk_running_sequences: ms.Tensor,
        topk_running_beam_indices: ms.Tensor,
        next_token_hits_stopping_criteria: ms.Tensor,
        num_beams: int,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """
        Given the top-K continuations, their scores, and whether they hit a stopping criteria, select the
        best non-finished beams to continue beam search in the next iteration.
        """
        # To prevent these just finished sequences from being used in subsequent iterations, set their log probs
        # to a very large negative value
        topk_running_log_probs = topk_log_probs + next_token_hits_stopping_criteria.to(ms.float32) * -1.0e9

        next_topk_indices = mint.topk(topk_running_log_probs, k=num_beams)[1]
        running_sequences = self._gather_beams(topk_running_sequences, next_topk_indices)
        running_beam_scores = self._gather_beams(topk_running_log_probs, next_topk_indices)
        running_beam_indices = self._gather_beams(topk_running_beam_indices, next_topk_indices)
        return running_sequences, running_beam_scores, running_beam_indices

    def _update_finished_beams(
        self,
        sequences: ms.Tensor,
        topk_running_sequences: ms.Tensor,
        beam_scores: ms.Tensor,
        topk_log_probs: ms.Tensor,
        beam_indices: ms.Tensor,
        topk_running_beam_indices: ms.Tensor,
        is_sent_finished: ms.Tensor,
        next_token_hits_stopping_criteria: ms.Tensor,
        top_num_beam_mask: ms.Tensor,
        num_beams: int,
        cur_len: int,
        decoder_prompt_len: int,
        length_penalty: float,
        early_stopping: Union[bool, str],
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        """
        Updates the finished beams if (and only if) there are new completed sequences that have a higher score than
        the current finished sequences.
        """
        # Only the top `num_beam` sequences can be considered for the final returned sequences. Remember: the
        # remaining sequences only exist as a backup to ensure that we have at least `num_beams` sequences to
        # continue.
        did_top_num_beams_just_finished = next_token_hits_stopping_criteria & top_num_beam_mask[None, :].to(ms.int32)

        # Further process topk logits for the finished beams
        # - add length penalty
        topk_log_probs = topk_log_probs / ((cur_len + 1 - decoder_prompt_len) ** length_penalty)
        # - make sure no scores can be added anymore if beam is full and early stopping is on
        beams_in_batch_are_full = ops.all(is_sent_finished, axis=-1, keep_dims=True) & ms.Tensor(
            early_stopping is True, ms.int32
        )
        topk_log_probs += beams_in_batch_are_full.to(ms.float32) * -1.0e9
        # - make sure still running sequences cannot be chosen as finalized beam
        topk_log_probs += (~did_top_num_beams_just_finished) * -1.0e9

        # Get finalized  `num_beam` sequences for the next generation step -- combine the previous finalized
        # data with the new finalized sequences (if any, non-finalized sequences have a very large negative score
        # in this step), and keep the best `num_beams` sequences.
        merged_sequences = mint.cat((sequences, topk_running_sequences), dim=1)
        merged_scores = mint.cat((beam_scores, topk_log_probs), dim=1)
        merged_beam_indices = mint.cat((beam_indices, topk_running_beam_indices), dim=1)
        merged_is_sent_finished = mint.cat((is_sent_finished, did_top_num_beams_just_finished.to(ms.bool_)), dim=1)
        topk_merged_indices = mint.topk(merged_scores, k=num_beams)[1]
        sequences = self._gather_beams(merged_sequences, topk_merged_indices)
        beam_scores = self._gather_beams(merged_scores, topk_merged_indices)
        beam_indices = self._gather_beams(merged_beam_indices, topk_merged_indices)
        is_sent_finished = self._gather_beams(merged_is_sent_finished, topk_merged_indices)
        return sequences, beam_scores, beam_indices, is_sent_finished

    # end of auxiliary functions for beam search

    def _beam_search(
        self,
        input_ids: ms.Tensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, ms.Tensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
        If it's the first time you're diving into Beam Search, we recommend you read the following blog post:
        https://huggingface.co/blog/how-to-generate (especially the beam search section).
        You can recompute the sequence scores from the individual scores using the `compute_transition_scores` function
        (https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores)
        Parameters:
            input_ids (`ms.Tensor` of shape `(batch_size*num_beams, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.
        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `ms.Tensor`: A `ms.Tensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """

        # 1. init beam_search values
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample
        early_stopping = generation_config.early_stopping
        length_penalty = generation_config.length_penalty
        max_length = generation_config.max_length
        num_beams = generation_config.num_beams
        num_return_sequences = generation_config.num_return_sequences

        batch_size_unflattened, cur_len = input_ids.shape
        batch_size = batch_size_unflattened // num_beams
        # TODO (joao): standardize special cases
        if self.__class__.__name__ == "MoshiDepthDecoder":
            vocab_size = self.config.audio_vocab_size
        elif self.__class__.__name__ == "ImageGPTForCausalImageModeling":
            vocab_size = self.get_output_embeddings().out_features
        else:
            vocab_size = self.config.get_text_config().vocab_size
        decoder_prompt_len = cur_len
        this_peer_finished = False

        # At each beam search step, we want to keep top K [K = (number of EOS tokens + 1) * `num_beams`] candidates
        # with the highest log-probabilities, or sample K continuations without replacement. We gather the top K
        # (as opposed to `num_beams`, or any number lower than K) so that we have at least `num_beams` sequences
        # non-finished to continue the live beam search, in case the top `num_beams` all select an EOS token.
        n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
        beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
        top_num_beam_mask = mint.cat(
            (mint.ones((num_beams), dtype=ms.bool_), mint.zeros((beams_to_keep - num_beams), dtype=ms.bool_)),
            dim=0,
        )

        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # (joao) feature lost in the refactor. Probably won't implement, hurts readbility with minimal gains (there
        # are newer low-memory alternatives like the offloaded cache)
        sequential = generation_config.low_memory
        if sequential:
            raise ValueError(
                "`low_memory=True` is not supported after the beam search refactor. Please check the discussion in "
                "#35802 *after the PR got merged*, and add a comment there if your questions are not yet answered."
            )

        # 2. init output tuples
        all_scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # 3. init running tensors and static-shaped placeholders

        # per batch, beam-item holding current token in loop and completed sequences
        output_fill_value = pad_token_id or eos_token_id[0] if eos_token_id is not None else -1
        running_sequences = mint.full(
            (batch_size, num_beams, max_length),
            fill_value=output_fill_value,
            dtype=ms.int64,
        )
        running_sequences[:, :, :cur_len] = self._unflatten_beam_dim(input_ids, batch_size, num_beams)
        sequences = running_sequences.copy()  # .detach()

        # per batch, beam-item score, logprobs
        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        running_beam_scores = mint.zeros((batch_size, num_beams), dtype=ms.float32)
        running_beam_scores[:, 1:] = -1e9
        beam_scores = mint.full((batch_size, num_beams), fill_value=-1e9, dtype=ms.float32)

        # per batch, beam-item state bit indicating if sentence has finished.
        is_sent_finished = mint.zeros((batch_size, num_beams), dtype=ms.bool_)

        # per batch, beam-item state bit indicating if there are valid continuations.
        next_token_hits_stopping_criteria = mint.zeros((batch_size, num_beams), dtype=ms.bool_)

        # per batch selected beam indices
        running_beam_indices = mint.full((batch_size, num_beams, max_length - cur_len), fill_value=-1, dtype=ms.int32)
        beam_indices = running_beam_indices.copy()  # .detach()

        # 4. run the generation loop
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus):
            # a. Forward current tokens, obtain the logits
            flat_running_sequences = self._flatten_beam_dim(running_sequences[:, :, :cur_len])
            model_inputs = self.prepare_inputs_for_generation(flat_running_sequences, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            model_outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                model_outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            logits = model_outputs.logits[:, -1, :].copy().float()  # copy is needed to avoid keeping a hanging ref

            # b. Compute log probs -- get log probabilities from logits, process logits with processors (*e.g.*
            # `temperature`, ...), and add new logprobs to existing running logprobs scores.
            log_probs = mint.nn.functional.log_softmax(logits, dim=-1)
            log_probs = logits_processor(flat_running_sequences, log_probs)

            # Store logits, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (logits.copy(),)
                if return_dict_in_generate and output_scores:
                    all_scores += (log_probs.copy(),)

                if output_attentions:
                    decoder_attentions += (
                        (model_outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (model_outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (model_outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (model_outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (model_outputs.hidden_states,)
                    )

            # This is needed to properly delete logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del model_outputs

            log_probs = self._unflatten_beam_dim(log_probs, batch_size, num_beams)
            log_probs = log_probs + running_beam_scores[:, :, None]
            log_probs = mint.reshape(log_probs, (batch_size, num_beams * vocab_size))

            # c. Retrieve top-K continuations, i.e. select the next token (greedy or sampling) and then keep the best
            # continuations among all beams based on the accumulated scores.
            topk_log_probs, topk_running_sequences, topk_running_beam_indices = self._get_top_k_continuations(
                accumulated_log_probs=log_probs,
                running_sequences=running_sequences,
                running_beam_indices=running_beam_indices,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                do_sample=do_sample,
                beams_to_keep=beams_to_keep,
                num_beams=num_beams,
                vocab_size=vocab_size,
                batch_size=batch_size,
            )

            # d. Check which running sequences have finished
            next_token_hits_stopping_criteria = stopping_criteria(
                self._flatten_beam_dim(topk_running_sequences[:, :, : cur_len + 1]),  # remove unfilled token indexes
                all_scores,
            )
            next_token_hits_stopping_criteria = self._unflatten_beam_dim(
                next_token_hits_stopping_criteria, batch_size, beams_to_keep
            )

            # e. Get the non-finished running `num_beams` sequences for the next generation step
            running_sequences, running_beam_scores, running_beam_indices = self._get_running_beams_for_next_iteration(
                topk_log_probs=topk_log_probs,
                topk_running_sequences=topk_running_sequences,
                topk_running_beam_indices=topk_running_beam_indices,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                num_beams=num_beams,
            )

            # f. Update the completed beams if a new high score in a finished sequence is found
            sequences, beam_scores, beam_indices, is_sent_finished = self._update_finished_beams(
                sequences=sequences,
                topk_running_sequences=topk_running_sequences,
                beam_scores=beam_scores,
                topk_log_probs=topk_log_probs,
                beam_indices=beam_indices,
                topk_running_beam_indices=topk_running_beam_indices,
                is_sent_finished=is_sent_finished,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                top_num_beam_mask=top_num_beam_mask,
                num_beams=num_beams,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

            # g. Prepare remaining data for the next iteration, including computing the stopping condition for
            # beam search as a whole (as opposed to individual beams, i.e. `stopping_criteria`)

            # pluck the cache from the beam indices that will be used in the next iteration
            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    past_key_values=model_kwargs["past_key_values"],
                    beam_idx=self._flatten_beam_dim(running_beam_indices[..., cur_len - decoder_prompt_len]),
                )

            cur_len = cur_len + 1
            this_peer_finished = not self._beam_search_has_unfinished_sequences(
                running_beam_scores,
                beam_scores,
                is_sent_finished,
                next_token_hits_stopping_criteria,
                cur_len,
                max_length,
                decoder_prompt_len,
                early_stopping,
                length_penalty,
            )

        # 5. prepare outputs
        # Take best beams for each batch (the score is sorted in descending order)
        sequences = self._flatten_beam_dim(sequences[:, :num_return_sequences, :])
        beam_scores = self._flatten_beam_dim(beam_scores[:, :num_return_sequences])
        beam_indices = self._flatten_beam_dim(beam_indices[:, :num_return_sequences, :])

        # Crop the static-shaped tensors to the actual size
        sequences = sequences[:, :cur_len]
        beam_indices = beam_indices[:, : cur_len - decoder_prompt_len]

        if return_dict_in_generate:
            if not output_scores:
                beam_scores = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequences,
                    sequences_scores=beam_scores,
                    scores=all_scores,
                    logits=raw_logits,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequences,
                    sequences_scores=beam_scores,
                    scores=all_scores,
                    logits=raw_logits,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequences
