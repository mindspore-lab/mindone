import copy
import inspect
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import logging
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.tokenization_utils import ExtensionsTrie
from transformers.utils.generic import ModelOutput

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops

from mindone.transformers.cache_utils import Cache, get_seq_length, init_static_cache, reset
from mindone.transformers.generation.logits_process import (
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from mindone.transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
)
from mindone.transformers.modeling_outputs import CausalLMOutputWithPast

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from mindone.transformers.modeling_utils import MSPreTrainedModel as PreTrainedModel

logger = logging.get_logger(__name__)


NEED_SETUP_CACHE_CLASSES_MAPPING = {}
QUANT_BACKEND_CLASSES_MAPPING = {}


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


class GenerationMixin:
    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`."
        )

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs.
        """

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # three conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same);
            # 3) the user must have set generation parameters in the model config.
            if (
                self.generation_config._from_model_config
                and self.generation_config._original_object_hash == hash(self.generation_config)
                and self.config._has_non_default_generation_parameters()
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled.
        strict_kwargs = kwargs.pop("strict_kwargs", False)
        if strict_kwargs:
            model_kwargs = kwargs
            generate_attributes_in_kwargs = [
                key for key, value in kwargs.items() if getattr(generation_config, key, None) != value
            ]
            if len(generate_attributes_in_kwargs) > 0:
                raise ValueError(
                    "exception: all generation configuration attributes must be passed within a "
                    f"`generation_config` instance passed to `generate` (found: {generate_attributes_in_kwargs})."
                )
        else:
            generation_config = copy.deepcopy(generation_config)
            model_kwargs = generation_config.update(**kwargs)

        return generation_config, model_kwargs

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
        inputs: ms.Tensor,
        pad_token_id: Optional[ms.Tensor],
        eos_token_id: Optional[ms.Tensor],
    ) -> ms.Tensor:
        # No information for attention mask inference -> return default attention mask
        default_attention_mask = ops.ones(inputs.shape[:2], dtype=ms.int32)
        if pad_token_id is None:
            return default_attention_mask

        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [ms.int32, ms.int64]
        if not is_input_ids:
            return default_attention_mask

        is_pad_token_in_inputs = (pad_token_id is not None) and (
            mnp.isin(element=inputs, test_elements=pad_token_id).any()
        )
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or ~(
            mnp.isin(element=eos_token_id, test_elements=pad_token_id).any()
        )
        can_infer_attention_mask = is_pad_token_in_inputs * is_pad_token_not_equal_to_eos_token_id
        attention_mask_from_padding = inputs.ne(pad_token_id).to(ms.int32)

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
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
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

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        if not model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = None
            return model_kwargs

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            if isinstance(cache, Tuple):
                past_length = get_seq_length(cache)

        if model_kwargs.get("attention_mask", None) is not None:
            attention_mask = model_kwargs["attention_mask"]
            if "inputs_embeds" in model_kwargs:
                max_len = model_kwargs["inputs_embeds"].shape[1]
            else:
                max_len = input_ids.shape[-1]
            cur_len = int(attention_mask.sum(-1).max())

            cache_position = ops.arange(past_length, cur_len, dtype=ms.int32)
            if (cur_len - past_length) < max_len:
                cache_position = ops.cat([cache_position, ops.zeros(max_len - (cur_len - past_length), ms.int32)])
        else:
            if "inputs_embeds" in model_kwargs:
                cur_len = model_kwargs["inputs_embeds"].shape[1]
            else:
                cur_len = input_ids.shape[-1]

            cache_position = ops.arange(past_length, cur_len, ms.int32)

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    def _get_cache(
        self, cache_implementation: str, max_batch_size: int, max_cache_len: int
    ) -> Tuple[Tuple[ms.Tensor, ms.Tensor]]:
        """
        Sets a cache for `generate`, that will persist across calls. A new cache will only be initialized a
        new `generate` call requires a larger cache.

        Returns the resulting cache object.
        """
        if cache_implementation == "sliding_window":
            max_cache_len = min(self.config.sliding_window, max_cache_len)

        need_new_cache = (
            not hasattr(self, "_cache")
            or (not isinstance(self._cache, tuple))
            or (not isinstance(self._cache[0][0], ms.Tensor))
            or self._cache[0][0].shape[0] != max_batch_size
            or self._cache[0][0].shape[2] < max_cache_len
        )

        if need_new_cache:
            if hasattr(self.config, "_pre_quantization_dtype"):
                cache_dtype = self.config._pre_quantization_dtype
            else:
                cache_dtype = self.dtype

            self._cache = init_static_cache(
                config=self.config,
                max_batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                dtype=cache_dtype,
            )
        else:
            self._cache = reset(self._cache)

        return self._cache

    def _supports_default_dynamic_cache(self) -> bool:
        """
        Return `True` if current model can use a `DynamicCache` instance when initializing the `past_key_values`.
        This is mostly the same as `_supports_cache_class` attribute, but add exception for `Jamba` model which
        uses its own `HybridMambaAttentionDynamicCache` and do not need to initialize the Cache in advance in
        order to save memory (because no back and forth `to_legacy_cache` and `from_legacy_cache` will be performed
        for `HybridMambaAttentionDynamicCache`).
        """
        return self._supports_cache_class and "jamba" not in self.__class__.__name__.lower()

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

        # Convert special tokens to tensors (if they exist either in kwargs or in self.config)
        def _tensor_or_none(token_kwargs, token_self):
            token = token_kwargs if token_kwargs is not None else token_self
            if token is None or isinstance(token, ms.Tensor):
                return token
            return ms.Tensor(token, dtype=ms.int32)

        bos_token_id = _tensor_or_none(generation_config.bos_token_id, self.generation_config.bos_token_id)
        eos_token_id = _tensor_or_none(generation_config.eos_token_id, self.generation_config.eos_token_id)
        pad_token_id = _tensor_or_none(generation_config.pad_token_id, self.generation_config.pad_token_id)
        decoder_start_token_id = _tensor_or_none(
            generation_config.decoder_start_token_id, self.generation_config.decoder_start_token_id
        )

        # for BC we also try to get `decoder_start_token_id` or `bos_token_id` (#30892)
        if self.config.is_encoder_decoder:
            decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else bos_token_id

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_id is not None and eos_token_id.ndim == 0:
            eos_token_id = eos_token_id.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_id is None and eos_token_id is not None:
            if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            pad_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_id} for open-end generation.")

        # we can't infer attn mask if pad token is set to be eos token in model's generation config
        if eos_token_id is not None and mnp.isin(element=eos_token_id, test_elements=pad_token_id).any():
            if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                logger.warning_once(
                    "The attention mask is not set and cannot be inferred from input because pad token is same as eos token."
                    "As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` "
                    "to obtain reliable results."
                )

        # Sanity checks/warnings
        if self.config.is_encoder_decoder and decoder_start_token_id is None:
            raise ValueError(
                "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
            )
        if eos_token_id is not None and (ops.is_floating_point(eos_token_id) or (eos_token_id < 0).any()):
            logger.warning(
                f"`eos_token_id` should consist of positive integers, but is {eos_token_id}. Your generation will not "
                "stop until the maximum length is reached. Depending on other flags, it may even crash."
            )

        # Update generation config with the updated special tokens tensors
        generation_config.bos_token_id = bos_token_id
        generation_config.eos_token_id = eos_token_id
        generation_config.pad_token_id = pad_token_id
        generation_config.decoder_start_token_id = decoder_start_token_id

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
                encoder_model_args = set(inspect.signature(encoder.forward).parameters)
                model_args |= encoder_model_args

            # allow decoder kwargs
            decoder = getattr(self, "decoder", None)
            if decoder is None and base_model is not None:
                decoder = getattr(base_model, "decoder", None)

            if decoder is not None:
                decoder_model_args = set(inspect.signature(decoder.forward).parameters)
                model_args |= {f"decoder_{x}" for x in decoder_model_args}

            # allow assistant_encoder_outputs to be passed if we're doing assisted generating
            if "assistant_encoder_outputs" in model_kwargs:
                model_args |= {"assistant_encoder_outputs"}

        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def _validate_assistant(self, assistant_model):
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

        if not self.config.vocab_size == assistant_model.config.vocab_size:
            raise ValueError("Make sure the main and assistant model use the same tokenizer")

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
        """Prepared max and min length in generaion configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )

            logger.warning(
                "Unlike the original transformers, `input_ids` will pad with inputs prompt token "
                "and `max_new_tokens` contains the length of the input, "
                "please set bigger `max_new_tokens` while considering the input length !"
            )

            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

            if generation_config.max_length < inputs_tensor.shape[1]:
                raise ValueError(
                    f"max_new_tokens `{generation_config.max_new_tokens}` is smaller than "
                    f"input length `{inputs_tensor.shape[1]}`."
                )

        # if both `inputs_embeds` and `input_ids` are passed, we do not correct the length
        # otherwise we need total length [inputs-embeds-len + new-tokens-len] to not go beyond indicated `max_length``
        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.max_length -= inputs_tensor.shape[1]

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
            # if the prompt is a single (non-pad) token, regenerate from bos
            if len(batch_ids[batch_ids != pad_token_id]) == 1:
                trimmed_ids[-1] = bos_token_id

            input_ids[batch_idx] = self.generate(trimmed_ids.unsqueeze(0), generation_config=generation_config)

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = ops.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], axis=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]

                cur_lens = attention_mask.sum(-1)
                for batch_idx in range(attention_mask.shape[0]):
                    cur_len = int(cur_lens[batch_idx])
                    if cur_len < attention_mask.shape[-1]:
                        attention_mask[batch_idx, cur_len] = 1
                    else:
                        attention_mask[batch_idx, :-1] = attention_mask[batch_idx, 1:]
                        attention_mask[batch_idx, -1:] = 1
                model_kwargs["attention_mask"] = attention_mask

                # model_kwargs["attention_mask"] = ops.cat(
                #     [attention_mask, ops.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)], axis=-1
                # )
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

        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            if (
                model_kwargs.get("attention_mask", None) is not None
                and model_kwargs["attention_mask"].shape[-1] == model_kwargs["cache_position"].shape[0]
            ):
                # `cache_position` obtain effective length after 1st step
                cur_idx = int(model_kwargs["attention_mask"].sum(-1).max()) - 1
                past_idx = cur_idx - 1
                model_kwargs["cache_position"] = (
                    model_kwargs["cache_position"][past_idx : past_idx + 1] + num_new_tokens
                )
            else:
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        return model_kwargs

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: ms.Tensor,
        prefix_allowed_tokens_fn: Callable[[int, ms.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
        device: str = None,
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
            raise NotImplementedError
        if generation_config.sequence_bias is not None:
            raise NotImplementedError

        if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
            raise NotImplementedError
        if (
            generation_config.encoder_repetition_penalty is not None
            and generation_config.encoder_repetition_penalty != 1.0
        ):
            raise NotImplementedError
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            raise NotImplementedError
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            raise NotImplementedError
        if (
            generation_config.encoder_no_repeat_ngram_size is not None
            and generation_config.encoder_no_repeat_ngram_size > 0
        ):
            raise NotImplementedError
        if generation_config.bad_words_ids is not None:
            raise NotImplementedError
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > 0
        ):
            processors.append(
                MinLengthLogitsProcessor(
                    generation_config.min_length,
                    generation_config.eos_token_id,
                )
            )
        if (
            generation_config.min_new_tokens is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_new_tokens > 0
        ):
            processors.append(
                MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length,
                    generation_config.min_new_tokens,
                    generation_config.eos_token_id,
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
            raise NotImplementedError
        if generation_config.forced_eos_token_id is not None:
            raise NotImplementedError
        if generation_config.remove_invalid_values is True:
            raise NotImplementedError
        if generation_config.exponential_decay_length_penalty is not None:
            raise NotImplementedError
        if generation_config.suppress_tokens is not None:
            raise NotImplementedError
        if generation_config.begin_suppress_tokens is not None:
            raise NotImplementedError
        if generation_config.forced_decoder_ids is not None:
            raise NotImplementedError
        if generation_config.watermarking_config is not None:
            raise NotImplementedError

        processors = self._merge_criteria_processor_list(processors, logits_processor)
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
        if generation_config.eos_token_id is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config.eos_token_id)) + 1)
        if generation_config.num_beams > 1:
            if isinstance(generation_config.eos_token_id, list):
                min_tokens_to_keep = len(generation_config.eos_token_id) + 1
            elif isinstance(generation_config.eos_token_id, ms.Tensor):
                min_tokens_to_keep = generation_config.eos_token_id.shape[0] + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.min_p is not None:
            # Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
            # warpers.append(MinPLogitsWarper(min_p=generation_config.min_p, min_tokens_to_keep=min_tokens_to_keep))
            raise NotImplementedError
        if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
            # warpers.append(
            #     TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
            # )
            raise NotImplementedError
        if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
            # warpers.append(
            #     EpsilonLogitsWarper(epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            # )
            raise NotImplementedError
        if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
            # warpers.append(
            #     EtaLogitsWarper(
            #         epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep
            #     )
            # )
            raise NotImplementedError
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers

    def _merge_criteria_processor_list(
        self,
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `.generate()` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[ms.Tensor] = None,
        **model_kwargs,
    ) -> Tuple[ms.Tensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

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
        **kwargs,
    ) -> Union[Tuple, ms.Tensor]:
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
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model)

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
                generation_config.pad_token_id is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and ops.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
            if not generation_config.use_cache:
                logger.warning("force `use_cache=True` when decoder-only and model_input_name is `inputs_embeds`.")
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

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
                decoder_start_token_id=generation_config.decoder_start_token_id,
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

        use_dynamic_cache_by_default = False
        if generation_config.cache_implementation is not None and model_kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Passing both `cache_implementation` (used to initialize certain caches) and `past_key_values` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs["past_key_values"] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    generation_config.max_length,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue."
                    )

                raise NotImplementedError("Not support Quantized Cache")
        # Use DynamicCache instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            raise NotImplementedError

        # Use static tuple cache by default.
        elif (
            generation_config.cache_implementation is None
            and not self._supports_default_dynamic_cache()
            and model_kwargs.get("use_cache", False)
        ):
            past = model_kwargs.get("past_key_values", None)
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
                model_kwargs["past_key_values"] = init_static_cache(
                    config=self.config,
                    max_batch_size=max_batch_size,
                    max_cache_len=max_cache_len,
                    dtype=cache_dtype,
                )
            else:
                model_kwargs["past_key_values"] = reset(past)

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        # 8. prepare distribution pre_processing samplers
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

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            raise NotImplementedError
        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            raise NotImplementedError
        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
            prepared_logits_warper = self._get_logits_warper(generation_config) if generation_config.do_sample else None

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
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
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

            # 14. unlike the original transformers, need delete the length of the input
            result = result[:, inputs_tensor.shape[1] :]

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            raise NotImplementedError
        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            raise NotImplementedError
        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            raise NotImplementedError
        else:
            raise NotImplementedError

        # Convert to legacy cache if needed
        if use_dynamic_cache_by_default and generation_config.return_legacy_cache:
            # if isinstance(result, ModelOutput) and hasattr(result, "past_key_values"):
            #     if isinstance(result.past_key_values, DynamicCache):
            #         result.past_key_values = result.past_key_values.to_legacy_cache()
            raise NotImplementedError

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

    def _sample(
        self,
        input_ids: ms.Tensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList] = None,
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
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
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
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

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

        # Padding inputs to avoid dynamic shape on MindSpore 2.3.1
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
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = ops.ones(batch_size, dtype=ms.int32)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        step = 0
        s_time = time.time()

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=False if ms.get_context("mode") == ms.GRAPH_MODE else True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            print(
                f"======> sampling, step: {step}, sample outputs shape: "
                f"{[o.shape for o in outputs if isinstance(o, ms.Tensor)]}, time cost: {time.time() - s_time:.3f}s"
            )
            s_time = time.time()
            step += 1

            if not isinstance(outputs, CausalLMOutputWithPast):
                outputs = CausalLMOutputWithPast(
                    loss=None,
                    logits=outputs[0],
                    past_key_values=outputs[1] if model_inputs.get("use_cache", False) else None,
                )

            if model_kwargs.get("attention_mask", None) is not None:
                attention_mask = model_kwargs["attention_mask"]
                cur_idx = int(attention_mask.sum(-1).max()) - 1

                if outputs.logits.shape[1] == attention_mask.shape[-1]:
                    next_token_logits = outputs.logits[:, cur_idx, :]  # (bs, seq, dim)
                else:
                    next_token_logits = outputs.logits[:, -1, :]

                # `input_ids` obtain effective length after 1st step
                if input_ids.shape[1] == attention_mask.shape[1]:
                    input_ids = input_ids[:, : cur_idx + 1]

            else:
                next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

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
                next_tokens = ops.multinomial(probs, num_samples=1).squeeze(1)
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

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~ms.Tensor(stopping_criteria(input_ids, scores), ms.bool_)
            this_peer_finished = unfinished_sequences.max() == 0

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
