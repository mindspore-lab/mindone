import os
from collections import OrderedDict

# from transformers.modeling_outputs import CausalLMOutputWithPast
# cong TODO: move this part to a suitable file, mindformers doesn't have this class
from dataclasses import dataclass
from typing import Any, ContextManager, Iterable, List, Optional, Tuple, Union

# from mindformers import LlamaConfig
from share4v.transformers.models.llama import LlamaForCausalLM, LlamaModel

import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import CrossEntropyLoss

# from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
#                           LlamaForCausalLM, LlamaModel)
# from mindformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
#                           LlamaForCausalLM, LlamaModel)


os.sys.path.append("/Users/congwang/Documents/M/ms_ShareGPT4V")
from share4v.pipeline import TextGenerator
from share4v.transformers.models.cache import Cache, DynamicCache

from ..share4v_arch import Share4VMetaForCausalLM, Share4VMetaModel

# @dataclass
# class CausalLMOutputWithPast(OrderedDict):
#     """
#     Base class for causal language model (or autoregressive) outputs.

#     Args:
#         loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
#             Language modeling loss (for next-token prediction).
#         logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
#             Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#         past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#             Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
#             `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

#             Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
#             `past_key_values` input) to speed up sequential decoding.
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
#             one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def __delitem__(self, *args, **kwargs):
#         raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

#     def setdefault(self, *args, **kwargs):
#         raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

#     def pop(self, *args, **kwargs):
#         raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

#     def update(self, *args, **kwargs):
#         raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

#     def __getitem__(self, k):
#         if isinstance(k, str):
#             inner_dict = dict(self.items())
#             return inner_dict[k]
#         else:
#             return self.to_tuple()[k]

#     def __setattr__(self, name, value):
#         if name in self.keys() and value is not None:
#             # Don't call self.__setitem__ to avoid recursion errors
#             super().__setitem__(name, value)
#         super().__setattr__(name, value)

#     def __setitem__(self, key, value):
#         # Will raise a KeyException if needed
#         super().__setitem__(key, value)
#         # Don't call self.__setattr__ to avoid recursion errors
#         super().__setattr__(key, value)

#     # def __reduce__(self):
#     #     if not is_dataclass(self):
#     #         return super().__reduce__()
#     #     callable, _args, *remaining = super().__reduce__()
#     #     args = tuple(getattr(self, field.name) for field in fields(self))
#         # return callable, args, *remaining

#     def to_tuple(self) -> Tuple[Any]:
#         """
#         Convert self to a tuple containing all the attributes/keys that are not `None`.
#         """
#         return tuple(self[k] for k in self.keys())

#     loss: Optional[float] = None
#     logits: float = None
#     past_key_values: Optional[Tuple[Tuple[float]]] = None
#     hidden_states: Optional[Tuple[float, ...]] = None
#     attentions: Optional[Tuple[float, ...]] = None
#     # loss: Optional[ms.float32] = None
#     # logits: ms.float32 = None
#     # past_key_values: Optional[Tuple[Tuple[ms.float32]]] = None
#     # hidden_states: Optional[Tuple[ms.float32, ...]] = None
#     # attentions: Optional[Tuple[ms.float32, ...]] = None


# this part used for mindformers.LlamaModel
# class Share4VConfig(LlamaConfig):
#     model_type = "share4v"
#     def __init__(self, **kwargs):
#         self.mm_vision_tower_path = kwargs.pop("mm_vision_tower_path", None)
#         # self.use_return_dict = kwargs.pop("use_return_dict", None)
#         # self.seq_length = kwargs.pop("seq_length", None)
#         super(Share4VConfig, self).__init__(**kwargs)


class Share4VLlamaModel(Share4VMetaModel, LlamaModel):
    # config_class = Share4VConfig

    # this func used for self-defined LlamaModel
    def __init__(self, config):
        super(Share4VLlamaModel, self).__init__(config)
        self.dtype = config.get("dtype")
        self.set_dtype(self.dtype)

    # this func used for mindformers.LlamaModel
    # def __init__(self, config: LlamaConfig):
    #     super(Share4VLlamaModel, self).__init__(config)

    # this func used for mindformers.LlamaModel
    # def embed_tokens(self, input_ids):
    #     if type(input_ids) is not ms.Tensor:
    #         print('Share4VLlamaModel.embed_tokens')
    #         print(type(input_ids), input_ids.shape, input_ids)
    #         # np.ndarray to ms.Tensor
    #         input_ids = ms.Tensor(input_ids, dtype=ms.int64)
    #     return self.tok_embeddings(input_ids)
    def set_train(self, mode: bool):
        """Set the model(llama model, vision_tower and mm_projector) to training mode or evaluation mode."""
        # cong TODO: edit alert message
        for param in self.get_parameters():
            param.requires_grad = mode

        vision_tower = self.get_vision_tower()
        if not vision_tower.is_loaded:
            print(f"set train to {mode}, but vision_tower is not load")

        # mm_projector = self.get_mm_projector()
        # if mm_projector is not None:
        #     for cell in mm_projector:
        #         for param in cell.get_parameters():
        #             param.requires_grad = mode
        # else:
        #     print("mm_projector is None")
        return self

    def set_dtype(self, dtype):
        """Set the model(llama model, vision_tower and mm_projector) to target data type"""
        self.dtype = dtype
        for param in self.get_parameters():
            param.set_dtype(self.dtype)


# class Share4VLlamaForCausalLM(LlamaForCausalLM, Share4VMetaForCausalLM, TextGenerator):
class Share4VLlamaForCausalLM(LlamaForCausalLM, Share4VMetaForCausalLM):
    # config_class = Share4VConfig

    def __init__(self, config):
        # print("Share4VLlamaForCausalLM init")

        self.vocab_size = int(config.get("vocab_size")) if config.get("vocab_size") else 32000
        config["vocab_size"] = self.vocab_size
        config["dtype"] = ms.float32 if config.get("vocab_size") == "float32" else ms.float16
        self.config = config

        super(LlamaForCausalLM, self).__init__(config)
        self.model = Share4VLlamaModel(config)
        # LlamaForCausalLM.construct()

        # self.model = LlamaModel(
        #     hidden_size=hidden_size,
        #     intermediate_size=intermediate_size,
        #     max_position_embeddings=max_position_embeddings,
        #     num_attention_heads=num_attention_heads,
        #     num_hidden_layers=num_hidden_layers,
        #     num_key_value_heads=num_key_value_heads,
        #     rms_norm_eps=rms_norm_eps,
        #     rope_theta=rope_theta,
        #     vocab_size=vocab_size,
        #     attention_dropout=attention_dropout,
        #     hidden_act=hidden_act,
        #     pad_token_id=pad_token_id,
        #     past_key_value_cache=past_key_value_cache,
        # )
        # print(type(self.model))
        # super(Share4VMetaForCausalLM, self).__init__(self.model)
        # super(TextGenerator, self).__init__(self.model)
        self.lm_head = nn.Dense(config["hidden_size"], config["vocab_size"], has_bias=False)
        # self.lm_head = nn.Dense(
        #     config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_model(self):
        return self.model

    # this func is used for self-defined LlamaModel
    def load_model(self, model_path, **kwargs):
        # cong TODO:  fix input args model_path to load from config.json
        # import json
        # with open(os.path.join(model_path, "config.json"), "r") as f:
        #     config = json.load(f)
        ms_source_data = ms.load_checkpoint(model_path)
        # due to version update reason, need to align the key name with latest version
        if "model.embed_tokens.embedding_table" in ms_source_data.keys():
            ms_source_data["model.embed_tokens.weight"] = ms_source_data["model.embed_tokens.embedding_table"]
        params_not_load = ms.load_param_into_net(self, ms_source_data, strict_load=False)
        print(f"Params not loaded: {params_not_load}")

    def set_train(self, mode: bool):
        self.get_model().set_train(mode)

        return self
        # # cong TODO: edit alert message

        # model = self.get_model()
        # if model is not None:
        #     for param in model.get_parameters():
        #         param.requires_grad = mode
        # else:
        #     print("model is None")

        # vision_tower = self.get_vision_tower()
        # if not vision_tower.is_loaded:
        #     vision_tower.set_train(mode)
        # else:
        #     print("vision_tower is not load")

        # mm_projector = self.get_mm_projector()
        # if mm_projector is not None:
        #     for cell in mm_projector:
        #         for param in cell.get_parameters():
        #             param.requires_grad = mode
        # else:
        #     print("mm_projector is None")

    def set_dtype(self, dtype):
        self.dtype = dtype
        self.lm_head.weight.set_dtype(dtype)
        self.get_model().set_dtype(dtype)
        # self.dtype = dtype
        # for param in self.get_model().get_parameters():
        #     param.set_dtype(self.dtype)

    def construct(
        self,
        # cong TODO: modify this area
        #     input_ids: ms.int64 = None,
        #     attention_mask: Optional[ms.Tensor] = None,
        #     past_key_values: Optional[List[ms.float32]] = None,
        #     inputs_embeds: Optional[ms.float32] = None,
        #     labels: Optional[ms.int64] = None,
        #     use_cache: Optional[bool] = None,
        #     output_attentions: Optional[bool] = None,
        #     output_hidden_states: Optional[bool] = None,
        #     images: Optional[ms.float32] = None,
        #     return_dict: Optional[bool] = None,
        # ) -> Union[Tuple, CausalLMOutputWithPast]:
        input_ids=None,
        attention_mask=None,
        # past_key_values = None,
        inputs_embeds=None,
        labels=None,
        # use_cache = None,
        past_key_cache_list=None,
        past_value_cache_list=None,
        images=None,
        # return_dict = None,
        return_key_value_cache: bool = False,
    ) -> Tuple:
        # output_attentions = output_attentions if output_attentions is not None else self.config['output_attentions']
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config['output_hidden_states']
        # )
        # return_dict = return_dict if return_dict is not None else self.config['use_return_dict']

        (
            input_ids,
            attention_mask,
            past_key_cache_list,
            past_value_cache_list,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_cache_list, past_value_cache_list, labels, images
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # outputs = self.model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )
        # print("input_ids", input_ids)
        # if attention_mask is not None:
        #     print("attention_mask", type(attention_mask), attention_mask.shape)
        # if inputs_embeds is not None:
        #     print("inputs_embeds", type(inputs_embeds), inputs_embeds.shape)

        # this part for self-defined LlamaModel
        # transformers.LlamaModel outputs:
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,

        # 0625 self-defined LlamaModel outputs:
        #     (hidden_states, key_cache_list, value_cache_list)
        # cong TODO: edit inputs for llamamodel, other args:
        #  position_ids
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            # use_cache=use_cache,
            past_key_cache_list=past_key_cache_list,
            past_value_cache_list=past_value_cache_list,
            return_key_value_cache=return_key_value_cache,
        )

        # cong TODO: for 0625 version, llama model return hidden_states, key_cache_list, value_cache_list
        hidden_states = outputs[0]
        # cong TODO: .to(ms.float32)?
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config["vocab_size"])
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            loss = loss_fct(shift_logits, shift_labels)

        # print('Share4VLlamaForCausalLM.construct return tuple...')
        # don't have hidden_states, attentions for current self-defined LlamaModel
        # original results: loss, logits, past_key_values, hidden_states, attentions
        # now results: loss, logits, key_cache_list, value_cache_list
        results = (loss, logits, outputs[1], outputs[2])
        return results

        # cong TODO: modify this area
        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    def prepare_inputs_for_generation(
        # self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        self,
        input_ids,
        **kwargs,
    ):
        # mindformers.LlamaForCausalLM.prepare_inputs_for_generation only takes (input_ids, **kwarg)

        # cong TODO: what is this doing?
        if kwargs.get("return_key_value_cache"):
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if kwargs.get("inputs_embeds") is not None and not kwargs.get("return_key_value_cache"):
            model_inputs = {"inputs_embeds": kwargs.get("inputs_embeds")}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                # "past_key_values": past_key_values if past_key_values is not None else kwargs.get("past_key_values"),
                "return_key_value_cache": kwargs.get("return_key_value_cache", None),
                "past_key_cache_list": kwargs.get("past_key_cache_list", None),
                "past_value_cache_list": kwargs.get("past_value_cache_list", None),
                "attention_mask": kwargs.get("attention_mask", None),
                "images": kwargs.get("images", None),
            }
        )
        # print("model_inputs.get('input_ids')", model_inputs.get('input_ids').shape if model_inputs.get('input_ids') is not None else None)
        # print("model_inputs.get('inputs_embeds')", model_inputs.get('inputs_embeds').shape if model_inputs.get('inputs_embeds') is not None  else None)
        # print("model_inputs.get('images')", model_inputs.get('images').shape if model_inputs.get('images') is not None  else None)
        # print("model_inputs.get('past_key_cache_list')", len(model_inputs.get('past_key_cache_list')) if model_inputs.get('past_key_cache_list') is not None else None)
        # print("model_inputs.get('past_value_cache_list')", len(model_inputs.get('past_value_cache_list')) if model_inputs.get('past_value_cache_list') is not None else None)
        # print("model_inputs.get('return_key_value_cache')", model_inputs.get('return_key_value_cache'))

        # print("model_inputs", model_inputs)
        return model_inputs

    # def init_weights(self, module=None):
    #     """Initialize the weights"""
    #     pass


# AutoConfig.register("share4v", Share4VConfig)
# AutoModelForCausalLM.register(Share4VConfig, Share4VLlamaForCausalLM)
