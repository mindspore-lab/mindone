from typing import Tuple

from share4v.transformers.models.llama import LlamaForCausalLM, LlamaModel

import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import CrossEntropyLoss

from ..share4v_arch import Share4VMetaForCausalLM, Share4VMetaModel


class Share4VLlamaModel(Share4VMetaModel, LlamaModel):
    def __init__(self, config):
        super(Share4VLlamaModel, self).__init__(config)
        self.dtype = config.get("dtype")
        self.set_dtype(self.dtype)

    def set_train(self, mode: bool):
        """Set the model(llama model, vision_tower and mm_projector) to training mode or evaluation mode."""
        # cong TODO: edit alert message
        for param in self.get_parameters():
            param.requires_grad = mode

        vision_tower = self.get_vision_tower()
        if not vision_tower.is_loaded:
            print(f"set train to {mode}, but vision_tower is not load")
        return self

    def set_dtype(self, dtype):
        """Set the model(llama model, vision_tower and mm_projector) to target data type"""
        self.dtype = dtype
        for param in self.get_parameters():
            param.set_dtype(self.dtype)


class Share4VLlamaForCausalLM(LlamaForCausalLM, Share4VMetaForCausalLM):
    def __init__(self, config):
        self.vocab_size = int(config.get("vocab_size")) if config.get("vocab_size") else 32000
        config["vocab_size"] = self.vocab_size
        config["dtype"] = ms.float32 if config.get("vocab_size") == "float32" else ms.float16
        self.config = config

        super(LlamaForCausalLM, self).__init__(config)
        self.model = Share4VLlamaModel(config)
        self.lm_head = nn.Dense(config["hidden_size"], config["vocab_size"], has_bias=False)

    def get_model(self):
        return self.model

    # this func is used for self-defined LlamaModel
    def load_model(self, model_path, **kwargs):
        ms_source_data = ms.load_checkpoint(model_path)
        # due to version update reason, need to align the key name with latest version
        if "model.embed_tokens.embedding_table" in ms_source_data.keys():
            ms_source_data["model.embed_tokens.weight"] = ms_source_data["model.embed_tokens.embedding_table"]
        params_not_load = ms.load_param_into_net(self, ms_source_data, strict_load=False)
        print(f"Params not loaded: {params_not_load}")

    def set_train(self, mode: bool):
        self.get_model().set_train(mode)

        return self

    def set_dtype(self, dtype):
        self.dtype = dtype
        self.lm_head.weight.set_dtype(dtype)
        self.get_model().set_dtype(dtype)

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        past_key_cache_list=None,
        past_value_cache_list=None,
        images=None,
        return_key_value_cache: bool = False,
    ) -> Tuple:
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

        hidden_states = outputs[0]
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

        # results: loss, logits, key_cache_list, value_cache_list
        results = (loss, logits, outputs[1], outputs[2])
        return results

    def prepare_inputs_for_generation(
        self,
        input_ids,
        **kwargs,
    ):
        if kwargs.get("return_key_value_cache"):
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if kwargs.get("inputs_embeds") is not None and not kwargs.get("return_key_value_cache"):
            model_inputs = {"inputs_embeds": kwargs.get("inputs_embeds")}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "return_key_value_cache": kwargs.get("return_key_value_cache", None),
                "past_key_cache_list": kwargs.get("past_key_cache_list", None),
                "past_value_cache_list": kwargs.get("past_value_cache_list", None),
                "attention_mask": kwargs.get("attention_mask", None),
                "images": kwargs.get("images", None),
            }
        )

        return model_inputs
