# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import mindspore as ms
from mindspore import mint, ops, nn, Tensor
from typing import Optional
from addict import Dict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
)
from mindone.transformers import (
    LlamaForCausalLM,
)
from mindone.transformers.modeling_utils import MSPreTrainedModel as PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector


class vision_head(nn.Cell):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = mint.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = nn.GELU()
        self.vision_head = mint.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def construct(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = Dict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = Dict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = Dict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = Dict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = Dict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    # since LlamaPreTrainedModel support FA
    _supports_flash_attn_2 = True


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        # FIXME: allow set attn impl in from_pretrained, or  default FA (current default eager)
        # language_config._attn_implementation = "flash_attention_2"
        # print("Set _attn_implementation = \"flash_attention_2\"")
        self.language_model = LlamaForCausalLM(language_config)

        self.cross_entropy_loss = nn.CrossEntropyLoss()  # TODO: allow setting ignore_idex, default is -100

    def prepare_inputs_embeds(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
        images_seq_mask: Tensor,
        images_emb_mask: Tensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (Tensor): [b, T]
            pixel_values (ms.float32):   [b, n_images, 3, h, w]
            images_seq_mask (ms.BoolTensor): [b, T]
            images_emb_mask (ms.BoolTensor): [b, n_images, n_image_tokens]

            assert ms.sum(images_seq_mask) == ms.sum(images_emb_mask)

        Returns:
            input_embeds (ms.Tensor): [b, T, D]
        """


        bs, n, c, h, w = pixel_values.shape
        images = ops.reshape(pixel_values, (bs*n, c, h, w))

        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        # (b n) t d -> b n t d -> b (n t) d
        bn, T, D = images_embeds.shape
        images_embeds = ops.reshape(images_embeds, (bs, n, T, D))
        images_embeds = ops.reshape(images_embeds, (bs, n*T, D))

        # [b, n, T2] -> [b, n x T2]
        # images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")
        _, Nm, Tm = images_emb_mask.shape
        images_emb_mask = ops.reshape(images_emb_mask, (bs, Nm * Tm)) 

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: Tensor):
        return self.gen_aligner(self.gen_embed(image_ids))

    # TODO: image_seq_masks can be extracted from labels != -100 or input_ids == img_placeholder_token
    def gen_with_loss(self,
        input_ids: Tensor = None,
        labels: Tensor = None,
        attention_masks: Optional[Tensor] = None,
        image_seq_masks: Optional[Tensor] = None, 
        pixel_values: Optional[Tensor] = None,
        image_tokens: Optional[Tensor] = None,
    ):
        r""" only compute loss on the image sequence
        Args:
            input_ids: input sequence of text and image tokens, shape (bs, seq_len)
                format like [BOS, tokens of instruction and prompt, BOI, image placeholder tokens, EOI, padded tokens]  
            labels: labels for computing the masked auto-regressive model loss. Indices should be either be in `[0, .., text_vocab_size + vision_vocab_size]` or -100 (see `input_ids` for docstring). 
                For T2I gen, only compute loss on image sequence. Thusthe value is like 
                            [-100, -100, ..., -100, image tokens, -100, -100, ...]
            attention_masks: shape (bs seq_len), where 1 for valid input seq, 0 for padded seq 
            image_seq_masks: 1 - image tokens (exclude BOI and EOI)
            pixel_values: images resized to (384, 384), shape (bs n_images 3 h w)
            image_tokens: image tokens encoded and quantized by VQ16, shape (bs n_images per_img_seq_len)

        Note: pre-compute VQ encoded tokens for efficiency
        """
        # prepare inputs
        # TODO: consider remove n_images dimension unless we need to generate videos for extension
        if image_tokens is None:
            bs, n, c, h, w = pixel_values.shape
            pixel_values = ops.reshape(pixel_values, (bs*n, c, h, w))
            image_tokens = self.gen_vision_model.encode(pixel_values)[0] 
            bs = image_tokens.shape[0]
            image_tokens = image_tokens.reshape(bs, -1)

        image_embed = self.gen_aligner(self.gen_embed(image_tokens))
        # [b x n, T2, D] -> [b, n x T2, D]
        bn, T, D = image_embeds.shape
        image_embeds = ops.reshape(image_embeds, (bs, n, T, D))
        image_embeds = ops.reshape(image_embeds, (bs, n*T, D))

        # TODO: set image (placeholder) tokens to 0? avoid being larger than lm vocab size  
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds[image_seq_masks] = image_embeds

        # LlamaModel forward
        outputs = self.lanugage_model.model(
            attention_mask=attention_masks,
            inputs_embeds=inputs_embeds,
            return_dict=False,
        )
        hidden_states = outputs[0]

        # gen head
        # since Janus use decouple heads for image and text, only image seq is meaningful input to gen head. mask before linear should save compute cost. 
        # TODO: tbc influence on gradient ?
        logits = self.gen_head(hidden_states[image_seq_masks])
        labels = labels[image_seq_masks]  # if so, can just output masked labels in dataloader

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = self.cross_entropy_loss(shift_logits, shift_labels)

    
    def und_with_loss(self,
        input_ids: Tensor = None,
        labels: Tensor = None,
        attention_masks: Optional[Tensor] = None,
        image_seq_masks: Optional[Tensor] = None, 
        pixel_values: Optional[Tensor] = None,
    ):
        r""" only compute loss on the text sequence
        Args:
            input_ids: input sequence of text and image tokens, shape (bs, seq_len)
                format varies from tasks, like 
                    - vqa       [BOS, tokens of instruction, BOI, 576 image placeholder tokens, EOI, question tokens, answer tokens, EOS, padded tokens],   
                    - caption   [BOS, BOI, 576 image placeholder tokens, EOI, tokens for "Describe the image in detail", caption tokens, EOS, padded tokens]
            labels: labels for computing the masked auto-regressive model loss. Indices should be either be in `[0, .., text_vocab_size + vision_vocab_size]` or -100 (see `input_ids` for docstring). 
                For mm und, only compute loss on text sequence. Thusthe value is like
                    - vqa       [-100, -100, ..., -100, answer tokens,  EOS, -100, ...]
                    - caption   [-100, -100, ..., -100, caption tokens, EOS, -100, ...] 
            attention_masks: shape (bs seq_len), where 1 for valid input seq, 0 for padded seq 
            pixel_values: images resized to (384, 384)

        Note: since sigLIP is trainable in stage 3, so we prefer not to pre-compute sigLIP features
        """
        # preapre inputs
        # TODO: consider remove n_images dimension unless we need to generate videos for extension
        bs, n, c, h, w = pixel_values.shape
        pixel_values = ops.reshape(pixel_values, (bs*n, c, h, w))
        image_features = self.vision_model(pixel_values)
        image_embeds = self.aligner(image_embeds)
        # [b x n, T2, D] -> [b, n x T2, D]
        bn, T, D = image_embeds.shape
        image_embeds = ops.reshape(image_embeds, (bs, n, T, D))
        image_embeds = ops.reshape(image_embeds, (bs, n*T, D))

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds[image_seq_masks] = image_embeds  # bprop? graph support?

        # LlamaModel forward
        outputs = self.language_model.model(
            attention_mask=attention_masks,
            inputs_embeds=inputs_embeds,
            return_dict=False,
        )
        hidden_states = outputs[0]
        
        # text head
        logits = self.language_mode.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = self.cross_entropy_loss(shift_logits, shift_labels)

        return loss


    def construct(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor],
        attention_masks: Optional[Tensor] = None,
        image_seq_masks: Optional[Tensor] = None, 
        pixel_values: Optional[Tensor] = None,
        image_tokens: Optional[Tensor] = None,
        is_gen_task = True,  # TODO: compatible with TrainOneStep() ?
    ):
        r"""
        Added for training
        Args:
            input_ids: input sequence of tokens, shape (bs seq_len). see transformers docstring for details  
            
        """

        if is_gen_task:
            loss = self.gen_with_loss(
                    input_ids,
                    labels,
                    attention_masks,
                    image_seq_masks, 
                    pixel_values,
                    image_tokens,
                    )
        else:
            loss = self.und_with_loss(
                input_ids,
                labels,
                attention_masks,
                image_seq_masks, 
                pixel_values,
                )

        return loss
            

AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
