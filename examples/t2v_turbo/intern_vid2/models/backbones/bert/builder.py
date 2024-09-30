# from .xbert import BertConfig, BertForMaskedLM, BertLMHeadModel, BertModel
import mindspore as ms
import numpy as np
from mindnlp.transformers import BertConfig, BertForMaskedLM, BertLMHeadModel, BertModel
from mindnlp.transformers.modeling_utils import CellUtilMixin
from typing import Tuple

import logging
logger = logging.getLogger(__name__)


class BertModelWrapper(BertModel):
    def __init__(self, config, add_pooling_layer=True, dtype=ms.float32):
        super().__init__(config, add_pooling_layer)
        self._dtype = dtype
    
    @property
    def dtype(self):
        return self._dtype
    
    def get_extended_attention_mask(
        self, attention_mask: ms.Tensor, input_shape: Tuple[int], dtype = None
    ) -> ms.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`ms.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            :obj:`ms.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """

        if dtype is None:
            dtype = self.dtype

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = CellUtilMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.astype(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) \
            * ms.Tensor(np.finfo(ms.dtype_to_nptype(dtype)).min)
        return extended_attention_mask

def build_bert(model_config, pretrain, checkpoint, encoder_width=None, dtype=ms.float32):
    """build text encoder.

    Args:
        model_config (dict): model config.
        pretrain (bool): Whether to do pretrain or finetuning.
        checkpoint (bool): whether to do gradient_checkpointing.

    Returns: TODO

    """
    bert_config = BertConfig.from_json_file(model_config.text_encoder.config)
    if encoder_width is None:
        bert_config.encoder_width = model_config.vision_encoder.d_model
    else:
        bert_config.encoder_width = encoder_width
        
    bert_config.gradient_checkpointing = checkpoint
    bert_config.fusion_layer = model_config.text_encoder.fusion_layer

    if not model_config.multimodal.enable:
        bert_config.fusion_layer = bert_config.num_hidden_layers

    if pretrain:
        try:
            text_encoder, loading_info = BertForMaskedLM.from_pretrained(
                model_config.text_encoder.pretrained,
                config=bert_config,
                output_loading_info=True, 
                local_files_only=True
            )
        except:
            text_encoder, loading_info = BertForMaskedLM.from_pretrained(
                model_config.text_encoder.pretrained,
                config=bert_config,
                output_loading_info=True, 
                local_files_only=False
            )
    else:
        try:
            text_encoder, loading_info = BertModelWrapper.from_pretrained(
                model_config.text_encoder.pretrained,
                config=bert_config,
                add_pooling_layer=False,
                output_loading_info=True,
                local_files_only=True,
                dtype=dtype,
            )
        except:
            text_encoder, loading_info = BertModelWrapper.from_pretrained(
                model_config.text_encoder.pretrained,
                config=bert_config,
                add_pooling_layer=False,
                output_loading_info=True,
                local_files_only=False,
                dtype=dtype,
            )

    return text_encoder


def build_bert_decoder(model_config, checkpoint, only_fusion_layer=True):
    """build text decoder the same as the multimodal encoder.

    Args:
        model_config (dict): model config.
        pretrain (bool): Whether to do pretrain or finetuning.
        checkpoint (bool): whether to do gradient_checkpointing.

    Returns: TODO

    """
    bert_config = BertConfig.from_json_file(model_config.text_encoder.config)
    bert_config.encoder_width = model_config.vision_encoder.d_model
    bert_config.gradient_checkpointing = checkpoint

    bert_config.fusion_layer = 0

    if only_fusion_layer:
        bert_config.num_hidden_layers = (
            bert_config.num_hidden_layers - model_config.text_encoder.fusion_layer
        )

    text_decoder, loading_info = BertLMHeadModel.from_pretrained(
        model_config.text_encoder.pretrained,
        config=bert_config,
        output_loading_info=True,
        local_files_only=True
    )

    return text_decoder

def build_lm_bert_decoder(model_config, checkpoint):
    """build text decoder the same as the multimodal encoder.

    Args:
        model_config (dict): model config.
        pretrain (bool): Whether to do pretrain or finetuning.
        checkpoint (bool): whether to do gradient_checkpointing.

    Returns: TODO

    """
    bert_config = BertConfig.from_json_file(model_config.text_encoder.config)
    bert_config.encoder_width = model_config.vision_encoder.d_model
    bert_config.gradient_checkpointing = checkpoint
    bert_config.fusion_layer = model_config.text_encoder.fusion_layer
    
    text_decoder, loading_info = BertLMHeadModel.from_pretrained(
        model_config.text_encoder.pretrained,
        config=bert_config,
        output_loading_info=True,
        local_files_only=True
    )

    return text_decoder
