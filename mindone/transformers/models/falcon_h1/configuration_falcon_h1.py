# Copyright 2025 Technology Innovation Institute and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...modeling_rope_utils import rope_config_validation


class FalconH1Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FalconH1Model`]. It is used to instantiate an
    FalconH1 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the FalconH1 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FalconH1Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        key_multiplier (`float`, *optional*, defaults to 1.0):
            The multiplier for the key projection.
        
        # Mamba/SSM parameters
        mamba_d_conv (`int`, *optional*, defaults to 4):
            Dimension of the Mamba convolution kernel.
        mamba_d_ssm (`int`, *optional*, defaults to None):
            Dimension of the SSM state. If None, uses mamba_expand * hidden_size.
        mamba_expand (`int`, *optional*, defaults to 2):
            Expansion factor for the Mamba model.
        mamba_n_groups (`int`, *optional*, defaults to 1):
            Number of groups for the Mamba model.
        mamba_d_state (`int`, *optional*, defaults to 16):
            Dimension of the SSM state in the Mamba model.
        mamba_n_heads (`int`, *optional*, defaults to 1):
            Number of heads for the Mamba model.
        mamba_d_head (`int`, *optional*, defaults to 64):
            Dimension of each head in the Mamba model.
        mamba_chunk_size (`int`, *optional*, defaults to 64):
            Chunk size for the Mamba model.
        mamba_conv_bias (`bool`, *optional*, defaults to True):
            Whether to use bias in the Mamba convolution.
        mamba_proj_bias (`bool`, *optional*, defaults to False):
            Whether to use bias in the Mamba projection.
        mamba_rms_norm (`bool`, *optional*, defaults to True):
            Whether to use RMS normalization in the Mamba model.
        mamba_norm_before_gate (`bool`, *optional*, defaults to True):
            Whether to apply normalization before the gate in the Mamba model.
            
        # Multiplier parameters for MuP
        mlp_multipliers (`float`, *optional*, defaults to 1.0):
            Multiplier for the MLP layers.
        ssm_multipliers (`float`, *optional*, defaults to 1.0):
            Multiplier for the SSM layers.
        ssm_in_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier for the SSM input.
        attention_in_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier for the attention input.
        ssm_out_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier for the SSM output.
        attention_out_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier for the attention output.
        embedding_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier for the embedding.
        lm_head_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier for the language model head.
            
        # Projection parameters
        projectors_bias (`bool`, *optional*, defaults to False):
            Whether to use bias in the projectors.
        mlp_bias (`bool`, *optional*, defaults to False):
            Whether to use bias in the MLP layers.
    """

    model_type = "falcon_h1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        key_multiplier=1.0,
        # Mamba/SSM parameters
        mamba_d_conv=4,
        mamba_d_ssm=None,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_d_state=16,
        mamba_n_heads=1,
        mamba_d_head=64,
        mamba_chunk_size=64,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_rms_norm=True,
        mamba_norm_before_gate=True,
        # Multiplier parameters
        mlp_multipliers=1.0,
        ssm_multipliers=1.0,
        ssm_in_multiplier=1.0,
        attention_in_multiplier=1.0,
        ssm_out_multiplier=1.0,
        attention_out_multiplier=1.0,
        embedding_multiplier=1.0,
        lm_head_multiplier=1.0,
        # Projection parameters
        projectors_bias=False,
        mlp_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.key_multiplier = key_multiplier

        # Mamba/SSM parameters
        self.mamba_d_conv = mamba_d_conv
        self.mamba_d_ssm = mamba_d_ssm
        self.mamba_expand = mamba_expand
        self.mamba_n_groups = mamba_n_groups
        self.mamba_d_state = mamba_d_state
        self.mamba_n_heads = mamba_n_heads
        self.mamba_d_head = mamba_d_head
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_rms_norm = mamba_rms_norm
        self.mamba_norm_before_gate = mamba_norm_before_gate

        # Multiplier parameters
        self.mlp_multipliers = mlp_multipliers
        self.ssm_multipliers = ssm_multipliers
        self.ssm_in_multiplier = ssm_in_multiplier
        self.attention_in_multiplier = attention_in_multiplier
        self.ssm_out_multiplier = ssm_out_multiplier
        self.attention_out_multiplier = attention_out_multiplier
        self.embedding_multiplier = embedding_multiplier
        self.lm_head_multiplier = lm_head_multiplier

        # Projection parameters
        self.projectors_bias = projectors_bias
        self.mlp_bias = mlp_bias

        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["FalconH1Config"]