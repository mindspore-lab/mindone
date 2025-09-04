- Task:
  - write a test script given the model's modeling script and configuration file

- Inputs collected
  - input_folder: `mindone/transformers/models/distilbert`

- Immediate checks
  - Verify `mindone/transformers/models/distilbert` exists (warn user if it is empty)
  - Verify `tests/transformers_tests/models/distilbert/` exists (create it if it is empty)

- Scope
  - Read all Python files under `mindone/transformers/models/distilbert`
  - Read all arguments of `mindone/transformers/models/distilbert/configuration_distilbert.py`

- Process
  - Read the following example of configuration file and test script for a model named `cohere2`:
```python
# Configuration for model cohere2
import warnings

from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...modeling_rope_utils import rope_config_validation


class Cohere2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CohereModel`]. It is used to instantiate an Cohere
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01) model.


    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Cohere model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CohereModel`]
        hidden_size (`int`, *optional*, defaults to 8192):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22528):
            Dimension of the MLP representations.
        logit_scale (`float`, *optional*, defaults to 0.0625):
            The scaling factor for the output logits.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
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
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 5):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 255001):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        sliding_window (`int`, *optional*, defaults to 4096):
            Size of the sliding window attention context.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
    """

    model_type = "cohere2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=8192,
        intermediate_size=22528,
        logit_scale=0.0625,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=5,
        eos_token_id=255001,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=4096,
        layer_types=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.logit_scale = logit_scale
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        # Need to specify head_dim in the config so it can be used in the attention forward functions
        self.head_dim = hidden_size // num_attention_heads

        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 4)

        if self.layer_types is None:
            # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
            self._sliding_window_pattern = getattr(self, "sliding_window_pattern", 4)
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

    @property
    def sliding_window_pattern(self):
        warnings.warn(
            "The `sliding_window_pattern` attribute is deprecated and will be removed in v4.55.0.",
            FutureWarning,
        )
        return self._sliding_window_pattern

    @sliding_window_pattern.setter
    def sliding_window_pattern(self, value):
        self._sliding_window_pattern = value


__all__ = ["Cohere2Config"]

```

```python
"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/cohere2/test_modeling_cohere2.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.

import inspect

import numpy as np
import pytest
import torch
from transformers import Cohere2Config

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
# MODES = [0, 1] # 0: graph mode, 1: pynative mode
# FIXME: Cohere2 does not support graph mode yet, so we only test in pynative mode.
MODES = [1]


class Cohere2ModelTester:
    config_class = Cohere2Config

    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        # config
        vocab_size=99,
        hidden_size=32,
        intermediate_size=37,
        logit_scale=0.0625,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=5,
        eos_token_id=255001,
        tie_word_embeddings=True,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=4096,
        sliding_window_pattern=4,
        cache_implementation="hybrid",
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.logit_scale = logit_scale
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias

        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.cache_implementation = cache_implementation
        self.head_dim = self.hidden_size // self.num_attention_heads

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = np.tril(np.ones_like(input_ids))

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_numpy([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_numpy([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_numpy([self.batch_size], self.num_choices)

        config = self.get_config()

        # set _attn_implementation
        config._attn_implementation = "eager"

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return self.config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            logit_scale=self.logit_scale,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            sliding_window=self.sliding_window,
            sliding_window_pattern=self.sliding_window_pattern,
            cache_implementation=self.cache_implementation,
        )


model_tester = Cohere2ModelTester()
(
    config,
    input_ids,
    token_type_ids,
    input_mask,
    sequence_labels,
    token_labels,
    choice_labels,
) = model_tester.prepare_config_and_inputs()


COHERE2_CASES = [
    [
        "Cohere2Model",
        "transformers.Cohere2Model",
        "mindone.transformers.Cohere2Model",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
        },
        {
            "last_hidden_state": 0,
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [
        case
        + [
            dtype,
        ]
        + [
            mode,
        ]
        for case in COHERE2_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    ms.set_context(mode=mode)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )
    # set `hidden_dtype` if requiring, for some modules always compute in float
    # precision and require specific `hidden_dtype` to cast before return
    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[ms_idx]
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )

```
  - Reduce the argument values (e.g., num_hidden_layers, vocab_size, batch_size, seq_len) in `configuration_distilbert.py` to initiate a reduced-size model for testing
  - Write the test script to `tests/transformers_tests/models/distilbert/test_modeling_distilbert.py`

- Validation checklist
  - Python AST syntax validation (parse each edited file)
  - Run test script

- Exit criteria
  - No syntax errors; imports resolve; no prohibited APIs remain.