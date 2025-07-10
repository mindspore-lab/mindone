# coding=utf-8
# Copyright 2024 Mindspore Team. All rights reserved.
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

import pytest

import mindspore
from mindspore import dtype as mstype

from mindone.transformers import LEDConfig
from mindone.transformers.models.led.modeling_led import (
    LEDForConditionalGeneration,
    LEDForQuestionAnswering,
    LEDForSequenceClassification,
    LEDModel,
)


class LEDModelTester:
    config_class = LEDConfig

    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=37,
        decoder_ffn_dim=37,
        activation_function="gelu",
        dropout=0.1,
        attention_dropout=0.1,
        max_encoder_position_embeddings=512,
        max_decoder_position_embeddings=512,
        init_std=0.02,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        classifier_dropout=0.0,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        attention_window=4,
        num_labels=3,
        num_choices=4,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_encoder_position_embeddings = max_encoder_position_embeddings
        self.max_decoder_position_embeddings = max_decoder_position_embeddings
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.attention_window = attention_window
        self.num_labels = num_labels
        self.num_choices = num_choices

    def prepare_config_and_inputs(self):
        input_ids = mindspore.Tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0]], dtype=mindspore.int64)

        attention_mask = None
        if self.use_input_mask:
            attention_mask = mindspore.Tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=mindspore.int64)

        decoder_input_ids = mindspore.Tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=mindspore.int64)

        config = self.get_config()

        return (
            config,
            input_ids,
            attention_mask,
            decoder_input_ids,
        )

    def get_config(self):
        return LEDConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            decoder_attention_heads=self.decoder_attention_heads,
            encoder_ffn_dim=self.encoder_ffn_dim,
            decoder_ffn_dim=self.decoder_ffn_dim,
            activation_function=self.activation_function,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_encoder_position_embeddings=self.max_encoder_position_embeddings,
            max_decoder_position_embeddings=self.max_decoder_position_embeddings,
            init_std=self.init_std,
            encoder_layerdrop=self.encoder_layerdrop,
            decoder_layerdrop=self.decoder_layerdrop,
            classifier_dropout=self.classifier_dropout,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            attention_window=self.attention_window,
        )


# Define test cases for different LED model variants
LED_CASES = [
    (
        "led_base",
        LEDModel,
        LEDModel,
        [],
        {"config": LEDModelTester().get_config()},
        [],
        {
            "input_ids": mindspore.Tensor([[1, 2, 3, 4]], dtype=mindspore.int64),
            "attention_mask": mindspore.Tensor([[1, 1, 1, 1]], dtype=mindspore.int64),
            "decoder_input_ids": mindspore.Tensor([[0, 1, 2, 3]], dtype=mindspore.int64),
        },
        {"last_hidden_state": (1, 4, 32)},
    ),
    (
        "led_for_conditional_generation",
        LEDForConditionalGeneration,
        LEDForConditionalGeneration,
        [],
        {"config": LEDModelTester().get_config()},
        [],
        {
            "input_ids": mindspore.Tensor([[1, 2, 3, 4]], dtype=mindspore.int64),
            "attention_mask": mindspore.Tensor([[1, 1, 1, 1]], dtype=mindspore.int64),
            "decoder_input_ids": mindspore.Tensor([[0, 1, 2, 3]], dtype=mindspore.int64),
        },
        {"logits": (1, 4, 99)},
    ),
    (
        "led_for_sequence_classification",
        LEDForSequenceClassification,
        LEDForSequenceClassification,
        [],
        {"config": LEDModelTester().get_config()},
        [],
        {
            "input_ids": mindspore.Tensor([[1, 2, 3, 4]], dtype=mindspore.int64),
            "attention_mask": mindspore.Tensor([[1, 1, 1, 1]], dtype=mindspore.int64),
            "decoder_input_ids": mindspore.Tensor([[0, 1, 2, 3]], dtype=mindspore.int64),
        },
        {"logits": (1, 3)},
    ),
    (
        "led_for_question_answering",
        LEDForQuestionAnswering,
        LEDForQuestionAnswering,
        [],
        {"config": LEDModelTester().get_config()},
        [],
        {
            "input_ids": mindspore.Tensor([[1, 2, 3, 4]], dtype=mindspore.int64),
            "attention_mask": mindspore.Tensor([[1, 1, 1, 1]], dtype=mindspore.int64),
            "decoder_input_ids": mindspore.Tensor([[0, 1, 2, 3]], dtype=mindspore.int64),
        },
        {
            "start_logits": (1, 4),
            "end_logits": (1, 4),
        },
    ),
]

DTYPE_AND_THRESHOLDS = {
    mstype.float32: 1e-4,
    mstype.float16: 1e-2,
}

MODES = ["graph", "pynative"]


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
        for case in LED_CASES
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
    """Test LED model variants."""
    mindspore.set_context(mode=mode)

    # Initialize model
    model = ms_module(*init_args, **init_kwargs)
    model.set_train(False)

    # Run forward pass
    outputs = model(*inputs_args, **inputs_kwargs)

    # Check output shapes
    for key, expected_shape in outputs_map.items():
        if hasattr(outputs, key):
            output = getattr(outputs, key)
            assert output.shape == expected_shape, f"Shape mismatch for {key}"
