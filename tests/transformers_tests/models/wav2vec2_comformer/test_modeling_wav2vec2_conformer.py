"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/wav2vec2/test_modeling_wav2vec2.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import math

import numpy as np
import pytest
import torch
from transformers import Wav2Vec2ConformerConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class Wav2Vec2ConformerModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=1024,  # speech is longer
        is_training=False,
        hidden_size=16,
        feat_extract_norm="group",
        feat_extract_dropout=0.0,
        feat_extract_activation="gelu",
        conv_dim=(32, 32, 32),
        conv_stride=(4, 4, 4),
        conv_kernel=(8, 8, 8),
        conv_bias=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout_prob=0.1,
        intermediate_size=20,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        initializer_range=0.02,
        mask_time_prob=0.5,
        mask_time_length=2,
        vocab_size=32,
        do_stable_layer_norm=False,
        num_adapter_layers=1,
        adapter_stride=2,
        tdnn_dim=(32, 32),
        tdnn_kernel=(5, 3),
        tdnn_dilation=(1, 2),
        xvector_output_dim=32,
        position_embeddings_type="relative",
        scope=None,
        attn_implementation="eager",
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_dropout = feat_extract_dropout
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.num_adapter_layers = num_adapter_layers
        self.adapter_stride = adapter_stride
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.scope = scope
        self.tdnn_dim = tdnn_dim
        self.tdnn_kernel = tdnn_kernel
        self.tdnn_dilation = tdnn_dilation
        self.xvector_output_dim = xvector_output_dim
        self.position_embeddings_type = position_embeddings_type
        self.attn_implementation = attn_implementation

        output_seq_length = self.seq_length
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

        self.adapter_output_seq_length = (self.output_seq_length - 1) // adapter_stride + 1

    def prepare_config_and_inputs(self):
        input_values = floats_numpy([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_values, attention_mask

    def get_config(self, position_embeddings_type="relative"):
        return Wav2Vec2ConformerConfig(
            hidden_size=self.hidden_size,
            feat_extract_norm=self.feat_extract_norm,
            feat_extract_dropout=self.feat_extract_dropout,
            feat_extract_activation=self.feat_extract_activation,
            conv_dim=self.conv_dim,
            conv_stride=self.conv_stride,
            conv_kernel=self.conv_kernel,
            conv_bias=self.conv_bias,
            mask_time_prob=self.mask_time_prob,
            mask_time_length=self.mask_time_length,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.hidden_dropout_prob,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            do_stable_layer_norm=self.do_stable_layer_norm,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
            num_adapter_layers=self.num_adapter_layers,
            adapter_stride=self.adapter_stride,
            tdnn_dim=self.tdnn_dim,
            tdnn_kernel=self.tdnn_kernel,
            tdnn_dilation=self.tdnn_dilation,
            xvector_output_dim=self.xvector_output_dim,
            position_embeddings_type=position_embeddings_type,
            attn_implementation=self.attn_implementation,
        )

    def prepare_config_and_inputs_for_seq_classifier_loss(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()

        input_values = input_values[:3]
        attention_mask = np.ones(input_values.shape).astype(np.int64)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_numpy((input_values.shape[0], 1), len(config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        return config, input_values, attention_mask, labels

    def prepare_config_and_inputs_for_xvector_training(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        config.ctc_zero_infinity = True

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_numpy((input_values.shape[0], 1), len(config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

        return config, input_values, labels

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


model_tester = Wav2Vec2ConformerModelTester()
(
    config,
    input_values,
    attention_mask,
) = model_tester.prepare_config_and_inputs()
(
    classifier_config,
    classifier_input_values,
    classifier_attention_mask,
    classifier_labels,
) = model_tester.prepare_config_and_inputs_for_seq_classifier_loss()
(
    xvector_config,
    xvector_input_values,
    xvector_labels,
) = model_tester.prepare_config_and_inputs_for_xvector_training()


TEST_CASES = [
    [
        "Wav2Vec2ConformerModel",
        "transformers.Wav2Vec2ConformerModel",
        "mindone.transformers.Wav2Vec2ConformerModel",
        (config,),
        {},
        (),
        {
            "input_values": input_values,
            "attention_mask": attention_mask,
        },
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "Wav2Vec2ConformerForCTC",
        "transformers.Wav2Vec2ConformerForCTC",
        "mindone.transformers.Wav2Vec2ConformerForCTC",
        (config,),
        {},
        (),
        {
            "input_values": input_values,
            "attention_mask": attention_mask,
        },
        {
            "logits": 0,
        },
    ],
    [
        "Wav2Vec2ConformerForSequenceClassification",
        "transformers.Wav2Vec2ConformerForSequenceClassification",
        "mindone.transformers.Wav2Vec2ConformerForSequenceClassification",
        (classifier_config,),
        {},
        (),
        {
            "input_values": classifier_input_values,
            "attention_mask": classifier_attention_mask,
            "labels": classifier_labels,
        },
        {
            "loss": 0,
        },
    ],
    [
        "Wav2Vec2ConformerForXVector",
        "transformers.Wav2Vec2ConformerForXVector",
        "mindone.transformers.Wav2Vec2ConformerForXVector",
        (xvector_config,),
        {},
        (),
        {
            "input_values": xvector_input_values,
            "labels": xvector_labels,
        },
        {
            "loss": 0,
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
        for case in TEST_CASES
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
