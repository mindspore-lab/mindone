# tests/models/llama/test_modeling_llama.py
import inspect
import numpy as np
import pytest
import torch
from transformers import DbrxConfig
import unittest
import mindspore as ms
from mindway.transformers import DbrxForCausalLM
from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]

def random_attention_mask(shape, rng=None, name=None):
    attn_mask = ids_numpy(shape, vocab_size=2, rng=None, name=None)
    # make sure that at least one token is attended to for each batch
    # we choose the 1st token so this property of `at least one being non-zero` still holds after applying causal mask
    attn_mask[:, 0] = 1
    return attn_mask

class DbrxModelTester:
    def __init__(
        self,
        hidden_size=32,
        ffn_hidden_size=32,
        num_attention_heads=4,
        kv_n_heads=4,
        num_hidden_layers=5,
        max_position_embeddings=512,
        type_vocab_size=16,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        use_cache=True,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        clip_qkv=8,
        rope_theta=500000,
        attn_config_model_type="",
        emb_pdrop=0.0,
        moe_jitter_eps=0,
        moe_loss_weight=0.05,
        moe_num_experts=16,
        moe_top_k=4,
        ffn_config_model_type="",
        ffn_act_fn_name="gelu",
        initializer_range=0.02,
        output_router_logits=False,
        resid_pdrop=0.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        vocab_size=99,
        is_decoder=True,
        pad_token_id=0,
    ):
        # Parameters unique to testing
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels

        # attn_config params
        self.clip_qkv = clip_qkv
        self.kv_n_heads = kv_n_heads
        self.rope_theta = rope_theta
        self.attn_config_model_type = attn_config_model_type

        # ffn_config params
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.ffn_config_model_type = ffn_config_model_type
        self.ffn_act_fn_name = ffn_act_fn_name

        # Other model params
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.emb_pdrop = emb_pdrop
        self.output_router_logits = output_router_logits
        self.resid_pdrop = resid_pdrop
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype
        self.is_decoder = is_decoder
        self.pad_token_id = pad_token_id

        # Make the dictionaries
        self.ffn_config = {
            "ffn_hidden_size": self.ffn_hidden_size,
            "moe_jitter_eps": self.moe_jitter_eps,
            "moe_loss_weight": self.moe_loss_weight,
            "moe_num_experts": self.moe_num_experts,
            "moe_top_k": self.moe_top_k,
            "model_type": self.ffn_config_model_type,
            "ffn_act_fn": {"name": self.ffn_act_fn_name},
        }
        self.attn_config = {
            "clip_qkv": self.clip_qkv,
            "kv_n_heads": self.kv_n_heads,
            "model_type": self.attn_config_model_type,
            "rope_theta": self.rope_theta,
        }

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

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

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        # Behind the scenes, `DbrxConfig` maps the parameters `hidden_size`, `num_hidden_layers`,
        # `num_attention_heads`, `max_position_embeddings` to the parameters `d_model`, `n_layers`,
        # `n_heads`, `max_seq_len` respectively. We use the first group of parameters because
        # other tests expect every model to have these parameters with these specific names.
        config = DbrxConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,  # mapped to `d_model`
            num_hidden_layers=self.num_hidden_layers,  # mapped to `n_layers`
            num_attention_heads=self.num_attention_heads,  # mapped to `n_heads`
            max_position_embeddings=self.max_position_embeddings,  # mapped to `max_seq_len`
            attn_config=self.attn_config,
            ffn_config=self.ffn_config,
            resid_pdrop=self.resid_pdrop,
            emb_pdrop=self.emb_pdrop,
            use_cache=self.use_cache,
            initializer_range=self.initializer_range,
            output_router_logits=self.output_router_logits,
            is_decoder=self.is_decoder,
            pad_token_id=self.pad_token_id,
        )
        return config


model_tester = DbrxModelTester()
config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels = model_tester.prepare_config_and_inputs()

LLAMA_CASES = [
    [
        "DbrxForCausalLM",
        "transformers.DbrxForCausalLM",
        "mindway.transformers.DbrxForCausalLM",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask":input_mask,
            "labels":token_labels
        },
        {
            "logits": 0,
        },
    ],
    [
        "DbrxModel",
        "transformers.DbrxModel",
        "mindway.transformers.DbrxModel",
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
        for case in LLAMA_CASES
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
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

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
            ms_output = ms_outputs[pt_key]
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


class DbrxModelIntegrationTest(unittest.TestCase):
    def test_tiny_model_logits(self):
        model = DbrxForCausalLM.from_pretrained("Rocketknight1/dbrx-tiny-random")
        input_ids = ms.tensor([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]
        vocab_size = model.vocab_size

        expected_shape = (1, 6, vocab_size)
        self.assertEqual(output.shape, expected_shape)

        expected_slice = np.array(
            [
                [
                    [-1.6300e-04, 5.0118e-04, 2.5437e-04],
                    [2.0422e-05, 2.7210e-04, -1.5125e-04],
                    [-1.5105e-04, 4.6879e-04, 3.3309e-04],
                ]
            ]
        )
        THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]
        diffs = np.linalg.norm(expected_slice - output[:, :3, :3].asnumpy()) / np.linalg.norm(expected_slice)
        assert (np.array(diffs) < THRESHOLD).all(), (
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )
