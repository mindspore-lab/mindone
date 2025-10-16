import inspect

import numpy as np
import pytest
import torch
from transformers import AriaConfig, AriaTextConfig, Idefics3VisionConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-3, "fp16": 6e-4, "bf16": 6e-3}
MODES = [1]


class AriaModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        use_cache=False,
        moe_num_experts=2,
        attn_implementation="eager",
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.moe_num_experts = moe_num_experts
        self.attn_implementation = attn_implementation
        self.image_token_index = self.vocab_size - 1

    def prepare_config_and_inputs(self):
        n_img_feat = 64
        input_ids = ids_numpy([self.batch_size, self.seq_length + n_img_feat], self.vocab_size - 1)
        input_ids[-1, -n_img_feat:] = self.image_token_index

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_numpy([self.batch_size, self.seq_length + n_img_feat], vocab_size=2)

        image_batch_size = 1
        num_channels = 3
        height = 64
        width = 64
        pixel_values = ids_numpy([image_batch_size, num_channels, height, width], vocab_size=256)
        pixel_values = (pixel_values.astype(np.float32) / 255.0) * 2 - 1  # in range [-1, 1]
        pixel_mask = np.ones(
            (pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3]),
            dtype=np.bool_,
        )

        text_config, config = self.get_config()

        return text_config, config, input_ids, input_mask, pixel_values, pixel_mask

    def get_config(self):
        text_config = AriaTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=None,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=self.use_cache,
            pad_token_id=2,
            bos_token_id=1,
            eos_token_id=2,
            moe_num_experts=self.moe_num_experts,
            attn_implementation=self.attn_implementation,
        )

        vision_config = Idefics3VisionConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_channels=3,
            image_size=64,
            patch_size=14,
            attn_implementation=self.attn_implementation,
        )

        config = AriaConfig(
            vision_config=vision_config,
            vision_feature_layer=-1,
            text_config=text_config,
            projector_patch_to_query_dict={
                "16": 64,  # (64//14) x (64//14)
                "1225": 128,
                "4900": 256,
            },
            image_token_index=self.image_token_index,
        )
        return text_config, config


model_tester = AriaModelTester()
text_config, config, input_ids, input_mask, pixel_values, pixel_mask = model_tester.prepare_config_and_inputs()

ARIA_CASES = [
    [  # test pure text Q&A
        "AriaTextForCausalLM",
        "transformers.AriaTextForCausalLM",
        "mindone.transformers.AriaTextForCausalLM",
        (text_config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        },
        {
            "logits": 0,
        },
    ],
    [  # test VQA, always compare with torch 32
        "AriaForConditionalGeneration",
        "transformers.AriaForConditionalGeneration",
        "mindone.transformers.AriaForConditionalGeneration",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
        },
        {
            "logits": 0,
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
        for case in ARIA_CASES
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
