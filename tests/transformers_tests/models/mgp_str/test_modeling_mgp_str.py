import inspect

import numpy as np
import pytest
import torch
from transformers import MgpstrConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-2, "fp16": 5e-2, "bf16": 5e-2}
MODES = [1]


class MgpstrModelTester:
    def __init__(
        self,
        batch_size=1,
        # vision
        image_size=(32, 32),
        patch_size=4,
        num_channels=3,
        # model
        hidden_size=64,  # divisible by 8, to satisfy A^3 grouped conv (groups=8)
        num_hidden_layers=2,
        num_attention_heads=8,  # hidden_size // num_attention_heads must be integer
        mlp_ratio=4.0,
        max_token_length=5,
        num_character_labels=11,
        num_bpe_labels=17,
        num_wordpiece_labels=13,
        # regularization
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        torch_dtype="float32",
    ):
        self.batch_size = batch_size

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio

        self.max_token_length = max_token_length
        self.num_character_labels = num_character_labels
        self.num_bpe_labels = num_bpe_labels
        self.num_wordpiece_labels = num_wordpiece_labels

        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias

        self.torch_dtype = torch_dtype

    def get_config(self):
        # A small, fast config for unit tests (keeps the three logits heads small)
        config = MgpstrConfig(
            image_size=list(self.image_size),
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            max_token_length=self.max_token_length,
            num_character_labels=self.num_character_labels,
            num_bpe_labels=self.num_bpe_labels,
            num_wordpiece_labels=self.num_wordpiece_labels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            torch_dtype=self.torch_dtype,
        )
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()

        B = self.batch_size
        C = self.num_channels
        H, W = self.image_size
        pixel_values = ids_numpy([B, C, H, W], vocab_size=256).astype(np.float32)
        pixel_values = (pixel_values / 255.0) * 2.0 - 1.0

        return (config, pixel_values)


model_tester = MgpstrModelTester()
config, pixel_values = model_tester.prepare_config_and_inputs()

TEST_CASES = [
    [
        # backbone encoder-only: compare last_hidden_state
        "MgpstrModel",
        "transformers.MgpstrModel",
        "mindone.transformers.MgpstrModel",
        (config,),
        {},
        (),
        {
            "pixel_values": pixel_values,
        },
        {
            "last_hidden_state": 0,  # BaseModelOutput ordering: index 0 is last_hidden_state
        },
    ],
    [
        # Full STR model with 3 heads: compare the tuple of logits (char/bpe/wp)
        "MgpstrForSceneTextRecognition",
        "transformers.MgpstrForSceneTextRecognition",
        "mindone.transformers.MgpstrForSceneTextRecognition",
        (config,),
        {},
        (),
        {
            "pixel_values": pixel_values,
        },
        {
            "logits": 0,  # MgpstrModelOutput ordering: index 0 is logits (tuple of 3)
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [case + [dtype] + [mode] for case in TEST_CASES for dtype in DTYPE_AND_THRESHOLDS.keys() for mode in MODES],
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
