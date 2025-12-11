import inspect

import numpy as np
import pytest
import torch
from transformers import InternVLConfig, InternVLVisionConfig, Qwen2Config

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-2, "bf16": 8e-3}
MODES = [1]


class InternVLModelTester:
    def __init__(
        self,
        batch_size=1,
        seq_length=7,
        # common
        is_training=False,
        use_attention_mask=True,
        use_cache=False,
        output_attentions=False,
        # text model
        vocab_size=99,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        hidden_act="silu",
        max_position_embeddings=512,
        # vision model
        image_size=(32, 32),  # 32x32 with 16x16 patches -> 2x2 patches -> 4 tokens
        patch_size=(16, 16),  # removing CLS => 4 -> reshape (2,2)
        downsample_ratio=0.5,  # pixel_shuffle(0.5) => (2,2) -> (1,1) and channels x4 => exactly 1 image feature vector
        # run-time impl
        attn_implementation="eager",
        torch_dtype="float32",
        image_token_id=5,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_cache = use_cache
        self.output_attentions = output_attentions

        self.vocab_size = vocab_size

        # shared dims
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings

        # vision dims
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio

        # impl & dtype
        self.attn_implementation = attn_implementation
        self.torch_dtype = torch_dtype
        self.image_token_id = image_token_id

    def get_config(self):
        text_config = Qwen2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=self.use_cache,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype,
        )

        vision_config = InternVLVisionConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_channels=3,
            image_size=list(self.image_size),
            patch_size=list(self.patch_size),
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype,
        )

        config = InternVLConfig(
            use_cache=self.use_cache,
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=self.image_token_id,
            attn_implementation=self.attn_implementation,
            downsample_ratio=self.downsample_ratio,
            torch_dtype=self.torch_dtype,
        )
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        # place exactly one <image> token per sample so it matches the 1 image feature vector produced
        # (with config above, image_features per sample is 1)
        image_pos = self.seq_length // 2
        input_ids[:, image_pos] = config.image_token_id

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy([self.batch_size, self.seq_length], vocab_size=2)

        # pixel_values in [-1, 1] (consistency with your Idefics3)
        num_channels, height, width = 3, self.image_size[0], self.image_size[1]
        pixel_values = ids_numpy([self.batch_size, num_channels, height, width], vocab_size=256)
        pixel_values = (pixel_values.astype(np.float32) / 255.0) * 2.0 - 1.0

        return (config, input_ids, attention_mask, pixel_values)


model_tester = InternVLModelTester()
config, input_ids, attention_mask, pixel_values = model_tester.prepare_config_and_inputs()


TEST_CASES = [
    [  # text Q&A
        "InternVLModel",
        "transformers.InternVLModel",
        "mindone.transformers.InternVLModel",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        {
            "last_hidden_state": 0,  # text_model, i.e., Qwen2Model
        },
    ],
    [  # VQA (multimodal)
        "InternVLModel",
        "transformers.InternVLModel",
        "mindone.transformers.InternVLModel",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,  # (B, C, H, W) for InternVL
        },
        {
            "last_hidden_state": 0,  # Qwen2Model
            "image_hidden_states": -1,  # Vision Transformer
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
