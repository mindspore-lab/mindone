import inspect
import numpy as np
import pytest
import torch

import mindspore as ms

from transformers import GotOcr2Config, GotOcr2VisionConfig, Qwen2Config

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


class GotOcr2ModelTester:
    def __init__(
        self,
        batch_size=1,
        seq_length=7,
        # common
        is_training=False,
        use_attention_mask=True,
        use_cache=False,
        output_attentions=False,
        # text model (Qwen2)
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        hidden_act="silu",
        max_position_embeddings=256,
        # vision model
        image_size=64,
        patch_size=16,
        vision_hidden_size=32,
        vision_output_channels=16,
        vision_num_hidden_layers=2,
        vision_num_attention_heads=4,
        vision_window_size=2,
        vision_global_attn_indexes=(1,),
        torch_dtype="float32",
        image_token_id=5,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_cache = use_cache
        self.output_attentions = output_attentions

        # text
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings

        # vision
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_hidden_size = vision_hidden_size
        self.vision_output_channels = vision_output_channels
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_window_size = vision_window_size
        self.vision_global_attn_indexes = list(vision_global_attn_indexes)

        self.torch_dtype = torch_dtype
        self.image_token_id = image_token_id

    def _image_seq_len(self):
        # After patch embedding: (H/patch, W/patch) = (image_size // patch_size)
        grid = self.image_size // self.patch_size
        # After two stride-2 convs in the projector: grid //= 2 twice
        grid //= 2
        grid //= 2
        return grid * grid

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
            torch_dtype=self.torch_dtype,
        )

        vision_config = GotOcr2VisionConfig(
            hidden_size=self.vision_hidden_size,
            output_channels=self.vision_output_channels,
            num_hidden_layers=self.vision_num_hidden_layers,
            num_attention_heads=self.vision_num_attention_heads,
            num_channels=3,
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_act="gelu",
            layer_norm_eps=1e-06,
            attention_dropout=0.0,
            initializer_range=1e-10,
            qkv_bias=True,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=self.vision_window_size,
            global_attn_indexes=self.vision_global_attn_indexes,
            mlp_dim=64,
        )

        config = GotOcr2Config(
            vision_config=vision_config,
            text_config=text_config,
            image_token_index=self.image_token_id,
            image_seq_length=self._image_seq_len(),
            torch_dtype=self.torch_dtype,
        )
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        # place exactly L image tokens per sample so it matches the number of image feature vectors
        L = self._image_seq_len()  # = 1 with the chosen geometry
        start = self.seq_length // 2
        for b in range(self.batch_size):
            input_ids[b, start:start + L] = config.image_token_id  # alias for image_token_index via attribute_map

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy([self.batch_size, self.seq_length], vocab_size=2)

        # pixel_values in [-1, 1], shape (B, C, H, W)
        num_channels = 3
        height = width = self.image_size
        pixel_values = ids_numpy([self.batch_size, num_channels, height, width], vocab_size=256)
        pixel_values = (pixel_values.astype(np.float32) / 255.0) * 2.0 - 1.0

        return (config, input_ids, attention_mask, pixel_values)


model_tester = GotOcr2ModelTester()
config, input_ids, attention_mask, pixel_values = model_tester.prepare_config_and_inputs()


TEST_CASES = [
    [  # text-only
        "GotOcr2Model",
        "transformers.GotOcr2Model",
        "mindone.transformers.GotOcr2Model",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        {
            "last_hidden_state": 0,  # Qwen2Model
        },
    ],
    [  # multimodal (OCR)
        "GotOcr2Model",
        "transformers.GotOcr2Model",
        "mindone.transformers.GotOcr2Model",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,  # (B, C, H, W)
        },
        {
            "last_hidden_state": 0,   # Qwen2Model
            "image_hidden_states": -1 # projected vision features (B, L, D)
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

    # pass dtype down if the model accepts it (some Qwen2 impls expose hidden_dtype)
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
