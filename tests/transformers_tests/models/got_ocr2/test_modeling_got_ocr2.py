import numpy as np
import pytest
import torch
from transformers import GotOcr2Config, GotOcr2VisionConfig, Qwen2Config

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
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
        # text model
        vocab_size=99,
        text_hidden_size=128,
        text_num_hidden_layers=2,
        text_num_attention_heads=4,
        text_num_kv_heads=4,
        text_intermediate_size=256,
        text_hidden_act="silu",
        max_position_embeddings=512,
        # vision model
        image_size=64,
        patch_size=16,
        vision_hidden_size=64,
        vision_output_channels=16,
        vision_num_hidden_layers=2,
        vision_num_attention_heads=4,
        vision_mlp_dim=128,
        window_size=14,
        attn_implementation="eager",
        torch_dtype="float32",
        image_token_index=5,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_cache = use_cache
        self.output_attentions = output_attentions

        # text
        self.vocab_size = vocab_size
        self.text_hidden_size = text_hidden_size
        self.text_num_hidden_layers = text_num_hidden_layers
        self.text_num_attention_heads = text_num_attention_heads
        self.text_num_kv_heads = text_num_kv_heads
        self.text_intermediate_size = text_intermediate_size
        self.text_hidden_act = text_hidden_act
        self.max_position_embeddings = max_position_embeddings

        # vision
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_hidden_size = vision_hidden_size
        self.vision_output_channels = vision_output_channels
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.window_size = window_size

        self.attn_implementation = attn_implementation
        self.torch_dtype = torch_dtype
        self.image_token_index = image_token_index

        # number of projected image tokens per image
        # grid = (image_size / patch_size), projector halves twice => /4 per dim
        s = self.image_size // self.patch_size
        self.tokens_per_image = (s // 4) ** 2

    def get_config(self):
        text_config = Qwen2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.text_hidden_size,
            num_hidden_layers=self.text_num_hidden_layers,
            num_attention_heads=self.text_num_attention_heads,
            num_key_value_heads=self.text_num_kv_heads,
            intermediate_size=self.text_intermediate_size,
            hidden_act=self.text_hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=self.use_cache,
            attn_implementation=self.attn_implementation,
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
            attention_dropout=0.0,
            qkv_bias=True,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=self.window_size,
            global_attn_indexes=[2, 5, 8, 11],
            mlp_dim=self.vision_mlp_dim,
        )

        config = GotOcr2Config(
            use_cache=self.use_cache,
            vision_config=vision_config,
            text_config=text_config,
            image_token_index=self.image_token_index,
            image_seq_length=self.tokens_per_image,
            torch_dtype=self.torch_dtype,
        )
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        assert self.tokens_per_image >= 1
        pos0 = self.seq_length // 2
        for b in range(self.batch_size):
            for k in range(self.tokens_per_image):
                pos = pos0 + k
                pos = min(pos, self.seq_length - 1)
                input_ids[b, pos] = config.image_token_index

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy([self.batch_size, self.seq_length], vocab_size=2)

        num_channels, height, width = 3, self.image_size, self.image_size
        pixel_values = ids_numpy([self.batch_size, num_channels, height, width], vocab_size=256)
        pixel_values = (pixel_values.astype(np.float32) / 255.0) * 2.0 - 1.0

        return (config, input_ids, attention_mask, pixel_values)


model_tester = GotOcr2ModelTester()
config, input_ids, attention_mask, pixel_values = model_tester.prepare_config_and_inputs()


TEST_CASES = [
    [  # text-only
        "GotOcr2ForConditionalGeneration",
        "transformers.GotOcr2ForConditionalGeneration",
        "mindone.transformers.GotOcr2ForConditionalGeneration",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        {
            "logits": 0,
        },
    ],
    [  # multimodal
        "GotOcr2ForConditionalGeneration",
        "transformers.GotOcr2ForConditionalGeneration",
        "mindone.transformers.GotOcr2ForConditionalGeneration",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        },
        {
            "logits": 0,
            "image_hidden_states": -1,
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
