# tests/models/llama/test_modeling_llama.py
import inspect
import unittest

import mindspore as ms
import numpy as np
import torch
from parameterized import parameterized
from transformers import AutoTokenizer, Qwen3Config
from transformers.testing_utils import slow

from mindone.transformers import Qwen3ForCausalLM

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [0, 1]


class Qwen3ModelTester:
    config_class = Qwen3Config

    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
        rms_norm_eps=1e-6,
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
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.rms_norm_eps = rms_norm_eps

        self.head_dim = self.hidden_size // self.num_attention_heads

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = np.tril(np.ones_like(input_ids))

        config = self.get_config()

        # set _attn_implementation to eager because flash-attention is not supported for torch in cpu
        config._attn_implementation = "eager"

        return config, input_ids, input_mask

    def get_config(self):
        return self.config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=False,
            sliding_window=None
        )


class Qwen3ModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = Qwen3ModelTester()
        self.pt_module = "transformers.Qwen3ForCausalLM"
        self.ms_module = "mindone.transformers.Qwen3ForCausalLM"

    @parameterized.expand(
        [(dtype,) + (mode,) for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_forward(self, dtype, mode):
        ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        config, input_ids, input_mask = self.model_tester.prepare_config_and_inputs()
        init_args = (config,)
        init_kwargs = {}
        inputs_args = (input_ids,)
        inputs_kwargs = {"attention_mask": input_mask}
        outputs_map = {"logits": 0}  # key: torch attribute, value: mindspore idx

        (
            pt_model,
            ms_model,
            pt_dtype,
            ms_dtype,
        ) = get_modules(self.pt_module, self.ms_module, dtype, *init_args, **init_kwargs)

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
        self.assertTrue(
            (np.array(diffs) < THRESHOLD).all(),
            f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )

    @parameterized.expand(
        [(dtype,) + (mode,) for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_generate(self, dtype, mode):
        ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        config, input_ids, _ = self.model_tester.prepare_config_and_inputs()
        init_args = (config,)
        init_kwargs = {}
        inputs_args = (input_ids,)
        inputs_kwargs = {"max_new_tokens": 5, "do_sample": False, "use_cache": False}

        (
            pt_model,
            ms_model,
            pt_dtype,
            ms_dtype,
        ) = get_modules(self.pt_module, self.ms_module, dtype, *init_args, **init_kwargs)

        pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
            pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
        )

        if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
            pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
            ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

        with torch.no_grad():
            pt_outputs = pt_model.generate(*pt_inputs_args, **pt_inputs_kwargs)
        ms_outputs = ms_model.generate(*ms_inputs_args, **ms_inputs_kwargs)
        pt_outputs_np, ms_outputs_np = pt_outputs.numpy(), ms_outputs.asnumpy()

        self.assertTrue(
            ms_outputs_np.shape == pt_outputs_np.shape and (ms_outputs_np == pt_outputs_np).all(),
            f"ms_outputs_shape: {ms_outputs_np.shape}, pt_outputs_shape: {pt_outputs_np.shape};"
            f"ms_outputs: {ms_outputs_np}, pt_outputs: {pt_outputs_np}"
        )


class Qwen3IntegrationTest(unittest.TestCase):
    @parameterized.expand(MODES)
    @slow
    def test_model_600m_logits(self, mode):
        ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
        input_ids = ms.tensor([input_ids], ms.int32)
        model.set_train(False)
        out_logits = model(input_ids, use_cache=False)[0].asnumpy()
        # Expected mean on dim = -1
        EXPECTED_MEAN = np.array(
            [[-1.378831, 1.302914, 3.826209, 3.463683, 2.87961, 1.835721, 2.12904, 2.181405]]).astype(np.float32)
        np.testing.assert_allclose(out_logits.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = np.array(
            [4.69043, 4.9242105, 4.710022, 3.205107, 2.268291, 1.6575601, 3.6528485, 3.97992, 3.260471, 2.6474714,
             3.0466843, 4.2294917, 5.744139, 4.893916, 4.4881663, 6.0321455, 7.4055367, 7.3707757, 6.837107, 6.6321025,
             6.711287, 6.3067284, 6.17496, 6.0414357, 6.0991864, 4.697408, 2.328598, 3.6386256, 2.0756602,
             1.981242]).astype(np.float32)
        np.testing.assert_allclose(out_logits[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @parameterized.expand(MODES)
    @slow
    def test_model_600m_generate(self, mode):
        ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        EXPECTED_TEXT_COMPLETION = """100% plain, unflavoured, and unadulterated. It is"""
        prompt = "My favourite condiment is "
        model_name = "Qwen/Qwen3-0.6B-Base"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = Qwen3ForCausalLM.from_pretrained(model_name)
        input_ids = ms.Tensor(tokenizer([prompt], return_tensors="np").input_ids, ms.int32)

        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
