import numpy as np
import pytest
import torch

import mindspore as ms

from ..modeling_test_utils import compute_diffs, generalized_parse_args, get_modules

THRESHOLD_FP16 = 1e-2
THRESHOLD_FP32 = 5e-3


@pytest.mark.parametrize(
    "name,mode,dtype",
    [
        ["T5FilmDecoder_graph_fp32", 0, "fp32"],
        ["T5FilmDecoder_graph_fp16", 0, "fp16"],
        ["T5FilmDecoder_pynative_fp32", 1, "fp32"],
        ["T5FilmDecoder_pynative_fp16", 1, "fp16"],
    ],
)
def test_t5_film_decoder(name, mode, dtype):
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)

    # init model
    pt_module = "diffusers.models.transformers.t5_film_transformer.T5FilmDecoder"
    ms_module = f"mindone.{pt_module}"

    init_args = ()
    init_kwargs = {
        "input_dims": 32,
        "d_model": 64,
        "num_heads": 4,
        "d_ff": 64,
        "targets_length": 8,
    }

    pt_model, ms_model, pt_dtype, ms_dtype = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)

    # get inputs
    inputs_args = (
        np.random.randn(2, 64, 64).astype(np.float32),
        np.random.randn(2, 64).astype(np.int32),
        np.random.randn(2, 8, 64).astype(np.float32),
        np.random.randn(2, 8).astype(np.int32),
    )
    inputs_kwargs = {
        "decoder_input_tokens": np.random.randn(2, 8, 32).astype(np.float32),
        "decoder_noise_time": np.array([0.99, 0.50]).astype(np.float32),
    }

    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )
    pt_inputs_kwargs["encodings_and_masks"] = (
        (pt_inputs_args[0], pt_inputs_args[1]),
        (pt_inputs_args[2], pt_inputs_args[3]),
    )
    ms_inputs_kwargs["encodings_and_masks"] = (
        (ms_inputs_args[0], ms_inputs_args[1]),
        (ms_inputs_args[2], ms_inputs_args[3]),
    )

    with torch.no_grad():
        pt_outputs = pt_model(**pt_inputs_kwargs)
    ms_outputs = ms_model(**ms_inputs_kwargs)

    diffs = compute_diffs(pt_outputs, ms_outputs)

    eps = THRESHOLD_FP16 if dtype == "fp16" else THRESHOLD_FP32
    assert (np.array(diffs) < eps).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {eps}"
