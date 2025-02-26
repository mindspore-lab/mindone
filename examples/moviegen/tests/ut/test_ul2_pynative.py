import os
import sys

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from transformers import T5EncoderModel as T5EncoderModel_PyTorch

import mindspore as ms

# FIXME: remove in future when mindone is ready for install
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from mindone.transformers.models.t5.modeling_t5 import T5EncoderModel, T5LayerNorm

ms.set_context(mode=ms.PYNATIVE_MODE)

fp32_tolerance = 1e-4
fp16_tolerance = 2e-2
bf16_tolerance = 2e-1

test_samples = [
    "Life is like a box of chocolates.",
    "La vie est comme une boîte de chocolat.",
    "Today is Monday.",
    "Aujourd'hui c'est lundi.",
]

tokenizer = AutoTokenizer.from_pretrained("google/ul2", local_files_only=True)
test_samples = tokenizer(test_samples, padding="max_length", return_tensors="np")


@pytest.fixture(scope="function")
def ul2_pt():
    return T5EncoderModel_PyTorch.from_pretrained("google/ul2", local_files_only=True)


@pytest.fixture(scope="function")
def ul2_ms():
    return T5EncoderModel.from_pretrained("google/ul2", local_files_only=True)


def test_fp32(ul2_ms, ul2_pt):
    # set models precision
    ul2_pt.to(torch.float32)

    ms_enc = ul2_ms(
        ms.Tensor(test_samples.input_ids, dtype=ms.int32), ms.Tensor(test_samples.attention_mask, dtype=ms.uint8)
    )
    ms_enc = ms_enc[0].asnumpy().astype(np.float32)
    pt_enc = ul2_pt(torch.tensor(test_samples.input_ids), torch.tensor(test_samples.attention_mask), return_dict=False)
    pt_enc = pt_enc[0].detach().numpy().astype(np.float32)
    assert np.allclose(ms_enc, pt_enc, atol=fp32_tolerance, rtol=0)


def test_fp16(ul2_ms, ul2_pt):
    # set models precision
    ul2_ms = ms.amp.custom_mixed_precision(ul2_ms, black_list=ms.amp.get_black_list() + [T5LayerNorm], dtype=ms.float16)
    ul2_pt.to(torch.float16)

    ms_enc = ul2_ms(
        ms.Tensor(test_samples.input_ids, dtype=ms.int32), ms.Tensor(test_samples.attention_mask, dtype=ms.uint8)
    )
    ms_enc = ms_enc[0].asnumpy().astype(np.float32)
    pt_enc = ul2_pt(torch.tensor(test_samples.input_ids), torch.tensor(test_samples.attention_mask), return_dict=False)
    pt_enc = pt_enc[0].detach().numpy().astype(np.float32)
    assert np.allclose(ms_enc, pt_enc, atol=fp16_tolerance, rtol=0)


def test_bf16(ul2_ms, ul2_pt):
    # set models precision
    ul2_ms = ms.amp.custom_mixed_precision(
        ul2_ms, black_list=ms.amp.get_black_list() + [T5LayerNorm], dtype=ms.bfloat16
    )
    ul2_pt.to(torch.bfloat16)

    ms_enc = ul2_ms(
        ms.Tensor(test_samples.input_ids, dtype=ms.int32), ms.Tensor(test_samples.attention_mask, dtype=ms.uint8)
    )
    ms_enc = ms_enc[0].astype(ms.float32).asnumpy()
    pt_enc = ul2_pt(torch.tensor(test_samples.input_ids), torch.tensor(test_samples.attention_mask), return_dict=False)
    pt_enc = pt_enc[0].detach().to(torch.float32).numpy()
    assert np.allclose(ms_enc, pt_enc, atol=bf16_tolerance, rtol=0)
