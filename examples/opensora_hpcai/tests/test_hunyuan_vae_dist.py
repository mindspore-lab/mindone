from typing import Callable

import numpy as np
import pytest
import yaml
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.models.hunyuan_vae import CausalVAE3D_HUNYUAN
from opensora.models.hunyuan_vae.distributed import Conv3dTPCol, Conv3dTPRow, GroupNormTP

from mindspore import GRAPH_MODE, PYNATIVE_MODE, Tensor
from mindspore import dtype as mstype
from mindspore import mint, ops, set_context, set_seed, tensor
from mindspore.communication import GlobalComm, get_group_size, get_rank, init

TOLERANCES = {
    "fp32": {"rtol": 1.3e-6, "atol": 1e-5},
    "fp16": {"rtol": 1e-3, "atol": 1e-5},
    "bf16": {"rtol": 1.6e-2, "atol": 1e-5},
}

DTYPES = {"fp32": mstype.float32, "fp16": mstype.float16, "bf16": mstype.bfloat16}

SEED = 42


@pytest.fixture(scope="module")
def gn_input() -> Tensor:
    set_seed(SEED)
    return tensor(np.random.randn(1, 128, 2, 32, 32), dtype=mstype.float32)


@pytest.fixture(scope="module")
def vanilla_gn(gn_input: Tensor) -> dict[str, np.ndarray]:
    def _gn(x: Tensor, dtype: mstype) -> np.ndarray:
        return GroupNormTP(32, 128, dtype=dtype).to_float(dtype)(x).asnumpy()

    return {k: _gn(gn_input, v) for k, v in DTYPES.items()}


@pytest.fixture(scope="module")
def conv3d_input() -> Tensor:
    set_seed(SEED)
    return tensor(np.random.randn(1, 32, 3, 258, 258), dtype=mstype.float32)


@pytest.fixture(scope="module")
def vanilla_conv3d(conv3d_input: Tensor) -> dict[str, np.ndarray]:
    def _conv3d(x: Tensor, dtype: mstype) -> np.ndarray:
        set_seed(SEED)
        return mint.nn.Conv3d(32, 128, 3, dtype=dtype).to_float(dtype)(x).asnumpy()

    return {k: _conv3d(conv3d_input, v) for k, v in DTYPES.items()}


@pytest.fixture(scope="module")
def vae_encode_input() -> Tensor:
    set_seed(SEED)
    return tensor(np.random.randn(1, 3, 33, 256, 256), dtype=mstype.float32)


@pytest.fixture(scope="module")
def vae_decode_input() -> Tensor:
    set_seed(SEED)
    return tensor(np.random.randn(1, 16, 5, 32, 32), dtype=mstype.float32)


@pytest.fixture(scope="module")
def vae_config() -> dict:
    config = "configs/opensora-v2-0/ae/hunyuan_vae.yaml"
    with open(config, "r") as file:
        config = yaml.safe_load(file)
    return config


@pytest.fixture(scope="module")
def vanilla_vae_encode(vae_encode_input: Tensor, vae_config: dict) -> dict[str, np.ndarray]:
    def _vae(x: Tensor, dtype: mstype) -> np.ndarray:
        vae_config["dtype"] = dtype
        vae = CausalVAE3D_HUNYUAN(**vae_config).set_train(False)
        return vae.encode(x.to(DTYPES[dtype])).asnumpy()

    return {k: _vae(vae_encode_input, k) for k in DTYPES.keys()}


@pytest.fixture(scope="module")
def vanilla_vae_decode(vae_decode_input: Tensor, vae_config: dict) -> dict[str, np.ndarray]:
    def _vae(x: Tensor, dtype: mstype) -> np.ndarray:
        vae_config["dtype"] = dtype
        vae = CausalVAE3D_HUNYUAN(**vae_config).set_train(False)
        return vae.decode(x.to(DTYPES[dtype])).asnumpy()

    return {k: _vae(vae_decode_input, k) for k in DTYPES.keys()}


@pytest.fixture(scope="module")
def init_parallel() -> tuple[int, int]:
    init()
    set_sequence_parallel_group(GlobalComm.WORLD_COMM_GROUP)
    return get_group_size(GlobalComm.WORLD_COMM_GROUP), get_rank(GlobalComm.WORLD_COMM_GROUP)


@pytest.fixture(scope="module")
def all_gather() -> Callable[[Tensor], Tensor]:
    all_gather_op = ops.AllGather(GlobalComm.WORLD_COMM_GROUP)

    def _all_gather(x: Tensor) -> Tensor:
        x = x.swapaxes(0, 1)
        x = all_gather_op(x)
        return x.swapaxes(1, 0)

    return _all_gather


@pytest.mark.parametrize("dtype", list(DTYPES.keys()))
@pytest.mark.parametrize("mode", [PYNATIVE_MODE, GRAPH_MODE], ids=["PyNative", "Graph"])
def test_groupnorm_tp(gn_input, vanilla_gn, init_parallel, all_gather, dtype, mode):
    set_context(mode=mode)
    tp_size, rank = init_parallel

    seq_len = gn_input.shape[1] // tp_size
    gn_input = gn_input[:, rank * seq_len : (rank + 1) * seq_len]

    gn = GroupNormTP(32, 128, dtype=DTYPES[dtype], enable_tp=True).split_weights().to_float(DTYPES[dtype])
    out = all_gather(gn(gn_input)).asnumpy()
    assert np.allclose(
        out.astype(np.float32), vanilla_gn[dtype].astype(np.float32), **TOLERANCES[dtype]
    ), f"GroupNormTP {dtype} test failed. Max error: {np.abs(out - vanilla_gn[dtype]).max()}"


@pytest.mark.parametrize("dtype", list(DTYPES.keys()))
@pytest.mark.parametrize("mode", [PYNATIVE_MODE, GRAPH_MODE], ids=["PyNative", "Graph"])
def test_conv3d_col(conv3d_input, vanilla_conv3d, all_gather, dtype, mode):
    set_context(mode=mode)
    set_seed(SEED)

    conv3d = Conv3dTPCol(32, 128, 3, dtype=DTYPES[dtype]).split_weights().to_float(DTYPES[dtype])
    out = all_gather(conv3d(conv3d_input)).asnumpy()
    assert np.allclose(
        out.astype(np.float32), vanilla_conv3d[dtype].astype(np.float32), **TOLERANCES[dtype]
    ), f"Conv3dTPCol {dtype} test failed. Max error: {np.abs(out - vanilla_conv3d[dtype]).max()}"


@pytest.mark.parametrize("dtype", list(DTYPES.keys()))
@pytest.mark.parametrize("mode", [PYNATIVE_MODE, GRAPH_MODE], ids=["PyNative", "Graph"])
def test_conv3d_row(conv3d_input, vanilla_conv3d, init_parallel, dtype, mode):
    set_context(mode=mode)
    set_seed(SEED)
    tp_size, rank = init_parallel

    in_channels = conv3d_input.shape[1] // tp_size
    conv3d_input = conv3d_input[:, rank * in_channels : (rank + 1) * in_channels]

    conv3d = Conv3dTPRow(32, 128, 3, dtype=DTYPES[dtype]).split_weights().to_float(DTYPES[dtype])
    out = conv3d(conv3d_input).asnumpy()
    assert np.allclose(
        out.astype(np.float32), vanilla_conv3d[dtype].astype(np.float32), **TOLERANCES[dtype]
    ), f"Conv3dTPRow {dtype} test failed. Max error: {np.abs(out - vanilla_conv3d[dtype]).max()}"


@pytest.mark.parametrize("dtype", list(DTYPES.keys()))
@pytest.mark.parametrize("mode", [PYNATIVE_MODE, GRAPH_MODE], ids=["PyNative", "Graph"])
def test_vae_encode(vae_encode_input, vanilla_vae_encode, vae_config, dtype, mode):
    set_context(mode=mode)

    vae_config["dtype"] = dtype
    vae = CausalVAE3D_HUNYUAN(**vae_config).set_train(False)
    out = vae.encode(vae_encode_input.to(DTYPES[dtype])).asnumpy()
    assert np.allclose(
        out.astype(np.float32), vanilla_vae_encode[dtype].astype(np.float32), **TOLERANCES[dtype]
    ), f"VAE Encode {dtype} test failed. Max error: {np.abs(out - vanilla_vae_encode[dtype]).max()}"


@pytest.mark.parametrize("dtype", list(DTYPES.keys()))
@pytest.mark.parametrize("mode", [PYNATIVE_MODE, GRAPH_MODE], ids=["PyNative", "Graph"])
def test_vae_decode(vae_decode_input, vanilla_vae_decode, vae_config, dtype, mode):
    set_context(mode=mode)

    vae_config["dtype"] = dtype
    vae = CausalVAE3D_HUNYUAN(**vae_config).set_train(False)
    out = vae.decode(vae_decode_input.to(DTYPES[dtype])).asnumpy()
    assert np.allclose(
        out.astype(np.float32), vanilla_vae_decode[dtype].astype(np.float32), **TOLERANCES[dtype]
    ), f"VAE Decode {dtype} test failed. Max error: {np.abs(out - vanilla_vae_decode[dtype]).max()}"
