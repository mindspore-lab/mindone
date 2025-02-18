import mindspore as ms
from mindspore import nn
from mindspore.train.amp import _auto_black_list

HALF_UNFRIENDLY_LAYERS = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.SiLU,
    nn.GELU,
    nn.Softmax,
    nn.Sigmoid,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.CrossEntropyLoss,
]


def auto_mixed_precision(network, amp_level="O0", dtype=ms.float16):
    """
    auto mixed precision function.

    Args:
        network (Cell): Definition of the network.
        amp_level (str): Supports ["O0", "O1", "O2", "O3"]. Default: "O0".

            - "O0": Do not change.
            - "O1": Cast the operators in white_list to float16, the remaining operators are kept in float32.
            - "O2": Cast network to float16, keep operators in black_list run in float32,
            - "O3": Cast network to float16.

    Raises:
        ValueError: If amp level is not supported.

    Examples:
        >>> network = LeNet5()
        >>> amp_level = "O2"
        >>> net = auto_mixed_precision(network, amp_level, dtype=ms.float16)
    """

    if not isinstance(network, nn.Cell):
        raise TypeError("The network type should be Cell.")

    if amp_level == "O0":
        pass
    elif amp_level == "O1":
        raise NotImplementedError
    elif amp_level == "O2":
        _auto_black_list(network, HALF_UNFRIENDLY_LAYERS, dtype)
    elif amp_level == "O3":
        network.to_float(dtype)
    else:
        raise ValueError("The amp level {} is not supported".format(amp_level))
    return network


def auto_convert_module_dtype(model: nn.Cell, dtype=ms.float16, keep_norm_fp32=True):
    dtype2str_map = {ms.float16: "fp16", ms.bfloat16: "bf16", ms.float32: "fp32"}

    if dtype not in (ms.float16, ms.bfloat16, ms.float32):
        raise ValueError(f"convert_module_dtype, not support dtype: {dtype}")

    if model is not None:
        assert isinstance(model, nn.Cell)

        k_num, c_num = 0, 0
        for _, p in model.parameters_and_names():
            # filter norm parameters
            if keep_norm_fp32 and ("norm" in p.name):
                k_num += 1
            # filter bool/int parameters
            elif p.dtype in (ms.bool_, ms.int32, ms.int64, ms.uint8):
                k_num += 1
            elif p.dtype == dtype:
                c_num += 1
            else:
                c_num += 1
                p.set_dtype(dtype)

        print(f"Convert `{type(model).__name__}` param to {dtype2str_map[dtype]}, keep/modify num {k_num}/{c_num}.")

    return model
