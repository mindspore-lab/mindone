import mindspore as ms
from mindspore import nn, mint
from mindspore.train.amp import AMP_BLACK_LIST, AMP_WHITE_LIST, _auto_black_list

try:
    from mindspore.train.amp import _auto_white_list

    NEW_AUTO_WHITE = False
except Exception:
    # API changed since ms2.3-20240219
    from mindspore.train.amp import _auto_mixed_precision_rewrite

    NEW_AUTO_WHITE = True


def auto_mixed_precision(network, amp_level="O0", dtype=ms.float16, custom_fp32_cells=[mint.nn.GroupNorm]):
    """
    auto mixed precision function.

    Args:
        network (Cell): Definition of the network.
        amp_level (str): Supports ["O0", "O1", "O2", "O3"]. Default: "O0".

            - "O0": Do not change.
            - "O1": Cast the operators in white_list to float16, the remaining operators are kept in float32.
            - "O2": Cast network to float16, keep operators in black_list run in float32,
            - "O3": Cast network to float16.
        dtype: ms.float16 or ms.bfloat16
        custom_fp32_cells: extra cells to keep in fp32 precision in O2, e.g. self-defined LayerNorm

    Raises:
        ValueError: If amp level is not supported.

    Examples:
        >>> from mindspore import amp, nn
        >>> network = LeNet5()
        >>> amp_level = "O1"
        >>> net = amp.auto_mixed_precision(network, amp_level)
    """

    if not isinstance(network, nn.Cell):
        raise TypeError("The network type should be Cell.")

    if amp_level == "O0":
        pass
    elif amp_level == "O1":
        if NEW_AUTO_WHITE:
            return _auto_mixed_precision_rewrite(network, dtype, white_list=AMP_WHITE_LIST)
        else:
            return _auto_white_list(network, AMP_WHITE_LIST, dtype=dtype)
    elif amp_level == "O2":
        try:
            _auto_black_list(
                network,
                AMP_BLACK_LIST + custom_fp32_cells,
                dtype,
            )
        except Exception:
            _auto_black_list(
                network,
                AMP_BLACK_LIST + custom_fp32_cells,
            )
    elif amp_level == "O3":
        network.to_float(dtype)
    else:
        raise ValueError("The amp level {} is not supported".format(amp_level))

    return network
