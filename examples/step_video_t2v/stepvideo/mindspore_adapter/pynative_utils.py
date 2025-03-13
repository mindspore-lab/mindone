import mindspore as ms


@ms.jit
def pynative_x_to_dtype(x: ms.Tensor, dtype: ms.Type = ms.float32):
    return x.to(dtype)
