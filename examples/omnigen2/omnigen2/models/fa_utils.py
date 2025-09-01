"""
Adapted from
https://github.com/Dao-AILab/flash-attention/blob/7321879fde54f09ed94f7f6ce9377e2f4cf1fac0/flash_attn/bert_padding.py#L64
"""
from math import prod

from einops import rearrange, repeat

import mindspore as ms
from mindspore import mint, nn
from mindspore.mint.nn import functional as F


class IndexFirstAxis(nn.Cell):
    @staticmethod
    def construct(input, indices):
        assert input.ndim >= 2
        first_axis_dim, other_shape = input.shape[0], input.shape[1:]  # noqa
        second_dim = prod(other_shape)
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return mint.gather(rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)).reshape(
            -1, *other_shape
        )

    @staticmethod
    def bprop(input, indices, out, dout):
        assert dout.ndim >= 2
        first_axis_dim = input.shape[0]
        other_shape = dout.shape[1:]
        grad_output = rearrange(dout, "b ... -> b (...)")
        grad_input = mint.zeros((first_axis_dim, grad_output.shape[1]), dtype=grad_output.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(first_axis_dim, *other_shape), None  # ops.zeros_like(indices)?


index_first_axis = IndexFirstAxis()  # FIXME


class IndexPutFirstAxis(nn.Cell):
    @staticmethod
    def construct(values, indices, first_axis_dim):
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = mint.zeros((first_axis_dim, *values.shape[1:]), dtype=values.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(values, indices, first_axis_dim, out, dout):
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = dout[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis()  # FIXME


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    # dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=ms.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=ms.int32)
    indices = mint.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(mint.cumsum(seqlens_in_batch, dim=0, dtype=ms.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )
