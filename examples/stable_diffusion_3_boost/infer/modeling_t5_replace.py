import mindspore as ms
from mindspore import Tensor, ops, mint


def t5_layernorm_construct(self, hidden_states):
    variance = mint.mean(hidden_states.to(ms.float32).pow(2), dim=-1, keepdim=True)
    hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
    
    # convert into half-precision if necessary
    if self.weight.dtype in [ms.float16, ms.bfloat16]:
        hidden_states = hidden_states.to(self.weight.dtype)

    return self.weight * hidden_states


def t5_attention_relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).long() * num_buckets
        relative_position = ops.abs(relative_position)
    else:
        relative_position = -ops.minimum(relative_position, ops.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)
    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = (
        max_exact
        + (mint.div(
            ops.log(mint.div(relative_position.float(), max_exact))
            , ops.log(Tensor(max_distance, ms.float32) / max_exact))
            * (num_buckets - max_exact)
        ).long()
    )
    relative_position_if_large = ops.minimum(
        relative_position_if_large, ops.full_like(relative_position_if_large, num_buckets - 1)
    )
    relative_buckets += ops.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets
