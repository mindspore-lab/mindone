import math
from enum import Enum

import numpy as np

import mindspore
from mindspore import Tensor, context, nn, ops
from mindspore.ops import operations as P


def is_pynative():
    """get whether the mode is pynative"""
    mode = context.get_context("mode")
    return mode == context.PYNATIVE_MODE


def _check_llama3_scaling_factor(scaling_factor, max_position_embedding):
    """check llama3 scaling factor"""
    if not isinstance(scaling_factor, dict):
        raise ValueError(
            f"`scaling_factor` must be a dict for {SeqExtendMethod.LLAMA3.value} rope extend method,"
            f" but got {scaling_factor}"
        )

    required_keys = {"factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
    received_keys = set(scaling_factor.keys())

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `scaling_factor` for 'extend_method' LLAMA3': {missing_keys}")
    unused_keys = received_keys - required_keys
    if unused_keys:
        raise KeyError(f"Unrecognized keys in `scaling_factor` for 'extend_method' LLAMA3': {unused_keys}")

    factor = scaling_factor["factor"]
    if not isinstance(factor, (int, float)) or factor < 1.0:
        raise ValueError(f"`scaling_factor`'s factor field must be a float >= 1, got {factor}")

    low_freq_factor = scaling_factor["low_freq_factor"]
    high_freq_factor = scaling_factor["high_freq_factor"]
    if not isinstance(low_freq_factor, (int, float)):
        raise ValueError(f"`scaling_factor`'s low_freq_factor field must be a float, got {low_freq_factor}")
    if not isinstance(high_freq_factor, (int, float)):
        raise ValueError(f"`scaling_factor`'s high_freq_factor field must be a float, got {high_freq_factor}")
    if high_freq_factor < low_freq_factor:
        raise ValueError(
            "`scaling_factor`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
            f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
        )

    original_max_position_embeddings = scaling_factor["original_max_position_embeddings"]
    if not isinstance(original_max_position_embeddings, int):
        raise ValueError(
            "`scaling_factor`'s original_max_position_embeddings field must be an integer, got "
            f"{original_max_position_embeddings}"
        )
    if original_max_position_embeddings >= max_position_embedding:
        raise ValueError(
            "`scaling_factor`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
            f"{original_max_position_embeddings} and max_position_embeddings={max_position_embedding}"
        )


def _check_yarn_scaling_factor(scaling_factor, max_position_embedding):
    """check YARN scaling factor"""
    if not isinstance(scaling_factor, dict):
        raise ValueError(
            f"`scaling_factor` must be a dict for {SeqExtendMethod.YARN.value} rope extend method,"
            f" but got {scaling_factor}"
        )

    required_keys = {"factor", "original_max_position_embeddings", "beta_slow", "beta_fast", "mscale", "mscale_all_dim"}
    received_keys = set(scaling_factor.keys())

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `scaling_factor` for 'extend_method' YARN': {missing_keys}")
    unused_keys = received_keys - required_keys
    if unused_keys:
        raise KeyError(f"Unrecognized keys in `scaling_factor` for 'extend_method' YARN': {unused_keys}")

    factor = scaling_factor["factor"]
    if not isinstance(factor, (int, float)) or factor < 1.0:
        raise ValueError(f"`scaling_factor`'s factor field must be a float >= 1, got {factor}")

    beta_slow = scaling_factor["beta_slow"]
    beta_fast = scaling_factor["beta_fast"]
    if not isinstance(beta_slow, (int, float)):
        raise ValueError(f"`scaling_factor`'s beta_slow field must be a float, got {beta_slow}")
    if not isinstance(beta_fast, (int, float)):
        raise ValueError(f"`scaling_factor`'s beta_fast field must be a float, got {beta_fast}")
    if beta_fast < beta_slow:
        raise ValueError(
            "`scaling_factor`'s beta_fast field must be greater than beta_slow, got beta_fast="
            f"{beta_fast} and beta_slow={beta_slow}"
        )

    original_max_position_embeddings = scaling_factor["original_max_position_embeddings"]
    if not isinstance(original_max_position_embeddings, int):
        raise ValueError(
            "`scaling_factor`'s original_max_position_embeddings field must be an integer, got "
            f"{original_max_position_embeddings}"
        )
    if original_max_position_embeddings > max_position_embedding:
        raise ValueError(
            "`scaling_factor`'s original_max_position_embeddings field must be not larger than max_position_embeddings,"
            f" got {original_max_position_embeddings} and max_position_embeddings={max_position_embedding}"
        )


def _check_linear_scaling_factor(scaling_factor):
    """check LINEAR scaling factor"""
    if not isinstance(scaling_factor, dict):
        raise ValueError(
            f"`scaling_factor` must be a dict for {SeqExtendMethod.LINEAR.value} rope extend method,"
            f" but got {scaling_factor}"
        )
    required_keys = {"factor"}
    received_keys = set(scaling_factor.keys())
    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `scaling_factor` for 'extend_method' LINEAR': {missing_keys}")
    unused_keys = received_keys - required_keys
    if unused_keys:
        raise KeyError(f"Unrecognized keys in `scaling_factor` for 'extend_method' LINEAR': {unused_keys}")
    factor = scaling_factor["factor"]
    if isinstance(factor, (int, float)) or factor < 1.0:
        raise ValueError(f"`scaling_factor`'s factor field must be a float >= 1, got {factor}")


def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """Inverse dim formula to find dim based on number of rotations"""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """Find dim range bounds based on rotations"""
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_linear_ramp_mask(min_, max_, dim):
    if min_ == max_:
        max_ += 0.001  # Prevent singularity

    linear_func = (np.arange(dim, dtype=np.float32) - min_) / (max_ - min_)
    ramp_func = np.clip(linear_func, 0, 1, out=None)
    return ramp_func


class SeqExtendMethod(Enum):
    """Stores the acceptable string identifiers for seq length extend method"""

    PI = "PI"
    NTK = "NTK"
    YARN = "YARN"
    NONE = "None"
    LLAMA3 = "LLAMA3"
    DYNMAIC_NTK = "DYNAMIC_NTK"
    LINEAR = "linear"


class FreqsMgr(nn.Cell):
    r"""freqs_cis manager."""

    def __init__(
        self,
        head_dim,
        seq_length=None,
        max_position_embedding=4096,
        rotary_dtype=mindspore.float16,
        theta=10000,
        scaling_factor=1.0,
        extend_method=SeqExtendMethod.NONE.value,
        parallel_config=None,
        is_dynamic=False,
        limit_not_apply_seq_pipe=False,
    ):
        super().__init__()
        self.is_pynative = is_pynative()
        if seq_length is not None and seq_length > max_position_embedding:
            max_position_embedding = seq_length
        if extend_method == SeqExtendMethod.NTK.value:
            theta *= scaling_factor
        freqs_base = np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32)  # (head_dim // 2, )
        freqs = 1.0 / (theta ** (freqs_base / head_dim))  # (head_dim // 2, )
        mscale = 1.0
        if extend_method == SeqExtendMethod.LINEAR.value:
            _check_linear_scaling_factor(scaling_factor)
            factor = scaling_factor["factor"]
            freqs /= factor

        if extend_method == SeqExtendMethod.YARN.value:
            _check_yarn_scaling_factor(scaling_factor, max_position_embedding)
            factor = scaling_factor["factor"]
            beta_fast = scaling_factor["beta_fast"]
            beta_slow = scaling_factor["beta_slow"]
            base = theta
            original_max_position_embeddings = scaling_factor["original_max_position_embeddings"]
            mscale_all_dim = scaling_factor["mscale_all_dim"]
            mscale_ = scaling_factor["mscale"]

            internal_freq_base = np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32)
            internal_freq = 1.0 / (factor * theta ** (internal_freq_base / head_dim))

            extra_freq_base = np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32)
            extra_freq = 1.0 / (theta ** (extra_freq_base / head_dim))

            low, high = _yarn_find_correction_range(
                beta_fast, beta_slow, head_dim, base, original_max_position_embeddings
            )
            inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, head_dim // 2)
            freqs = internal_freq * (1 - inv_freq_mask) + extra_freq * inv_freq_mask
            mscale = float(_yarn_get_mscale(factor, mscale_) / _yarn_get_mscale(factor, mscale_all_dim))

        if extend_method == SeqExtendMethod.LLAMA3.value:
            _check_llama3_scaling_factor(scaling_factor, max_position_embedding)

            factor = scaling_factor["factor"]
            if factor is None or not isinstance(factor, float) or factor < 1.0:
                raise ValueError(f"`scaling_factor`'s factor field must be a float >= 1, got {factor}")

            factor = scaling_factor["factor"]
            low_freq_factor = scaling_factor["low_freq_factor"]
            high_freq_factor = scaling_factor["high_freq_factor"]
            old_context_len = scaling_factor["original_max_position_embeddings"]

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            new_freqs = []
            for freq in freqs:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / factor)
                else:
                    if low_freq_wavelen == high_freq_wavelen:
                        raise ValueError(
                            f"low_freq_wavelen should not equal high_freq_wavelen, "
                            f"but low_freq_wavelen got {low_freq_wavelen},"
                            f"high_freq_wavelen got {high_freq_wavelen}."
                        )
                    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
            freqs = np.array(new_freqs, dtype=freqs.dtype)

        if extend_method == SeqExtendMethod.PI.value:
            t = np.arange(0, max_position_embedding / scaling_factor, 1 / scaling_factor).astype(np.float32)
        else:
            t = np.arange(0, max_position_embedding, 1).astype(np.float32)

        self.freqs = Tensor(freqs.reshape(1, 1, 1, -1), dtype=rotary_dtype)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb) * mscale  # (seq_len, head_dim)
        freqs_sin = np.sin(emb) * mscale  # (seq_len, head_dim)
        swap_mask = FreqsMgr.get_swap_mask(head_dim)

        if parallel_config is not None and parallel_config.context_parallel > 1:
            self.context_parallel = parallel_config.context_parallel
        else:
            self.context_parallel = 1
        self.head_dim = head_dim
        self.is_dynamic = is_dynamic
        self.freqs_cos = Tensor(freqs_cos, dtype=rotary_dtype)
        self.freqs_sin = Tensor(freqs_sin, dtype=rotary_dtype)
        self.swap_mask = Tensor(swap_mask, dtype=rotary_dtype)

        self.slice = P.StridedSlice()
        self.gather = P.Gather()
        self.tile = P.Tile()
        self.seq_pipe = (
            parallel_config
            and parallel_config.seq_split_num
            and parallel_config.seq_split_num > 1
            and not limit_not_apply_seq_pipe
        )
        if self.seq_pipe:
            self.seq_split_num = parallel_config.seq_split_num
            self.seq_seg_len = seq_length // self.seq_split_num
            np_range = np.arange(self.seq_seg_len)
            self.seq_seg_range = Tensor(np_range, dtype=mindspore.int32)
            self.add_seq = P.Add()

    def construct(self, seq_length=None, position_ids=None, seq_chunk=None):
        """Get freqs_cos and freqs_sin"""
        if position_ids is None:
            if self.seq_pipe:
                seg_seq_range = self.add_seq(self.seq_seg_range, self.seq_seg_len * seq_chunk)
                freqs_cos = self.gather(self.freqs_cos, seg_seq_range, 0)
                freqs_sin = self.gather(self.freqs_sin, seg_seq_range, 0)
            else:
                freqs_cos = self.slice(self.freqs_cos, (0, 0), (seq_length, self.head_dim), (1, 1))
                freqs_sin = self.slice(self.freqs_sin, (0, 0), (seq_length, self.head_dim), (1, 1))
        else:
            bs, seq = position_ids.shape
            freqs = position_ids.reshape((bs, 1, seq, 1)) * self.freqs
            emb = ops.concat((freqs, freqs), axis=-1)
            freqs_cos = ops.cos(emb)
            freqs_sin = ops.sin(emb)
        freqs_cos = freqs_cos.reshape((-1, 1, seq_length, self.head_dim))
        freqs_sin = freqs_sin.reshape((-1, 1, seq_length, self.head_dim))
        return freqs_cos, freqs_sin, self.swap_mask

    def prefill(self, bs, seq_length):
        if self.is_dynamic and not self.is_pynative:
            return self.freqs_cos, self.freqs_sin, self.swap_mask
        freqs_cos = self.tile(self.slice(self.freqs_cos, (0, 0), (seq_length, self.head_dim), (1, 1)), (bs, 1))
        freqs_sin = self.tile(self.slice(self.freqs_sin, (0, 0), (seq_length, self.head_dim), (1, 1)), (bs, 1))
        return freqs_cos, freqs_sin, self.swap_mask

    def increment(self, batch_valid_length):
        indices = batch_valid_length - 1
        freqs_cos = self.gather(self.freqs_cos, indices, 0)
        freqs_sin = self.gather(self.freqs_sin, indices, 0)
        return freqs_cos, freqs_sin, self.swap_mask

    def increment_multi_ids(self, indices):
        indices = indices.reshape(-1)
        freqs_cos = self.gather(self.freqs_cos, indices, 0)
        freqs_sin = self.gather(self.freqs_sin, indices, 0)
        return freqs_cos, freqs_sin, self.swap_mask

    def chunk_with_decode(self, seq_range):
        """Obtain the position encoding of chunks and increments"""
        freqs_cos = self.gather(self.freqs_cos, seq_range, 0)
        freqs_sin = self.gather(self.freqs_sin, seq_range, 0)
        return freqs_cos, freqs_sin, self.swap_mask

    @staticmethod
    def get_swap_mask(head_dim):
        """Swap matrix"""
        zero_block = np.zeros((head_dim // 2, head_dim // 2), dtype=np.float32)
        id_block = np.identity(head_dim // 2, dtype=np.float32)
        return np.block([[zero_block, id_block], [-id_block, zero_block]])
