"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

from math import ceil, log2

import mindspore as ms
from mindspore import nn, ops


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(axis=-1)


# cosine sim linear


class CosineSimLinear(nn.Cell):
    def __init__(
        self,
        dim_in,
        dim_out,
        scale=1.0,
        dtype=ms.float32,
    ):
        super().__init__()
        self.scale = scale
        self.weight = ms.Parameter(ops.randn((dim_in, dim_out), dtype=dtype))

    def construct(self, x):
        x = ops.L2Normalize(axis=-1, epsilon=1e-12)(x)
        w = ops.L2Normalize(axis=0, epsilon=1e-12)(self.weight)
        return (x @ w) * self.scale


# class


class LFQ(nn.Cell):
    def __init__(
        self,
        config,
        return_loss_breakdown=False,
        is_training=False,
        dtype=ms.float32,
    ):
        super(LFQ, self).__init__()

        dim = config.dim
        codebook_size = config.codebook_size
        entropy_loss_weight = config.entropy_loss_weight
        commitment_loss_weight = config.commitment_loss_weight
        diversity_gamma = config.diversity_gamma
        num_codebooks = config.num_codebooks
        keep_num_codebooks_dim = config.keep_num_codebooks_dim
        codebook_scale = config.codebook_scale  # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy = (
            config.frac_per_sample_entropy
        )  # make less than 1. to only use a random fraction of the probs for per sample entropy
        inv_temperature = config.inv_temperature
        soft_clamp_input_value = config.soft_clamp_input_value
        cosine_sim_project_in = config.cosine_sim_project_in
        cosine_sim_project_in_scale = config.cosine_sim_project_in_scale

        # some assert validations

        assert exists(dim) or exists(codebook_size), "either dim or codebook_size must be specified for LFQ"
        assert (
            not exists(codebook_size) or log2(codebook_size).is_integer()
        ), f"your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})"

        codebook_size = default(codebook_size, lambda: 2**dim)
        codebook_dim = int(log2(codebook_size))

        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in_scale = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = CosineSimLinear(dim, codebook_dims, scale=cosine_sim_project_in_scale)
        else:
            project_in_klass = nn.Dense(dim, codebook_dims, dtype=dtype)

        has_projections = dim != codebook_dims
        self.project_in = project_in_klass if has_projections else nn.Identity()
        self.project_out = nn.Dense(codebook_dims, dim, dtype=dtype) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        self.return_loss_breakdown = return_loss_breakdown

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # straight through activation

        self.activation = nn.Identity()

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.0
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # for no auxiliary loss, during inference

        self.mask = ops.pow(2, ops.arange(codebook_dim - 1, -1, -1))

        # temperature
        self.inv_temperature = inv_temperature

        # codes

        all_codes = ops.arange(codebook_size)
        bits = ((all_codes[..., None] & self.mask) != 0).float().astype(dtype)
        self.codebook = self.bits_to_codes(bits)

        # training
        self.is_training = is_training

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(self, indices, project_out=True):
        if not self.keep_num_codebooks_dim:
            # indices = rearrange(indices, '... -> ... 1')
            indices = indices.unsqueeze(-1)

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).float()

        codes = self.bits_to_codes(bits)

        # codes = rearrange(codes, '... c d -> ... (c d)')
        b, h, w, c, d = codes.shape
        codes = codes.reshape(b, h, w, c * d)

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        # codes = rearrange(codes, 'b ... d -> b d ...')
        codes = codes.permute(0, 4, 1, 2, 3)

        return codes

    def _forward(self, x):
        x_shape = x.shape
        # x, ps = pack_one(x, 'b * d')
        b = x.shape[0]
        d = x.shape[-1]
        x = x.reshape(b, -1, d)

        assert x.shape[-1] == self.dim, f"expected dimension of {self.dim} but received {x.shape[-1]}"

        x = self.project_in(x)

        # maybe soft clamp

        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        # split out number of codebooks

        # x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)
        b, n, _ = x.shape
        x = x.reshape(b, n, self.num_codebooks, -1)

        # quantize by eq 3.

        original_input = x

        codebook_value = ops.ones_like(x) * self.codebook_scale
        quantized = ops.where(x > 0, codebook_value, -codebook_value)

        # use straight-through gradients (optionally with custom activation fn) if training

        if self.is_training:
            x = self.activation(x)
            x = x + ops.stop_gradient(quantized - x)
        else:
            x = quantized

        # calculate indices
        indices = ops.sum((x > 0).int() * self.mask.int(), dim=-1)

        # entropy aux loss

        if self.is_training:
            # the same as euclidean distance up to a constant
            # distance = -2 * einsum('... i d, j d -> ... i j', original_input, self.codebook)
            distance = -2 * ops.matmul(original_input, self.codebook.t())

            prob = ops.softmax(-distance * self.inv_temperature, axis=-1)

            b, n, c, d = prob.shape
            prob = prob.reshape(b * n, c, d)

            # whether to only use a fraction of probs, for reducing memory

            if self.frac_per_sample_entropy < 1.0:
                num_tokens = prob.shape[0]
                num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
                rand_mask = ops.randn(num_tokens).argsort(dim=-1) < num_sampled_tokens
                per_sample_probs = prob[rand_mask]
            else:
                per_sample_probs = prob

            # calculate per sample entropy

            per_sample_entropy = entropy(per_sample_probs).mean()

            # distribution over all available tokens in the batch

            avg_prob = ops.mean(per_sample_probs, axis=0)
            codebook_entropy = entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

            entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
        else:
            entropy_aux_loss = ms.Tensor(0.0)
            per_sample_entropy = ms.Tensor(0.0)
            codebook_entropy = ms.Tensor(0.0)

        # commit loss

        if self.is_training:
            commit_loss = ops.mse_loss(original_input, ops.stop_gradient(quantized), reduction="mean")
            commit_loss = commit_loss.mean()
        else:
            commit_loss = ms.Tensor(0.0)

        # complete aux loss

        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        # merge back codebook dim

        # x = rearrange(x, 'b n c d -> b n (c d)')
        b, n, c, d = x.shape
        x = x.reshape(b, n, c * d)

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        x = x.reshape(*x_shape)

        return x, indices, aux_loss

    def construct(
        self,
        x,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        # standardize image or video into (batch, seq, dimension)
        # x = rearrange(x, 'b d ... -> b ... d')
        x = x.permute(0, 2, 3, 4, 1)

        x, indices, aux_loss = self._forward(x=x)

        x = x.permute(0, 4, 1, 2, 3)

        indices = indices.squeeze(-1)

        return (x, indices, aux_loss)


class LFQ2d(LFQ):
    def construct(self, x):
        x = x.permute(0, 2, 3, 1)
        x, indices, aux_loss = self._forward(x=x)
        x = x.permute(0, 3, 1, 2)
        return (x, indices, aux_loss)
