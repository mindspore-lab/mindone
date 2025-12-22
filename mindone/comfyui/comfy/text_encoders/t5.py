import math

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device
from mindspore_patch.utils import dtype_to_max

import mindspore
from mindspore import mint


class T5LayerNorm(mindspore.nn.Cell):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, operations=None):
        super().__init__()
        self.weight = mindspore.Parameter(mint.empty(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def construct(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * mint.rsqrt(variance + self.variance_epsilon)
        return comfy.ops.cast_to_input(self.weight, x) * x


activations = {
    "gelu_pytorch_tanh": lambda a: mint.functional.gelu(a, approximate="tanh"),
    "relu": mint.functional.relu,
}


class T5DenseActDense(mindspore.nn.Cell):
    def __init__(self, model_dim, ff_dim, ff_activation, dtype, operations):
        super().__init__()
        self.wi = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype)
        self.wo = operations.Linear(ff_dim, model_dim, bias=False, dtype=dtype)
        # self.dropout = nn.Dropout(config.dropout_rate)
        self.act = activations[ff_activation]

    def construct(self, x):
        x = self.act(self.wi(x))
        # x = self.dropout(x)
        x = self.wo(x)
        return x


class T5DenseGatedActDense(mindspore.nn.Cell):
    def __init__(self, model_dim, ff_dim, ff_activation, dtype, operations):
        super().__init__()
        self.wi_0 = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype)
        self.wi_1 = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype)
        self.wo = operations.Linear(ff_dim, model_dim, bias=False, dtype=dtype)
        # self.dropout = nn.Dropout(config.dropout_rate)
        self.act = activations[ff_activation]

    def construct(self, x):
        hidden_gelu = self.act(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        # x = self.dropout(x)
        x = self.wo(x)
        return x


class T5LayerFF(mindspore.nn.Cell):
    def __init__(self, model_dim, ff_dim, ff_activation, gated_act, dtype, operations):
        super().__init__()
        if gated_act:
            self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, ff_activation, dtype, operations)
        else:
            self.DenseReluDense = T5DenseActDense(model_dim, ff_dim, ff_activation, dtype, operations)

        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, operations=operations)
        # self.dropout = nn.Dropout(config.dropout_rate)

    def construct(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        # x = x + self.dropout(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(mindspore.nn.Cell):
    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, operations):
        super().__init__()

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype)
        self.k = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype)
        self.v = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype)
        self.o = operations.Linear(inner_dim, model_dim, bias=False, dtype=dtype)
        self.num_heads = num_heads

        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = operations.Embedding(
                self.relative_attention_num_buckets, self.num_heads, dtype=dtype
            )

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(mindspore.long) * num_buckets
            relative_position = mint.abs(relative_position)
        else:
            relative_position = -mint.min(relative_position, mint.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            mint.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(mindspore.long)
        relative_position_if_large = mint.min(
            relative_position_if_large, mint.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += mint.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, dtype):
        """Compute binned relative position bias"""
        context_position = mint.arange(query_length, dtype=mindspore.long)[:, None]
        memory_position = mint.arange(key_length, dtype=mindspore.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket, out_dtype=dtype
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values.contiguous()

    def construct(self, x, mask=None, past_bias=None, optimized_attention=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.dtype)

        if past_bias is not None:
            if mask is not None:
                mask = mask + past_bias
            else:
                mask = past_bias

        out = optimized_attention(q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask)
        return self.o(out), past_bias


class T5LayerSelfAttention(mindspore.nn.Cell):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, operations):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, operations)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, operations=operations)
        # self.dropout = nn.Dropout(config.dropout_rate)

    def construct(self, x, mask=None, past_bias=None, optimized_attention=None):
        output, past_bias = self.SelfAttention(
            self.layer_norm(x), mask=mask, past_bias=past_bias, optimized_attention=optimized_attention
        )
        # x = x + self.dropout(attention_output)
        x += output
        return x, past_bias


class T5Block(mindspore.nn.Cell):
    def __init__(
        self,
        model_dim,
        inner_dim,
        ff_dim,
        ff_activation,
        gated_act,
        num_heads,
        relative_attention_bias,
        dtype,
        operations,
    ):
        super().__init__()
        self.layer = mindspore.nn.CellList()
        self.layer.append(
            T5LayerSelfAttention(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, operations)
        )
        self.layer.append(T5LayerFF(model_dim, ff_dim, ff_activation, gated_act, dtype, operations))

    def construct(self, x, mask=None, past_bias=None, optimized_attention=None):
        x, past_bias = self.layer[0](x, mask, past_bias, optimized_attention)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(mindspore.nn.Cell):
    def __init__(
        self,
        num_layers,
        model_dim,
        inner_dim,
        ff_dim,
        ff_activation,
        gated_act,
        num_heads,
        relative_attention,
        dtype,
        operations,
    ):
        super().__init__()

        self.block = mindspore.nn.CellList(
            [
                T5Block(
                    model_dim,
                    inner_dim,
                    ff_dim,
                    ff_activation,
                    gated_act,
                    num_heads,
                    relative_attention_bias=((not relative_attention) or (i == 0)),
                    dtype=dtype,
                    operations=operations,
                )
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, operations=operations)
        # self.dropout = nn.Dropout(config.dropout_rate)

    def construct(
        self,
        x,
        attention_mask=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=None,
        embeds_info=[],
    ):
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ).expand((attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]))
            mask = mask.masked_fill(mask.to(mindspore.bool), -dtype_to_max(x.dtype))

        intermediate = None
        optimized_attention = optimized_attention_for_device(None, mask=attention_mask is not None, small_input=True)
        past_bias = None

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.block) + intermediate_output

        for i, l in enumerate(self.block):
            x, past_bias = l(x, mask, past_bias, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        return x, intermediate


class T5(mindspore.nn.Cell):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_layers"]
        model_dim = config_dict["d_model"]
        inner_dim = config_dict["d_kv"] * config_dict["num_heads"]

        self.encoder = T5Stack(
            self.num_layers,
            model_dim,
            inner_dim,
            config_dict["d_ff"],
            config_dict["dense_act_fn"],
            config_dict["is_gated_act"],
            config_dict["num_heads"],
            config_dict["model_type"] != "umt5",
            dtype,
            operations,
        )
        self.dtype = dtype
        self.shared = operations.Embedding(config_dict["vocab_size"], model_dim, dtype=dtype)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, embeddings):
        self.shared = embeddings

    def construct(self, input_ids, attention_mask, embeds=None, num_tokens=None, **kwargs):
        if input_ids is None:
            x = embeds
        else:
            x = self.shared(input_ids, out_dtype=kwargs.get("dtype", mindspore.float32))
        if self.dtype not in [mindspore.float32, mindspore.float16, mindspore.bfloat16]:
            x = mint.nan_to_num(x)  # Fix for fp8 T5 base
        return self.encoder(x, attention_mask=attention_mask, **kwargs)
