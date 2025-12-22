import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device

import mindspore
from mindspore import mint
from mindspore.mint import functional as F

from mindone.transformers.mindspore_adapter.utils import dtype_to_max


class CLIPAttention(mindspore.nn.Cell):
    def __init__(self, embed_dim, heads, dtype, operations):
        super().__init__()

        self.heads = heads
        self.q_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype)
        self.k_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype)
        self.v_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype)

        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype)

    def construct(self, x, mask=None, optimized_attention=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out = optimized_attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


ACTIVATIONS = {
    "quick_gelu": lambda a: a * mint.sigmoid(1.702 * a),
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda a: F.gelu(a, approximate="tanh"),
}


class CLIPMLP(mindspore.nn.Cell):
    def __init__(self, embed_dim, intermediate_size, activation, dtype, operations):
        super().__init__()
        self.fc1 = operations.Linear(embed_dim, intermediate_size, bias=True, dtype=dtype)
        self.activation = ACTIVATIONS[activation]
        self.fc2 = operations.Linear(intermediate_size, embed_dim, bias=True, dtype=dtype)

    def construct(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CLIPLayer(mindspore.nn.Cell):
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, operations):
        super().__init__()
        self.layer_norm1 = operations.LayerNorm(embed_dim, dtype=dtype)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, operations)
        self.layer_norm2 = operations.LayerNorm(embed_dim, dtype=dtype)
        self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, operations)

    def construct(self, x, mask=None, optimized_attention=None):
        x += self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(mindspore.nn.Cell):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, operations):
        super().__init__()
        self.layers = mindspore.nn.CellList(
            [
                CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, operations)
                for i in range(num_layers)
            ]
        )

    def construct(self, x, mask=None, intermediate_output=None):
        optimized_attention = optimized_attention_for_device(None, mask=mask is not None, small_input=True)

        all_intermediate = None
        if intermediate_output is not None:
            if intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
            if all_intermediate is not None:
                all_intermediate.append(x.unsqueeze(1).clone())

        if all_intermediate is not None:
            intermediate = mint.cat(all_intermediate, dim=1)

        return x, intermediate


class CLIPEmbeddings(mindspore.nn.Cell):
    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, operations=None):
        super().__init__()
        self.token_embedding = operations.Embedding(vocab_size, embed_dim, dtype=dtype)
        self.position_embedding = operations.Embedding(num_positions, embed_dim, dtype=dtype)

    def construct(self, input_tokens, dtype=mindspore.float32):
        return self.token_embedding(input_tokens, out_dtype=dtype) + comfy.ops.cast_to(
            self.position_embedding.weight, dtype=dtype
        )


class CLIPTextModel_(mindspore.nn.Cell):
    def __init__(self, config_dict, dtype, device, operations):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        num_positions = config_dict["max_position_embeddings"]
        self.eos_token_id = config_dict["eos_token_id"]

        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, num_positions=num_positions, dtype=dtype, operations=operations)
        self.encoder = CLIPEncoder(
            num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, operations
        )
        self.final_layer_norm = operations.LayerNorm(embed_dim, dtype=dtype)

    def construct(
        self,
        input_tokens=None,
        attention_mask=None,
        embeds=None,
        num_tokens=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=mindspore.float32,
        embeds_info=[],
    ):
        if embeds is not None:
            x = embeds + comfy.ops.cast_to(self.embeddings.position_embedding.weight, dtype=dtype)
        else:
            x = self.embeddings(input_tokens, dtype=dtype)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ).expand((attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]))
            mask = mask.masked_fill(mask.to(mindspore.bool), -dtype_to_max(x.type))

        causal_mask = mint.full((x.shape[1], x.shape[1]), -dtype_to_max(x.dtype), dtype=x.dtype).triu(1)

        if mask is not None:
            mask += causal_mask
        else:
            mask = causal_mask

        x, i = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)

        if num_tokens is not None:
            pooled_output = x[list(range(x.shape[0])), list(map(lambda a: a - 1, num_tokens))]
        else:
            pooled_output = x[
                mint.arange(x.shape[0]),
                (mint.round(input_tokens).to(dtype=mindspore.int) == self.eos_token_id).int().argmax(dim=-1),
            ]
        return x, i, pooled_output


class CLIPTextModel(mindspore.nn.Cell):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, None, operations)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def construct(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class CLIPVisionEmbeddings(mindspore.nn.Cell):
    def __init__(
        self, embed_dim, num_channels=3, patch_size=14, image_size=224, model_type="", dtype=None, operations=None
    ):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2
        if model_type == "siglip_vision_model":
            self.class_embedding = None
            patch_bias = True
        else:
            num_patches = num_patches + 1
            self.class_embedding = mindspore.Parameter(mint.empty(embed_dim, dtype=dtype))
            patch_bias = False

        self.patch_embedding = operations.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=patch_bias,
            dtype=dtype,
        )

        self.position_embedding = operations.Embedding(num_patches, embed_dim, dtype=dtype)

    def construct(self, pixel_values):
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        if self.class_embedding is not None:
            embeds = mint.cat(
                [comfy.ops.cast_to_input(self.class_embedding, embeds).expand((pixel_values.shape[0], 1, -1)), embeds],
                dim=1,
            )
        return embeds + comfy.ops.cast_to_input(self.position_embedding.weight, embeds)


class CLIPVision(mindspore.nn.Cell):
    def __init__(self, config_dict, dtype, operations):
        super().__init__()
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        model_type = config_dict["model_type"]

        self.embeddings = CLIPVisionEmbeddings(
            embed_dim,
            config_dict["num_channels"],
            config_dict["patch_size"],
            config_dict["image_size"],
            model_type=model_type,
            dtype=dtype,
            operations=operations,
        )
        if model_type == "siglip_vision_model":
            self.pre_layrnorm = lambda a: a
            self.output_layernorm = True
        else:
            self.pre_layrnorm = operations.LayerNorm(embed_dim)
            self.output_layernorm = False
        self.encoder = CLIPEncoder(
            num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, operations
        )
        self.post_layernorm = operations.LayerNorm(embed_dim)

    def construct(self, pixel_values, attention_mask=None, intermediate_output=None):
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)
        # TODO: attention_mask?
        x, i = self.encoder(x, mask=None, intermediate_output=intermediate_output)
        if self.output_layernorm:
            x = self.post_layernorm(x)
            pooled_output = x
        else:
            pooled_output = self.post_layernorm(x[:, 0, :])
        return x, i, pooled_output


class LlavaProjector(mindspore.nn.Cell):
    def __init__(self, in_dim, out_dim, dtype, operations):
        super().__init__()
        self.linear_1 = operations.Linear(in_dim, out_dim, bias=True, dtype=dtype)
        self.linear_2 = operations.Linear(out_dim, out_dim, bias=True, dtype=dtype)

    def construct(self, x):
        return self.linear_2(F.gelu(self.linear_1(x[:, 1:])))


class CLIPVisionModelProjection(mindspore.nn.Cell):
    def __init__(self, config_dict, dtype, operations):
        super().__init__()
        self.vision_model = CLIPVision(config_dict, dtype, operations)
        if "projection_dim" in config_dict:
            self.visual_projection = operations.Linear(
                config_dict["hidden_size"], config_dict["projection_dim"], bias=False
            )
        else:
            self.visual_projection = lambda a: a

        if "llava3" == config_dict.get("projector_type", None):
            self.multi_modal_projector = LlavaProjector(config_dict["hidden_size"], 4096, dtype, operations)
        else:
            self.multi_modal_projector = None

    def construct(self, *args, **kwargs):
        x = self.vision_model(*args, **kwargs)
        out = self.visual_projection(x[2])
        projected = None
        if self.multi_modal_projector is not None:
            projected = self.multi_modal_projector(x[1])

        return (x[0], x[1], out, projected)
