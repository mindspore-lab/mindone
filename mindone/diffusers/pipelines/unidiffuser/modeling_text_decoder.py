from typing import Optional

import numpy as np
from transformers import GPT2Config

import mindspore as ms
from mindspore import mint

from mindone.transformers import GPT2LMHeadModel

from ....transformers.modeling_utils import ModuleUtilsMixin
from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin


# Modified from ClipCaptionModel in https://github.com/thu-ml/unidiffuser/blob/main/libs/caption_decoder.py
class UniDiffuserTextDecoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    """
    Text decoder model for a image-text [UniDiffuser](https://arxiv.org/pdf/2303.06555.pdf) model. This is used to
    generate text from the UniDiffuser image-text embedding.

    Parameters:
        prefix_length (`int`):
            Max number of prefix tokens that will be supplied to the model.
        prefix_inner_dim (`int`):
            The hidden size of the incoming prefix embeddings. For UniDiffuser, this would be the hidden dim of the
            CLIP text encoder.
        prefix_hidden_dim (`int`, *optional*):
            Hidden dim of the MLP if we encode the prefix.
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.
    """

    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.bias", r"h\.\d+\.attn\.masked_bias"]

    @register_to_config
    def __init__(
        self,
        prefix_length: int,
        prefix_inner_dim: int,
        prefix_hidden_dim: Optional[int] = None,
        vocab_size: int = 50257,  # Start of GPT2 config args
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        scale_attn_by_inverse_layer_idx: bool = False,
        reorder_and_upcast_attn: bool = False,
    ):
        super().__init__()

        self.wte_lm_share = False
        self.prefix_length = prefix_length

        if prefix_inner_dim != n_embd and prefix_hidden_dim is None:
            raise ValueError(
                f"`prefix_hidden_dim` cannot be `None` when `prefix_inner_dim`: {prefix_hidden_dim} and"
                f" `n_embd`: {n_embd} are not equal."
            )

        self.prefix_inner_dim = prefix_inner_dim
        self.prefix_hidden_dim = prefix_hidden_dim

        self.encode_prefix = (
            mint.nn.Linear(self.prefix_inner_dim, self.prefix_hidden_dim)
            if self.prefix_hidden_dim is not None
            else mint.nn.Identity()
        )
        self.decode_prefix = (
            mint.nn.Linear(self.prefix_hidden_dim, n_embd) if self.prefix_hidden_dim is not None else mint.nn.Identity()
        )

        gpt_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )
        self.transformer = GPT2LMHeadModel(gpt_config)

    def construct(
        self,
        input_ids: ms.Tensor,
        prefix_embeds: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
    ):
        """
        Args:
            input_ids (`ms.Tensor` of shape `(N, max_seq_len)`):
                Text tokens to use for inference.
            prefix_embeds (`ms.Tensor` of shape `(N, prefix_length, 768)`):
                Prefix embedding to preprend to the embedded tokens.
            attention_mask (`ms.Tensor` of shape `(N, prefix_length + max_seq_len, 768)`, *optional*):
                Attention mask for the prefix embedding.
            labels (`ms.Tensor`, *optional*):
                Labels to use for language modeling.
        """
        embedding_text = self.transformer.transformer.wte(input_ids)
        hidden = self.encode_prefix(prefix_embeds)
        prefix_embeds = self.decode_prefix(hidden)
        embedding_cat = mint.cat((prefix_embeds, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(input_ids.shape[0])
            labels = mint.cat((dummy_token, input_ids), dim=1)
        out = self.transformer(inputs_embeds=embedding_cat, labels=labels, attention_mask=attention_mask)
        if self.prefix_hidden_dim is not None:
            return out, hidden
        else:
            return out

    def get_dummy_token(self, batch_size: int) -> ms.Tensor:
        return mint.zeros(batch_size, self.prefix_length, dtype=ms.int64)

    def encode(self, prefix):
        return self.encode_prefix(prefix)

    def generate_captions(self, features, eos_token_id):
        """
        Generate captions given text embedding features. Returns list[L].

        Args:
            features (`ms.Tensor` of shape `(B, L, D)`):
                Text embedding features to generate captions from.
            eos_token_id (`int`):
                The token ID of the EOS token for the text decoder model.

        Returns:
            `List[str]`: A list of strings generated from the decoder model.
        """

        features = mint.split(features, 1, dim=0)
        generated_tokens = []
        generated_seq_lengths = []
        for feature in features:
            feature = self.decode_prefix(feature)  # back to the clip feature
            # Only support beam search for now
            output_tokens, seq_lengths = self.generate_beam(input_embeds=feature, eos_token_id=eos_token_id)
            generated_tokens.append(output_tokens[0])
            generated_seq_lengths.append(seq_lengths[0])
        generated_tokens = mint.stack(generated_tokens)
        generated_seq_lengths = mint.stack(generated_seq_lengths)
        return generated_tokens, generated_seq_lengths

    def set_share_weight(self, flag):
        self.wte_lm_share = flag

    def generate_beam(
        self,
        input_ids=None,
        input_embeds=None,
        beam_size: int = 5,
        entry_length: int = 67,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
    ):
        """
        Generates text using the given tokenizer and text prompt or token embedding via beam search. This
        implementation is based on the beam search implementation from the [original UniDiffuser
        code](https://github.com/thu-ml/unidiffuser/blob/main/libs/caption_decoder.py#L89).

        Args:
            eos_token_id (`int`, *optional*):
                The token ID of the EOS token for the text decoder model.
            input_ids (`ms.Tensor` of shape `(batch_size, input_ids_length)`, *optional*):
                Tokenizer indices of input sequence tokens in the vocabulary. One of `input_ids` and `input_embeds`
                must be supplied.
            input_embeds (`ms.Tensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                An embedded representation to directly pass to the transformer as a prefix for beam search. One of
                `input_ids` and `input_embeds` must be supplied.
            beam_size (`int`, *optional*, defaults to `5`):
                The number of best states to store during beam search.
            entry_length (`int`, *optional*, defaults to `67`):
                The number of iterations to run beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The temperature to use when performing the softmax over logits from the decoding model.

        Returns:
            `Tuple(ms.Tensor, ms.Tensor)`: A tuple of tensors where the first element is a tensor of generated
            token sequences sorted by score in descending order, and the second element is the sequence lengths
            corresponding to those sequences.
        """
        # Generates text until stop_token is reached using beam search with the desired beam size.
        stop_token_index = eos_token_id
        tokens = None
        scores = None
        seq_lengths = mint.ones(beam_size, dtype=ms.int32)
        is_stopped = mint.zeros(beam_size, dtype=ms.bool_)

        if input_embeds is not None:
            generated = input_embeds
        else:
            generated = self.transformer.transformer.wte(input_ids)

        for i in range(entry_length):
            outputs = self.transformer(inputs_embeds=generated)
            logits = outputs[0]
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand((beam_size, *generated.shape[1:]))
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand((beam_size, *tokens.shape[1:]))
                    tokens = mint.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = mint.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            next_token_embed = self.transformer.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = mint.cat((generated, next_token_embed), dim=1)
            is_stopped = mint.add(is_stopped, next_tokens.eq(stop_token_index).squeeze())
            if is_stopped.all():
                break

        scores = scores / seq_lengths
        order = scores.argsort(descending=True)
        # tokens tensors are already padded to max_seq_length
        output_texts = [tokens[i] for i in order]
        output_texts = mint.stack(output_texts, dim=0)
        seq_lengths = ms.tensor([seq_lengths[i] for i in order], dtype=seq_lengths.dtype)
        return output_texts, seq_lengths
