from typing import Optional

import numpy as np
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

# CLIP model
from transformers import CLIPTokenizer

import mindspore as ms
from mindspore.common.initializer import TruncatedNormal, initializer


class FrozenCLIPEmbedder_CLIPTokenizer(FrozenCLIPEmbedder):
    """
    A wrapper over FrozenCLIPEmbedder that uses CLIPTokenizer as the tokenizer instead of Simple Tokenizer
    """

    def __init__(
        self,
        *args,
        version="openai/clip-vit-large-patch14",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        setattr(self.tokenizer, "context_length", self.context_length)
        # change to pad with zeros, not eos
        self.tokenizer._pad_token = "!"
        assert (
            self.tokenizer.pad_token_id == 0
        ), f"Expect FrozenOpenCLIPEmbedder2 pads with zeros, not {self.tokenizer.pad_token_id}"

    # rewrite the tokenize function
    def tokenize(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.context_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
        )
        tokens = np.array(batch_encoding["input_ids"], np.int32)
        return ms.Tensor(tokens)

    def get_input_embeddings(self) -> ms.Parameter:
        return self.transformer.embedding_table

    def set_input_embeddings(self, new_embedding_table):
        self.transformer.embedding_table = new_embedding_table

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> ms.Parameter:
        """
        Resizes input token embeddings matrix of the `CLIPTextTransformer` if `new_num_tokens != config.vocab_size`.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `ms.Parameter` module of the model without doing anything.

        Return:
            `ms.Parameter`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        # self.config.vocab_size = model_embeds.shape[0]
        self.vocab_size = model_embeds.shape[0]
        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens) -> ms.Parameter:
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings,
        new_num_tokens: Optional[int] = None,
    ) -> ms.Parameter:
        """Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``mindspore.Parameter``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.shape
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        old_dtype = old_embeddings.dtype
        # Build new embeddings
        new_embeddings = ms.Parameter(
            initializer(TruncatedNormal(0.02), [new_num_tokens, old_embedding_dim], dtype=old_dtype)
        )

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.data[:num_tokens_to_copy, :] = old_embeddings.data[:num_tokens_to_copy, :]

        # align the parameter status
        old_name = old_embeddings.name
        old_requires_grad = old_embeddings.requires_grad
        new_embeddings.name = old_name
        new_embeddings.requires_grad = old_requires_grad
        return new_embeddings
