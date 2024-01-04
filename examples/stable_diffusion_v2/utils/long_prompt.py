"""
long prompt handling functions
"""
from typing import List, Optional, Union

import mindspore as ms


def parse_prompt_attention(text):
    """
    parse a string with attention tokens and returns a list of paris: text and its associated weight
    """
    raise NotImplementedError


def get_tokenize_functions(tokenizer, context_length, pad_with_eos=False):
    """
    The tokenizer accepts a string as the input and returns a list of tokenized ids.
    Returns:
        A tokenize function that handles the text prompts exceeding the context length by re-organize it into chunks
        of token ids, where each group has N = context_length.
    """
    SOT_TEXT = tokenizer.sot_text
    EOT_TEXT = tokenizer.eot_text
    bos, eos = tokenizer.encoder[SOT_TEXT], tokenizer.encoder[EOT_TEXT]

    def get_prompt_token_ids(prompt: str):
        """
        Args:
            prompt, string.
        Returns:
            token_ids, ms.Tensor, shape is (n_chunks, context_length)
        """
        if len(prompt) > 0:
            token_ids = tokenizer.encode(prompt)
        else:
            token_ids = [bos, eos]  # simple tokenizer cannot handle empty string
        if token_ids[0] == bos:
            token_ids.pop(0)
        if token_ids[-1] == eos:
            token_ids.pop(-1)
        new_token_ids = []
        while len(token_ids) >= context_length - 2:
            temp_token_ids = [bos] + [token_ids.pop(0) for _ in range(75)] + [eos]
            new_token_ids.append(temp_token_ids)
        # padding the left
        if len(token_ids) > 0 or len(new_token_ids) == 0:
            pad_token_id = eos if pad_with_eos else 0
            pad_len = context_length - 2 - len(token_ids)
            temp_token_ids = (
                [bos] + token_ids + [eos] + [pad_token_id] * pad_len
            )  # diffuser seems to insert padding token before eos
            new_token_ids.append(temp_token_ids)
        new_token_ids = ms.Tensor(new_token_ids, dtype=ms.int64)  # (n_chunks, context_len)
        return new_token_ids

    def get_prompt_token_ids_batch(prompts, return_tensor=True):
        """
        prompts, list of strings, representing a batch of strings. The length of each string may vary.
        return_tensor, bool. If False, the returned item will be a list of tensors, where each tensor has the shape (n_chunks, context_len).
           Note that `n_chunks` may vary.
           If `return_tensor=True`, it will force the number of chunks equals to the maximum of n_chunks in this batch.
           Then the returned tensor shape is (bs, max_n_chunks, context_len)
        """
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]

        group_ids = []
        for prompt in prompts:
            new_token_ids = get_prompt_token_ids(prompt)
            group_ids.append(new_token_ids)

        if return_tensor:
            max_len = max([x.shape[0] for x in group_ids])
            pad_token_ids = get_prompt_token_ids("")
            new_group_ids = []
            for token_ids in group_ids:
                if token_ids.shape[0] < max_len:
                    token_ids = ms.ops.concat([token_ids] + [pad_token_ids] * (max_len - token_ids.shape[0]), axis=0)
                new_group_ids.append(token_ids)
            group_ids = ms.ops.stack(new_group_ids, axis=0)
        return group_ids

    return get_prompt_token_ids, get_prompt_token_ids_batch


def get_text_embedding_functions(text_encoder_embedding_function):
    def get_prompt_text_embedding(token_ids: ms.Tensor):
        """
        Args:
            token_ids, Tensor, shape is (n, context_length)
        """
        return text_encoder_embedding_function(token_ids)  # (n_chunks, context_length, hidden_size)

    def get_prompt_text_embedding_batch(group_token_ids, return_tensor=True):
        """
        Args:
            group_token_ids, Optional[List, Tensor]. If it is a tensor, its shape is (bs, n_chunks, context_len).
                If it is a list, it consists of `bs` tensors, and each tensor's shape is (n_chunks, context_len).
                Note that `n_chunks` may vary.
            return_tensor, bool. If False, the returned item will be a list of tensors, where each tensor has the shape
                (n_chunks, context_len, hidden_size).  Note that `n_chunks` may vary.
                If `return_tensor=True`, it will force the number of chunks equals to the maximum of n_chunks in this batch.
                Then the returned tensor shape is (bs, max_n_chunks, context_len, hidden_size)
        """
        text_embeddings = []
        if isinstance(group_token_ids, (list, tuple)):
            # the group_token_ids is a list of tensors, where each tensor's shape is (n_chunks, context_len)
            for token_ids in group_token_ids:
                text_embed = get_prompt_text_embedding(token_ids)
                text_embeddings.append(text_embed)
            if return_tensor:
                max_len = max([x.shape[0] for x in text_embeddings])
                new_text_embeddings = []
                for text_embed in text_embeddings:
                    if text_embed.shape[0] < max_len:
                        pad_tensor = ms.ops.zeros(
                            (max_len - text_embed.shape[0], text_embed.shape[1], text_embed.shape[2]),
                            dtpe=text_embed.dtype,
                        )
                        text_embed = ms.ops.concat([text_embed, pad_tensor], axis=0)
                    new_text_embeddings.append(text_embed)
                text_embeddings = ms.ops.stack(
                    new_text_embeddings, axis=0
                )  # (bs, max_n_chunks, context_len, hidden_size)
        elif isinstance(group_token_ids, ms.Tensor):
            assert (
                len(group_token_ids.shape) == 3
            ), f"Expect group_token_ids to have three dimensions, but got {len(group_token_ids.shape)} dims"
            bs, n_chunks, context_len = group_token_ids.shape
            group_token_ids = group_token_ids.reshape((bs * n_chunks, context_len))
            text_embeddings = get_prompt_text_embedding(group_token_ids)
            text_embeddings = text_embeddings.reshape((bs, n_chunks * context_len, -1))
            if not return_tensor:
                text_embeddings = [t for t in text_embeddings]

        return text_embeddings

    return get_prompt_text_embedding, get_prompt_text_embedding_batch


def get_text_embeddings(
    model,
    prompts: Union[str, List[str]],
    negative_prompts: Optional[Union[str, List[str]]] = None,
    support_long_prompts: Optional[bool] = True,
    return_tensor: Optional[bool] = True,
    max_n_chunks: Optional[int] = None,
):
    """
    Handling long text prompts and extracting text embeddings.
    Args:
        model: diffusion model from mindone.
        prompts: Union[str, List[str]]. A string or a list of strings as the conditions for image generation.
        negative_prompts: Optional[Union[str, List[str]]]. The unconditional text prompts. If provided, will
           return the text embeddings of the negative_prompts.
        support_long_prompts: Optional[bool]. Whether to support long prompts. If False, the token ids of each
            long text prompt will be truncated to the context length, so that the token ids' shape will be
            (context_len, ). If True, the token ids of each long text prompt will be divided into chunks. Thus
            the token ids' shape will be 2d (n_chunks, context_len).
        return_tensor: Optional[bool]. Whether to return tensor for both the token ids and the text embeddings.
            If False, it will tolerate the case where each prompt in the batch may have different lengths. If
            True, it will always pad the token ids or the text embeddings to the longest length in the batch.
            It only applies when support_long_prompts is True.
        max_n_chunks: Optional[int]. If not None, the `max_n_chunks` specifies the maximum number of chunks being
            divided from the long prompt. If None, it puts no constraints on the prompt length. It only applies
            when support_long_prompts is True.
    """
    if not isinstance(prompts, (list, tuple)):
        prompts = [prompts]
    if negative_prompts is not None:
        if not isinstance(negative_prompts, (list, tuple)):
            negative_prompts = [negative_prompts]
        assert len(prompts) == len(
            negative_prompts
        ), "prompts and negative_prompts should have the same number of prompts in a batch!"

    if not support_long_prompts:
        #  will truncate prompts to the context len if support_long_prompts=False
        tokenized_prompts = model.tokenize(prompts)
        c = model.get_learned_conditioning(tokenized_prompts)
        if negative_prompts is not None:
            negative_tokenized_prompts = model.tokenize(negative_prompts)
            uc = model.get_learned_conditioning(negative_tokenized_prompts)
            assert c.shape == uc.shape, "text embeddings and negative text embeddings have different shapes!"
        else:
            uc = None
    else:
        tokenizer = model.cond_stage_model.tokenizer
        context_length = model.cond_stage_model.context_length
        tokenize_func, tokenize_func_batch = get_tokenize_functions(tokenizer, context_length, pad_with_eos=False)
        _, text_embedding_func_batch = get_text_embedding_functions(model.get_learned_conditioning)
        group_token_ids = tokenize_func_batch(prompts, return_tensor=return_tensor)
        pad_token_ids = tokenize_func("")  # (1, 77)
        if negative_prompts is not None:
            negative_group_token_ids = tokenize_func_batch(negative_prompts, return_tensor=return_tensor)
            # ensure that token ids of the prompts and negative prompts have the same shape
            new_group_token_ids, new_negative_group_token_ids = [], []
            for i in range(len(group_token_ids)):
                token_ids, negative_token_ids = group_token_ids[i], negative_group_token_ids[i]
                if token_ids.shape[0] > negative_token_ids.shape[0]:
                    num = token_ids.shape[0] - negative_token_ids.shape[0]
                    negative_token_ids = ms.ops.concat([negative_token_ids] + [pad_token_ids] * num, axis=0)
                elif token_ids.shape[0] < negative_token_ids.shape[0]:
                    num = negative_token_ids.shape[0] - token_ids.shape[0]
                    token_ids = ms.ops.concat([token_ids] + [pad_token_ids] * num, axis=0)
                assert token_ids.shape == negative_token_ids.shape
                new_group_token_ids.append(token_ids)
                new_negative_group_token_ids.append(negative_token_ids)
            group_token_ids = new_group_token_ids
            negative_group_token_ids = new_negative_group_token_ids
            if return_tensor:
                group_token_ids = ms.ops.stack(group_token_ids, axis=0)
                negative_group_token_ids = ms.ops.stack(negative_group_token_ids, axis=0)

        if max_n_chunks is not None:
            assert max_n_chunks >= 1, "Expect that max_n_chunks should be at least one."
            print(
                f"The token length of long text prompts should be no longer than {max_n_chunks*group_token_ids.shape[-1]}. "
                "Longer tokens will be truncated to this length."
            )
            assert max_n_chunks >= 1, "Expect that max_n_chunks should be at least one."
            if return_tensor:
                group_token_ids = group_token_ids[:, :max_n_chunks, :]
                if negative_prompts is not None:
                    negative_group_token_ids = negative_group_token_ids[:, :max_n_chunks, :]
            else:
                group_token_ids = [ids[:max_n_chunks] for ids in group_token_ids]
                if negative_prompts is not None:
                    negative_group_token_ids = [ids[:max_n_chunks] for ids in negative_group_token_ids]

        c = text_embedding_func_batch(group_token_ids, return_tensor=return_tensor)
        if negative_prompts is not None:
            uc = text_embedding_func_batch(negative_group_token_ids)
            assert c.shape == uc.shape, "text embeddings and negative text embeddings have different shapes!"
        else:
            uc = None

    return c, uc
