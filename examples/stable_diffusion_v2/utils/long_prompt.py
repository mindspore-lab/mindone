"""
long prompt handling functions
"""
import mindspore as ms


def get_long_prompts_tokenize_function(tokenizer, context_length, pad_with_eos=False):
    """
    The tokenizer accepts a string as the input and returns a list of tokenized ids.
    Returns:
        A tokenize function that handles the text prompts exceeding the context length by re-organize it into groups
        of token ids, where each group has N = context_length.

    """
    SOT_TEXT = tokenizer.sot_text
    EOT_TEXT = tokenizer.eot_text
    bos, eos = tokenizer.encoder[SOT_TEXT], tokenizer.encoder[EOT_TEXT]

    def get_long_prompts_token_ids(prompts, force_n_groups=None):
        """
        prompts, list of strings.
        force_n_groups, Optional(int). If None, it will divide long prompts (length=N) into different chunks (N//context_len + 1).
           If force_n_groups is provided, it will force the number of chunks equals to force_n_groups.
        """
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        if force_n_groups is not None:
            assert isinstance(force_n_groups, int) and force_n_groups >= 1, "force_n_groups should be at least 1"
        group_ids = []
        for prompt in prompts:
            token_ids = tokenizer.encode(prompt)
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

            if force_n_groups is not None:
                if len(new_token_ids) > force_n_groups:
                    num = len(new_token_ids) - force_n_groups
                    for _ in range(num):
                        new_token_ids.pop(-1)
                elif len(new_token_ids) < force_n_groups:
                    pad_token_id = eos if pad_with_eos else 0
                    num = force_n_groups - len(new_token_ids)
                    temp_token_ids = [bos] + [eos] + [pad_token_id] * (context_length - 2)
                    for _ in range(num):
                        new_token_ids.append(temp_token_ids)
            new_token_ids = ms.Tensor(new_token_ids, dtype=ms.int64)
            group_ids.append(new_token_ids)
        return (
            group_ids  # a list of tensors, where each tensor's shape is (n_groups, context_length). n_groups may vary
        )

    return get_long_prompts_token_ids


def get_long_prompts_text_embedding_function(text_encoder_embedding_function):
    def get_long_prompts_text_embedding(group_token_ids):
        text_embeddings = []
        if not isinstance(group_token_ids, (list, tuple)):
            group_token_ids = [group_token_ids]

        for token_ids in group_token_ids:
            text_embed = text_encoder_embedding_function(token_ids)  # (n_groups, context_length, hidden_size)
            text_embed = ms.ops.concat([x for x in text_embed], axis=0)  # (n_group*context_length, hidden_size)
            text_embeddings.append(text_embed)

        return text_embeddings

    return get_long_prompts_text_embedding
