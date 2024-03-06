"""
long prompt handling functions
"""
from typing import Dict, List, Optional

import numpy as np
from gm.helpers import get_batch, get_unique_embedder_keys_from_conditioner
from gm.modules.embedders.open_clip.tokenizer import SimpleTokenizer
from gm.util import expand_dims_like

import mindspore as ms
from mindspore import Tensor, ops


def parse_prompt_attention(text):
    """
    parse a string with attention tokens and returns a list of paris: text and its associated weight
    """
    raise NotImplementedError


def get_tokenize_functions(tokenizer, context_length, pad_with_eos=True):
    """
    The tokenizer accepts a string as the input and returns a list of tokenized ids.
    Returns:
        A tokenize function that handles the text prompts exceeding the context length by re-organize it into chunks
        of token ids, where each group has N = context_length.
    """
    if tokenizer is None:
        # use SimpleTokenizer()
        tokenizer = SimpleTokenizer()
        SOT_TEXT = "<start_of_text>"
        EOT_TEXT = "<end_of_text>"
        bos, eos = tokenizer.encoder[SOT_TEXT], tokenizer.encoder[EOT_TEXT]
    else:
        bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id

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
        pooled_text_embeddings = []  # a placeholder for pooled text embeddings
        if isinstance(group_token_ids, (list, tuple)):
            # the group_token_ids is a list of tensors, where each tensor's shape is (n_chunks, context_len)
            for token_ids in group_token_ids:
                text_embed = get_prompt_text_embedding(token_ids)
                pooled_text_embedding = None
                if isinstance(text_embed, (list, tuple)):
                    if len(text_embed) == 2:
                        # text embedding and pooled_text_embedding
                        pooled_text_embedding = text_embed[1][-1]  # always take the last chunk's pooled_text_embedding
                        text_embed = text_embed[0]
                    else:
                        raise ValueError(
                            f"Expect to have one text embedding and one pooled text embedding, but got {len(text_embed)} embeddings!"
                        )
                text_embeddings.append(text_embed)
                if pooled_text_embedding is not None:
                    pooled_text_embeddings.append(pooled_text_embedding)

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
            pooled_text_embeddings = None
            if isinstance(text_embeddings, (list, tuple)):
                if len(text_embeddings) == 2:
                    # text embedding and pooled_text_embedding
                    pooled_text_embeddings = text_embeddings[1]
                    text_embeddings = text_embeddings[0]
                else:
                    raise ValueError(
                        f"Expect to have one text embedding and one pooled text embedding, but got {len(text_embeddings)} embeddings!"
                    )
            text_embeddings = text_embeddings.reshape((bs, n_chunks * context_len, -1))
            if pooled_text_embeddings is not None:
                pooled_text_embeddings = pooled_text_embeddings.reshape((bs, n_chunks, -1))[
                    :, -1
                ]  # always take the last chunk's pooled_text_embedding
            if not return_tensor:
                text_embeddings = [t for t in text_embeddings]
                pooled_text_embeddings = [t for t in pooled_text_embeddings]
        if pooled_text_embeddings is None or len(pooled_text_embeddings) == 0:
            return text_embeddings
        return text_embeddings, pooled_text_embeddings

    return get_prompt_text_embedding, get_prompt_text_embedding_batch


def _force_n_chunks_token_ids(token_ids, pad_token_ids, force_n_chunks):
    if token_ids.shape[0] < force_n_chunks:
        num = force_n_chunks - token_ids.shape[0]
        token_ids = ms.ops.concat([token_ids] + [pad_token_ids] * num, axis=0)
    elif token_ids.shape[0] > force_n_chunks:
        token_ids = token_ids[:force_n_chunks]
    return token_ids


def _text_embedder_tokenize_long_prompt(embedder, input_value, force_n_chunks=None, return_tensor=True):
    tokenizer = embedder.tokenizer if hasattr(embedder, "tokenizer") else None
    context_length = embedder.max_length
    # pad_with_eos=True for FrozenCLIPEmbedder; pad_with_eos=False for FrozenOpenCLIPEmbedder2
    # TODO: A better judgment term
    pad_with_eos = hasattr(embedder, "tokenizer")
    tokenize_func, tokenize_func_batch = get_tokenize_functions(tokenizer, context_length, pad_with_eos=pad_with_eos)
    emb_token = tokenize_func_batch(input_value, return_tensor=return_tensor)
    if force_n_chunks is not None:
        pad_token_ids = tokenize_func("")
        emb_token = [_force_n_chunks_token_ids(ids, pad_token_ids, force_n_chunks) for ids in emb_token]
        if return_tensor:
            emb_token = ms.ops.stack(emb_token, axis=0)
    return emb_token


def _embedding_group_token_ids(model, group_token_ids, force_zero_embeddings=None):
    vector, crossattn, concat = embedding(model, *group_token_ids, force_zero_embeddings=force_zero_embeddings)
    embeddings_dict = {}
    for k, v in zip(("vector", "crossattn", "concat"), (vector, crossattn, concat)):
        if v is not None:
            embeddings_dict[k] = v

    return embeddings_dict


def tokenize(model, batch: Dict, force_n_chunks: Optional[int] = None, return_tensor: bool = True):
    tokens = []
    if force_n_chunks is not None:
        assert force_n_chunks >= 1, "Expect force_n_chunks is at least one."
    for embedder in model.embedders:
        if hasattr(embedder, "input_key") and (embedder.input_key is not None):
            if embedder.legacy_ucg_val is not None:
                batch = model.possibly_get_ucg_val(embedder, batch)
            input_value = batch[embedder.input_key]
            # frozenclip text embedder or FrozenOpenCLIPEmbedder2
            # TODO: a better judgemental condition
            is_text_encoder = hasattr(embedder, "max_length")
            if is_text_encoder:
                emb_token = _text_embedder_tokenize_long_prompt(
                    embedder, input_value, force_n_chunks, return_tensor=return_tensor
                )
            else:
                # ConcatTimestepEmbedderND's tokenize func is an identity func
                emb_token, _ = embedder.tokenize(input_value)
        else:
            raise AttributeError("embedder does not have attribute input_key/input_keys.")

        assert isinstance(
            emb_token, (Tensor, np.ndarray, list, tuple)
        ), f"tokens must be Tensor, np.ndarray or a sequence, but got {type(emb_token)}"

        tokens.append(emb_token)
    return tokens


def embedding(model, *tokens, force_zero_embeddings=None):
    assert len(tokens) == len(model.embedders), (
        f"tokens and model.embedders length is not equal, " f"{len(tokens)}, {len(model.embedders)}"
    )

    vector, crossattn, concat = None, None, None
    if force_zero_embeddings is None:
        force_zero_embeddings = ()

    for i in range(len(model.embedders)):
        embedder = model.embedders[i]
        token = tokens[i]
        token = token if isinstance(token, (list, tuple)) else (token,)
        # TODO: a better judgemental condition
        is_text_encoder = hasattr(embedder, "max_length")
        if is_text_encoder:
            _, get_text_embed_func = get_text_embedding_functions(embedder.construct)
            emb_out = get_text_embed_func(*token, return_tensor=True)
        else:
            emb_out = embedder(*token)

        if not isinstance(emb_out, (list, tuple)):
            emb_out = (emb_out,)
        for emb in emb_out:
            if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                emb = (
                    expand_dims_like(
                        ops.bernoulli((1.0 - embedder.ucg_rate) * ops.ones(emb.shape[0], dtype=emb.dtype)),
                        emb,
                    )
                    * emb
                )
            if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                emb = ops.zeros_like(emb)

            # CONCAT
            # OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
            # KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}
            assert emb.dim() in (2, 3, 4, 5)
            if emb.dim() == 2:  # vector
                if vector is None:
                    vector = emb
                else:
                    vector = ops.concat((vector, emb), 1)
            elif emb.dim() == 3:  # crossattn
                if crossattn is None:
                    crossattn = emb
                else:
                    crossattn = ops.concat((crossattn, emb), 2)
            else:  # concat
                if concat is None:
                    concat = emb
                else:
                    concat = ops.concat((concat, emb), 1)

    return vector, crossattn, concat


def tokenize_embedding(
    model, batch: Dict, batch_uc: Dict, force_zero_embeddings: Optional[List] = None, max_n_chunks: Optional[int] = None
) -> Dict:
    # tokenize
    group_token_ids = tokenize(model, batch, return_tensor=True)
    negative_group_token_ids = tokenize(model, batch_uc, return_tensor=True)
    # ensure the prompts and negative prompts have the same sequence length of their token ids
    n_chunks = group_token_ids[0].shape[1]
    negative_n_chunks = negative_group_token_ids[0].shape[1]
    if n_chunks > negative_n_chunks:
        negative_group_token_ids = tokenize(model, batch_uc, force_n_chunks=n_chunks, return_tensor=True)
    elif n_chunks < negative_n_chunks:
        group_token_ids = tokenize(model, batch_uc, force_n_chunks=negative_n_chunks, return_tensor=True)

    if max_n_chunks is not None:
        assert max_n_chunks >= 1, "Expect that max_n_chunks should be at least one."
        print(
            f"The token length of long text prompts should be no longer than {max_n_chunks*group_token_ids.shape[-1]}. "
            "Longer tokens will be truncated to this length."
        )
        group_token_ids = group_token_ids[:, :max_n_chunks, :]
        negative_group_token_ids = negative_group_token_ids[:, :max_n_chunks, :]

    c = _embedding_group_token_ids(model, group_token_ids)
    uc = _embedding_group_token_ids(model, negative_group_token_ids, force_zero_embeddings)
    return c, uc


def get_unconditional_conditioning(
    model,
    batch_c,
    batch_uc=None,
    force_uc_zero_embeddings=None,
    max_n_chunks: Optional[int] = None,
):
    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    ucg_rates = []
    for embedder in model.embedders:
        ucg_rates.append(embedder.ucg_rate)
        embedder.ucg_rate = 0.0
    c, uc = tokenize_embedding(
        model, batch_c, batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings, max_n_chunks=max_n_chunks
    )

    for embedder, rate in zip(model.embedders, ucg_rates):
        embedder.ucg_rate = rate
    return c, uc


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: List = None,
    batch2model_input: List = None,
    return_latents=False,
    filter=None,
    adapter_states: Optional[List[ms.Tensor]] = None,
    amp_level="O0",
    max_n_chunks: Optional[int] = None,
    init_latent_path=None,  # '/path/to/sdxl_init_latent.npy'
    **kwargs,
):
    """
    Args:
        reference: gm.models.diffusion.DiffusionEngine.do_sample
        max_n_chunks: Optional[int]. If not None, the `max_n_chunks` specifies the maximum number of chunks being
            divided from the long prompt. If None, it puts no constraints on the prompt length.

    """
    print("Sampling")
    if kwargs:
        print(
            "Some key arguments are fed but not supported in the long text prompt sampling function"
            " ".join(list(kwargs.keys()))
        )

    dtype = ms.float32 if amp_level not in ("O2", "O3") else ms.float16

    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    if batch2model_input is None:
        batch2model_input = []

    num_samples = [num_samples]
    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples, dtype=dtype
    )
    for key in batch:
        if isinstance(batch[key], Tensor):
            print(key, batch[key].shape)
        elif isinstance(batch[key], list):
            print(key, [len(i) for i in batch[key]])
        else:
            print(key, batch[key])
    print("Get Condition Done.")

    print("Embedding Starting (long text prompts are supported)...")
    c, uc = get_unconditional_conditioning(
        model.conditioner,
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=force_uc_zero_embeddings,
        max_n_chunks=max_n_chunks,
    )
    print("Embedding Done.")

    for k in c:
        if not k == "crossattn":
            c[k], uc[k] = map(
                lambda y: y[k][: int(np.prod(num_samples))],
                (c, uc)
                # lambda y: y[k][: math.prod(num_samples)], (c, uc)
            )

    additional_model_inputs = {}
    for k in batch2model_input:
        additional_model_inputs[k] = batch[k]

    shape = (np.prod(num_samples), C, H // F, W // F)
    if init_latent_path is not None:
        print("Loading latent noise from ", init_latent_path)
        randn = Tensor(np.load(init_latent_path), ms.float32)
        # assert randn.shape==shape, 'unmatch shape due to loaded noise'
    else:
        randn = Tensor(np.random.randn(*shape), ms.float32)

    print("Sample latent Starting...")
    samples_z = sampler(model, randn, cond=c, uc=uc, adapter_states=adapter_states)
    print("Sample latent Done.")

    print("Decode latent Starting...")
    samples_x = model.decode_first_stage(samples_z)
    samples_x = samples_x.asnumpy()
    print("Decode latent Done.")

    samples = np.clip((samples_x + 1.0) / 2.0, a_min=0.0, a_max=1.0)

    if filter is not None:
        print("Filter Starting...")
        samples = filter(samples)
        print("Filter Done.")

    if return_latents:
        return samples, samples_z
    return samples
