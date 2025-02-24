# Copyright 2025 StepFun Inc. All Rights Reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter, mint

import numpy as np
from typing import List


class LLaMaEmbedding(nn.Cell):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 cfg,
                 ):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.params_dtype = cfg.params_dtype
        self.fp32_residual_connection = cfg.fp32_residual_connection 
        self.embedding_weights_in_fp32 = cfg.embedding_weights_in_fp32
        self.word_embeddings = mint.nn.Embedding(
            cfg.padded_vocab_size, self.hidden_size,
        )
        self.embedding_dropout = mint.nn.Dropout(p=cfg.hidden_dropout)

    def construct(self, input_ids):
        # Embeddings.
        if self.embedding_weights_in_fp32:
            self.word_embeddings = self.word_embeddings.to(ms.float32)
        embeddings = self.word_embeddings(input_ids)
        if self.embedding_weights_in_fp32:
            embeddings = embeddings.to(self.params_dtype)
            self.word_embeddings = self.word_embeddings.to(self.params_dtype)

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = mint.swapaxes(embeddings, 0, 1)

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        return embeddings



class StepChatTokenizer:
    """Step Chat Tokenizer"""

    def __init__(
        self, model_file, name="StepChatTokenizer",
        bot_token="<|BOT|>",  # Begin of Turn
        eot_token="<|EOT|>",  # End of Turn
        call_start_token="<|CALL_START|>",      # Call Start
        call_end_token="<|CALL_END|>",          # Call End
        think_start_token="<|THINK_START|>",    # Think Start
        think_end_token="<|THINK_END|>",        # Think End
        mask_start_token="<|MASK_1e69f|>",      # Mask start
        mask_end_token="<|UNMASK_1e69f|>",      # Mask end
    ):
        import sentencepiece

        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)

        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        for idx in range(self._tokenizer.get_piece_size()):
            text = self._tokenizer.id_to_piece(idx)
            self._inv_vocab[idx] = text
            self._vocab[text] = idx

            if self._tokenizer.is_control(idx) or self._tokenizer.is_unknown(idx):
                self._special_tokens[text] = idx
                self._inv_special_tokens[idx] = text

        self._unk_id = self._tokenizer.unk_id()
        self._bos_id = self._tokenizer.bos_id()
        self._eos_id = self._tokenizer.eos_id()

        for token in [
            bot_token, eot_token, call_start_token, call_end_token,
            think_start_token, think_end_token
        ]:
            assert token in self._vocab, f"Token '{token}' not found in tokenizer"
            assert token in self._special_tokens, f"Token '{token}' is not a special token"

        for token in [mask_start_token, mask_end_token]:
            assert token in self._vocab, f"Token '{token}' not found in tokenizer"

        self._bot_id = self._tokenizer.piece_to_id(bot_token)
        self._eot_id = self._tokenizer.piece_to_id(eot_token)
        self._call_start_id = self._tokenizer.piece_to_id(call_start_token)
        self._call_end_id = self._tokenizer.piece_to_id(call_end_token)
        self._think_start_id = self._tokenizer.piece_to_id(think_start_token)
        self._think_end_id = self._tokenizer.piece_to_id(think_end_token)
        self._mask_start_id = self._tokenizer.piece_to_id(mask_start_token)
        self._mask_end_id = self._tokenizer.piece_to_id(mask_end_token)

        self._underline_id = self._tokenizer.piece_to_id("\u2581")
        
    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size()

    def tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode_as_ids(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode_ids(token_ids)

    
class Tokens:
    def __init__(self, input_ids, cu_input_ids, attention_mask, cu_seqlens, max_seq_len) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.cu_input_ids = cu_input_ids
        self.cu_seqlens = cu_seqlens
        self.max_seq_len = max_seq_len
    

class Wrapped_StepChatTokenizer(StepChatTokenizer):
    def __call__(self, text, max_length=320, padding="max_length", truncation=True, return_tensors="np"):
        # [bos, ..., eos, pad, pad, ..., pad]
        self.BOS = 1
        self.EOS = 2
        self.PAD = 2
        out_tokens = []
        attn_mask = []
        if len(text) == 0:
            part_tokens = [self.BOS] + [self.EOS]
            valid_size = len(part_tokens)
            if len(part_tokens) < max_length:
                part_tokens += [self.PAD] * (max_length - valid_size)
            out_tokens.append(part_tokens)
            attn_mask.append([1]*valid_size+[0]*(max_length-valid_size))
        else:
            for part in text:
                part_tokens = self.tokenize(part)
                part_tokens = part_tokens[:(max_length - 2)] # leave 2 space for bos and eos
                part_tokens = [self.BOS] + part_tokens + [self.EOS]
                valid_size = len(part_tokens)
                if len(part_tokens) < max_length:
                    part_tokens += [self.PAD] * (max_length - valid_size)
                out_tokens.append(part_tokens)
                attn_mask.append([1]*valid_size+[0]*(max_length-valid_size))

        if return_tensors == "np":

            out_tokens = np.array(out_tokens, dtype=np.int32)
            attn_mask = np.array(attn_mask, dtype=np.int32)

            # padding y based on tp size
            padded_len = 0
            padded_flag = True if padded_len > 0 else False
            if padded_flag:
                pad_tokens = np.array([[self.PAD] * max_length])
                pad_attn_mask = np.array([[1]*padded_len+[0]*(max_length-padded_len)])
                out_tokens = np.concatenate([out_tokens, pad_tokens], axis=0)
                attn_mask = np.concatenate([attn_mask, pad_attn_mask], axis=0)

            # cu_seqlens
            # cu_out_tokens = out_tokens.masked_select(attn_mask != 0).unsqueeze(0)
            cu_out_tokens = out_tokens[attn_mask != 0].reshape((1, -1))

            seqlen = attn_mask.sum(axis=1).tolist()
            cu_seqlens = np.cumsum(np.array([0]+seqlen), 0).astype(dtype=np.int32)
            max_seq_len = max(seqlen)
            
            return Tokens(out_tokens, cu_out_tokens, attn_mask, cu_seqlens, max_seq_len)
        
        else:
            raise NotImplementedError
