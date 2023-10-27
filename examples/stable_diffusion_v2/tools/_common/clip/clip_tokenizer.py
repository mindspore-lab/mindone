"""
CLIP tokenizer
"""
import gzip
import html
from collections import defaultdict
from functools import lru_cache
from typing import List, Optional, Union

import ftfy
import numpy as np
import regex as re

from mindspore import Tensor


def get_pairs(input_wd):
    r"""Get_pairs"""
    output = set()
    prev_char = input_wd[0]
    for char in input_wd[1:]:
        output.add((prev_char, char))
        prev_char = char
    return output


@lru_cache()
def bytes_to_unicode():
    r"""Bytes_to_unicode"""
    input_bt = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    output_cd = input_bt[:]
    num = 0
    for item in range(2**8):
        if item not in input_bt:
            input_bt.append(item)
            output_cd.append(2**8 + num)
            num += 1
    output_cd = [chr(item) for item in output_cd]
    return dict(zip(input_bt, output_cd))


def whitespace_clean(input_text):
    r"""Whitespace clean"""
    input_text = re.sub(r"\s+", " ", input_text)
    input_text = input_text.strip()
    return input_text


def basic_clean(input_text):
    r"""Basic_clean"""
    input_text = ftfy.fix_text(input_text)
    input_text = html.unescape(html.unescape(input_text))
    return input_text.strip()


class TempTokenizer:
    r"""Simple Tokenizer"""

    def __init__(self, merges, vocab, flag_dict, pat):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.flag_dict = flag_dict
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.pat = pat

    def tokenize_alg(self, input_tk):
        r"""Bpe"""
        if input_tk in self.flag_dict:
            return self.flag_dict[input_tk]
        word = tuple(input_tk[:-1]) + (input_tk[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return input_tk + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.flag_dict[input_tk] = word
        return word

    def decode(self, input_ids):
        r"""Decode"""
        output_text = "".join([self.decoder[input_id] for input_id in input_ids])
        output_text = (
            bytearray([self.byte_decoder[c] for c in output_text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return output_text

    def encode(self, content):
        r"""Encode"""
        output_ids = []
        content = whitespace_clean(basic_clean(content)).lower()
        for token in re.findall(self.pat, content):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            output_ids.extend(self.encoder[bpe_token] for bpe_token in self.tokenize_alg(token).split(" "))
        print("res is:", output_ids)
        return output_ids


class CLIPTokenizer:
    r"""
    CLIP Tokenizer

    Args:
        vocab_file (str): File path of vocab.
        eos_token (str): Eos_token.
        bos_token (str): Bos_token.
        pad_token (str): Pad_token.
        unk_token (str): Unk_token.
    """
    MODEL_INPUT_NAME = ["input_ids", "attention_mask"]
    VOCAB_FILES = {"vocab_file": ["vocab.txt", "bpe_simple_vocab_16e6.txt.gz"]}
    FILE_LIST = ["tokenizer_config.json"]
    """clip tokenizer"""

    def __init__(
        self,
        vocab_file: str,
        eos_token: str = "<|endoftext|>",
        bos_token: str = "<|startoftext|>",
        pad_token: str = "<|endoftext|>",
        unk_token: str = "<|endoftext|>",
    ):
        # SpecialTokensMixin
        self._eos_token = eos_token
        self._bos_token = bos_token
        self._pad_token = pad_token
        self._unk_token = unk_token
        self._pad_token_type_id = 0

        self.path = vocab_file
        merges = self._read_merge_files(vocab_file)
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend([bos_token, eos_token])

        flag_dict = {bos_token: bos_token, eos_token: eos_token}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|
        've|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.tool = TempTokenizer(merges, vocab, flag_dict, self.pat)

        self.model_inputs = self.MODEL_INPUT_NAME

    # SpecialTokensMixin
    @property
    def pad_token(self):
        return self._pad_token

    @property
    def pad_token_id(self):
        return self._convert_tokens_to_ids(self._pad_token)

    @property
    def unk_token(self):
        return self._unk_token

    @property
    def unk_token_id(self):
        return self._convert_tokens_to_ids(self._unk_token)

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def eos_token_id(self):
        return self._convert_tokens_to_ids(self._eos_token)

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def bos_token_id(self):
        return self._convert_tokens_to_ids(self._bos_token)

    @property
    def pad_token_type_id(self):
        return self._pad_token_type_id

    # BaseTokenizer
    def __call__(
        self,
        text: Optional[Union[str, List[str]]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: str = False,
        truncation: bool = False,
        return_tensors: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        Tokenize the input string and convert them into the ids.

        Args:
            text(str, list(str)) : To be converted text strings. It can be string or a list of strings.
            text_pair(str, list(str)): To be converted text pair strings. It can be string or a list of strings.
            add_special_tokens(bool): Whether to add special tokens such as CLS and EOS to the token list. The subclass
                can determine the behavior of the adding by overriding the method `build_inputs_with_special_tokens`.
                If True, the special token will be added. Default True.
            max_length (int): max length of tokenizer's output . Default None.
            padding(False / "max_length"): padding for max_length. Default None.
            truncation(bool): To truncate the sequence if the length exceeds the max_length. Default False.
            return_tensors(str): Specific the returned tensor type. If set None, the returned tensor will be
                `numpy.ndarray`. If set `ms`, the returned tensor will be of `mindspore.Tensor`.
                Support 'ms' and None. Default None.
            **kwargs: The other kwargs passed to the internal behaivor, currently not used.

        Outputs:
            A dict contains the processed ids, attention_mask that specific by the member `MODEL_INPUT_NAME`
            of the subclass.
        """
        return_batch = True
        if isinstance(text, str):
            return_batch = False
        output_dict = self.batch_encode_plus(
            text,
            text_pair=text_pair,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            return_batch=return_batch,
            **kwargs,
        )

        return output_dict

    def truncate_sequences(self, ids, id_pairs, nums_tokens_to_remove):
        if nums_tokens_to_remove <= 0:
            return ids, id_pairs
        if id_pairs:
            raise ValueError("The id_pairs do not support truncation, please set it to be a empty list or None.")

        ids = ids[:-nums_tokens_to_remove]
        return ids, id_pairs

    def convert_tokens_to_ids(self, input_tokens):
        """Convert the tokens to ids using vocab mapping"""
        return self._convert_tokens_to_ids(input_tokens)

    def _get_token_ids(self, text):
        """Get the token_ids"""
        if not isinstance(text, list):
            tokens = self.tokenize(text)
            res = self.convert_tokens_to_ids(tokens)
            return res
        output = []
        for item in text:
            tokens = self.tokenize(item)
            res = self.convert_tokens_to_ids(tokens)
            output.append(res)
        if len(text) == 1 and isinstance(text[0], str) and output and isinstance(output[0], list):
            output = output[0]
        return output

    def batch_encode_plus(
        self,
        text: Optional[Union[str, List[str]]],
        text_pair: Optional[Union[str, List[str]]] = None,
        max_length: Optional[int] = None,
        padding: Optional[str] = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[bool] = None,
        return_batch: bool = True,
        **kwargs,
    ):
        r"""
        The core function of the __call__ method. The method aims to tokenizer the input strings and then convert them
        into ids.

        Args:
            text(str) : To be converted text strings. It can be string or a list of strings.
            text_pair(str): To be converted text pair strings. It can be string or a list of strings.
            max_length (int): max length of tokenizers output . Default None.
            padding(bool, str): padding for max_length. Default None.
            truncation(bool): To truncate the sequence if the length exceeds the max_length. Default False.
            add_special_tokens(bool): Whether to add special tokens such as CLS and EOS to the token list. The subclass
                can determine the behavior of the adding by overriding the method `build_inputs_with_special_tokens`.
                If True, the special token will be added. Default True.
            return_token_type_ids(bool): Whether to add `token_type_ids` in the returned dict. If True,
                `token_type_ids` will be added, otherwise not. If None, it will be added if it is in the
                MODEL_INPUT_NAME. Default None.
            return_attention_mask(bool): Whether to add `return_attention_mask` in the returned dict. If True,
                `return_attention_mask` will be added, otherwise not. If None, it will be added if it is in the
                MODEL_INPUT_NAME. Default None.
            return_tensors(str): Specific the returned tensor type. If support `ms` only, the returned value in the
                dict will be converted to the mindspore.Tensor, otherwise it will return the list.
                Default None.
            return_batch(bool): Whether the returned the list should be added batch dimension. Default True.
            **kwargs: The other kwargs passed to the internal method, currently not used.

        Outputs:
            A dict contains the processed ids, attention_mask that specific by the member `MODEL_INPUT_NAME`
            of the subclass.
        """
        if padding and padding != "max_length":
            raise ValueError("padding only supports `max_length` or `None`.")
        padding_strategy = None
        if padding:
            padding_strategy = "max_length"
        if max_length and not padding:
            print("If you want to enable the padding, please set padding to `max_length`.")
        # if input text is only one list, we should prepare it into a tensor with batch size 1.
        text = self._prepare_input_to_list(text)
        text_pair = self._prepare_input_to_list(text_pair)
        tokens = self._batch_encode_plus(
            text,
            text_pair=text_pair,
            max_length=max_length,
            padding_strategy=padding_strategy,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_batch=return_batch,
            **kwargs,
        )

        return tokens

    def _batch_encode_plus(
        self,
        text,
        text_pair=None,
        max_length=None,
        padding_strategy="do_not_pad",
        add_special_tokens=True,
        truncation=False,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_batch=True,
        **kwargs,
    ):
        """Convert the text into the converted id. text should be batched. For example, [["hello world"]]"""
        if not isinstance(text, list) and not isinstance(text[0], list):
            raise ValueError(
                "For _batch_encode_plus, the input `text` should be batched, " "for example: [['hello world']]."
            )

        text_ids = [self._get_token_ids(item) for item in text]
        text_pair_ids = [self._get_token_ids(item) for item in text_pair] if text_pair else None
        processed_output = self._batch_postprocess_ids(
            ids=text_ids,
            pair_ids=text_pair_ids,
            max_length=max_length,
            truncation=truncation,
            padding_strategy=padding_strategy,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_batch=return_batch,
        )
        return processed_output

    def _batch_postprocess_ids(
        self,
        ids,
        pair_ids=None,
        add_special_tokens=True,
        max_length=None,
        truncation=False,
        padding_strategy="do_not_pad",
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_batch=True,
    ):
        """Convert the input_ids to the format of model inputs"""
        if return_tensors and return_tensors != "ms":
            raise ValueError("You should set return_tensors to be `ms`.")
        if not return_batch and len(ids) != 1:
            raise ValueError(
                f"If `return_batch` is False, the length of input ids should be 1. But found {len(ids)}. "
                f"Input ids is: {ids}. To fix this, you can set the return_batch=True"
            )
        if pair_ids:
            paired_ids = zip(ids, pair_ids)
        else:
            paired_ids = zip(ids, [None] * len(ids))
        output = defaultdict(list)
        for per_ids, per_pair_ids in paired_ids:
            per_output = self.postprocess_ids(
                ids=per_ids,
                pair_ids=per_pair_ids,
                padding_strategy="do_not_pad",
                return_tensors=None,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
            )
            if not return_batch:
                output = per_output
            else:
                for k, v in per_output.items():
                    output[k].append(v)
        output_map = self._pad(
            output,
            max_length=max_length,
            padding_strategy=padding_strategy,
            return_attention_mask=return_attention_mask,
        )
        if return_tensors:
            for k in output_map.keys():
                v = np.array(output_map[k])
                if v.dtype == np.int64:
                    v = v.astype(np.int32)
                output_map[k] = Tensor(v)
        return output_map

    def _pad(self, id_dict, max_length, padding_strategy="do_not_pad", return_attention_mask=None):
        """Do padding according to the max_length"""
        is_batch = False
        if (
            isinstance(id_dict["input_ids"], list)
            and len(id_dict["input_ids"]) > 0
            and isinstance(id_dict["input_ids"][0], list)
        ):
            is_batch = True
            length_each = [len(line) for line in id_dict["input_ids"]]
            for item in length_each:
                if length_each[0] != item and (not max_length or padding_strategy != "max_length"):
                    raise ValueError(
                        f"You should set `max_length` to {max(length_each)} "
                        f"and padding_strategy to `max_length`, as the length in the batch "
                        f"is different, which should be padded."
                    )

        if return_attention_mask is not False:
            return_attention_mask = True

        if return_attention_mask and "attention_mask" in self.model_inputs:
            if is_batch:
                id_dict["attention_mask"] = [[1] * len(line) for line in id_dict["input_ids"]]
            else:
                id_dict["attention_mask"] = [1] * len(id_dict["input_ids"])

        if not max_length or padding_strategy != "max_length":
            return id_dict

        def _pad_batch(source_ids, pad_value):
            if not is_batch:
                source_ids = [source_ids]
            for i in range(len(source_ids)):
                if max_length < len(source_ids[i]):
                    print(
                        f"WARNING: length of input tokens {len(source_ids[i])} "
                        f"exceeds the max_length {max_length} and will be truncated. "
                    )
                    source_ids[i] = source_ids[i][:max_length]
                    source_ids[i][-1] = self.eos_token_id
                else:
                    source_ids[i] += [pad_value] * (max_length - len(source_ids[i]))
            if not is_batch:
                source_ids = source_ids[0]

            return source_ids

        id_dict["input_ids"] = _pad_batch(id_dict["input_ids"], pad_value=self.pad_token_id)
        if "attention_mask" in id_dict:
            id_dict["attention_mask"] = _pad_batch(id_dict["attention_mask"], pad_value=0)
        if "token_type_ids" in id_dict:
            id_dict["token_type_ids"] = _pad_batch(id_dict["token_type_ids"], pad_value=self.pad_token_type_id)

        return id_dict

    def num_special_tokens_to_add(self):
        """Return the special tokens to be added to the ids and ids_pair"""
        ids = []
        ids_pair = []
        output = self.build_inputs_with_special_tokens(ids, ids_pair)
        return len(output)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1):
        if token_ids_1:
            return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)
        # cls and sep is 1
        return [0] * (len(token_ids_0) + 1 + 1)

    def postprocess_ids(
        self,
        ids,
        pair_ids=None,
        add_special_tokens=True,
        max_length=None,
        truncation=False,
        padding_strategy="do_not_pad",
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
    ):
        """
        Insert the special ids into the input ids, generate the attention mask and do padding.
        """
        if not isinstance(ids, list):
            raise ValueError("The input ids should be a list.")
        output_map = dict()

        def process_token_id(ids, par_ids=None):
            sentence_b_type_ids = []
            if par_ids:
                sentence_b_type_ids = [1] * len(par_ids)
            return [0] * len(ids) + sentence_b_type_ids

        length_of_each = len(ids)

        if max_length and truncation:
            num_sp_tokens = self.num_special_tokens_to_add() if add_special_tokens else 0
            num_tokens = length_of_each + num_sp_tokens - max_length
            ids, pair_ids = self.truncate_sequences(ids, pair_ids, num_tokens)

        if add_special_tokens:
            # add cls and sep: [cls] ids [seq] pair_ids
            input_ids_output = self.build_inputs_with_special_tokens(ids, pair_ids)
            type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            input_ids_output = [ids + pair_ids] if pair_ids else ids
            type_ids = process_token_id(ids, pair_ids)

        output_map["input_ids"] = input_ids_output
        if return_token_type_ids or "token_type_ids" in self.model_inputs:
            output_map["token_type_ids"] = type_ids

        output_map = self._pad(
            output_map,
            max_length=max_length,
            padding_strategy=padding_strategy,
            return_attention_mask=return_attention_mask,
        )

        if return_tensors and return_tensors != "ms":
            raise ValueError("You should set return_tensors to be `ms`.")
        if return_tensors:
            for k, v in output_map.items():
                v = np.array(v)
                if v.dtype == np.int64:
                    v = v.astype(np.int32)
                output_map[k] = Tensor(v)
        return output_map

    def _prepare_input_to_list(self, inputs):
        """put the input into the list"""
        if inputs is None:
            return inputs
        if not isinstance(inputs, list):
            inputs = [inputs]
        return inputs

    @staticmethod
    def _read_merge_files(text_path, start_pos=1, end_pos=49152 - 256 - 2 + 1):
        r"""Read the merge files"""
        with gzip.open(text_path) as fp:
            data = fp.read()
        merges = data.decode("utf-8").split("\n")
        merges = merges[start_pos:end_pos]
        new_list = []
        for item in merges:
            new_list.append(tuple(item.split()))
        return new_list

    def _tokenize(self, text, **kwargs):
        r"""Tokenize"""
        output_ids = []
        content = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, content):
            token = "".join(self.tool.byte_encoder[b] for b in token.encode("utf-8"))
            output_ids.extend(self.tool.tokenize_alg(token).split(" "))
        return output_ids

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Insert the special tokens to the input_ids. Currently, we support token_ids_0 is a list of ids.
        """
        if token_ids_1:
            raise ValueError("The token_ids_1 is not supported yet.")
        # if not token_ids_0:
        #     raise ValueError("The length of the token_ids should be larger than 0.")
        res = [self.bos_token_id]
        res.extend(token_ids_0)
        res.extend([self.eos_token_id])
        return res

    def tokenize(self, text):
        r"""Tokenizer the input_text"""
        if not isinstance(text, str):
            raise ValueError("Text should be type str, but found type", type(text))
        return self._tokenize(text)

    def _convert_tokens_to_ids(self, input_tokens):
        r"""Convert_tokens_to_ids"""
        if input_tokens is None:
            raise ValueError(f"Input token {input_tokens} is None.")
        if len(input_tokens) == 0:
            return []
        if isinstance(input_tokens, str):
            return self.tool.encoder[input_tokens]
        return [self.tool.encoder[bpe_token] for bpe_token in input_tokens]

    @property
    def vocab_size(self):
        r"""Get the vocab size"""
        return len(self.tool.encoder)
