# coding=utf-8
# Copyright 2025 MMaDA team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


reserved_token_mapping = {
    "<|soi|>": 126084,
    "<|eoi|>": 126085,
    "<|sov|>": 126086,
    "<|eov|>": 126087,
    "<|t2i|>": 126088,
    "<|mmu|>": 126089,
    "<|t2v|>": 126090,
    "<|v2v|>": 126091,
    "<|lvg|>": 126092,
    "[iPAD]": 126093,
    "<|r2i|>": 126094,
}


import mindspore as ms
from mindspore import mint


class UniversalPrompting:
    def __init__(
        self,
        text_tokenizer,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
        ),
        max_text_len=8000,
        max_seq_len=377,
        ignore_id=-100,
        cond_dropout_prob=0.1,
        use_reserved_token=False,
    ):
        """
        :param text_tokenizer: original text tokenizer
        """
        if not use_reserved_token:
            self.text_tokenizer = text_tokenizer
            self.text_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.text_tokenizer.add_tokens(list(special_tokens))
            self.sptids_dict = {
                token: ms.tensor(self.text_tokenizer.convert_tokens_to_ids([token])) for token in special_tokens
            }
            self.sptids_dict["<|sot|>"] = ms.tensor([self.text_tokenizer.bos_token_id])
            self.sptids_dict["<|eot|>"] = ms.tensor([self.text_tokenizer.eos_token_id])
            self.sptids_dict["<|pad|>"] = ms.tensor([self.text_tokenizer.pad_token_id])
        else:
            self.text_tokenizer = text_tokenizer
            self.sptids_dict = {}
            for token, token_id in reserved_token_mapping.items():
                self.sptids_dict[token] = ms.tensor([token_id])
            self.sptids_dict["<|sot|>"] = ms.tensor([self.text_tokenizer.bos_token_id])
            self.sptids_dict["<|eot|>"] = ms.tensor([self.text_tokenizer.eos_token_id])
            end_header_tokens = self.text_tokenizer.convert_tokens_to_ids(["<|end_header_id|>"])
            if end_header_tokens and len(end_header_tokens) > 0 and end_header_tokens[0]:
                self.sptids_dict["<|end_header_id|>"] = ms.tensor(end_header_tokens)
                self.sptids_dict["<|eot_id|>"] = ms.tensor(self.text_tokenizer.convert_tokens_to_ids(["<|eot_id|>"]))
                self.sptids_dict["<|start_header_id|>"] = ms.tensor(
                    self.text_tokenizer.convert_tokens_to_ids(["<|start_header_id|>"])
                )
            else:
                # special_tokens_dict = {
                #    "additional_special_tokens": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
                # }
                # num_added = self.text_tokenizer.add_special_tokens(special_tokens_dict)
                new_token_id = self.text_tokenizer.convert_tokens_to_ids(["<|end_header_id|>"])
                self.sptids_dict["<|end_header_id|>"] = ms.tensor(new_token_id)
                self.sptids_dict["<|eot_id|>"] = ms.tensor(self.text_tokenizer.convert_tokens_to_ids(["<|eot_id|>"]))
                self.sptids_dict["<|start_header_id|>"] = ms.tensor(
                    self.text_tokenizer.convert_tokens_to_ids(["<|start_header_id|>"])
                )
        # plus 1 because at this time we add a task token before
        print(f"self.sptids_dict: {self.sptids_dict}")
        self.max_text_len = max_text_len + 1
        self.pad_id = reserved_token_mapping["[iPAD]"]
        self.ignore_id = ignore_id
        self.cond_dropout_prob = cond_dropout_prob

    def t2i_prompt(self, text_ids, image_ids, labels):
        sequence_ids = []
        attention_masks = []
        label_ids = []
        probs = mint.rand(len(text_ids))
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = [int(self.sptids_dict["<|t2i|>"])] + text_ids[i] + [self.text_tokenizer.eos_token_id]

            # randomly dropout text condition
            if probs[i] < self.cond_dropout_prob:
                temp_ids = [
                    int(self.sptids_dict["<|t2i|>"]),
                    self.text_tokenizer.bos_token_id,
                    self.text_tokenizer.eos_token_id,
                ]

            if self.max_text_len >= len(temp_ids):
                old_len = len(temp_ids)
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - old_len) + [1] * (old_len + image_ids.shape[-1] + 2)
            else:
                # should add the eos token
                temp_ids = temp_ids[: self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 2)  # +2 for two special tokens
            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = mint.cat(
                [
                    # should we predict text tokens when doing image reconstruction?
                    ms.tensor(temp_ids),
                    self.sptids_dict["<|soi|>"],
                    labels[i],
                    self.sptids_dict["<|eoi|>"],
                ],
                dim=0,
            )

            temp_label_ids = mint.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            temp_ids = mint.cat(
                [
                    ms.tensor(temp_ids),
                    self.sptids_dict["<|soi|>"],
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"],
                ],
                dim=0,
            )

            # sequence_ids: [pad]...[pad] <|t2i|> <bos> text_1 ... text_n <eos> <|soi|> image_1 ... image_m <|eoi|>
            temp_masks = ms.tensor(temp_masks)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return mint.cat(sequence_ids, dim=0), mint.cat(attention_masks, dim=0), mint.cat(label_ids, dim=0)

    def t2i_gen_prompt(self, text_ids, image_ids):
        sequence_ids = []
        attention_masks = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            temp_ids = [int(self.sptids_dict["<|t2i|>"])] + text_ids[i] + [self.text_tokenizer.eos_token_id]
            if self.max_text_len >= len(temp_ids):
                old_len = len(temp_ids)
                temp_ids = [self.pad_id] * (self.max_text_len - len(temp_ids)) + temp_ids
                temp_masks = [0] * (self.max_text_len - old_len) + [1] * (old_len + image_ids.shape[-1] + 2)
            else:
                # should add the eos token
                temp_ids = temp_ids[: self.max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 2)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = mint.cat(
                [
                    ms.tensor(temp_ids),
                    self.sptids_dict["<|soi|>"],
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"],
                ],
                dim=0,
            )

            temp_masks = ms.tensor(temp_masks)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))

        return mint.cat(sequence_ids, dim=0), mint.cat(attention_masks, dim=0)

    # language modeling
    def lm_prompt(self, text_ids, max_seq_len):
        sequence_ids = []
        attention_masks = []
        label_ids = []
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_seq_len >= len(temp_ids):
                temp_labels_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (max_seq_len - len(temp_ids))
                temp_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (max_seq_len - len(temp_ids))
                temp_masks = [1] * len(temp_ids) + [0] * (max_seq_len - len(temp_ids))
            else:
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]
                temp_masks = [1] * len(temp_ids)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = ms.tensor(temp_ids)
            temp_masks = ms.tensor(temp_masks)
            temp_labels_ids = ms.tensor(temp_labels_ids)
            sequence_ids.append(temp_ids.unsqueeze(0))
            attention_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return mint.cat(sequence_ids, dim=0), mint.cat(attention_masks, dim=0), mint.cat(label_ids, dim=0)

    # language modeling
    def lm_chat_prompt(self, text_ids, max_seq_len):
        sequence_ids = []
        prompt_masks = []
        label_ids = []

        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_seq_len >= len(temp_ids):
                temp_labels_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (max_seq_len - len(temp_ids))
                temp_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (max_seq_len - len(temp_ids))
            else:
                # In language modeling, we only process text tokens. We do not add the eos token if the text length
                # exceeds the max sequence length
                temp_labels_ids = temp_ids[:max_seq_len]
                temp_ids = temp_ids[:max_seq_len]

            end_header_id = int(self.sptids_dict["<|end_header_id|>"])
            end_header_pos = -1
            for pos in range(len(temp_ids) - 1, -1, -1):  # 尝试从文本序列中寻找<|end_header_id|>
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos
                    break
            if end_header_pos != -1:
                prompt_length = end_header_pos + 1
            else:
                prompt_length = 0
            temp_masks = [1] * prompt_length + [0] * (len(temp_ids) - prompt_length)

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_ids = ms.tensor(temp_ids)
            temp_masks = ms.tensor(temp_masks)
            temp_labels_ids = ms.tensor(temp_labels_ids)
            sequence_ids.append(temp_ids.unsqueeze(0))
            prompt_masks.append(temp_masks.unsqueeze(0))
            label_ids.append(temp_labels_ids.unsqueeze(0))

        # input_ids, masks, labels
        return mint.cat(sequence_ids, dim=0), mint.cat(prompt_masks, dim=0), mint.cat(label_ids, dim=0)

    def mmu_prompt(self, image_ids, text_ids):
        sequence_ids = []
        prompt_masks = []
        label_ids = []
        max_text_len = self.max_text_len - 1
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []

            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                # minus 1 because task token was prepended to the former image tokens
                temp_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (max_text_len - len(temp_ids))
                # temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3) + [0] * (max_text_len - len(temp_ids))
            else:
                # should add the eos token
                temp_ids = temp_ids[: max_text_len - 1] + [self.text_tokenizer.eos_token_id]
                # temp_masks = [1] * (len(temp_ids) + image_ids.shape[-1] + 3)  # +2 for two special tokens

            # prompting -- [task token] [sot] [text tokens] [eot] [soi] [image tokens] [eoi]
            temp_label_ids = mint.cat(
                [
                    ms.tensor([self.ignore_id]),
                    ms.tensor([self.ignore_id]),
                    mint.ones_like(image_ids[i]) * self.ignore_id,
                    ms.tensor([self.ignore_id]),
                    ms.tensor(temp_ids),
                ],
                dim=0,
            )

            temp_label_ids = mint.where(temp_label_ids == self.pad_id, self.ignore_id, temp_label_ids)

            return_temp_ids = mint.cat(
                [
                    self.sptids_dict["<|mmu|>"],  # task token
                    self.sptids_dict["<|soi|>"],
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"],
                    ms.tensor(temp_ids),
                ],
                dim=0,
            )
            end_header_id = int(self.sptids_dict["<|end_header_id|>"])
            end_header_pos = -1
            for pos in range(len(temp_ids) - 1, -1, -1):
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos
                    break
            if end_header_pos != -1:
                prompt_length = len(return_temp_ids) - len(temp_ids) + end_header_pos + 1
            else:
                prompt_length = len(return_temp_ids) - len(temp_ids)
            predict_length = len(return_temp_ids) - prompt_length
            prompt_mask = [1] * prompt_length + [0] * predict_length
            prompt_mask = ms.tensor(prompt_mask)
            sequence_ids.append(return_temp_ids.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
            label_ids.append(temp_label_ids.unsqueeze(0))

        return mint.cat(sequence_ids, dim=0), mint.cat(prompt_masks, dim=0), mint.cat(label_ids, dim=0)

    def mmu_gen_prompt(self, image_ids, text_ids):
        sequence_ids = []
        prompt_masks = []
        max_text_len = self.max_text_len - 1
        for i in range(len(text_ids)):
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]

            temp_ids = text_ids[i] + [self.text_tokenizer.eos_token_id]

            if max_text_len >= len(temp_ids):
                # minus 1 because task token was prepended to the former image tokens
                temp_ids = temp_ids + [self.text_tokenizer.eos_token_id] * (max_text_len - len(temp_ids))
            else:
                # should add the eos token
                temp_ids = temp_ids[: max_text_len - 1] + [self.text_tokenizer.eos_token_id]

            # print(f"mmu temp_ids: {temp_ids}")
            return_temp_ids = mint.cat(
                [
                    self.sptids_dict["<|mmu|>"],  # task token
                    self.sptids_dict["<|soi|>"],
                    image_ids[i],
                    self.sptids_dict["<|eoi|>"],
                    ms.tensor(temp_ids),
                ],
                dim=0,
            )

            end_header_id = int(self.sptids_dict["<|end_header_id|>"])
            end_header_pos = -1
            for pos in range(len(temp_ids) - 1, -1, -1):
                if temp_ids[pos] == end_header_id:
                    end_header_pos = pos
                    break
            if end_header_pos != -1:
                prompt_length = len(return_temp_ids) - len(temp_ids) + end_header_pos + 1
            else:
                prompt_length = len(return_temp_ids) - len(temp_ids)
            predict_length = len(temp_ids) - prompt_length
            print(
                f"prompt_length: {prompt_length}, predict_length: {predict_length}, all length: {len(return_temp_ids)}, {return_temp_ids[-predict_length:]}"
            )
            prompt_mask = [1] * prompt_length + [0] * predict_length
            prompt_mask = ms.tensor(prompt_mask)
            sequence_ids.append(return_temp_ids.unsqueeze(0))
            prompt_masks.append(prompt_mask.unsqueeze(0))
        return mint.cat(sequence_ids, dim=0), mint.cat(prompt_masks, dim=0)

    def r2i_prompt(self, image_ids, text_ids):
        sequence_ids = []
        prompt_masks = []
        # label_ids = []
        r2i_id = int(self.sptids_dict["<|r2i|>"])
        soi_id = int(self.sptids_dict["<|soi|>"])
        eoi_id = int(self.sptids_dict["<|eoi|>"])
        max_text_len = self.max_text_len - 1  # 512，include BOS text EOS
        for i in range(len(text_ids)):
            # note that, llama3 tokenizer automatically add the bot token at first but without eot
            # for empty list []
            if len(text_ids[i]) == 0:
                text_ids[i] = [self.text_tokenizer.bos_token_id]
            elif text_ids[i][0] != self.text_tokenizer.bos_token_id:
                text_ids[i] = [self.text_tokenizer.bos_token_id] + text_ids[i]
            text_ids_with_bos_eos = text_ids[i] + [self.text_tokenizer.eos_token_id]
            if max_text_len >= len(text_ids_with_bos_eos):
                # minus 1 because task token was prepended to the former image tokens
                text_ids_full_len = text_ids_with_bos_eos + [self.text_tokenizer.eos_token_id] * (
                    max_text_len - len(text_ids_with_bos_eos)
                )
            else:
                # should add the eos token
                text_ids_full_len = text_ids_with_bos_eos[: max_text_len - 1] + [self.text_tokenizer.eos_token_id]

            sequence_ids.append(
                mint.cat(
                    [
                        ms.tensor([r2i_id]),  # task token
                        ms.tensor(text_ids_full_len),
                        ms.tensor([soi_id]),
                        image_ids[i],
                        ms.tensor([eoi_id]),
                    ],
                    dim=0,
                ).unsqueeze(0)
            )

            end_header_id = int(self.sptids_dict["<|end_header_id|>"])
            end_header_pos = -1
            for pos in range(len(text_ids_full_len) - 1, -1, -1):
                if text_ids_full_len[pos] == end_header_id:
                    end_header_pos = pos
                    break
            prompt_mask = mint.zeros(sequence_ids[i].size(1))
            prompt_mask[0] = 1  # task_id
            if end_header_pos != -1:
                prompt_mask[1 : end_header_pos + 2] = 1
            else:
                prompt_mask[1 : len(text_ids_full_len) + 1] = 1
            prompt_mask[len(text_ids_full_len) + 1] = 1
            prompt_mask[len(text_ids_full_len) + 2 + len(image_ids[i])] = 1
            prompt_masks.append(prompt_mask.unsqueeze(0))

        return mint.cat(sequence_ids, dim=0), mint.cat(prompt_masks, dim=0), mint.cat(sequence_ids, dim=0)

    def mask_prompt(self):
        pass

    def __call__(self, input, task, padding=True, config=None):
        """
        input (tuple) : data pairs contain text(str), image(tensor), or videos(tensor).
        task (str) : a flag indicates the current task.
        """
        if task == "t2i":
            text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids, image_ids, input[2])

        elif task == "t2v":
            text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2v_prompt(text_ids, image_ids, input[2])

        elif task == "t2i_plus_lm":
            text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_prompt(text_ids[: config.training.batch_size], image_ids, input[2])
            sequence_ids_with_masks_lm = self.lm_prompt(text_ids[config.training.batch_size :], input[3])
            return sequence_ids_with_masks, sequence_ids_with_masks_lm

        elif task == "t2i_gen":
            text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2i_gen_prompt(text_ids, image_ids)

        elif task == "t2v_gen":
            text_ids = self.text_tokenizer(input[0])["input_ids"]  # (B, max_len)
            image_ids = input[1]  # (B, #tokens)
            sequence_ids_with_masks = self.t2v_gen_prompt(text_ids, image_ids)

        elif task == "lm":
            text_ids = self.text_tokenizer(input[0], truncation=True)["input_ids"]  # (B, max_len)
            sequence_ids_with_masks = self.lm_prompt(text_ids, input[1])

        elif task == "lm_chat":
            text_ids = self.text_tokenizer(input[0], truncation=True)["input_ids"]  # (B, max_len)
            sequence_ids_with_masks = self.lm_chat_prompt(text_ids, input[1])

        elif task == "mmu":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])["input_ids"]
            sequence_ids_with_masks = self.mmu_prompt(image_ids, text_ids)

        elif task == "r2i":
            image_ids = input[0]
            text_ids = self.text_tokenizer(input[1])["input_ids"]
            sequence_ids_with_masks = self.r2i_prompt(image_ids, text_ids)

        else:
            raise NotImplementedError

        return sequence_ids_with_masks


if __name__ == "__main__":
    pass
