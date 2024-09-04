import random
from typing import Optional

import gradio as gr

import mindspore as ms
from mindspore import ops

from mindone.diffusers.models.attention_processor import Attention


@ms.jit_class
class SpatialAttnProcessor2_0:
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        id_length=4,
        dtype=ms.float16,
        attention_masks={},
        write=False,
        cur_step=0,
    ):
        super().__init__()

        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}
        self.attention_masks = attention_masks
        assert len(self.attention_masks) > 0, "attention_masks must not be empty"
        self.write = write  # if true, will save hidden states to id_bank; otherwise, use the concatenated id_bank and hidden_states as key and query
        self.cur_step = cur_step  # the counter the number of inference steps

    def scaled_dot_product_attention(
        self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, training=False
    ) -> ms.Tensor:
        L, S = query.shape[-2], key.shape[-2]
        scale_factor = 1 / (query.shape[-1] ** 0.5) if scale is None else scale
        _dtype = query.dtype
        attn_bias = ops.zeros((L, S), dtype=ms.float32)
        if is_causal:
            assert attn_mask is None
            temp_mask = ops.ones((L, S), dtype=ms.bool_).tril(diagonal=0)
            attn_bias = attn_bias.masked_fill(~temp_mask, -1e5)
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == ms.bool_:
                attn_bias = attn_bias.masked_fill(~attn_mask, -1e5)
            else:
                attn_bias += attn_mask
        attn_weight = ops.matmul(query, key.swapaxes(-2, -1)) * scale_factor
        attn_weight = attn_weight.to(ms.float32)
        attn_weight += attn_bias
        attn_weight = ops.softmax(attn_weight, axis=-1)
        attn_weight = ops.dropout(attn_weight, p=dropout_p, training=training)
        out = ops.matmul(attn_weight.to(_dtype), value)
        out = out.astype(_dtype)
        return out

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        if self.write:
            self.id_bank[self.cur_step] = [hidden_states[: self.id_length], hidden_states[self.id_length :]]
        else:
            encoder_hidden_states = ops.cat(
                [self.id_bank[self.cur_step][0], hidden_states[:1], self.id_bank[self.cur_step][1], hidden_states[1:]]
            )
        # skip in early step
        if self.cur_step < 5:
            hidden_states = self.__call2__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:  # 256 1024 4096
            random_number = random.random()
            if self.cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                nums_tokens = hidden_states.shape[1]
                assert (
                    nums_tokens in self.attention_masks
                ), f"The input num_tokens is not supported. Supported num_tokens: { self.attention_masks.keys()}"
                attention_mask = self.attention_masks[nums_tokens]
                target_len = attention_mask.shape[0] // self.total_length * self.id_length
                if not self.write:
                    attention_mask = attention_mask[target_len:]
                else:
                    attention_mask = attention_mask[:target_len, :target_len]
                hidden_states = self.__call1__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)

        return hidden_states

    def __call1__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(
            -1, img_nums * nums_token, channel
        )

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, nums_token, channel).reshape(
                -1, (self.id_length + 1) * nums_token, channel
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        hidden_states = self.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # if input_ndim == 4:
        #     tile_hidden_states = tile_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     tile_hidden_states = tile_hidden_states + residual

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def __call2__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        # original AttnProcessor
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, sequence_length, channel = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, sequence_length, channel
            ).reshape(-1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = ops.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def cal_attn_mask(total_length, id_length, sa16, sa32, sa64, dtype=ms.float16):
    bool_matrix256 = ops.rand((1, total_length * 256), dtype=dtype) < sa16
    bool_matrix1024 = ops.rand((1, total_length * 1024), dtype=dtype) < sa32
    bool_matrix4096 = ops.rand((1, total_length * 4096), dtype=dtype) < sa64
    bool_matrix256 = ops.cat([bool_matrix256] * total_length, axis=0)
    bool_matrix1024 = ops.cat([bool_matrix1024] * total_length, axis=0)
    bool_matrix4096 = ops.cat([bool_matrix4096] * total_length, axis=0)
    for i in range(total_length):
        bool_matrix256[i : i + 1, id_length * 256 :] = False
        bool_matrix1024[i : i + 1, id_length * 1024 :] = False
        bool_matrix4096[i : i + 1, id_length * 4096 :] = False
        bool_matrix256[i : i + 1, i * 256 : (i + 1) * 256] = True
        bool_matrix1024[i : i + 1, i * 1024 : (i + 1) * 1024] = True
        bool_matrix4096[i : i + 1, i * 4096 : (i + 1) * 4096] = True
    mask256 = ops.cat([bool_matrix256.unsqueeze(1)] * 256, axis=1).reshape(-1, total_length * 256)
    mask1024 = ops.cat([bool_matrix1024.unsqueeze(1)] * 1024, axis=1).reshape(-1, total_length * 1024)
    mask4096 = ops.cat([bool_matrix4096.unsqueeze(1)] * 4096, axis=1).reshape(-1, total_length * 4096)
    return mask256, mask1024, mask4096


def cal_attn_mask_xl(total_length, id_length, sa32, sa64, height, width, dtype=ms.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = ops.rand((1, total_length * nums_1024), dtype=dtype) < sa32
    bool_matrix4096 = ops.rand((1, total_length * nums_4096), dtype=dtype) < sa64
    bool_matrix1024 = ops.cat([bool_matrix1024] * total_length, axis=0)
    bool_matrix4096 = ops.cat([bool_matrix4096] * total_length, axis=0)
    for i in range(total_length):
        bool_matrix1024[i : i + 1, id_length * nums_1024 :] = False
        bool_matrix4096[i : i + 1, id_length * nums_4096 :] = False
        bool_matrix1024[i : i + 1, i * nums_1024 : (i + 1) * nums_1024] = True
        bool_matrix4096[i : i + 1, i * nums_4096 : (i + 1) * nums_4096] = True
    mask1024 = ops.cat([bool_matrix1024.unsqueeze(1)] * nums_1024, axis=1).reshape(-1, total_length * nums_1024)
    mask4096 = ops.cat([bool_matrix4096.unsqueeze(1)] * nums_4096, axis=1).reshape(-1, total_length * nums_4096)
    return mask1024, mask4096


def calculate_attention_mask(total_length, id_length, down_scale, height, width, threshold=0.5, dtype=ms.float16):
    nums = (height // down_scale) * (width // down_scale)
    bool_matrix = ops.rand((1, total_length * nums), dtype=dtype) < threshold
    bool_matrix = ops.cat([bool_matrix] * total_length, axis=0)  # (N, N*num_image_tokens)
    for i in range(total_length):  # create a blockwise attention mask
        bool_matrix[i : i + 1, id_length * nums :] = False
        bool_matrix[i : i + 1, i * nums : (i + 1) * nums] = True
    mask = ops.cat([bool_matrix.unsqueeze(1)] * nums, axis=1).reshape(
        -1, total_length * nums
    )  # (N*num_image_tokens, N*num_image_tokens)
    return mask


def get_attention_mask_dict(
    down_scales_list, thresholds_list, total_length, id_length, height, width, dtype=ms.float16
):
    atten_masks = {}
    down_scales_list = set(down_scales_list)
    assert len(down_scales_list) == len(thresholds_list)
    for down_scale, threshold in zip(down_scales_list, thresholds_list):
        nums_tokens = (height // down_scale) * (width // down_scale)
        atten_masks[nums_tokens] = calculate_attention_mask(
            total_length, id_length, down_scale, height, width, threshold, dtype
        )
    return atten_masks


def cal_attn_indice_xl_effcient_memory(total_length, id_length, sa32, sa64, height, width, dtype=ms.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = ops.rand((total_length, nums_1024), dtype=dtype) < sa32
    bool_matrix4096 = ops.rand((total_length, nums_4096), dtype=dtype) < sa64
    # 用nonzero()函数获取所有为True的值的索引
    indices1024 = [ops.nonzero(bool_matrix1024[i])[0] for i in range(total_length)]
    indices4096 = [ops.nonzero(bool_matrix4096[i])[0] for i in range(total_length)]

    return indices1024, indices4096


# 将列表转换为字典的函数
def character_to_dict(general_prompt):
    character_dict = {}
    generate_prompt_arr = general_prompt.splitlines()
    character_list = []
    for ind, string in enumerate(generate_prompt_arr):
        # 分割字符串寻找key和value
        start = string.find("[")
        end = string.find("]")
        if start != -1 and end != -1:
            key = string[start : end + 1]
            value = string[end + 1 :]
            if "#" in value:
                value = value.rpartition("#")[0]
            if key in character_dict:
                raise gr.Error("duplicate character descirption: " + key)
            character_dict[key] = value
            character_list.append(key)

    return character_dict, character_list


def get_id_prompt_index(character_dict, id_prompts):
    replace_id_prompts = []
    character_index_dict = {}
    invert_character_index_dict = {}
    for ind, id_prompt in enumerate(id_prompts):
        for key in character_dict.keys():
            if key in id_prompt:
                if key not in character_index_dict:
                    character_index_dict[key] = []
                character_index_dict[key].append(ind)
                invert_character_index_dict[ind] = key
                replace_id_prompts.append(id_prompt.replace(key, character_dict[key]))

    return character_index_dict, invert_character_index_dict, replace_id_prompts


def get_cur_id_list(real_prompt, character_dict, character_index_dict):
    list_arr = []
    for keys in character_index_dict.keys():
        if keys in real_prompt:
            list_arr = list_arr + character_index_dict[keys]
            real_prompt = real_prompt.replace(keys, character_dict[keys])
    return list_arr, real_prompt


def process_original_prompt(character_dict, prompts, id_length):
    replace_prompts = []
    character_index_dict = {}
    invert_character_index_dict = {}
    for ind, prompt in enumerate(prompts):
        for key in character_dict.keys():
            if key in prompt:
                if key not in character_index_dict:
                    character_index_dict[key] = []
                character_index_dict[key].append(ind)
                if ind not in invert_character_index_dict:
                    invert_character_index_dict[ind] = []
                invert_character_index_dict[ind].append(key)
        cur_prompt = prompt
        if ind in invert_character_index_dict:
            for key in invert_character_index_dict[ind]:
                cur_prompt = cur_prompt.replace(key, character_dict[key] + " ")
        replace_prompts.append(cur_prompt)
    ref_index_dict = {}
    ref_totals = []
    print(character_index_dict)
    for character_key in character_index_dict.keys():
        if character_key not in character_index_dict:
            raise gr.Error("{} not have prompt description, please remove it".format(character_key))
        index_list = character_index_dict[character_key]
        index_list = [index for index in index_list if len(invert_character_index_dict[index]) == 1]
        if len(index_list) < id_length:
            raise gr.Error(
                f"{character_key} not have enough prompt description, need no less than {id_length}, but you give {len(index_list)}"
            )
        ref_index_dict[character_key] = index_list[:id_length]
        ref_totals = ref_totals + index_list[:id_length]
    return character_index_dict, invert_character_index_dict, replace_prompts, ref_index_dict, ref_totals


def get_ref_character(real_prompt, character_dict):
    list_arr = []
    for keys in character_dict.keys():
        if keys in real_prompt:
            list_arr = list_arr + [keys]
    return list_arr
