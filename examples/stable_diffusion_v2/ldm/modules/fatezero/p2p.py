import abc
import copy
from typing import Union, Tuple, List, Callable, Dict, Optional
import mindspore as ms
import numpy as np

from .seq_aligner import get_replacement_mapper
from .utils import get_word_inds, get_time_words_attention_alpha
from ..attention import default, exists

CROSS_ATTENTION_NAME = 'CrossAttention'
INPUT = 'input_blocks'
MIDDLE = 'middle_block'
OUTPUT = 'output_blocks'


def register_attention_control(unet, controller):
    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    def ca_forward(self, place_in_unet, attention_type):
        def _attention(q, k, v, mask, is_cross):
            sim = ms.ops.matmul(q, self.transpose(k, (0, 2, 1))) * self.attention.scale

            if exists(mask):
                mask = self.reshape(mask, (mask.shape[0], -1))
                if sim.dtype == ms.float16:
                    finfo_type = np.float16
                else:
                    finfo_type = np.float32
                max_neg_value = -np.finfo(finfo_type).max
                mask = mask.repeat(self.heads, axis=0)
                mask = ms.ops.expand_dims(mask, axis=1)
                sim.masked_fill(mask, max_neg_value)

            if self.attention.upcast:
                # use fp32 for exponential inside
                attn = self.attention.softmax(sim.astype(ms.float32)).astype(v.dtype)
            else:
                attn = self.attention.softmax(sim)

            attn = controller(attn, is_cross=is_cross, place_in_unet=place_in_unet)

            out = ms.ops.matmul(attn, v)
            return out

        def forward(x, context=None, mask=None):
            is_cross = context is not None
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            def rearange_in(x):
                # (b, n, h*d) -> (b*h, n, d)
                h = self.heads
                b, n, d = x.shape
                d = d // h

                x = self.reshape(x, (b, n, h, d))
                x = self.transpose(x, (0, 2, 1, 3))
                x = self.reshape(x, (b * h, n, d))
                return x

            q = rearange_in(q)
            k = rearange_in(k)
            v = rearange_in(v)

            if self.use_flash_attention and q.shape[1] % 16 == 0 and k.shape[1] % 16 == 0:
                out = self.flash_attention(q, k, v)
            else:
                out = _attention(q, k, v, mask, is_cross=is_cross)

            def rearange_out(x):
                # (b*h, n, d) -> (b, n, h*d)
                h = self.heads
                b, n, d = x.shape
                b = b // h

                x = self.reshape(x, (b, h, n, d))
                x = self.transpose(x, (0, 2, 1, 3))
                x = self.reshape(x, (b, n, h * d))
                return x

            out = rearange_out(out)
            return self.to_out(out)

        def sparse_forward(x, context=None, mask=None, video_length=None):
            is_cross = context is not None
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            def rearange_in(x):
                # (b, n, h*d) -> (b*h, n, d)
                h = self.heads
                b, n, d = x.shape
                d = d // h

                x = self.reshape(x, (b, n, h, d))
                x = self.transpose(x, (0, 2, 1, 3))
                x = self.reshape(x, (b * h, n, d))
                return x

            former_frame_index = ms.ops.arange(video_length) - 1
            former_frame_index[0] = 0

            k = self.concat_first_previous_features(k, video_length, former_frame_index)
            v = self.concat_first_previous_features(v, video_length, former_frame_index)

            q = rearange_in(q)
            k = rearange_in(k)
            v = rearange_in(v)

            if mask is not None:
                if mask.shape[-1] != q.shape[1]:
                    target_length = q.shape[1]
                    ndim = len(mask.shape)
                    paddings = [[0, 0] for i in range(ndim - 1)] + [0, target_length]
                    mask = ms.nn.Pad(paddings)(mask)
                    mask = mask.repeat_interleave(self.heads, axis=0)

            if self.use_flash_attention and q.shape[1] % 16 == 0 and k.shape[1] % 16 == 0:
                out = self.flash_attention(q, k, v)
            else:
                out = _attention(q, k, v, mask, is_cross=is_cross)

            def rearange_out(x):
                # (b*h, n, d) -> (b, n, h*d)
                h = self.heads
                b, n, d = x.shape
                b = b // h

                x = self.reshape(x, (b, h, n, d))
                x = self.transpose(x, (0, 2, 1, 3))
                x = self.reshape(x, (b, n, h * d))
                return x

            out = rearange_out(out)
            return self.to_out(out)

        return forward if attention_type == 'CrossAttention' else sparse_forward

    if controller is None:
        controller = DummyController()

    cross_att_count = 0
    for n in unet.cells_and_names():

        if hasattr(n[1], "attention_type") and n[1].attention_type in ['CrossAttention', 'SparseCausalAttention']:
            if INPUT in n[0]:
                key = INPUT
            elif MIDDLE in n[0]:
                key = MIDDLE
            elif OUTPUT in n[0]:
                key = OUTPUT

            cross_att_count += 1
            n[1].construct = ca_forward(n[1], place_in_unet=key, attention_type=n[1].attention_type)
    controller.num_att_layers = cross_att_count


class AttentionStore():
    def step_callback(self, x_t):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.attention_store_all_step.append(copy.deepcopy(self.step_store))
        self.step_store = self.get_empty_store()
        return x_t

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0:
            attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        return attn

    @staticmethod
    def get_empty_store():
        return {F"{INPUT}_cross": [], F"{OUTPUT}_cross": [], F"{MIDDLE}_cross": [],
                F"{INPUT}_self": [], F"{OUTPUT}_self": [], F"{MIDDLE}_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            print(f"Store attention map {key} of shape {attn.shape}")

            self.step_store[key].append(attn)
        return attn

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()

        self.latents_store = []
        self.attention_store_all_step = []
#
#
# class AttentionControlEdit(AttentionStore, abc.ABC):
#
#     def step_callback(self, x_t):
#         if self.local_blend is not None:
#             x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
#         return x_t
#
#     def replace_self_attention(self, attn_base, att_replace):
#         if att_replace.shape[2] <= 16 ** 2:
#             return attn_base.unsqueeze(0).broadcast_to((att_replace.shape[0],) + attn_base.shape)
#         else:
#             return att_replace
#
#     @abc.abstractmethod
#     def replace_cross_attention(self, attn_base, att_replace):
#         raise NotImplementedError
#
#     # @ms.jit
#     def f(self, attn, is_cross: bool):
#         if is_cross or (self.num_self_replace[0] <= self.cur_step and self.cur_step < self.num_self_replace[1]):
#             h = attn.shape[0] // (self.batch_size)
#             attn = attn.reshape((self.batch_size, h) + attn.shape[1:])
#             # attn = attn.reshape((self.batch_size, h, *attn.shape[1:]))
#             attn_base, attn_repalce = attn[0], attn[1:]
#             if is_cross:
#                 alpha_words = self.cross_replace_alpha[self.cur_step]
#                 attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
#                         1 - alpha_words) * attn_repalce
#                 attn[1:] = attn_repalce_new
#             else:
#                 attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
#             # attn = attn.reshape((self.batch_size * h, attn.shape[2]))
#             attn = attn.reshape((self.batch_size * h,) + attn.shape[2:])
#         return attn
#
#     def forward(self, attn, is_cross: bool, place_in_unet: int):
#         super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
#         return self.f(attn, is_cross)
#
#     def __init__(self, prompts, num_steps: int,
#                  cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
#                  self_replace_steps: Union[float, Tuple[float, float]],
#                  local_blend):
#         super(AttentionControlEdit, self).__init__()
#         self.batch_size = len(prompts)
#         self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps)
#         if type(self_replace_steps) is float:
#             self_replace_steps = 0, self_replace_steps
#         self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
#         self.local_blend = local_blend
#
#
#
# class AttentionReplace(AttentionControlEdit):
#
#     # @ms.jit
#     def replace_cross_attention(self, attn_base, att_replace):
#         dtype = attn_base.dtype
#
#         h, p, w = attn_base.shape
#         b, w, n = self.mapper.shape
#         i = attn_base.to(ms.float16).permute(1, 0, 2).reshape((h * p, -1))
#         j = self.mapper.to(ms.float16).permute(0, 2, 1)
#         r = ms.ops.matmul(i, j).to(dtype)
#         r = r.reshape((p, h, b, n)).permute(2, 1, 0, 3)
#         return r
#         # return ms.ops.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
#
#     def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
#                  local_blend=None, **kwargs):
#         super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
#                                                local_blend)
#         self.mapper = get_replacement_mapper(prompts)
