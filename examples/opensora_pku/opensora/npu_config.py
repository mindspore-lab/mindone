import bisect
import gc
import math
import os
import subprocess
from contextlib import contextmanager

from opensora.utils.ms_utils import init_env

import mindspore as ms
from mindspore import mint, ops

from mindone.utils.version_control import choose_flash_attention_dtype


@contextmanager
def set_run_dtype(x, dtype=None):
    # 保存原始环境变量的值（如果存在）
    npu_config.original_run_dtype = x.dtype
    # 设置环境变量为指定的值
    npu_config.current_run_dtype = dtype
    try:
        # Yield control back to the body of the `with` statement
        yield
    finally:
        # 恢复原始的环境变量值
        npu_config.current_run_dtype = None
        npu_config.original_run_dtype = None


class NPUConfig:
    N_NPU_PER_NODE = 8

    def __init__(self):
        self.on_npu = True
        self.node_world_size = self.N_NPU_PER_NODE
        self.profiling = False
        self.profiling_step = 5
        self.enable_FA = True
        self.enable_FP32 = False
        self.load_pickle = True
        self.use_small_dataset = False
        self.current_run_dtype = None
        self.original_run_dtype = None

        self.replaced_type = ms.float32
        self.conv_dtype = ms.bfloat16  # FIXME: torch uses float16
        self.norm_dtype = ms.bfloat16  # use bf16 for group_norm, layer_norm and batch_norm. Set to fp32 when training
        if self.enable_FA and self.enable_FP32:
            self.inf_float = -10000.0
        else:
            self.inf_float = -10000.0

        if self.use_small_dataset:
            self.load_pickle = False

        self._loss = []
        self.work_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pickle_save_path = f"{self.work_path}/pickles"

        gc.set_threshold(700, 10, 10000)
        self.fa_mask_dtype = choose_flash_attention_dtype()
        self.flash_attn_valid_head_dims = [64, 80, 96, 120, 128, 256]
        self.FA_dtype = ms.bfloat16
        assert self.FA_dtype in [ms.float16, ms.bfloat16], f"Unsupported flash-attention dtype: {self.FA_dtype}"

    def set_npu_env(self, args):
        rank_id, device_num = init_env(
            mode=args.mode,
            device_target=args.device,
            distributed=getattr(args, "use_parallel", False),
            precision_mode=getattr(args, "precision_mode", None),
            jit_level=getattr(args, "jit_level", None),
            jit_syntax_level=getattr(args, "jit_syntax_level", "strict"),
        )
        self.rank = rank_id
        self.bind_thread_to_cpu()
        return rank_id, device_num

    def get_total_cores(self):
        try:
            total_cores = os.sysconf("SC_NPROCESSORS_ONLN")
        except (AttributeError, ValueError):
            total_cores = os.cpu_count()
        return total_cores

    def bind_thread_to_cpu(self):
        total_cores = self.get_total_cores()
        # 每个卡的核心数量
        cores_per_rank = total_cores // 8
        # 计算本地rank
        local_rank = self.rank % 8
        # 计算当前 rank 的 CPU 核范围
        start_core = local_rank * cores_per_rank
        end_core = start_core + cores_per_rank - 1
        # 构建 CPU 核范围字符串
        cpu_cores_range = f"{start_core}-{end_core}"
        pid = os.getpid()
        command = f"taskset -cp {cpu_cores_range} {pid}"

        subprocess.run(command, shell=True, check=True)
        return f"Binding Cores: {self.rank} {pid}  {cpu_cores_range}"

    def get_attention_mask(self, attention_mask, repeat_num):
        if self.on_npu and attention_mask is not None:
            if npu_config.enable_FA:
                attention_mask = attention_mask.to(ms.float16)
            attention_mask = attention_mask.repeat_interleave(repeat_num, dim=-2)
        return attention_mask

    def set_current_run_dtype(self, variables):
        if variables[0].dtype != self.current_run_dtype and self.current_run_dtype is not None:
            for index, var in enumerate(variables):
                variables[index] = var.to(self.current_run_dtype)
        return tuple(variables)

    def restore_dtype(self, x):
        if x.dtype != self.original_run_dtype and self.original_run_dtype is not None:
            x = x.to(self.original_run_dtype)
        return x

    def get_node_id(self):
        return self.rank // self.node_world_size

    def get_node_size(self):
        return self.world_size // self.node_world_size

    def get_local_rank(self):
        return self.rank % self.N_NPU_PER_NODE

    def _run(self, operator, x, tmp_dtype, out_dtype=None):
        if self.on_npu:
            if out_dtype is None:
                out_dtype = x.dtype
            x = operator(x.to(tmp_dtype))
            x = x.to(out_dtype)
            return x
        else:
            return operator(x)

    def run_group_norm(self, operator, x):
        return self._run(operator, x, self.norm_dtype)

    def run_layer_norm(self, operator, x):
        return self._run(operator, x, self.norm_dtype)

    def run_batch_norm(self, operator, x):
        return self._run(operator, x, self.norm_dtype)

    def run_conv3d(self, operator, x, out_dtype):
        return self._run(operator, x, self.conv_dtype, out_dtype)

    def run_pool_2d(self, operator, x, kernel_size, stride):
        if self.on_npu:
            x_dtype = x.dtype
            x = x.to(self.replaced_type)
            x = operator(x, kernel_size=kernel_size, stride=stride)
            x = x.to(x_dtype)
        else:
            x = operator(x, kernel_size=kernel_size, stride=stride)
        return x

    def run_pad_2d(self, operator, x, pad, mode="constant"):
        if self.on_npu:
            x_dtype = x.dtype
            x = x.to(self.replaced_type)
            x = operator(x, pad, mode)
            x = x.to(x_dtype)
        else:
            x = operator(x, pad, mode)
        return x

    def run_interpolate(self, operator, x, scale_factor=None):
        if self.on_npu:
            x_dtype = x.dtype
            x = x.to(self.replaced_type)
            x = operator(x, scale_factor=scale_factor)
            x = x.to(x_dtype)
        else:
            x = operator(x, scale_factor=scale_factor)
        return x

    def run_attention(self, query, key, value, attention_mask, input_layout, head_dim, head_num):
        if self.enable_FA:
            hidden_states = self.ms_flash_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                input_layout=input_layout,
                scale=1 / math.sqrt(head_dim),
                head_num=head_num,
            )
        else:
            hidden_states = self.scaled_dot_product_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                input_layout=input_layout,
                scale=1 / math.sqrt(head_dim),
                head_num=head_num,
            )
        return hidden_states

    def ms_flash_attention(
        self,
        query,
        key,
        value,
        attention_mask,
        head_num,
        scale,
        input_layout="BSH",
        attention_dropout: float = 0.0,
    ):
        # Memory efficient attention on mindspore uses flash attention under the hoods.
        # Flash attention implementation is called `FlashAttentionScore`
        # which is an experimental api with the following limitations:
        # 1. Sequence length of query must be divisible by 16 and in range of [1, 32768].
        # 2. Head dimensions must be one of [64, 80, 96, 120, 128, 256].
        # 3. The input dtype must be float16 or bfloat16.
        # Sequence length of query must be checked in runtime.
        if input_layout not in ["BSH", "BNSD"]:
            raise ValueError(f"input_layout must be in ['BSH', 'BNSD'], but get {input_layout}.")
        Bs, query_tokens, inner_dim = query.shape
        assert query_tokens % 16 == 0, f"Sequence length of query must be divisible by 16, but got {query_tokens}."
        key_tokens = key.shape[1]
        heads = head_num
        query = query.view(Bs, query_tokens, heads, -1)
        key = key.view(Bs, key_tokens, heads, -1)
        value = value.view(Bs, key_tokens, heads, -1)

        head_dim = inner_dim // heads
        if head_dim in self.flash_attn_valid_head_dims:
            head_dim_padding = 0
        else:
            minimum_larger_index = bisect.bisect_right(self.flash_attn_valid_head_dims, head_dim)
            if minimum_larger_index >= len(self.flash_attn_valid_head_dims):
                head_dim_padding = -1  # head_dim is bigger than the largest one, we cannot do padding
            else:
                head_dim_padding = self.flash_attn_valid_head_dims[minimum_larger_index] - head_dim
        # Head dimension is checked in Attention.set_use_memory_efficient_attention_xformers. We maybe pad on head_dim.
        if head_dim_padding > 0:
            query_padded = mint.nn.functional.pad(query, (0, head_dim_padding), mode="constant", value=0.0)
            key_padded = mint.nn.functional.pad(key, (0, head_dim_padding), mode="constant", value=0.0)
            value_padded = mint.nn.functional.pad(value, (0, head_dim_padding), mode="constant", value=0.0)
        else:
            query_padded, key_padded, value_padded = query, key, value
        flash_attn = ops.operations.nn_ops.FlashAttentionScore(
            scale_value=scale, head_num=heads, input_layout=input_layout, keep_prob=1 - attention_dropout
        )
        if attention_mask is not None:
            # flip mask, since ms FA treats 1 as discard, 0 as retain.
            attention_mask = ~attention_mask if attention_mask.dtype == ms.bool_ else 1 - attention_mask
            # (b, 1, 1, k_n) - > (b, 1, q_n, k_n), manual broadcast
            if attention_mask.shape[-2] == 1:
                attention_mask = mint.tile(attention_mask.bool(), (1, 1, query_tokens, 1))
            attention_mask = attention_mask.to(self.fa_mask_dtype)

        if input_layout == "BNSD":
            # (b s n d) -> (b n s d)
            query_padded = query_padded.swapaxes(1, 2)
            key_padded = key_padded.swapaxes(1, 2)
            value_padded = value_padded.swapaxes(1, 2)
        elif input_layout == "BSH":
            query_padded = query_padded.view(Bs, query_tokens, -1)
            key_padded = key_padded.view(Bs, key_tokens, -1)
            value_padded = value_padded.view(Bs, key_tokens, -1)
        hidden_states_padded = flash_attn(
            query_padded.to(self.FA_dtype),
            key_padded.to(self.FA_dtype),
            value_padded.to(self.FA_dtype),
            None,
            None,
            None,
            attention_mask,
        )[3]
        # If we did padding before calculate attention, undo it!
        if head_dim_padding > 0:
            if input_layout == "BNSD":
                hidden_states = hidden_states_padded[:, :, :, :head_dim]
            else:
                hidden_states = hidden_states_padded.view(Bs, query_tokens, heads, -1)[:, :, :, :head_dim]
                hidden_states = hidden_states.view(Bs, query_tokens, -1)
        else:
            hidden_states = hidden_states_padded
        if input_layout == "BNSD":
            # b n s d -> b s n d
            hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = hidden_states.reshape(Bs, query_tokens, -1)
        hidden_states = hidden_states.to(query.dtype)
        return hidden_states

    def scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        input_layout,
        head_num=None,
        attention_mask=None,
        scale=None,
        dropout_p=0.0,
        is_causal=False,
    ) -> ms.Tensor:
        def trans_tensor_shape(x, layout, head_num):
            if layout == "BSH":
                batch = x.shape[0]
                x = x.view(batch, -1, head_num, x.shape[-1] // head_num).swapaxes(1, 2)
            elif layout == "SBH":
                batch = x.shape[1]
                x = x.view(-1, batch * head_num, x.shape[-1] // head_num).swapaxes(0, 1)
                x = x.view(batch, head_num, -1, x.shape[-1])
            return x

        query = trans_tensor_shape(query, input_layout, head_num)
        key = trans_tensor_shape(key, input_layout, head_num)
        value = trans_tensor_shape(value, input_layout, head_num)

        attn_weight = query @ key.swapaxes(-2, -1) * scale
        attn_bias = mint.zeros_like(attn_weight, dtype=query.dtype)
        if is_causal:
            assert attention_mask is None
            temp_mask = mint.zeros_like(attn_weight, dtype=ms.bool).tril(diagonal=0)
            attn_bias.masked_fill(~temp_mask, npu_config.inf_float)
            attn_bias.to(query.dtype)

        if attention_mask is not None:
            assert (
                not self.enable_FA
            ) and attention_mask.dtype != ms.bool, "attention_mask must not be bool type when use this function"

        attn_weight += attn_bias
        attn_weight = mint.nn.functional.softmax(attn_weight, dim=-1)
        attn_weight = mint.nn.functional.dropout(attn_weight, p=dropout_p, training=True)
        output = attn_weight @ value
        if input_layout == "BSH":
            output = output.swapaxes(1, 2).view(output.shape[0], -1, head_num * output.shape[-1])
        else:
            output = output.view(output.shape[0] * head_num, -1, output.shape[-1]).swapaxes(0, 1)
            output = output.view(output.shape[0], -1, head_num * output.shape[-1])
        return output


npu_config = NPUConfig()
