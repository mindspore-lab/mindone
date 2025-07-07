from time import time

import numpy as np
from cogvideox.acceleration import create_parallel_group, get_sequence_parallel_group
from cogvideox.models.cogvideox_transformer_3d_sp import CogVideoXTransformer3DModel_SP

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.communication import get_group_size, init

from mindone.diffusers import CogVideoXTransformer3DModel
from mindone.models.modules.parallel import PARALLEL_MODULES
from mindone.trainers.zero import _prepare_network
from mindone.utils.seed import set_random_seed

THRESHOLD_FP16 = 5e-3
THRESHOLD_BF16 = 5e-2


class MeanNet(nn.Cell):
    def __init__(self, net: nn.Cell) -> None:
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        output = self.net(*inputs)
        return output[0].mean()


def get_sample_data(use_rotary_positional_embeddings=False, dtype=ms.bfloat16):
    batch_size = 1
    frame = (77 - 1) // 4 + 1
    w = 1360 // 8
    h = 768 // 8
    max_s = 224
    x = mint.rand([batch_size, frame, 16, h, w], dtype=dtype)  # (B, C, T, H, W)
    timestep = ms.Tensor(
        [
            32,
        ]
        * batch_size,
        dtype=ms.int64,
    )
    y = mint.rand(batch_size, max_s, 4096, dtype=dtype)
    if use_rotary_positional_embeddings:
        image_rotary_emb = mint.rand(2, frame * h * w // 8, 64, dtype=dtype)
    else:
        image_rotary_emb = None
    return dict(
        hidden_states=x,
        encoder_hidden_states=y,
        timestep=timestep,
        timestep_cond=None,
        ofs=None,
        image_rotary_emb=image_rotary_emb,
        attention_kwargs=None,
        return_dict=None,
    )


def get_model_config():
    config = {
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 64,
        "dropout": 0.0,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "in_channels": 16,
        "max_text_seq_length": 226,
        "norm_elementwise_affine": True,
        "norm_eps": 1e-05,
        "num_attention_heads": 48,
        "num_layers": 42,
        "out_channels": 16,
        "patch_bias": False,
        "patch_size": 2,
        "patch_size_t": 2,
        "sample_frames": 81,
        "sample_height": 300,
        "sample_width": 300,
        "spatial_interpolation_scale": 1.875,
        "temporal_compression_ratio": 4,
        "temporal_interpolation_scale": 1.0,
        "text_embed_dim": 4096,
        "time_embed_dim": 512,
        "timestep_activation_fn": "silu",
        "use_learned_positional_embeddings": False,
        "use_rotary_positional_embeddings": True,
    }
    return config


def run_performence(
    mode: int = 0,
    jit_level="O0",
    dtype=ms.bfloat16,
    zero_stage=0,
    dist_model=True,
    run_perf_step=10,
    use_rotary_positional_embeddings=True,
    gradient_checkpointing=True,
    fa_checkpointing=True,
    get_profile=False,
    output_path="./data",
):
    ms.set_context(mode=mode)
    ms.set_context(jit_config={"jit_level": jit_level})
    init()
    ms.context.set_auto_parallel_context(parallel_mode="data_parallel")

    # prepare data
    set_random_seed(1024)
    create_parallel_group(get_group_size())
    data = get_sample_data(use_rotary_positional_embeddings=use_rotary_positional_embeddings, dtype=dtype)

    # single model
    set_random_seed(1024)
    model_cfg = get_model_config()
    model_cfg["fa_checkpointing"] = fa_checkpointing
    print(f"model_cfg:\n {model_cfg}")
    if dist_model:
        model_cfg["enable_sequence_parallelism"] = True
        dist_model = CogVideoXTransformer3DModel_SP(**model_cfg).to(dtype)
        transformer_parameters = dist_model.trainable_params()
        num_trainable_parameters = sum(param.numel() for param in transformer_parameters)
        if zero_stage == 3:
            dist_model = _prepare_network(dist_model, "hccl_world_group", PARALLEL_MODULES)
        dist_model(**data)
        print(f"=== Num trainable parameters = {num_trainable_parameters/1e9:.2f} B")
        print("=" * 10, "run_pref", "=" * 10)
        start = time()
        for i in range(run_perf_step):
            dist_model(**data)
        print(f"=== dist_out forward cost: {(time() - start) / run_perf_step} s")

        # test backward
        if gradient_checkpointing:
            dist_model.enable_gradient_checkpointing()
        dist_mean_net = MeanNet(dist_model)

        dist_grad_fn = ops.value_and_grad(dist_mean_net, grad_position=None, weights=dist_mean_net.trainable_params())
        dist_grad_fn(*data.values())

        print("=" * 10, "run_pref", "=" * 10)
        if get_profile:
            profiler = ms.Profiler(
                start_profile=True, output_path=output_path, profiler_level=ms.profiler.ProfilerLevel.Level1
            )
        start = time()
        for i in range(run_perf_step):
            dist_grad_fn(*data.values())
        print(f"=== dist_out forward & backward cost: {(time() - start) / run_perf_step} s")
        if get_profile:
            profiler.stop()
            profiler.analyse()
    else:
        model_cfg["enable_sequence_parallelism"] = False
        non_dist_model = CogVideoXTransformer3DModel_SP(**model_cfg).to(dtype)
        transformer_parameters = non_dist_model.trainable_params()
        num_trainable_parameters = sum(param.numel() for param in transformer_parameters)
        non_dist_model(**data)
        print(f"=== Num trainable parameters = {num_trainable_parameters / 1e9:.2f} B")
        print("=" * 10, "run_pref", "=" * 10)
        start = time()
        for i in range(run_perf_step):
            non_dist_model(**data)
        print(f"=== non_dist_out forward cost: {(time() - start) / run_perf_step} s")

        # test backward
        if gradient_checkpointing:
            non_dist_model.enable_gradient_checkpointing()
        non_dist_mean_net = MeanNet(non_dist_model)

        non_dist_grad_fn = ops.value_and_grad(
            non_dist_mean_net, grad_position=None, weights=non_dist_mean_net.trainable_params()
        )
        non_dist_grad_fn(*data.values())

        print("=" * 10, "run_pref", "=" * 10)
        if get_profile:
            profiler = ms.Profiler(
                start_profile=True, output_path=output_path, profiler_level=ms.profiler.ProfilerLevel.Level1
            )
        start = time()
        for i in range(run_perf_step):
            non_dist_grad_fn(*data.values())
        print(f"=== non_dist_out forward & backward cost: {(time() - start) / run_perf_step} s")
        if get_profile:
            profiler.stop()
            profiler.analyse()


def run_model(
    mode: int = 0,
    jit_level="O0",
    dtype=ms.bfloat16,
    zero_stage=0,
    use_rotary_positional_embeddings=True,
    run_pref=True,
    run_perf_step=10,
    gradient_checkpointing=True,
    fa_checkpointing=True,
):
    ms.set_context(mode=mode)
    ms.set_context(jit_config={"jit_level": jit_level})
    init()
    ms.context.set_auto_parallel_context(parallel_mode="data_parallel")

    # prepare data
    set_random_seed(1024)
    data = get_sample_data(use_rotary_positional_embeddings=use_rotary_positional_embeddings, dtype=dtype)
    threshold = THRESHOLD_BF16 if dtype == ms.bfloat16 else THRESHOLD_FP16
    # single model
    set_random_seed(1024)
    model_cfg = get_model_config()
    print(f"model_cfg:\n {model_cfg}")
    non_dist_model = CogVideoXTransformer3DModel(**model_cfg).to(dtype)

    # sequence parallel model
    create_parallel_group(get_group_size())
    set_random_seed(1024)
    model_cfg["fa_checkpointing"] = fa_checkpointing
    model_cfg["enable_sequence_parallelism"] = True
    dist_model = CogVideoXTransformer3DModel_SP(**model_cfg).to(dtype)

    for (_, w0), (_, w1) in zip(non_dist_model.parameters_and_names(), dist_model.parameters_and_names()):
        w1.set_data(w0)  # FIXME: seed does not work
        np.testing.assert_allclose(w0.value().to(ms.float32).asnumpy(), w1.value().to(ms.float32).asnumpy())

    if zero_stage == 3:
        dist_model = _prepare_network(dist_model, "hccl_world_group", PARALLEL_MODULES)

    # test forward
    non_dist_out = non_dist_model(**data)
    dist_out = dist_model(**data)

    if isinstance(non_dist_out, ms.Tensor):
        non_dist_out = (non_dist_out,)
    if isinstance(dist_out, ms.Tensor):
        dist_out = (dist_out,)

    for non_dist_o, dist_o in zip(non_dist_out, dist_out):
        np.testing.assert_allclose(non_dist_o.to(ms.float32).asnumpy(), dist_o.to(ms.float32).asnumpy(), atol=threshold)

    if run_pref:
        print("=" * 10, "run_pref", "=" * 10)
        start = time()
        for i in range(run_perf_step):
            non_dist_out = non_dist_model(**data)
        print(f"=== non_dist_out forward cost: {(time() - start)/run_perf_step} s")
        start = time()
        for i in range(run_perf_step):
            dist_out = dist_model(**data)
        print(f"=== dist_out forward cost: {(time() - start) / run_perf_step} s")
    print("Test 1 (Forward): Passed.")

    # test backward
    if gradient_checkpointing:
        non_dist_model.enable_gradient_checkpointing()
        dist_model.enable_gradient_checkpointing()
    non_dist_mean_net = MeanNet(non_dist_model)
    dist_mean_net = MeanNet(dist_model)

    non_dist_grad_fn = ops.value_and_grad(
        non_dist_mean_net, grad_position=None, weights=non_dist_mean_net.trainable_params()
    )
    non_dist_loss, non_dist_grads = non_dist_grad_fn(*data.values())

    dist_grad_fn = ops.value_and_grad(dist_mean_net, grad_position=None, weights=dist_mean_net.trainable_params())
    dist_loss, dist_grads = dist_grad_fn(*data.values())

    # take mean around different ranks
    sp_group = get_sequence_parallel_group()
    reduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=sp_group)
    gather = ops.AllGather(group=sp_group)
    num = get_group_size()
    syn_dist_grads = list()
    for x, y in zip(dist_grads, non_dist_grads):
        if zero_stage != 3:
            syn_dist_grads.append(reduce(x) / num)
        elif x.shape != y.shape:
            syn_dist_grads.append(gather(x))
        else:
            syn_dist_grads.append(x)

    np.testing.assert_allclose(
        non_dist_loss.to(ms.float32).asnumpy(), dist_loss.to(ms.float32).asnumpy(), atol=threshold
    )

    for grad_0, grad_1 in zip(non_dist_grads, syn_dist_grads):
        np.testing.assert_allclose(grad_0.to(ms.float32).asnumpy(), grad_1.to(ms.float32).asnumpy(), atol=threshold)
    if run_pref:
        print("=" * 10, "run_pref", "=" * 10)
        start = time()
        for i in range(run_perf_step):
            non_dist_loss, non_dist_grads = non_dist_grad_fn(*data.values())
        print(f"=== non_dist_out forward & backward cost: {(time() - start) / run_perf_step} s")
        start = time()
        for i in range(run_perf_step):
            dist_loss, dist_grads = dist_grad_fn(*data.values())
        print(f"=== dist_out forward & backward cost: {(time() - start) / run_perf_step} s")
    print("Test 2 (Backward): Passed.")


if __name__ == "__main__":
    mode = ms.GRAPH_MODE
    jit_level = "O1"
    dtype = ms.bfloat16
    zero_stage = 3

    # get max diff compare to original diffuser
    # Note: layer num = 42 will OOM, when run acc, layer num better be set 12.
    run_acc = False
    gradient_checkpointing = True
    fa_checkpointing = False

    # get performence when run_acc or run standalone
    run_perf = True
    # only apply when run_acc=False and run_perf=True
    run_perf_step = 3
    dist_model = True
    get_profile = True
    output_path = "data"
    if run_acc:
        run_model(
            mode=mode,
            jit_level=jit_level,
            dtype=dtype,
            zero_stage=zero_stage,
            run_pref=run_perf,
            run_perf_step=run_perf_step,
            gradient_checkpointing=gradient_checkpointing,
            fa_checkpointing=fa_checkpointing,
        )
    elif run_perf:
        run_performence(
            mode=mode,
            jit_level=jit_level,
            dtype=dtype,
            zero_stage=zero_stage,
            dist_model=dist_model,
            run_perf_step=run_perf_step,
            gradient_checkpointing=gradient_checkpointing,
            fa_checkpointing=fa_checkpointing,
            get_profile=get_profile,
            output_path=output_path,
        )
    else:
        print("Nothing to Running!!!")
