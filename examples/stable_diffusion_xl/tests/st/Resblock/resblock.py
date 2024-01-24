import os
import time
import ast
import argparse
import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor, context, Profiler
from mindspore.communication.management import get_group_size, get_rank, init
import mindspore.dataset as de
from mindspore.train.amp import AMP_BLACK_LIST, AMP_WHITE_LIST, _auto_white_list, _auto_black_list

import numpy as np

import mindspore as ms
from mindspore import nn, ops
from typing import Iterable





class ZeroInitModule(nn.Cell):
    def __init__(self, module):
        super(ZeroInitModule, self).__init__(auto_prefix=False)
        self.module = module
        for n, p in self.parameters_and_names():
            ops.assign(p, ops.zeros_like(p))

    def construct(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


def make_beta_schedule(
        schedule,
        n_timestep,
        linear_start=1e-4,
        linear_end=2e-2,
):
    if schedule == "linear":
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float) ** 2
    else:
        raise NotImplementedError

    return betas


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False, dtype=ms.float32):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = ops.exp(
            -ops.log(ops.ones(1, dtype=ms.float32) * max_period)
            * ops.arange(start=0, end=half, dtype=ms.float32)
            / half
        )
        args = timesteps[:, None].astype(ms.float32) * freqs[None]
        embedding = ops.concat((ops.cos(args), ops.sin(args)), axis=-1)
        if dim % 2:
            embedding = ops.concat((embedding, ops.zeros_like(embedding[:, :1])), axis=-1)
    else:
        embedding = ops.broadcast_to(timesteps[:, None], (-1, dim))
    return embedding.astype(dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for n, p in module.parameters_and_names():
        ops.assign(p, ops.zeros_like(p))
    return module


def normalization(channels, eps=1e-5):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels, eps)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(
            *args, has_bias=kwargs.pop("has_bias", True), pad_mode=kwargs.pop("pad_mode", "valid"), **kwargs
        )
    elif dims == 2:
        return nn.Conv2d(
            *args, has_bias=kwargs.pop("has_bias", True), pad_mode=kwargs.pop("pad_mode", "valid"), **kwargs
        )
    elif dims == 3:
        return nn.Conv3d(
            *args, has_bias=kwargs.pop("has_bias", True), pad_mode=kwargs.pop("pad_mode", "valid"), **kwargs
        )
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Dense(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Upsample(nn.Cell):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_up=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding, pad_mode="pad")

    def construct(self, x):
        # assert x.shape[1] == self.channels
        if self.dims == 3:
            t_factor = 1 if not self.third_up else 2

            # x = ops.interpolate(x, size=(t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest",)
            x = ops.ResizeNearestNeighbor(
                size=(t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
            )(x)
        else:
            # x = ops.interpolate(x, size=(x.shape[-2] * 2, x.shape[-1] * 2), mode="nearest")  # scale_factor=2., (not support with ms2.1)
            x = ops.ResizeNearestNeighbor(
                size=(x.shape[-2] * 2, x.shape[-1] * 2),
            )(x)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, third_down=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        if use_conv:
            print(f"Building a Downsample layer with {dims} dims.")
            print(
                f"  --> settings are: \n in-chn: {self.channels}, out-chn: {self.out_channels}, "
                f"kernel-size: 3, stride: {stride}, padding: {padding}"
            )
            if dims == 3:
                print(f"  --> Downsampling third axis (time): {third_down}")
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding, pad_mode="pad")
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def construct(self, x):
        # assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock():
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            up=False,
            down=False,
            kernel_size=3,
            exchange_temb_dims=False,
            skip_t_emb=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.SequentialCell(
            [
                normalization(channels),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding, pad_mode="pad"),
            ]
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        if self.skip_t_emb:
            print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.SequentialCell(
                [
                    nn.SiLU(),
                    linear(
                        emb_channels,
                        self.emb_out_channels,
                    ),
                ]
            )

        self.out_layers = nn.SequentialCell(
            [
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, pad_mode="pad")
                ),
            ]
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding, pad_mode="pad"
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def construct(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = ops.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = ops.chunk(emb_out, 2, axis=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                # emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                emb_out = emb_out.swapaxes(1, 2)  # (b, t, c, ...) -> (b, c, t, ...)
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def create_loader(
        total_batch_size,
        size=(),
        dtypes=None,
        rank=None,
        rank_size=None,
        num_parallel_workers=1,
        shuffle=True,
        drop_remainder=True,
        python_multiprocessing=False,
        seed=1,
        dataset_column_names=['data']
):
    dataset = Dataset(size=size, dtypes=dtypes)

    de.config.set_seed(seed)
    if rank_size is not None and rank_size > 1:
        ds = de.GeneratorDataset(
            dataset,
            column_names=dataset_column_names,
            num_parallel_workers=min(8, num_parallel_workers),
            shuffle=shuffle,
            python_multiprocessing=python_multiprocessing,
            num_shards=rank_size,
            shard_id=rank,
        )
        per_batch_size = max(total_batch_size // rank_size, 1)
    else:
        ds = de.GeneratorDataset(
            dataset,
            column_names=dataset_column_names,
            num_parallel_workers=min(8, num_parallel_workers),
            shuffle=shuffle,
            python_multiprocessing=python_multiprocessing,
        )
        per_batch_size = total_batch_size

    ds = ds.batch(
        per_batch_size,
        drop_remainder=drop_remainder,
    )
    ds = ds.repeat(1)

    return ds


class Dataset:
    def __init__(
            self,
            size=(),
            dtypes=None,
    ):
        super().__init__()
        self.size = size
        self.dtyps = dtypes
        self.input_num = len(self.size)

        assert self.input_num > 0
        assert (self.dtyps is None) or (len(self.dtyps) == len(self.size))

    def __getitem__(self, idx):
        out = ()
        for i in range(len(self.size)):
            s = self.size[i]  # delete batch dim
            dtype = np.float32 if self.dtyps is None else self.dtyps[i]
            if len(s) > 1:
                s = s[1:]
                out += (np.random.randn(*s).astype(dtype),)
            else:
                # timestep
                out += (np.array(np.random.randint(0, 1000), dtype=dtype),)
        return out

    def __len__(self):
        return 100


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.network = network

    def shard(self, dp=1, mp=1):
        self.network.shard(dp=dp, mp=mp)

    def construct(self, *args, **kwargs):
        out = self.network(*args, **kwargs)
        loss = ((out - 1) ** 2).mean()
        return loss


def main(args):
    def auto_mixed_precision(network, amp_level="O0"):
        if not isinstance(network, nn.Cell):
            raise TypeError("The network type should be Cell.")

        if amp_level == "O0":
            pass
        elif amp_level == "O1":
            return _auto_white_list(network, AMP_WHITE_LIST)
        elif amp_level == "O2":
            _auto_black_list(
                network,
                AMP_BLACK_LIST
                + [
                    nn.GroupNorm,
                ],
            )
        elif amp_level == "O3":
            network.to_float(ms.float16)
        else:
            raise ValueError("The amp level {} is not supported".format(amp_level))
        return network
    # set context
    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_target="Ascend",
        device_id=int(os.getenv("DEVICE_ID", 0)),
        save_graphs=args.save_graphs,
        save_graphs_path=args.save_graphs_path
    )
    if args.is_parallel:
        init()
        args.rank, args.rank_size = get_rank(), get_group_size()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.SEMI_AUTO_PARALLEL,
            enable_parallel_optimizer=True,
            parallel_optimizer_config={
                "gradient_accumulation_shard": False,
                "parallel_optimizer_threshold": 64
            },
            strategy_ckpt_save_file="./test_src_strategy.ckpt",
            device_num=args.rank_size,
            gradients_mean=False,
            enable_alltoall=False,
            full_batch=True,
            search_mode="sharding_propagation",
        )
    else:
        args.rank, args.rank_size = 0, 1

    if args.profiler:
        profiler = Profiler()

    # run with backward
    run_bp = True
    input_dtype = None

    # create train network
    if args.net == "ResBlock":
        mult, model_channels, use_scale_shift_norm = 2, 320, False
        _net = ResBlock(
            model_channels,
            model_channels * 4,
            dropout=0.0,
            out_channels=mult * model_channels,
            dims=2,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        net = NetWithLoss(_net)

        net = auto_mixed_precision(net, amp_level="O2")
        input_size = ((1, model_channels, 128, 128), (1, 1280))  # x, emb
        dataset_column_names = ["data1", "data2"]

    if args.is_parallel:
        net.shard(dp=args.dp, mp=args.mp)

    if run_bp:
        optimizer = nn.SGD(net.trainable_params(), 1e-2)
        train_step_cell = nn.TrainOneStepCell(net, optimizer)
    else:
        train_step_cell = net

    if args.is_parallel:
        dataloader = create_loader(total_batch_size=args.bs, size=input_size, dtypes=input_dtype,
                                   rank_size=None, rank=None,
                                   dataset_column_names=dataset_column_names)
    else:
        dataloader = create_loader(total_batch_size=args.bs, size=input_size, dtypes=input_dtype,
                                   rank_size=args.rank_size, rank=args.rank,
                                   dataset_column_names=dataset_column_names)
    loader = dataloader.create_tuple_iterator(num_epochs=1)
    s_time = time.time()
    for i, data in enumerate(loader):
        loss = train_step_cell(*data)
        s_run_mode = "run fp/bp" if run_bp else "run fp"
        print(f"Step: {i}, input data shape: {[d.shape for d in data]}, "
              f"{s_run_mode}, "
              f"loss: {loss}, time cost: {(time.time() - s_time) * 1000:.2f} ms")
        if i > 9 or (args.profiler and i > 3):
            break  # early stop
        s_time = time.time()

    if args.save_checkpoint:
        os.makedirs(args.save_checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(args.save_checkpoint_path, f"rank_{args.rank}"), exist_ok=True)
        ms.save_checkpoint(net, os.path.join(
            args.save_checkpoint_path,
            f"rank_{args.rank}",
            f"{args.net}_{args.rank}.ckpt"
        ))

    if args.profiler:
        profiler.analyse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model parallel example")
    parser.add_argument("--is_parallel", type=ast.literal_eval, default=True)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--mp", type=int, default=2)
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "ResBlock", "BasicTransformerBlock", "SpatialTransformer", "UNetModel",
            "VAE-Encoder", "GeneralConditioner", "ConcatTimestepEmbedderND", "FrozenCLIPEmbedder",
            "FrozenOpenCLIPEmbedder2",
            "SDXL", "SDXL_MultiGraph", "SDXL_MultiGraph_Dev",
            "MemoryEfficientCrossAttention", "BasicTransformerBlockFA", "SpatialTransformerFA", "UNetModelFA"
        ],
        default="ResBlock"
    )
    # parser.add_argument("--save_checkpoint", type=ast.literal_eval, default=False)
    # parser.add_argument("--save_checkpoint_path", type=str, default="./test_module_weights")
    # parser.add_argument("--save_graphs", type=ast.literal_eval, default=False)
    # parser.add_argument("--save_graphs_path", type=str, default="./irs")
    # parser.add_argument("--profiler", type=ast.literal_eval, default=False)
    args, _ = parser.parse_known_args()
    print("=" * 100)
    print("Args: ", args)
    print("=" * 100)
    main(args)
