import argparse
import ast
import os

import numpy as np

import mindspore as ms
import mindspore.dataset as de
from mindspore import Profiler, nn


def create_loader(
    total_batch_size,
    size=(),
    dtypes=None,
    num_parallel_workers=1,
    shuffle=True,
    drop_remainder=True,
    python_multiprocessing=False,
    seed=1,
    dataset_column_names=["data"],
):
    dataset = Dataset(size=size, dtypes=dtypes)

    de.config.set_seed(seed)
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

    def construct(self, *args, **kwargs):
        out = self.network(*args, **kwargs)
        loss = ((out - 1) ** 2).mean()
        return loss


def main(args):
    # set context
    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_target="Ascend",
        device_id=int(os.getenv("DEVICE_ID", 0)),
        save_graphs=args.save_graphs,
        save_graphs_path=args.save_graphs_path,
    )

    args.rank, args.rank_size = 0, 1

    if args.profiler:
        profiler = Profiler()

    # run with backward
    input_dtype = None

    # create train network
    if args.net == "FrozenOpenCLIPEmbedder2":
        from gm.modules.embedders.modules import FrozenOpenCLIPEmbedder2
        from gm.util.util import auto_mixed_precision

        class FrozenOpenCLIPEmbedder2Wrapper(FrozenOpenCLIPEmbedder2):
            def construct(self, *args, **kwargs):
                outs = super(FrozenOpenCLIPEmbedder2Wrapper, self).construct(*args, **kwargs)
                return outs

        net = FrozenOpenCLIPEmbedder2Wrapper(
            arch="ViT-bigG-14-Text",
            freeze=True,
            layer="penultimate",
            always_return_pooled=True,
            legacy=False,
            require_pretrained=False,
        )
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 77),)
        input_dtype = (np.int32,)
        dataset_column_names = [
            "data1",
        ]
    else:
        raise NotImplementedError

    dataloader = create_loader(
        total_batch_size=args.bs,
        size=input_size,
        dtypes=input_dtype,
        rank_size=args.rank_size,
        rank=args.rank,
        dataset_column_names=dataset_column_names,
    )

    # loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    loader = dataloader.create_tuple_iterator(num_epochs=1)

    for i, data in enumerate(loader):
        out = net(*data)
        print(out)
        np.save("out.npy", out.asnumpy())
        break

    if args.save_checkpoint:
        os.makedirs(args.save_checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(args.save_checkpoint_path, f"rank_{args.rank}"), exist_ok=True)
        ms.save_checkpoint(
            net, os.path.join(args.save_checkpoint_path, f"rank_{args.rank}", f"{args.net}_{args.rank}.ckpt")
        )

    if args.profiler:
        profiler.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model parallel example")
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--mp", type=int, default=2)
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "ResBlock",
            "BasicTransformerBlock",
            "SpatialTransformer",
            "UNetModel",
            "VAE-Encoder",
            "GeneralConditioner",
            "ConcatTimestepEmbedderND",
            "FrozenCLIPEmbedder",
            "FrozenOpenCLIPEmbedder2",
            "SDXL",
            "MemoryEfficientCrossAttention",
            "BasicTransformerBlockFA",
            "SpatialTransformerFA",
            "UNetModelFA",
        ],
        default="FrozenOpenCLIPEmbedder2",
    )
    parser.add_argument("--save_checkpoint", type=ast.literal_eval, default=False)
    parser.add_argument("--save_checkpoint_path", type=str, default="./test_module_weights")
    parser.add_argument("--save_graphs", type=ast.literal_eval, default=False)
    parser.add_argument("--save_graphs_path", type=str, default="./irs")
    parser.add_argument("--profiler", type=ast.literal_eval, default=False)
    args, _ = parser.parse_known_args()

    print("=" * 100)
    print("Args: ", args)
    print("=" * 100)

    main(args)
