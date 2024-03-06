import argparse
import ast
import os

import numpy as np
from common import NetWithLoss, create_loader

import mindspore as ms
from mindspore import Profiler


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
    if args.net == "UNetModel":
        from gm.modules.diffusionmodules.openaimodel import UNetModel
        from gm.util.util import auto_mixed_precision

        # succes: use_recompute=False
        # fail: use_recompute=True, stream out limit
        _net = UNetModel(
            in_channels=4,
            out_channels=4,
            model_channels=320,
            attention_resolutions=[4, 2],
            num_res_blocks=2,
            channel_mult=[1, 2, 4],
            num_head_channels=64,
            use_spatial_transformer=True,
            use_linear_in_transformer=True,
            transformer_depth=[1, 2, 10],  # [1, 2, 10]
            context_dim=2048,
            adm_in_channels=2816,
            spatial_transformer_attn_type="vanilla",
            num_classes="sequential",
            legacy=False,
            use_recompute=False,
        )
        net = NetWithLoss(_net)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 4, 128, 128), (1,), (1, 77, 2048), (1, 2816))
        dataset_column_names = ["data1", "data2", "data3", "data4"]

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
        net = _net
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
        default="UNetModel",
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
