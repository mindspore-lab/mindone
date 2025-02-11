import argparse
import ast
import os
import time

from common import NetWithLoss, create_loader

import mindspore as ms
from mindspore import Profiler, nn


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
    run_bp = True
    input_dtype = None

    # create train network
    if args.net == "BasicTransformerBlockFA":
        from gm.modules.attention import BasicTransformerBlock
        from gm.util.util import auto_mixed_precision

        _net = BasicTransformerBlock(
            dim=640,
            n_heads=10,
            d_head=64,
            dropout=0.0,
            context_dim=2048,
            gated_ff=True,
            disable_self_attn=False,
            attn_mode="flash-attention",
            dp=args.dp,
            mp=args.mp,
        )
        net = NetWithLoss(_net)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 4096, 640), (1, 77, 2048))
        dataset_column_names = ["data1", "data2"]
    else:
        raise NotImplementedError

    if run_bp:
        optimizer = nn.SGD(net.trainable_params(), 1e-2)
        train_step_cell = nn.TrainOneStepCell(net, optimizer)
    else:
        train_step_cell = net

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
    s_time = time.time()
    for i, data in enumerate(loader):
        # x = Tensor(data['data'], ms.float32)
        loss = train_step_cell(*data)
        s_run_mode = "run fp/bp" if run_bp else "run fp"
        print(
            f"Step: {i}, input data shape: {[d.shape for d in data]}, "
            f"{s_run_mode}, "
            f"loss: {loss}, time cost: {(time.time() - s_time)*1000:.2f} ms"
        )
        if i > 9 or (args.profiler and i > 3):
            break  # early stop
        s_time = time.time()

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
        default="BasicTransformerBlockFA",
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
