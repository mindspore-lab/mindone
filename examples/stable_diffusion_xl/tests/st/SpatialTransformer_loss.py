import argparse
import ast
import os
import time

import numpy as np

import mindspore as ms
import mindspore.dataset as de
from mindspore import Profiler, nn, ops


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
    run_bp = True
    input_dtype = None

    # create train network
    if args.net == "ResBlock":
        from gm.modules.diffusionmodules.openaimodel import ResBlock
        from gm.util.util import auto_mixed_precision

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
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, model_channels, 128, 128), (1, 1280))  # x, emb
        dataset_column_names = ["data1", "data2"]
    elif args.net == "MemoryEfficientCrossAttention":
        from gm.modules.attention import MemoryEfficientCrossAttention
        from gm.util.util import auto_mixed_precision

        _net = MemoryEfficientCrossAttention(
            query_dim=640, heads=10, dim_head=64, dropout=0.0, context_dim=None, dp=args.dp, mp=args.mp
        )
        net = NetWithLoss(_net)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 4096, 640),)
        dataset_column_names = [
            "data1",
        ]
    elif args.net == "BasicTransformerBlock":
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
            attn_mode="vanilla",
        )
        net = NetWithLoss(_net)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 4096, 640), (1, 77, 2048))
        dataset_column_names = ["data1", "data2"]
    elif args.net == "BasicTransformerBlockFA":
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
    elif args.net == "SpatialTransformer":
        from gm.modules.attention import SpatialTransformer
        from gm.util.util import auto_mixed_precision

        _net = SpatialTransformer(
            in_channels=640,
            n_heads=10,
            d_head=64,
            depth=2,
            dropout=0.0,
            context_dim=2048,
            disable_self_attn=False,
            use_linear=True,
            attn_type="vanilla",
        )
        net = NetWithLoss(_net)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 640, 64, 64), (1, 77, 2048))
        dataset_column_names = ["data1", "data2"]
    elif args.net == "SpatialTransformerFA":
        from gm.modules.attention import SpatialTransformer
        from gm.util.util import auto_mixed_precision

        _net = SpatialTransformer(
            in_channels=640,
            n_heads=10,
            d_head=64,
            depth=10,
            dropout=0.0,
            context_dim=2048,
            disable_self_attn=False,
            use_linear=True,
            attn_type="flash-attention",
            dp=args.dp,
            mp=args.mp,
        )
        net = NetWithLoss(_net)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 640, 64, 64), (1, 77, 2048))
        dataset_column_names = ["data1", "data2"]
    elif args.net == "UNetModel":
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
    elif args.net == "UNetModelFA":
        from gm.modules.diffusionmodules.openaimodel import UNetModel
        from gm.util.util import auto_mixed_precision

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
            spatial_transformer_attn_type="flash-attention",
            num_classes="sequential",
            legacy=False,
            use_recompute=False,
            dp=args.dp,
            mp=args.mp,
        )
        net = NetWithLoss(_net)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 4, 128, 128), (1,), (1, 77, 2048), (1, 2816))
        dataset_column_names = ["data1", "data2", "data3", "data4"]
    elif args.net == "VAE-Encoder":
        from gm.models.autoencoder import AutoencoderKLInferenceWrapper
        from gm.util.util import auto_mixed_precision

        ae = AutoencoderKLInferenceWrapper(
            embed_dim=4,
            monitor="val/rec_loss",
            ddconfig={
                "attn_type": "vanilla",
                "double_z": True,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 2, 4, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
                "decoder_attn_dtype": "fp16",
            },
        )

        class EncoderWrapper(nn.Cell):
            def __init__(self, ae):
                super(EncoderWrapper, self).__init__()
                self.ae = ae

            def construct(self, x):
                return self.ae.encode(x)

        _net = EncoderWrapper(ae)
        net = NetWithLoss(_net)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 3, 1024, 1024),)
        dataset_column_names = [
            "data1",
        ]
        run_bp = False
    elif args.net == "ConcatTimestepEmbedderND":
        from gm.modules.embedders.modules import ConcatTimestepEmbedderND
        from gm.util.util import auto_mixed_precision

        class ConcatTimestepEmbedderNDWrapper(ConcatTimestepEmbedderND):
            pass

        _net = ConcatTimestepEmbedderNDWrapper(outdim=256)
        net = NetWithLoss(_net)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 2),)
        dataset_column_names = [
            "data1",
        ]
        run_bp = False
    elif args.net == "FrozenCLIPEmbedder":
        from gm.modules.embedders.modules import FrozenCLIPEmbedder
        from gm.util.util import auto_mixed_precision

        class FrozenCLIPEmbedderWrapper(FrozenCLIPEmbedder):
            pass

        _net = FrozenCLIPEmbedderWrapper(layer="hidden", layer_idx=11, version="openai/clip-vit-large-patch14")
        net = NetWithLoss(_net)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 77),)
        input_dtype = (np.int32,)
        dataset_column_names = [
            "data1",
        ]
        run_bp = False
    elif args.net == "FrozenOpenCLIPEmbedder2":
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
        run_bp = False
    elif args.net == "GeneralConditioner":
        from gm.modules.embedders.modules import GeneralConditioner
        from gm.util.util import auto_mixed_precision
        from omegaconf import OmegaConf

        config = OmegaConf.load("./configs/inference/sd_xl_base.yaml")
        conditioner_config = config.model.get("params", dict())["conditioner_config"]

        class Wrapper(GeneralConditioner):
            def construct(self, *args, **kwargs):
                vector, crossattn, concat = super(Wrapper, self).construct(*args, **kwargs)
                return vector, crossattn

        net = Wrapper(**conditioner_config.get("params", dict()))
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 77), (1, 77), (1, 2), (1, 2), (1, 2))
        input_dtype = (np.int32, np.int32, np.float32, np.float32, np.float32)
        dataset_column_names = ["data1", "data2", "data3", "data4", "data5"]
        run_bp = False
    elif args.net == "SDXL":
        from gm.models.autoencoder import AutoencoderKLInferenceWrapper
        from gm.modules.diffusionmodules.openaimodel import UNetModel
        from gm.util.util import auto_mixed_precision

        ae = AutoencoderKLInferenceWrapper(
            embed_dim=4,
            monitor="val/rec_loss",
            ddconfig={
                "attn_type": "vanilla",
                "double_z": True,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 2, 4, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
                "decoder_attn_dtype": "fp16",
            },
        )
        unet = UNetModel(
            in_channels=4,
            out_channels=4,
            model_channels=320,
            attention_resolutions=[4, 2],
            num_res_blocks=2,
            channel_mult=[1, 2, 4],
            num_head_channels=64,
            use_spatial_transformer=True,
            use_linear_in_transformer=True,
            transformer_depth=[1, 2, 2],  # [1, 2, 10]
            context_dim=2048,
            adm_in_channels=2816,
            spatial_transformer_attn_type="vanilla",
            num_classes="sequential",
            legacy=False,
            use_recompute=True,
            dp=args.dp,
            mp=args.mp,
        )

        class SDXLWrapper(nn.Cell):
            def __init__(self, ae, unet):
                super(SDXLWrapper, self).__init__()
                self.ae = ae
                self.unet = NetWithLoss(unet)
                optimizer = nn.SGD(unet.trainable_params(), learning_rate=1e-2)
                self.train_net = nn.TrainOneStepCell(self.unet, optimizer)

                # # freeze ae parameters
                # for p in self.get_parameters():
                #     p.requires_grad = False

            def construct(self, *args):
                x = args[0]
                x = self.ae.encode(x)
                x = ops.stop_gradient(x)
                # out = self.unet(x, *args[1:])
                out = self.train_net(x, *args[1:])
                return out

        net = SDXLWrapper(ae, unet)
        net = auto_mixed_precision(net, amp_level="O0")
        input_size = ((1, 3, 1024, 1024), (1,), (1, 77, 2048), (1, 2816))
        dataset_column_names = ["data1", "data2", "data3", "data4"]
        run_bp = False
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
        default="SpatialTransformer",
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
