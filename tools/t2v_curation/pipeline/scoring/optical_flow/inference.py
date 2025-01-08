import argparse
import os

import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import Tensor, context, load_checkpoint, load_param_into_net
from mindspore.communication import get_group_size, get_rank, init
from tqdm import tqdm

from pipeline.datasets.utils import extract_frames
from pipeline.scoring.optical_flow.unimatch import UniMatch
from pipeline.scoring.utils import merge_scores


class VideoTextDataset:
    def __init__(self, meta_path, points=(0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.points = points

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample["path"]

        # extract frames
        images = extract_frames(path, points=self.points, backend="decord")

        # transform
        images = [np.array(img) for img in images]  # Convert PIL images to numpy arrays
        images = np.stack(images)  # [N, H, W, C]
        images = images.transpose((0, 3, 1, 2))  # [N, C, H, W]

        images = images.astype(np.float32)
        N, C, H, W = images.shape
        if H > W:
            images = images.transpose((0, 1, 3, 2))  # [N, C, W, H]

        images = Tensor(images)
        images = ops.interpolate(images, size=(320, 576), mode = 'bilinear', align_corners=True)

        return index, images

    def __len__(self):
        return len(self.meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--use_cpu", action="store_true", help="Whether to use CPU")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")  # don't use too large bs for UniMatch
    parser.add_argument("--skip_if_existing", action="store_true")
    parser.add_argument(
        "--ckpt_path_unimatch",
        type=str,
        default="pretrained_models/unimatch.ckpt",
        help="Load UniMatch model checkpoint.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_flow{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    if not args.use_cpu:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        init()
        rank_id = get_rank()
        rank_size = get_group_size()

    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task="flow",
    )
    param_dict = load_checkpoint(args.ckpt_path_unimatch)
    load_param_into_net(model, param_dict)
    model.set_train(False)

    dataset_generator = VideoTextDataset(meta_path=args.meta_path)
    if not args.use_cpu:
        dataset = ds.GeneratorDataset(
            source=dataset_generator,
            column_names=['index', 'images'],
            shuffle=False,
            num_shards=rank_size,
            shard_id=rank_id,
        )
    else:
        dataset = ds.GeneratorDataset(
            source=dataset_generator,
            column_names=['index', 'images'],
            shuffle=False,
        )
    dataset = dataset.batch(args.bs, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs=1)

    # compute optical flow scores
    indices_list = []
    scores_list = []
    for batch in tqdm(iterator):
        indices = batch["index"].asnumpy().tolist()
        images = batch["images"]

        B, N, C, H, W = images.shape

        batch_0 = images[:, :-1, :, :, :]
        batch_1 = images[:, 1:, :, :, :]

        batch_0 = batch_0.reshape((-1, C, H, W)).contiguous()
        batch_1 = batch_1.reshape((-1, C, H, W)).contiguous()

        res = model(
            batch_0,
            batch_1,
            attn_type="swin",
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            task="flow",
            pred_bidir_flow=False,
        )
        flow_preds = res["flow_preds"][-1]  # [B*(N-1), 2, H, W]
        flow_maps = flow_preds.reshape((B, N - 1, 2, H, W))
        flow_maps = flow_maps.transpose(0, 1, 3, 4, 2)  # [B, N-1, H, W, 2]

        abs_flow_maps = ops.Abs()(flow_maps)
        flow_scores = abs_flow_maps.mean(axis=(1, 2, 3, 4))  # Mean over N-1, H, W, C
        flow_scores = flow_scores.asnumpy().tolist()

        indices_list.extend(indices)
        scores_list.extend(flow_scores)

    # Allgather results if necessary
    if not args.use_cpu:
        allgather = ops.AllGather()
        indices_tensor = Tensor(indices_list, ms.int64)
        scores_tensor = Tensor(scores_list, ms.float32)
        indices_list = allgather(indices_tensor).asnumpy().tolist()
        scores_list = allgather(scores_tensor).asnumpy().tolist()

    if args.use_cpu or (not args.use_cpu and rank_id == 0):
        meta_new = merge_scores([(indices_list, scores_list)], dataset_generator.meta, column="flow")
        meta_new.to_csv(out_path, index=False)
        print(f"New meta with optical flow scores saved to '{out_path}'.")

if __name__ == "__main__":
    main()
