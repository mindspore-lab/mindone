# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------

import argparse
import logging
import os

import numpy as np
from omegaconf import OmegaConf
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset, get_pred_name
from src.util import metric
from src.util.alignment import align_depth_least_square, depth2disparity, disparity2depth
from src.util.metric import MetricTracker
from tabulate import tabulate
from tqdm.auto import tqdm

import mindspore.dataset as ds
from mindspore import context
from mindspore import dtype as mstype

eval_metrics = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        choices=["CPU", "Ascend"],
        help="Device to run the evaluation.",
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Directory of depth predictions",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    # LS depth alignment
    parser.add_argument(
        "--alignment",
        choices=[None, "least_square", "least_square_disparity"],
        default=None,
        help="Method to estimate scale and shift between predictions and ground truth.",
    )
    parser.add_argument(
        "--alignment_max_res",
        type=int,
        default=None,
        help="Max operating resolution used for LS alignment",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda")

    args = parser.parse_args()

    prediction_dir = args.prediction_dir
    output_dir = args.output_dir

    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir

    alignment = args.alignment
    alignment_max_res = args.alignment_max_res

    no_cuda = args.no_cuda
    pred_suffix = ".npy"

    os.makedirs(output_dir, exist_ok=True)

    # -------------------- Device --------------------
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device)
    logging.info(f"Evaluation device: {args.device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL, dtype=mstype.float32
    )

    dataloader = ds.GeneratorDataset(
        source=dataset,
        column_names=["dict"],
        num_parallel_workers=1,
    ).batch(1)

    # -------------------- Eval metrics --------------------
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    # -------------------- Per-sample metric file head --------------------
    per_sample_filename = os.path.join(output_dir, "per_sample_metrics.csv")
    # write title
    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")

    # -------------------- Evaluate --------------------
    for data in tqdm(dataloader, desc="Evaluating"):
        data = data[0]
        # GT data
        depth_raw = data["depth_raw_linear"].squeeze().asnumpy()
        valid_mask = data["valid_mask_raw"].squeeze().asnumpy()
        rgb_name = data["rgb_relative_path"].item()

        # Load predictions
        rgb_basename = os.path.basename(rgb_name)
        pred_basename = get_pred_name(rgb_basename, dataset.name_mode, suffix=pred_suffix)
        pred_name = os.path.join(os.path.dirname(rgb_name), pred_basename)
        pred_path = os.path.join(prediction_dir, pred_name)
        depth_pred = np.load(pred_path)

        if not os.path.exists(pred_path):
            logging.warning(f"Can't find prediction: {pred_path}")
            continue

        # Align with GT using least square
        if "least_square" == alignment:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=depth_raw,
                pred_arr=depth_pred,
                valid_mask_arr=valid_mask,
                return_scale_shift=True,
                max_resolution=alignment_max_res,
            )
        elif "least_square_disparity" == alignment:
            # convert GT depth -> GT disparity
            gt_disparity, gt_non_neg_mask = depth2disparity(depth=depth_raw, return_mask=True)
            # LS alignment in disparity space
            pred_non_neg_mask = depth_pred > 0
            valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

            disparity_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_disparity,
                pred_arr=depth_pred,
                valid_mask_arr=valid_nonnegative_mask,
                return_scale_shift=True,
                max_resolution=alignment_max_res,
            )
            # convert to depth
            disparity_pred = np.clip(disparity_pred, a_min=1e-3, a_max=None)  # avoid 0 disparity
            depth_pred = disparity2depth(disparity_pred)

        # Clip to dataset min max
        depth_pred = np.clip(depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth)

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

        # Evaluate
        sample_metric = []

        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred, depth_raw, valid_mask).item()
            sample_metric.append(_metric.__str__())
            metric_tracker.update(_metric_name, _metric)

        # Save per-sample metric
        with open(per_sample_filename, "a+") as f:
            f.write(pred_name + ",")
            f.write(",".join(sample_metric))
            f.write("\n")

    # -------------------- Save metrics to file --------------------
    eval_text = f"Evaluation metrics:\n\
    of predictions: {prediction_dir}\n\
    on dataset: {dataset.disp_name}\n\
    with samples in: {dataset.filename_ls_path}\n"

    eval_text += f"min_depth = {dataset.min_depth}\n"
    eval_text += f"max_depth = {dataset.max_depth}\n"

    eval_text += tabulate([metric_tracker.result().keys(), metric_tracker.result().values()])

    metrics_filename = "eval_metrics"
    if alignment:
        metrics_filename += f"-{alignment}"
    metrics_filename += ".txt"

    _save_to = os.path.join(output_dir, metrics_filename)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
        logging.info(f"Evaluation metrics saved to {_save_to}")
