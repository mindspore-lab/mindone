import os
import warnings

import numpy as np
import pandas as pd
from config import parse_args
from pipeline.datasets.utils import extract_frames, is_video, pil_loader
from pipeline.scoring.utils import merge_scores
from text_system import TextSystem
from tqdm import tqdm

import mindspore as ms
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.mint.distributed import all_gather, all_gather_object, get_rank, get_world_size, init_process_group


class VideoTextDataset:
    # By default, we use the middle frame for OCR
    def __init__(self, meta_path, transform=None):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample["path"]

        # extract frames
        if not is_video(path):
            images = [pil_loader(path)]
        else:
            num_frames = sample["num_frames"] if "num_frames" in sample else None
            images = extract_frames(path, points=[0.5], backend="decord", num_frames=num_frames)

        # transform & stack
        if self.transform is not None:
            images = [self.transform(img) for img in images]

        # height and width if exist
        height = sample["height"] if "height" in sample else -1
        width = sample["width"] if "width" in sample else -1

        return index, images, height, width

    def __len__(self):
        return len(self.meta)


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_ocr{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init_process_group()

    # initialize the TextSystem
    text_system = TextSystem(args)

    raw_dataset = VideoTextDataset(args.meta_path)
    rank_id = get_rank()
    rank_size = get_world_size()
    dataset = ds.GeneratorDataset(
        source=raw_dataset,
        column_names=["index", "images", "height", "width"],
        shuffle=False,
        num_shards=rank_size,
        shard_id=rank_id,
    )
    # DO NOT set Batch size > 1 unless all the images have the same shape, else error
    dataset = dataset.batch(args.bs, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs=1)

    # lists to store results
    indices_list = []
    ocr_results_list = []
    num_boxes_list = []
    max_single_percentage_list = []
    total_text_percentage_list = []

    compute_num_boxes = args.num_boxes
    compute_max_single_text_box_area_percentage = args.max_single_percentage
    compute_total_text_area_percentage = args.total_text_percentage

    # check for 'height' and 'width' columns if needed
    if compute_max_single_text_box_area_percentage or compute_total_text_area_percentage:
        if "height" not in raw_dataset.meta.columns or "width" not in raw_dataset.meta.columns:
            warnings.warn(
                "Columns 'height' and 'width' are not available in the input CSV. "
                "Cannot compute text area percentages."
            )
            compute_max_single_text_box_area_percentage = False
            compute_total_text_area_percentage = False

    # OCR detection and recognition
    for batch in tqdm(iterator):
        indices = batch["index"].asnumpy()
        images_batch = batch["images"].asnumpy()
        heights = batch["height"].asnumpy()
        widths = batch["width"].asnumpy()

        batch_ocr_results = []
        batch_num_boxes = []
        batch_max_single_percentage = []
        batch_total_text_percentage = []

        for image, height, width in zip(images_batch, heights, widths):
            image = image.squeeze()

            # OCR
            boxes, text_scores, _ = text_system(image)
            ocr_result = repr({"boxes": boxes, "texts": text_scores})  # as string for storing in csv
            batch_ocr_results.append(ocr_result)

            if compute_num_boxes:
                num_boxes = len(boxes)
            else:
                num_boxes = None
            batch_num_boxes.append(num_boxes)

            if (
                (compute_max_single_text_box_area_percentage or compute_total_text_area_percentage)
                and height is not None
                and width is not None
            ):
                image_area = height * width

                box_areas = []
                for box in boxes:
                    x = box[:, 0]
                    y = box[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    box_areas.append(area)

                if compute_total_text_area_percentage:
                    total_text_area = sum(box_areas)
                    total_text_area_percentage = total_text_area / image_area if image_area > 0 else 0
                else:
                    total_text_area_percentage = None

                if compute_max_single_text_box_area_percentage:
                    if box_areas:
                        max_single_text_box_area = max(box_areas)
                        max_single_text_box_area_percentage = (
                            max_single_text_box_area / image_area if image_area > 0 else 0
                        )
                    else:
                        max_single_text_box_area_percentage = 0
                else:
                    max_single_text_box_area_percentage = None
            else:
                total_text_area_percentage = None
                max_single_text_box_area_percentage = None

            batch_total_text_percentage.append(total_text_area_percentage)
            batch_max_single_percentage.append(max_single_text_box_area_percentage)

        indices_list.extend(indices.tolist())
        ocr_results_list.extend(batch_ocr_results)
        num_boxes_list.extend(batch_num_boxes)
        max_single_percentage_list.extend(batch_max_single_percentage)
        total_text_percentage_list.extend(batch_total_text_percentage)

    if rank_size > 1:
        # indices
        indices_list = Tensor(np.array(indices_list), dtype=ms.int64)
        indices_list_all = [Tensor(np.zeros(indices_list.shape, dtype=np.int64)) for _ in range(rank_size)]
        all_gather(indices_list_all, indices_list)
        indices_list_all = ops.Concat(axis=0)(indices_list_all).asnumpy().tolist()

        # num_boxes
        if compute_num_boxes:
            num_boxes_list = Tensor(np.array(num_boxes_list), dtype=ms.int32)
            num_boxes_list_all = [Tensor(np.zeros(num_boxes_list.shape, dtype=np.int32)) for _ in range(rank_size)]
            all_gather(num_boxes_list_all, num_boxes_list)
            num_boxes_list_all = ops.Concat(axis=0)(num_boxes_list_all).asnumpy().tolist()
        else:
            num_boxes_list_all = None

        # max_single_percentage_list
        if compute_max_single_text_box_area_percentage:
            max_single_percentage_list = Tensor(np.array(max_single_percentage_list), dtype=ms.float32)
            max_single_percentage_list_all = [
                Tensor(np.zeros(max_single_percentage_list.shape, dtype=np.float32)) for _ in range(rank_size)
            ]
            all_gather(max_single_percentage_list_all, max_single_percentage_list)
            max_single_percentage_list_all = ops.Concat(axis=0)(max_single_percentage_list_all).asnumpy().tolist()
        else:
            max_single_percentage_list_all = None

        # total_text_percentage_list
        if compute_total_text_area_percentage:
            total_text_percentage_list = Tensor(np.array(total_text_percentage_list), dtype=ms.float32)
            total_text_percentage_list_all = [
                Tensor(np.zeros(total_text_percentage_list.shape, dtype=np.float32)) for _ in range(rank_size)
            ]
            all_gather(total_text_percentage_list_all, total_text_percentage_list)
            total_text_percentage_list_all = ops.Concat(axis=0)(total_text_percentage_list_all).asnumpy().tolist()
        else:
            total_text_percentage_list_all = None

        # ocr_results_list
        ocr_results_list_all = [None] * rank_size
        all_gather_object(ocr_results_list_all, ocr_results_list)
        # Flatten the list-of-lists from each process into a single list
        ocr_results_list_all = sum(ocr_results_list_all, [])

        if rank_id == 0:
            meta_local = merge_scores([(indices_list_all, ocr_results_list_all)], raw_dataset.meta, column="ocr")
            if compute_num_boxes:
                meta_local = merge_scores([(indices_list_all, num_boxes_list_all)], meta_local, column="num_boxes")
            if compute_max_single_text_box_area_percentage:
                meta_local = merge_scores(
                    [(indices_list_all, max_single_percentage_list_all)], meta_local, column="max_single_percentage"
                )
            if compute_total_text_area_percentage:
                meta_local = merge_scores(
                    [(indices_list_all, total_text_percentage_list_all)], meta_local, column="total_text_percentage"
                )

    elif rank_size == 1:  # store directly without gathering
        meta_local = raw_dataset.meta.copy()
        meta_local["ocr"] = ocr_results_list
        if compute_num_boxes:
            meta_local["num_boxes"] = num_boxes_list
        if compute_max_single_text_box_area_percentage:
            meta_local["max_single_percentage"] = max_single_percentage_list
        if compute_total_text_area_percentage:
            meta_local["total_text_percentage"] = total_text_percentage_list

    if rank_id == 0:
        meta_local.to_csv(out_path, index=False)
        print(meta_local)
        print(f"New meta with OCR results saved to '{out_path}'.")


if __name__ == "__main__":
    main()
