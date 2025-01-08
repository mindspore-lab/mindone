import os
import warnings
import pandas as pd
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size, init
from tqdm import tqdm

from pipeline.datasets.utils import extract_frames, pil_loader, is_video
from config import parse_args
from text_system import TextSystem

class VideoTextDataset:
    # By default, we use the middle frame for OCR
    def __init__(self, meta_path, transform=None):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample['path']

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
        height = sample['height'] if 'height' in sample else -1
        width = sample['width'] if 'width' in sample else -1

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
    init()

    # initialize the TextSystem
    text_system = TextSystem(args)

    raw_dataset = VideoTextDataset(args.meta_path)
    rank_id = get_rank()
    rank_size = get_group_size()
    dataset = ds.GeneratorDataset(
        source=raw_dataset,
        column_names=['index', 'images', 'height', 'width'],
        shuffle=False,
        num_shards=rank_size,
        shard_id=rank_id
    )
    # TODO: Batch size > 1 only supports images with the same shapes
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
    if (compute_max_single_text_box_area_percentage or compute_total_text_area_percentage):
        if 'height' not in raw_dataset.meta.columns or 'width' not in raw_dataset.meta.columns:
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
            ocr_result = {"boxes": boxes, "texts": text_scores}
            batch_ocr_results.append(ocr_result)

            if compute_num_boxes:
                num_boxes = len(boxes)
            else:
                num_boxes = None
            batch_num_boxes.append(num_boxes)

            if (compute_max_single_text_box_area_percentage or compute_total_text_area_percentage) and height is not None and width is not None:
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
                        max_single_text_box_area_percentage = max_single_text_box_area / image_area if image_area > 0 else 0
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

    meta_local = raw_dataset.meta.copy()
    meta_local['ocr'] = ocr_results_list

    if compute_num_boxes:
        meta_local['num_boxes'] = num_boxes_list
    if compute_max_single_text_box_area_percentage:
        meta_local['max_single_percentage'] = max_single_percentage_list
    if compute_total_text_area_percentage:
        meta_local['total_text_percentage'] = total_text_percentage_list

    meta_local.to_csv(out_path, index=False)
    print(meta_local)
    print(f"New meta with OCR results saved to '{out_path}'.")

if __name__ == "__main__":
    main()
