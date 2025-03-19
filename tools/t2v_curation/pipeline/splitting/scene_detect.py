import argparse
import os

import cv2
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scenedetect import AdaptiveDetector, ContentDetector, detect
from tqdm import tqdm

tqdm.pandas()


def convert_frames_to_timecode(frames, fps):
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def process_single_row(row, detector_type="adaptive", max_cutscene_len=None):
    assert detector_type in [
        "adaptive",
        "content",
    ], f"Detector type should be 'adaptive' or 'content', got {detector_type}."

    video_path = row["path"]

    # default option in hpcai-OpenSora
    if detector_type == "adaptive":
        detector = AdaptiveDetector(
            adaptive_threshold=3.0,
        )
    # default option in Panda-70M
    elif detector_type == "content":
        detector = ContentDetector(threshold=25, min_scene_len=15)

    try:
        scene_list = detect(video_path, detector, start_in_scene=True)
        # default option for hpcai-OpenSora
        if max_cutscene_len is None:
            timestamp = [(s.get_timecode(), t.get_timecode()) for s, t in scene_list]
            return True, str(timestamp)
        # default value for Panda-70M is 5
        else:
            end_frame_idx = [0]
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            for scene in scene_list:
                new_end_frame_idx = scene[1].get_frames()
                while (new_end_frame_idx - end_frame_idx[-1]) > (max_cutscene_len + 2) * fps:
                    end_frame_idx.append(end_frame_idx[-1] + int(max_cutscene_len * fps))
                end_frame_idx.append(new_end_frame_idx)

            cutscenes = [(end_frame_idx[i], end_frame_idx[i + 1]) for i in range(len(end_frame_idx) - 1)]
            timestamps = [
                (convert_frames_to_timecode(start, fps), convert_frames_to_timecode(end, fps))
                for start, end in cutscenes
            ]

            return True, str(timestamps)
    except Exception as e:
        print(f"Video '{video_path}' encountered an error: {e}")
        return False, ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--num_workers", type=int, default=None, help="#workers for pandarallel")
    parser.add_argument(
        "--detector",
        type=str,
        default="adaptive",
        choices=["adaptive", "content"],
        help="Type of scene detector to use (adaptive or content).",
    )
    parser.add_argument("--max_cutscene_len", type=float, default=None, help="Maximum length for the cut scenes")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    if args.num_workers is not None:
        pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
    else:
        pandarallel.initialize(progress_bar=True)

    meta = pd.read_csv(meta_path)
    ret = meta.parallel_apply(
        process_single_row, axis=1, detector_type=args.detector, max_cutscene_len=args.max_cutscene_len
    )

    succ, timestamps = list(zip(*ret))
    meta["timestamp"] = timestamps
    meta = meta[np.array(succ)]

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_timestamp{ext}"
    meta.to_csv(out_path, index=False)
    print(f"New meta (shape={meta.shape}) with timestamp saved to '{out_path}'.")


if __name__ == "__main__":
    main()
