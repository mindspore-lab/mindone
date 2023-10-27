import logging
import os
import subprocess
import time

import cv2
import numpy as np
from mvextractor.videocap import VideoCap

from ...utils.misc import rand_name

__all__ = [
    "draw_motion_vectors",
    "extract_motion_vectors",
]

__dir__ = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.abspath(os.path.join(__dir__, "../../../tmp_videos/"))
os.makedirs(cache_dir, exist_ok=True)

_logger = logging.getLogger(__name__)


def draw_motion_vectors(frame, motion_vectors):
    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 3], mv[0, 4])
            end_pt = (mv[0, 5], mv[0, 6])
            cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.1)
    return frame


def extract_motion_vectors(
    input_video, fps=4, viz=False, dump=False, verbose=False, skip_long_videos=False, max_duration=30
):
    tmp_name = rand_name()
    # tmp_video = os.path.splitext(input_video)[0] + f"_{tmp_name}" + os.path.splitext(input_video)[-1]
    cache_video_name = os.path.basename(input_video).split(".")[0] + f"_{tmp_name}" + os.path.splitext(input_video)[-1]
    tmp_video = os.path.join(cache_dir, cache_video_name)
    videocapture = cv2.VideoCapture(input_video)
    frames_num = videocapture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_video = videocapture.get(cv2.CAP_PROP_FPS)

    if skip_long_videos:
        dur = frames_num / fps_video
        if dur > max_duration:
            videocapture.release()
            raise ValueError(
                f"{input_video} is too long. frames: {frames_num}, video_fps: {fps_video}, dur: {dur}. Will be skipped! Please trim it in pre-processing"
            )

    # check if enough frames
    if frames_num / fps_video * fps > 16:
        fps = max(fps, 1)
    else:
        fps = int(16 / (frames_num / fps_video)) + 1
    ffmpeg_cmd = (
        f"ffmpeg -threads 8 -loglevel error -i {input_video} -filter:v fps={fps} -c:v mpeg4 -f rawvideo {tmp_video}"
    )
    if os.path.exists(tmp_video):
        os.remove(tmp_video)

    subprocess.run(args=ffmpeg_cmd, shell=True, timeout=120)

    cap = VideoCap()
    # open the video file
    ret = cap.open(tmp_video)
    if not ret:
        _logger.warning(f"Could not open {tmp_video}")

    step = 0
    times = []
    frame_types = []
    frames = []
    mvs = []
    mvs_visual = []

    # continuously read and display video frames and motion vectors
    while True:
        if verbose:
            _logger.info(f"Frame: {step}")
        t_start = time.perf_counter()
        # read next video frame and corresponding motion vectors
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()

        t_end = time.perf_counter()
        times.append(t_end - t_start)
        # if there is an error reading the frame
        if not ret:
            if verbose:
                _logger.info("No frame read. Stopping.")
            break

        # visualization of motion vectors
        mv_visual = np.zeros(frame.shape, dtype=np.uint8)

        if viz:
            mv_visual = draw_motion_vectors(mv_visual, motion_vectors)

        if frame.shape[1] >= frame.shape[0]:
            w_half = (frame.shape[1] - frame.shape[0]) // 2
            mv_visual = mv_visual[:, w_half:-w_half]
        else:
            h_half = (frame.shape[0] - frame.shape[1]) // 2
            mv_visual = mv_visual[h_half:-h_half, :]

        h, w = frame.shape[:2]
        mv = np.zeros((h, w, 2))
        position = motion_vectors[:, 5:7].clip((0, 0), (w - 1, h - 1))
        mv[position[:, 1], position[:, 0]] = motion_vectors[:, 0:1] * motion_vectors[:, 7:9] / motion_vectors[:, 9:]

        step += 1
        frame_types.append(frame_type)
        frames.append(frame)
        mvs.append(mv)
        mvs_visual.append(mv_visual)

    if verbose:
        _logger.info(f"average dt: {np.mean(times)}")

    cap.release()
    videocapture.release()

    if os.path.exists(tmp_video):
        os.remove(tmp_video)
    if dump:
        dump_path = os.path.splitext(input_video)[0]
        _logger.info(f"Dumping visualization of motion vectors and frames to {dump_path}")
        os.makedirs(dump_path, exist_ok=True)
        for i, (frame, mv_visual) in enumerate(zip(frames, mvs_visual)):
            cv2.imwrite(os.path.join(dump_path, f"frame-{i}.jpg"), frame)
            cv2.imwrite(os.path.join(dump_path, f"frame-mv-{i}.jpg"), mv_visual)

    return frame_types, frames, mvs, mvs_visual
