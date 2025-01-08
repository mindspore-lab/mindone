import re
import mindspore as ms
import numpy as np
from PIL import Image

import math
import os
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union
import av

MAX_NUM_FRAMES = 2500

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# from torchvision.datasets.folder
def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def is_video(filename):
    ext = os.path.splitext(filename)[-1].lower()
    return ext in VID_EXTENSIONS


def extract_frames(
    video_path,
    frame_inds=None,
    points=None,
    seconds=None,
    backend="opencv",
    return_length=False,
    num_frames=None,
    return_timestamps=False, # only for `seconds`
):
    """
    Extract frames from a video using the specified backend.

    Args:
        video_path (str): Path to the video file.
        frame_inds (List[int], optional): Indices of frames to extract.
        points (List[float], optional): Values within [0, 1); multiply by total frames to get frame indices.
        seconds (float, optional): Interval in seconds to extract frames at.
        backend (str, optional): Backend to use ('av', 'opencv', or 'decord'). Defaults to 'opencv'.
        return_length (bool, optional): Whether to return the total number of frames in the video. Defaults to False.
        num_frames (int, optional): Total number of frames in the video. If not provided, it will be determined automatically.

    Returns:
        List[PIL.Image.Image]: List of extracted frames as PIL Images.
        int (optional): Total number of frames in the video (if return_length is True).

    Remark:
        The `seconds` parameter is designed to extract multiple frames for motion analysis
        (e.g., optical flow or perceptual similarity) with a more sophisticated extraction mechanism.
        We always extract the first frame when the input is "seconds"
    """
    assert backend in ["av", "opencv", "decord"], f"Unsupported backend: {backend}"
    assert (
        (frame_inds is not None and points is None and seconds is None)
        or (frame_inds is None and points is not None and seconds is None)
        or (frame_inds is None and points is None and seconds is not None)
    ), "Only one of frame_inds, points, or seconds should be provided."

    def get_frame_indices_from_points(points, total_frames):
        for p in points:
            if p > 1:
                print(f"[Warning] Point value {p} is greater than 1. Returning the last frame.")
        return [min(int(p * total_frames), total_frames - 1) for p in points]

    def create_black_frame(width=256, height=256):
        return Image.new("RGB", (width, height), (0, 0, 0))

    def handle_frame_read_error(prev_frame, frames, return_length, total_frames):
        if prev_frame is not None:
            print("[Info] Previous frame exists. Skipping appending any frame.")
            return True # continue processing
        else:
            print("[Warning] No previous frame available. Creating a black frame.")
            frame = create_black_frame()
            frames.append(frame)
            return False # finish processing

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if backend == "av":
        import av

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base  # fraction representing seconds per frame
            fps = float(stream.average_rate)

            if num_frames is not None:
                total_frames = num_frames
            else:
                total_frames = stream.frames
                if total_frames == 0:
                    # in some cases, total_frames might not be available
                    total_frames = int(stream.duration * fps * time_base)

            total_duration = total_frames / fps  # in seconds

            # compute frame indices based on the provided parameter
            if seconds is not None:
                if total_duration < seconds:
                    print(
                        f"[Warning] Video duration {total_duration:.2f}s is shorter than input seconds {seconds}. Returning one black frame."
                    )
                    frames = [create_black_frame()]
                    if return_length and return_timestamps:
                        return frames, total_frames, [0.0]
                    if return_length:
                        return frames, total_frames
                    if return_timestamps:
                        return frames, [0.0]
                    return frames
                else:
                    timestamps = []
                    t = 0
                    while t < total_duration:
                        timestamps.append(t)
                        t += seconds

                    frames = []
                    frame_timestamps = []
                    prev_frame = None
                    for t in timestamps:
                        pts = int(t / time_base)
                        try:
                            container.seek(pts, any_frame=False, backward=True, stream=stream)
                            for packet in container.demux(stream):
                                for frame in packet.decode():
                                    if frame.pts >= pts:
                                        img = frame.to_image()
                                        frames.append(img)
                                        frame_timestamps.append(frame.pts * time_base)
                                        prev_frame = img
                                        break
                                else:
                                    continue
                                break
                            else:
                                raise ValueError("No frame found at the specified timestamp.")
                        except Exception as e:
                            print(
                                f"[Warning] Error reading frame at {t:.2f}s from {video_path}: {e}."
                            )
                            continue_processing = handle_frame_read_error(prev_frame, frames, return_length, total_frames)
                            if not continue_processing:
                                if return_length and return_timestamps:
                                    return frames, total_frames, frame_timestamps
                                elif return_length:
                                    return frames, total_frames
                                elif return_timestamps:
                                    return frames, frame_timestamps
                                else:
                                    return frames
                    if return_length and return_timestamps:
                        return frames, total_frames, frame_timestamps
                    elif return_length:
                        return frames, total_frames
                    elif return_timestamps:
                        return frames, frame_timestamps
                    else:
                        return frames

            # handle frame_inds and points
            if points is not None:
                frame_inds = get_frame_indices_from_points(points, total_frames)

            frames = []
            for idx in frame_inds:
                try:
                    if idx >= total_frames:
                        idx = total_frames - 1
                    timestamp = idx / fps  # convert frame index to time in seconds
                    pts = int(timestamp / time_base)
                    container.seek(pts, any_frame=False, backward=True, stream=stream)
                    for packet in container.demux(stream):
                        for frame in packet.decode():
                            if frame.pts >= pts:
                                img = frame.to_image()
                                frames.append(img)
                                break
                        else:
                            continue
                        break
                    else:
                        raise ValueError("No frame found at the specified frame index.")
                except Exception:
                    try:
                        print(f"[Warning] Error reading frame {idx} from {video_path}. Try reading the first frame.")
                        container.seek(0, any_frame=False, backward=True, stream=stream)
                        frame = next(container.decode(video=0)).to_image()
                        frames.append(frame)
                    except Exception:
                        print(f"[Warning] Error reading first frame from {video_path}. Returning a black frame.")
                        frames = [create_black_frame()] # end here directly
                        if return_length:
                            return frames, total_frames
                        return frames
            if return_length:
                return frames, total_frames
            return frames

    elif backend == "decord":
        import decord

        decord.bridge.set_bridge("native")
        try:
            container = decord.VideoReader(video_path, num_threads=1)
        except Exception as e:
            print(f"[Error] Unable to open video file with decord: {e}")
            frames = [create_black_frame()]
            if return_length and return_timestamps and seconds is not None:
                return frames, 0, [0.0]
            if return_length:
                return frames, 0
            if return_timestamps and seconds is not None:
                return frames, [0.0]
            return frames

        if num_frames is not None:
            total_frames = num_frames
        else:
            total_frames = len(container)

        fps = container.get_avg_fps()
        total_duration = total_frames / fps

        # compute frame indices based on the provided parameter
        if seconds is not None:
            if total_duration < seconds:
                print(
                    f"[Warning] Video duration {total_duration:.2f}s is shorter than input seconds {seconds}. Returning one black frame."
                )
                frames = [create_black_frame()]
                if return_length and return_timestamps:
                    return frames, total_frames, [0.0]
                if return_length:
                    return frames, total_frames
                if return_timestamps:
                    return frames, [0.0]
                return frames
            else:
                timestamps = []
                t = 0
                while t < total_duration:
                    timestamps.append(t)
                    t += seconds

                frame_inds = [int(t * fps) for t in timestamps]
                frame_inds = np.clip(frame_inds, 0, total_frames - 1)

                frames = []
                frame_timestamps = []
                prev_frame = None
                for idx, t in zip(frame_inds, timestamps):
                    try:
                        frame = container[idx].asnumpy()
                        img = Image.fromarray(frame)
                        frames.append(img)
                        frame_timestamps.append(t)
                        prev_frame = img
                    except Exception as e:
                        print(
                            f"[Warning] Error reading frame at index {idx} from {video_path}: {e}."
                        )
                        continue_processing = handle_frame_read_error(prev_frame, frames, return_length, total_frames)
                        if not continue_processing:
                            if return_length and return_timestamps:
                                return frames, total_frames, frame_timestamps
                            elif return_length:
                                return frames, total_frames
                            elif return_timestamps:
                                return frames, frame_timestamps
                            else:
                                return frames
                if return_length and return_timestamps:
                    return frames, total_frames, frame_timestamps
                elif return_length:
                    return frames, total_frames
                elif return_timestamps:
                    return frames, frame_timestamps
                else:
                    return frames

        # handle frame_inds and points
        if points is not None:
            frame_inds = get_frame_indices_from_points(points, total_frames)

        frame_inds = np.clip(frame_inds, 0, total_frames - 1)

        try:
            frames_array = container.get_batch(frame_inds).asnumpy()
            frames = [Image.fromarray(frame) for frame in frames_array]
        except Exception as e:
            print(f"[Warning] Error reading frames from {video_path}: {e}")
            frames = []
            for idx in frame_inds:
                try:
                    frame = container[idx].asnumpy()
                    img = Image.fromarray(frame)
                    frames.append(img)
                except Exception as e:
                    try:
                        print(f"[Warning] Error reading frame {idx} from {video_path}. Try reading the first frame.")
                        frame = container[0].asnumpy()
                        frame = Image.fromarray(frame)
                        frames.append(frame)
                    except Exception:
                        print(f"[Warning] Error reading first frame from {video_path}. Returning a black frame.")
                        frames = [create_black_frame()] # end here directly
                        if return_length:
                            return frames, total_frames
                        return frames

        if return_length:
            return frames, total_frames
        return frames

    elif backend == "opencv":
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Unable to open video file with OpenCV: {video_path}")
            frames = [create_black_frame()]
            if return_length and return_timestamps and seconds is not None:
                return frames, 0, [0.0]
            elif return_length:
                return frames, 0
            elif return_timestamps and seconds is not None:
                return frames, [0.0]
            else:
                return frames

        try:
            if num_frames is not None:
                total_frames = num_frames
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    # handle cases where CAP_PROP_FRAME_COUNT is not available
                    total_frames = None

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps != fps:  # NaN check
                fps = 30  # in the case where the default fps is not available
            total_duration = total_frames / fps if total_frames else None

            # compute frame indices based on the provided parameter
            if seconds is not None:
                if total_duration is None or total_duration < seconds:
                    print(
                        f"[Warning] Video duration {total_duration:.2f}s is shorter than input seconds {seconds}. Returning one black frame."
                    )
                    frames = [create_black_frame()]
                    if return_length and return_timestamps:
                        return frames, total_frames, [0.0]
                    if return_length:
                        return frames, total_frames
                    if return_timestamps:
                        return frames, [0.0]
                    return frames
                else:
                    timestamps = []
                    t = 0
                    while t < total_duration:
                        timestamps.append(t)
                        t += seconds

                    frames = []
                    frame_timestamps = []
                    prev_frame = None
                    for t in timestamps:
                        ms = t * 1000  # convert seconds to milliseconds
                        cap.set(cv2.CAP_PROP_POS_MSEC, ms)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            print(
                                f"[Warning] Error reading frame at {t:.2f}s from {video_path}."
                            )
                            continue_processing = handle_frame_read_error(prev_frame, frames, return_length, total_frames)
                            if not continue_processing:
                                if return_length and return_timestamps:
                                    return frames, total_frames, frame_timestamps
                                elif return_length:
                                    return frames, total_frames
                                elif return_timestamps:
                                    return frames, frame_timestamps
                                else:
                                    return frames
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame)
                            frames.append(img)
                            frame_timestamps.append(t)
                            prev_frame = img
                    if return_length and return_timestamps:
                        return frames, total_frames, frame_timestamps
                    elif return_length:
                        return frames, total_frames
                    elif return_timestamps:
                        return frames, frame_timestamps
                    else:
                        return frames

            # handle frame_inds and points
            if points is not None:
                # if total_frames is not known, try to estimate it
                if total_frames is None:
                    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                    total_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_inds = get_frame_indices_from_points(points, total_frames)

            frames = []
            for idx in frame_inds:
                if total_frames and idx >= total_frames:
                    idx = total_frames - 1

                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None: # failure case
                    print(f"[Warning] Error reading frame {idx} from {video_path}")
                    try:
                        print(f"[Warning] Try reading first frame.")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(frame)
                        frames.append(frame)
                    except Exception as e:
                        print(
                            f"[Warning] Error in reading first frame from {video_path}: {e}. Returning a black frame.")
                        frames = [create_black_frame()] # end here directly
                        if return_length:
                            return frames, total_frames
                        return frames
                else: # success case
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    frames.append(img)

            if return_length:
                return frames, total_frames
            return frames
        finally:
            cap.release()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

# used in PLLaVA captioning
def read_video_av(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "sec",
    output_format: str = "THWC",
) -> Tuple[ms.Tensor, ms.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames (as Mindspore Tensors) and empty audio frames.

    Args:
        filename (str): Path to the video file.
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video.
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time.
        pts_unit (str, optional): Unit in which start_pts and end_pts values will be interpreted.
        output_format (str, optional): The format of the output video tensors, either "THWC" or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames.
        info (Dict): Metadata for the video.
    """
    # format
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")
    # file existence
    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")
    # pts check
    if end_pts is None:
        end_pts = float("inf")
    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")
    # ==get video info==
    with av.open(filename, metadata_errors="ignore") as container:
        video_stream = container.streams.video[0]
        video_fps = video_stream.average_rate
        info = {'video_fps': float(video_fps)} if video_fps is not None else {}
        # total frame calculation
        if video_stream.frames > 0:
            total_frames = video_stream.frames
        else:
            if video_stream.duration and video_stream.average_rate:
                total_frames = int(video_stream.duration * video_stream.average_rate * video_stream.time_base)
            else:
                total_frames = MAX_NUM_FRAMES

        video_frames = np.zeros(
            (total_frames, video_stream.height, video_stream.width, 3), dtype=np.uint8
        )

    try:
        with av.open(filename, metadata_errors="ignore") as container:
            assert container.streams.video is not None
            video_frames = _read_from_stream(
                video_frames,
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],
                {"video": 0},
                filename=filename,
            )
            for stream in container.streams:
                # Issue: https://github.com/PyAV-Org/PyAV/issues/1117
                stream.close() # explicitly close stream to avoid memory leak
    except av.AVError as e:
        print(f"[Warning] Error while reading video {filename}: {e}")
        return None, info

    if output_format == "TCHW":
        video_frames = video_frames.transpose(0, 3, 1, 2)  # [T, H, W, C] to [T, C, H, W]

    return video_frames, info

def _read_from_stream(
    video_frames,
    container: "av.container.Container",
    start_offset: float,
    end_offset: float,
    pts_unit: str,
    stream: "av.stream.Stream",
    stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]],
    filename: Optional[str] = None,
) -> List["av.frame.Frame"]:
    if pts_unit == "sec":
        # TODO: we should change all of this from ground up to simply take
        # sec and convert to MS in C++
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn("The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")

    should_buffer = True
    max_buffer_size = 5
    if stream.type == "video":
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything
        # properly
        extradata = stream.codec_context.extradata
        # overly complicated way of finding if `divx_packed` is set, following
        # https://github.com/FFmpeg/FFmpeg/commit/d5a21172283572af587b3d939eba0091484d3263
        if extradata and b"DivX" in extradata:
            # can't use regex directly because of some weird characters sometimes...
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(rb"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(rb"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        # FIXME this is kind of a hack, but we will jump to the previous keyframe
        # so this will be safe
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        # TODO check if stream needs to always be the video stream here or not
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError as e:
        print(f"[Warning] Error while seeking video {filename}: {e}")
        return video_frames[:0] # return empty array

    # == main ==
    buffer_count = 0
    frames_pts = []
    cnt = 0
    try:
        for frame in container.decode(**stream_name):
            frames_pts.append(frame.pts)
            video_frames[cnt] = frame.to_rgb().to_ndarray()
            cnt += 1
            if cnt >= len(video_frames):
                break
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError as e:
        print(f"[Warning] Error while reading video {filename}: {e}")

    # ensure that the results are sorted wrt the pts
    # NOTE: here we assert frames_pts is sorted
    start_ptr = 0
    end_ptr = cnt
    while start_ptr < end_ptr and frames_pts[start_ptr] < start_offset:
        start_ptr += 1
    while start_ptr < end_ptr and frames_pts[end_ptr - 1] > end_offset:
        end_ptr -= 1
    if start_offset > 0 and start_offset not in frames_pts[start_ptr:end_ptr]:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        if start_ptr > 0:
            start_ptr -= 1
    result = video_frames[start_ptr:end_ptr].copy()
    return result