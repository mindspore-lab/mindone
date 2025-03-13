import os

import numpy as np
from PIL import Image

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
    return_timestamps=False,  # only for `seconds`
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
            return True  # continue processing
        else:
            print("[Warning] No previous frame available. Creating a black frame.")
            frame = create_black_frame()
            frames.append(frame)
            return False  # finish processing

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
                            print(f"[Warning] Error reading frame at {t:.2f}s from {video_path}: {e}.")
                            continue_processing = handle_frame_read_error(
                                prev_frame, frames, return_length, total_frames
                            )
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
                        frames = [create_black_frame()]  # end here directly
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
                        print(f"[Warning] Error reading frame at index {idx} from {video_path}: {e}.")
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
                except Exception:
                    try:
                        print(f"[Warning] Error reading frame {idx} from {video_path}. Try reading the first frame.")
                        frame = container[0].asnumpy()
                        frame = Image.fromarray(frame)
                        frames.append(frame)
                    except Exception:
                        print(f"[Warning] Error reading first frame from {video_path}. Returning a black frame.")
                        frames = [create_black_frame()]  # end here directly
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
                            print(f"[Warning] Error reading frame at {t:.2f}s from {video_path}.")
                            continue_processing = handle_frame_read_error(
                                prev_frame, frames, return_length, total_frames
                            )
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
                if not ret or frame is None:  # failure case
                    print(f"[Warning] Error reading frame {idx} from {video_path}")
                    try:
                        print("[Warning] Try reading first frame.")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(frame)
                        frames.append(frame)
                    except Exception as e:
                        print(
                            f"[Warning] Error in reading first frame from {video_path}: {e}. Returning a black frame."
                        )
                        frames = [create_black_frame()]  # end here directly
                        if return_length:
                            return frames, total_frames
                        return frames
                else:  # success case
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
