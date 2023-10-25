import cv2
import numpy as np


class VideoReader:
    def __init__(self, video_path: str):
        self._video_path = video_path
        self._cap = None
        self.shape = (0, 0)
        self.fps = 0

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(self._video_path, apiPreference=cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise IOError(f"Video {self._video_path} cannot be opened.")
        self.shape = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        return self

    def __exit__(self, *args):
        self._cap.release()

    def __len__(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def fetch_frames(self, num: int = 0, start_pos: int = 0, step: int = 1) -> np.ndarray:
        """
        Fetches a sequence of frames from a video starting at a specified position with a specified step.

        Parameters:
            num: The number of frames to fetch.
            start_pos: The frame index to start fetching from. Default: 0.
            step: The number of frames to skip after fetching a frame. Default: 1.

        Returns:
            np.ndarray: An array containing the fetched frames.

        Raises:
            ValueError: If the requested number of frames exceeds the video
        """
        if num * step > len(self):
            raise ValueError(f"Number of frames to fetch ({num * step}) must be less than video length ({len(self)}).")

        if start_pos:
            start_pos = min(start_pos, len(self) - num * step)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

        frames = []
        i = start_pos
        ret, frame = self._cap.read()
        while ret and len(frames) < num:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if step > 1:
                i += step
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self._cap.read()

        if len(frames) != num:
            raise RuntimeError(f"Failed to read {num} frames from {self._video_path}.")

        return np.stack(frames)
