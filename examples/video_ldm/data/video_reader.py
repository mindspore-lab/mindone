import cv2
import numpy as np


class VideoReader:
    def __init__(self, video_path):
        self._video_path = video_path
        self._cap = None
        self.shape = (0, 0)
        self.fps = 0

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(self._video_path, apiPreference=cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise IOError(f"Video {self._video_path} cannot be opened.")
        self.shape = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        return self

    def __exit__(self, *args):
        self._cap.release()

    def __len__(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def fetch_frames(self, num: int, start_pos: int = 0, step: int = 1) -> np.ndarray:
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

        return np.stack(frames)
