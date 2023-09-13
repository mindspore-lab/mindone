import decord
import numpy as np
import os

decord.bridge.set_bridge('torch')


def video_convert(video_path, width, height):
    # load and reshape the video using decord
    video_reader = decord.VideoReader(
        video_path,
        width=width,
        height=height,
    )

    # convert the video to np.ndarray
    sample_index = list(range(len(video_reader)))
    video = video_reader.get_batch(sample_index).numpy()

    # save the converted video
    np.save(f"{os.path.splitext(video_path)[0]}.npy", video)


if __name__ == "__main__":
    video_params_list = [
        {"video_path": "data/car-turn.mp4", "width": 512, "height": 512},
        {"video_path": "data/girl-glass.mp4", "width": 512, "height": 512},
        {"video_path": "data/girl.mp4", "width": 512, "height": 512},
    ]

    for video_params in video_params_list:
        video_convert(**video_params)
