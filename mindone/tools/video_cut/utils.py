import os

from moviepy.editor import VideoFileClip


def check_video_integrity(filepath):
    try:
        VideoFileClip(filepath)
        return True
    except Exception as e:
        print(f"Error occurred while checking video {filepath} integrity:", str(e))
        return False


def get_video_path(paths):
    if os.path.isdir(paths) and os.path.exists(paths):
        paths = [
            os.path.join(root, file)
            for root, _, file_list in os.walk(os.path.join(paths))
            for file in file_list
            if file.endswith(".mp4")
        ]
        paths.sort()
        paths = paths
    else:
        paths = [paths]

    return paths
