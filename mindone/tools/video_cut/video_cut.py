import argparse
import os

import av
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg
from tqdm import tqdm


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


class KeyFrameExtractUtils:
    def __init__(self, video_data_path=None, keyframe_path=None):
        self.video_data_path = video_data_path
        self.keyframe_path = keyframe_path

    def save_keyframe(self):
        vdieo_dir = get_video_path(self.video_data_path)
        print("save keyframes")
        for path in tqdm(vdieo_dir):
            if self.keyframe_path is None:
                keyframe_dir = os.path.join(os.path.splitext(path)[0])
            else:
                keyframe_dir = os.path.join(self.keyframe_path, os.path.splitext(os.path.basename(path))[0])
            if not os.path.exists(keyframe_dir):
                os.makedirs(keyframe_dir)

            self.keyframe_extract(path, keyframe_dir)

    def keyframe_extract(self, video_path, keyframe_dir):
        container = av.open(video_path)
        # only want to look at keyframes.
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        for frame in container.decode(stream):
            frame.to_image().save(keyframe_dir + "/" + "frame.{:04d}.jpg".format(frame.pts), quality=80)


class SceneCutUtils:
    """cut video scene refer to https://github.com/Breakthrough/PySceneDetect.
    Args:
        video_data_path(str): video data path.
        Detectortype(str): scene detection algorithms type, contains three algorithms:
            - ContentDetector: Detects shot changes by considering pixel changes in the HSV colorspace.
            - AdaptiveDetector: Two-pass version of ContentDetector that handles fast camera movement better in some cases.
            - ThresholdDetector: Detects transitions below a set pixel intensity (cuts or fades to black).
    """

    def __init__(self, video_data_path, Detectortype="ContentDetector"):
        self.video_data_path = video_data_path
        self.Detectortype = Detectortype
        if self.Detectortype == "ContentDetector":
            self.Detector = ContentDetector()
        elif self.Detectortype == "AdaptiveDetector":
            self.Detector = AdaptiveDetector()
        elif self.Detectortype == "ThresholdDetector":
            self.Detector = ThresholdDetector()
        else:
            raise ValueError(f"Unsupported Detector type: {self.Detectortype}.")

    def save_scene(self):
        vdieo_dir = get_video_path(self.video_data_path)
        for video_path in vdieo_dir:
            # Create our video & scene managers, then add the detector.
            video_manager = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(self.Detector)

            # Start the video manager and perform the scene detection.
            scene_manager.detect_scenes(video_manager, show_progress=True)
            # save scene
            scene_list = scene_manager.get_scene_list()
            if len(scene_list):
                split_video_ffmpeg(video_path, scene_list, show_progress=False)
                for index, scene in enumerate(scene_list):
                    print(
                        f"Scene {index + 1}: Start {scene[0].get_timecode()} / Frame {scene[0].get_frames()}",
                        f"End {scene[1].get_timecode()} / Frame {scene[1].get_frames()}",
                    )
            else:
                print(f"{video_path} only detects one scene, no need to cut it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_data_path", type=str, default=None, help="path to video data." "Default: None")
    parser.add_argument(
        "--keyframe_save_dir",
        type=str,
        default=None,
        help="path for save keyframe images. if None, keyframes will save in video_data_path." "Default: None",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="keyframe",
        choices=["keyframe", "scene"],
        help="Choose how to process videos." "Default: keframe",
    )
    parser.add_argument(
        "--Detectortype",
        type=str,
        default="ContentDetector",
        choices=["ContentDetector", "AdaptiveDetector", "ThresholdDetector"],
        help="scene detection algorithms type." "Default: ContentDetector",
    )
    args = parser.parse_args()

    if args.task == "keyframe":
        keyframe = KeyFrameExtractUtils(args.video_data_path, args.keyframe_save_dir)
        keyframe.save_keyframe()
    elif args.task == "scene":
        cut_scene = SceneCutUtils(args.video_data_path, args.Detectortype)
        cut_scene.save_scene()
    else:
        raise ValueError(f"Unsupported tasks: {args.task}.")
