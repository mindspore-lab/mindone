import argparse
import os
from multiprocessing import Pool

import av
from scenedetect import detect
from scenedetect.detectors import AdaptiveDetector, ContentDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg
from tqdm import tqdm
from utils import check_video_integrity, get_video_path


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

    def __init__(self, video_data_path, Detectortype="ContentDetector", save_dir=None, num_processing=1):
        self.video_data_path = video_data_path
        self.Detectortype = Detectortype
        self.num_processing = num_processing
        self.save_dir = save_dir
        if self.Detectortype == "ContentDetector":
            self.Detector = ContentDetector
        elif self.Detectortype == "AdaptiveDetector":
            self.Detector = AdaptiveDetector
        elif self.Detectortype == "ThresholdDetector":
            self.Detector = ThresholdDetector
        else:
            raise ValueError(f"Unsupported Detector type: {self.Detectortype}.")

    def split_list(self, lst, n):
        length = len(lst) // n + (1 if len(lst) % n != 0 else 0)
        result = []
        for i in range(n):
            start = i * length
            end = start + length
            sublist = lst[start:end]
            result.append(sublist)

        return result

    def scenes_detect(self):
        video_dir = get_video_path(self.video_data_path)
        if self.save_dir is None:
            self.save_dir = os.path.join(os.path.splitext(self.video_data_path)[0])
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        if self.num_processing > 1:
            video_list = self.split_list(video_dir, self.num_processing)
            with Pool(processes=self.num_processing) as pool:
                pool.starmap(self.save_scene, [(video, self.save_dir) for video in video_list])
        else:
            self.save_scene(video_dir, self.save_dir)

    def do_detect_scenes(self, path, detector_type=ContentDetector):
        return detect(str(path), detector_type(), show_progress=True)

    def save_scene(self, video_dir, save_dir):
        for video_path in video_dir:
            # Create our video & scene managers, then add the detector.
            if not check_video_integrity(video_path):
                continue
            scene_list = self.do_detect_scenes(video_path, detector_type=self.Detector)
            if len(scene_list):
                split_video_ffmpeg(video_path, scene_list, output_dir=save_dir, show_progress=False)
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
        "--save_dir",
        type=str,
        default=None,
        help="path for save keyframe images or sences. if None, keyframes or sences will save in video_data_path, ."
        "Default: None",
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
        default="AdaptiveDetector",
        choices=["ContentDetector", "AdaptiveDetector", "ThresholdDetector"],
        help="scene detection algorithms type." "Default: AdaptiveDetector",
    )
    parser.add_argument(
        "--num_processing",
        type=int,
        default=1,
        help="multiple processesr",
    )
    args = parser.parse_args()

    if args.task == "keyframe":
        keyframe = KeyFrameExtractUtils(args.video_data_path, args.save_dir)
        keyframe.save_keyframe()
    elif args.task == "scene":
        cut_scene = SceneCutUtils(args.video_data_path, args.Detectortype, args.save_dir, args.num_processing)
        cut_scene.scenes_detect()
    else:
        raise ValueError(f"Unsupported tasks: {args.task}.")
