"""
VC training/finetuning
"""
import json
import logging
import os

# import datetime
import sys
import time

import cv2
import pandas as pd

# from omegaconf import OmegaConf
from vc.config import Config
from vc.utils import convert_to_abspath, setup_logger

import mindspore as ms
from mindspore import context
from mindspore import dataset as ds
from mindspore.communication.management import get_group_size, get_rank, init

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../stable_diffusion_v2/")))

from ldm.modules.train.parallel_config import ParallelConfig
from ldm.modules.train.tools import set_random_seed

# os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def get_video_info(input_video):
    videocapture = cv2.VideoCapture(input_video)
    frames_num = videocapture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_video = videocapture.get(cv2.CAP_PROP_FPS)
    dur = frames_num / fps_video

    videocapture.release()
    return frames_num, fps_video, dur


class VidReader(object):
    def __init__(
        self,
        cfg=None,
        root_dir=None,
        max_words=30,
        feature_framerate=1,
        max_frames=16,
        image_resolution=224,
        transforms=None,
        mv_transforms=None,
        misc_transforms=None,
        vit_transforms=None,
        vit_image_size=336,
        misc_size=384,
        mvs_visual=False,
        tokenizer=None,
        conditions_for_train=None,
        rank_id=0,
    ):
        """
        Args:
            root_dir: dir containing csv file which records video path and caption.
        """

        self.cfg = cfg

        self.tokenizer = tokenizer
        self.max_words = max_words
        self.feature_framerate = feature_framerate
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        self.transforms = transforms
        self.mv_transforms = mv_transforms
        self.misc_transforms = misc_transforms
        self.vit_transforms = vit_transforms
        self.canny_detector = None  # canny is not used by UNetSD
        self.vit_image_size = vit_image_size
        self.misc_size = misc_size
        self.mvs_visual = mvs_visual
        self.conditions_for_train = conditions_for_train

        video_paths, captions, rel_video_paths = get_video_paths_captions(root_dir)
        num_samples = len(video_paths)
        self.video_cap_pairs = [[video_paths[i], captions[i]] for i in range(num_samples)]

        self.rel_video_paths = rel_video_paths
        self.tokenizer = tokenizer  # bpe

        self.stat_fp = os.path.join(cfg.output_dir, f"data_stat_rank_{rank_id}.csv")
        header = ",".join(["video_path", "caption", "frames", "fps", "duration"])
        with open(self.stat_fp, "w", encoding="utf-8") as fp:
            fp.write(header + "\n")

    def __len__(self):
        return len(self.video_cap_pairs)

    def __getitem__(self, index):
        video_key, cap_txt = self.video_cap_pairs[index]
        rel_video_path = self.rel_video_paths[index]

        if os.path.exists(video_key):
            try:
                frames_num, fps_video, dur = get_video_info(video_key)

                _stat = f'{rel_video_path},"{cap_txt}",{frames_num},{fps_video},{dur}'
                with open(self.stat_fp, "a", encoding="utf-8") as fp:
                    fp.write(_stat + "\n")

            except Exception as e:
                print("Load video {} fails, Error: {}".format(video_key, e), flush=True)
        else:  # use dummy data
            logger.warning(
                f"Fail to load {video_key}, video data could be broken, which will be replaced with dummy data."
            )

        return rel_video_path, cap_txt, frames_num, fps_video, dur


def get_video_paths_captions(data_dir, only_use_csv_anno=False):
    """
    JSON files have higher priority, i.e., if both JSON and csv annotion files exist, only JSON files will be loaded.
    To force to read CSV annotation, please parse only_use_csv_anno=True.
    """
    csv_anno_list = sorted(
        [os.path.join(data_dir, f) for f in list(filter(lambda x: x.endswith(".csv"), os.listdir(data_dir)))]
    )
    json_anno_list = sorted(
        [os.path.join(data_dir, f) for f in list(filter(lambda x: x.endswith(".json"), os.listdir(data_dir)))]
    )

    video_paths = []
    all_captions = []
    if (len(json_anno_list) == 0) or only_use_csv_anno:
        logger.info("Reading annotation from csv files: {}".format(csv_anno_list))
        db_list = [pd.read_csv(f) for f in csv_anno_list]
        for db in db_list:
            video_paths.extend(list(db["video"]))
            all_captions.extend(list(db["caption"]))
        # _logger.info(f"Before filter, Total number of training samples: {len(video_paths)}")
    elif len(json_anno_list) > 0:
        logger.info("Reading annotation from json files: {}".format(json_anno_list))
        for json_fp in json_anno_list:
            with open(json_fp, "r", encoding="utf-8") as fp:
                datasets_dict = json.load(fp)
                for dataset in datasets_dict:
                    rel_path_caption_pair_list = datasets_dict[dataset]
                    for rel_path_caption_pair in rel_path_caption_pair_list:
                        video_paths.append(rel_path_caption_pair[0])
                        all_captions.append(rel_path_caption_pair[1])

    assert len(video_paths) == len(all_captions)
    abs_video_paths = [os.path.join(data_dir, f) for f in video_paths]
    print("D--: ", video_paths, all_captions)

    return abs_video_paths, all_captions, video_paths


def build_dataset(cfg, device_num, rank_id, tokenizer):
    dataset = VidReader(
        cfg=cfg,
        root_dir=cfg.root_dir,
        max_words=cfg.max_words,
        feature_framerate=cfg.feature_framerate,
        max_frames=cfg.max_frames,
        image_resolution=cfg.resolution,
        rank_id=rank_id,
    )

    print("Total number of samples: ", len(dataset))

    dataloader = ds.GeneratorDataset(
        source=dataset,
        num_shards=device_num,
        column_names=["vid_name", "cap", "frames", "fps", "dur"],
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=cfg.shuffle,
        num_parallel_workers=cfg.num_parallel_workers,
        max_rowsize=128,  # video data require larger rowsize
    )

    dl = dataloader.batch(
        cfg.batch_size,
        drop_remainder=False,
    )

    return dl, dataset.stat_fp


def init_env(args):
    # rank_id - global card id, device_num - num of cards
    set_random_seed(args.seed)

    ms.set_context(mode=args.ms_mode)  # needed for MS2.0
    if args.use_parallel:
        init()
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = get_group_size()
        ParallelConfig.dp = device_num
        rank_id = get_rank()
        args.rank = rank_id
        logger.debug("Device_id: {}, rank_id: {}, device_num: {}".format(device_id, rank_id, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        device_num = 1
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        args.rank = rank_id

    context.set_context(
        mode=args.ms_mode,
        device_target="Ascend",
        device_id=device_id,
        # max_device_memory="30GB", # adapt for 910b
    )
    ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})  # Only effective on Ascend 901B

    # logger
    # ct = datetime.datetime.now().strftime("_%y%m%d_%H_%M")
    # args.output_dir += ct
    setup_logger(output_dir=args.output_dir, rank=args.rank)

    return rank_id, device_id, device_num


def check_config(cfg):
    # prev_cond_idx = -1
    for cond in cfg.conditions_for_train:
        if cond not in cfg.video_compositions:
            raise ValueError(f"Unknown condition: {cond}. Available conditions are: {cfg.video_compositions}")
            # idx = cfg.video_compositions.index(cond)
    print("===> Conditions used for training: ", cfg.conditions_for_train)

    # turn to abs path if it's relative path, for modelarts running
    cfg.root_dir = convert_to_abspath(cfg.root_dir, __dir__)
    cfg.cfg_file = convert_to_abspath(cfg.cfg_file, __dir__)
    cfg.resume_checkpoint = convert_to_abspath(cfg.resume_checkpoint, __dir__)


def main(cfg):
    check_config(cfg)

    # 1. init
    rank_id, device_id, device_num = init_env(cfg)

    # 2. build model components for ldm
    # 2.1 clip - text encoder, and image encoder (optional)

    dataloader, stat_fp = build_dataset(cfg, device_num, rank_id, tokenizer=None)

    num_batches = dataloader.get_dataset_size()

    num_tries = num_batches
    start = time.time()
    warmup = 0
    warmup_steps = 2
    warmup_steps = min(num_tries - 1, warmup_steps)
    iterator = dataloader.create_dict_iterator()
    for i, batch in enumerate(iterator):
        logger.info(f"{i}/{num_batches}")
        # for k in batch:
        #    print(k, batch[k].shape)  # , batch[k].min(), batch[k].max())
        if i == warmup_steps - 1:
            warmup = time.time() - start
    tot_time = time.time() - start - warmup
    mean = tot_time / (num_tries - warmup_steps)
    print("Avg batch loading time: ", mean)

    # saving csv annotation

    df = pd.read_csv(stat_fp)
    max_duration = 30
    short_df = df[df["duration"] <= max_duration]
    print("Filter by max_duration ", max_duration)
    print(short_df)
    short_df = short_df[["video_path", "caption"]]

    save_fp = os.path.join(cfg.output_dir, f"video_caption_short_rank_{rank_id}.csv")
    short_df.to_csv(save_fp, index=False, sep=",")


if __name__ == "__main__":
    # 0. parse config
    from configs.train_base import cfg  # base config from train_base.py

    args_for_update = Config(load=True).cfg_dict  # config args from CLI (arg parser) and yaml files

    # update base config
    for k, v in args_for_update.items():
        cfg[k] = v

    print(cfg)
    main(cfg)
