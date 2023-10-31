"""
VC training/finetuning
"""
import logging
import os
import sys
import time

# from omegaconf import OmegaConf
from vc.config import Config
from vc.data.dataset_train import build_dataset
from vc.utils import convert_to_abspath, setup_logger

import mindspore as ms
from mindspore import context
from mindspore.communication.management import get_group_size, get_rank, init

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../stable_diffusion_v2/")))

from ldm.modules.train.parallel_config import ParallelConfig
from ldm.modules.train.tools import set_random_seed
from tools._common.clip import CLIPTokenizer

logger = logging.getLogger(__name__)


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

    # TODO: set sink_size and epochs to solve it
    # assert not (
    #    cfg.step_mode and cfg.dataset_sink_mode
    # ), f"step_mode is enabled, dataset_sink_mode should be set to False, but got {cfg.dataset_sink_mode})"


def main(cfg):
    check_config(cfg)

    # 1. init
    rank_id, device_id, device_num = init_env(cfg)

    # 2. build model components for ldm
    # 2.1 clip - text encoder, and image encoder (optional)
    tokenizer = CLIPTokenizer(os.path.join(__dir__, "model_weights/bpe_simple_vocab_16e6.txt.gz"))

    dataloader = build_dataset(cfg, device_num, rank_id, tokenizer, record_data_stat=True)

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


if __name__ == "__main__":
    # 0. parse config
    from configs.train_base import cfg  # base config from train_base.py

    args_for_update = Config(load=True).cfg_dict  # config args from CLI (arg parser) and yaml files

    # update base config
    for k, v in args_for_update.items():
        cfg[k] = v

    print(cfg)
    main(cfg)
