# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import datetime
import math
import argparse
import shutil
import mindspore as ms 
from mindspore import nn, ops, mint
from mindspore.experimental.optim.lr_scheduler import LRScheduler

import safetensors
from omegaconf import OmegaConf
from abc import abstractmethod
from contextlib import contextmanager
from logging import getLogger
from openlrm.utils import seed_everything, str2bool

# from openlrm.utils.logging import configure_logger
from openlrm.runners.abstract import Runner

from mindone.utils import init_train_env, set_logger, count_params
from mindone.safetensors import load_file

logger = getLogger(__name__)


def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser = parse_train_args(parser)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="resume from checkpoint with path",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="configs/instant-nerf-large-train.yaml",
        help="path to base configs",
    )
    parser.add_argument(
        "--log_interval",
        default=1,
        type=int,
        help="log interval in the unit of data sink size.. E.g. if data sink size = 10, log_inteval=2, log every 20 steps",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When debugging, set it true. Dumping files will overlap to avoid trashing your storage.",
    )
    args = parser.parse_args()
    return args


def parse_train_args(parser):
    parser.add_argument("--mode", default=1, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument(
        "--dtype",
        default="fp32",  # if amp level O0/1, must pass fp32
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what computation data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--global_bf16",
        default=False,
        help="Experimental. If True, dtype will be overrided, operators will be computered in bf16 if they are supported by CANN",
    )
    parser.add_argument(
        "--amp_level",
        default="O0",  # cannot amp for InstantMesh training, easily grad nan
        type=str,
        help="mindspore amp level, O1: most fp32, only layers in whitelist compute in fp16 (dense, conv, etc); \
                        O2: most fp16, only layers in blacklist compute in fp32 (batch norm etc)",
    )
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs")
    parser.add_argument(
        "--profile",
        default=False,  # deactivate as profiler says NOT supporting PyNative
        type=str2bool,
        help="Profile or not",
    )
    parser.add_argument(
        "--loss_scaler_type", default=None, type=str, help="dynamic or static"  # loss scale only used in amp O1/O2
    )
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--ckpt_max_keep", default=5, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--output_path", default="outputs/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--num_parallel_workers", default=12, type=int, help="num workers for data loading")
    parser.add_argument(
        "--data_multiprocessing",
        default=False,
        help="If True, use multiprocessing for data processing. Default: multithreading.",
    )
    parser.add_argument("--max_rowsize", default=64, type=int, help="max rowsize for data loading")
    parser.add_argument(
        "--train_steps", default=-1, type=int, help="If not -1, limit the number of training steps to the set value"
    )
    parser.add_argument(
        "--epochs",
        default=7000,
        type=int,
        help="epochs. If dataset_sink_mode is on, epochs is with respect to dataset sink size. Otherwise, it's w.r.t the dataset size.",
    )
    parser.add_argument(
        "--ckpt_save_steps",
        default=-1,
        type=int,
        help="save checkpoint every this steps. If -1, use ckpt_save_interval will be used.",
    )
    parser.add_argument("--step_mode", default=False, help="whether save ckpt by steps. If False, save ckpt by epochs.")
    # optimizer param
    parser.add_argument("--use_ema", default=False, help="whether use EMA")
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--clip_grad", default=True, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--start_learning_rate", default=4e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=4e-5, type=float, help="The end learning rate for Adam.")
    parser.add_argument(
        "--decay_steps", default=9e3, type=int, help="lr decay steps."
    )  # with dataset_size==3, it's 3k epochs
    parser.add_argument("--scheduler", default="cosine_annealing_warm_restarts_lr", type=str, help="scheduler.")
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.9, 0.95],
        help="Specify the [beta1, beta2] parameter for the AdamW optimizer.",
    )
    parser.add_argument(
        "--optim_eps", type=float, default=1e-8, help="Specify the eps parameter for the AdamW optimizer."
    )
    parser.add_argument(
        "--group_strategy",
        type=str,
        default="not_grouping",
        help="Grouping strategy for weight decay. If `norm_and_bias`, weight decay filter list is [beta, gamma, bias]. \
                                If None, filter list is [layernorm, bias]. Default: norm_and_bias",
    )
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
    parser.add_argument("--seed", default=42, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    # dataloader param
    parser.add_argument("--dataset_sink_mode", default=False, help="sink mode")
    parser.add_argument("--sink_size", default=-1, type=int, help="dataset sink size. If -1, sink size = dataset size.")

    return parser

# def parse_configs():
#     # Define argparse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='./assets/config.yaml')
#     args, unknown = parser.parse_known_args()

#     # Load configuration file
#     cfg = OmegaConf.load(args.config)

#     # Override with command-line arguments
#     cli_cfg = OmegaConf.from_cli(unknown)
#     cfg = OmegaConf.merge(cfg, cli_cfg)

    # return cfg


class Trainer(Runner):

    def __init__(self):
        super().__init__()
        
        logger.debug("process id:", os.getpid())

        
        # read configs 
        self.args = parse_args()
        self.cfg = OmegaConf.load(config)

        # attributes with defaults
        self.model : nn.Cell = None
        self.optimizer: nn.optim.Optimizer = None
        self.scheduler: LRScheduler = None
        self.train_loader = None
        self.val_loader = None
        self.N_max_global_steps: int = None
        self.N_global_steps_per_epoch: int = None
        self.global_step: int = 0
        self.current_epoch: int = 0

    def __enter__(self):
        # self.prepare_everything()
        # self.log_inital_info()
        return self
        

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


    def prepare_everything(self, is_dist_validation: bool = False):

        # prepare stats
        N_total_batch_size = self.cfg.train.batch_size * self.cfg.train.accum_steps #* self.accelerator.num_processes
        self.N_global_steps_per_epoch = math.ceil(len(self.train_loader) / self.cfg.train.accum_steps)
        self.N_max_global_steps = self.N_global_steps_per_epoch * self.cfg.train.epochs
        if self.cfg.train.debug_global_steps is not None:
            logger.warning(f"Overriding max global steps from {self.N_max_global_steps} to {self.cfg.train.debug_global_steps}")
            self.N_max_global_steps = self.cfg.train.debug_global_steps
        logger.info(f"======== Statistics ========")
        logger.info(f"** N_max_global_steps: {self.N_max_global_steps}")
        logger.info(f"** N_total_batch_size: {N_total_batch_size}")
        logger.info(f"** N_epochs: {self.cfg.train.epochs}")
        logger.info(f"** N_global_steps_per_epoch: {self.N_global_steps_per_epoch}")
        logger.debug(f"** Prepared loader length: {len(self.train_loader)}")
        logger.info(f"** Distributed validation: {is_dist_validation}")
        logger.info(f"============================")
        logger.info(f"======== Trainable parameters ========")
        logger.info(f"** Total: {sum([p.size for p in model.get_parameters() if p.requires_grad])}")
        # for sub_name, sub_module in self.accelerator.unwrap_model(self.model).named_children():
        #     logger.info(f"** {sub_name}: {sum(p.numel() for p in sub_module.parameters() if p.requires_grad)}")
        for cell_name, cell in self.model.name_cells().items():
            logger.info(f"** {cell_name}: {sum(p.size for p in cell.get_parameters() if p.requires_grad)}")
        logger.info(f"=====================================")
        
        # load checkpoint or model
        self.load_ckpt_or_auto_resume_(self.cfg)
        # register hooks
        self.register_hooks()

    @abstractmethod
    def register_hooks(self):
        pass

    def auto_resume_(self, cfg) -> bool:
        ckpt_root = os.path.join(
            cfg.saver.checkpoint_root,
            cfg.experiment.parent, cfg.experiment.child,
        )
        if not os.path.exists(ckpt_root):
            return False
        ckpt_dirs = os.listdir(ckpt_root)
        if len(ckpt_dirs) == 0:
            return False
        ckpt_dirs.sort()
        latest_ckpt = ckpt_dirs[-1]
        latest_ckpt_dir = os.path.join(ckpt_root, latest_ckpt)
        logger.info(f"======== Auto-resume from {latest_ckpt_dir} ========")
        self.load_ckpt_(latest_ckpt_dir)
        # self.accelerator.load_state(latest_ckpt_dir)
        self.global_step = int(latest_ckpt)
        self.current_epoch = self.global_step // self.N_global_steps_per_epoch
        return True

    def load_ckpt_(self, ckpt_dir):
        if ckpt_dir.endswith(".ckpt"):  # ms.ckpt
            state_dict = ms.load_checkpoint(ckpt_dir)
        elif ckpt_dir.endswith(".safetensors"):
            state_dict = load_file(ckpt_dir)
        else:
            raise AssertionError(
                f"Cannot recognize checkpoint file {ckpt_dir}, only support MS *.ckpt and *.safetensors"
            )
        param_not_load, ckpt_not_load = ms.load_param_into_net(self.model, state_dict, strict_load=True)
        print(f"Loaded checkpoint: param_not_load {param_not_load}, ckpt_not_load {ckpt_not_load}")

    def load_model_(self, cfg):
        logger.info(f"======== Loading model from {cfg.saver.load_model} ========")

        self.load_ckpt_(cfg.saver.load_model)

        logger.info(f"======== Model loaded ========")

    def load_ckpt_or_auto_resume_(self, cfg):
        # auto resume has higher priority, load model from path if auto resume is not available
        if cfg.saver.auto_resume:
            successful_resume = self.auto_resume_(cfg)
            if successful_resume:
                return
        if cfg.saver.load_model:
            successful_load = self.load_model_(cfg)
            if successful_load:
                return
        logger.debug(f"======== No checkpoint or model is loaded ========")

    def save_checkpoint(self):
        ckpt_dir = os.path.join(
            self.cfg.saver.checkpoint_root,
            self.cfg.experiment.parent, self.cfg.experiment.child,
            f"{self.global_step:06d}",
        )
        # self.accelerator.save_state(output_dir=ckpt_dir, safe_serialization=True)
        # TODO: save optimizer & grad scaler etc. 
        os.makedirs(ckpt_dir, exist_ok=True)
        output_model_file = os.path.join(ckpt_dir, "latest.ckpt")
        ms.save_checkpoint(self.model, output_model_file)
        logger.info(f"Saved state to {ckpt_dir}")

        logger.info(f"======== Saved checkpoint at global step {self.global_step} ========")
        # manage stratified checkpoints
        ckpt_dirs = os.listdir(os.path.dirname(ckpt_dir))
        ckpt_dirs.sort()
        max_ckpt = int(ckpt_dirs[-1])
        ckpt_base = int(self.cfg.saver.checkpoint_keep_level)
        ckpt_period = self.cfg.saver.checkpoint_global_steps
        logger.debug(f"Checkpoint base: {ckpt_base}")
        logger.debug(f"Checkpoint period: {ckpt_period}")
        cur_order = ckpt_base ** math.floor(math.log(max_ckpt // ckpt_period, ckpt_base))
        cur_idx = 0
        while cur_order > 0:
            cur_digit = max_ckpt // ckpt_period // cur_order % ckpt_base
            while cur_idx < len(ckpt_dirs) and int(ckpt_dirs[cur_idx]) // ckpt_period // cur_order % ckpt_base < cur_digit:
                if int(ckpt_dirs[cur_idx]) // ckpt_period % cur_order != 0:
                    shutil.rmtree(os.path.join(os.path.dirname(ckpt_dir), ckpt_dirs[cur_idx]))
                    logger.info(f"Removed checkpoint {ckpt_dirs[cur_idx]}")
                cur_idx += 1
            cur_order //= ckpt_base

    @property
    def global_step_in_epoch(self):
        return self.global_step % self.N_global_steps_per_epoch

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _build_optimizer(self):
        pass

    @abstractmethod
    def _build_scheduler(self):
        pass

    @abstractmethod
    def _build_dataloader(self):
        pass

    @abstractmethod
    def _build_loss_fn(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @staticmethod
    def _get_str_progress(epoch: int = None, step: int = None):
        if epoch is not None:
            log_type = 'epoch'
            log_progress = epoch
        elif step is not None:
            log_type = 'step'
            log_progress = step
        else:
            raise ValueError('Either epoch or step must be provided')
        return log_type, log_progress

   
    def log_scalar_kwargs(self, epoch: int = None, step: int = None, split: str = None, **scalar_kwargs):
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f'/{split}' if split else ''
        for key, value in scalar_kwargs.items():
            # self.accelerator.log({f'{key}{split}/{log_type}': value}, log_progress)
            logger.info(f'{log_progress} - {key}{split}/{log_type}: {value}')


    def log_images(self, values: dict, step: int | None = None, log_kwargs: dict | None = {}):
        pass
        # for tracker in self.accelerator.trackers:
        #     if hasattr(tracker, 'log_images'):
        #         tracker.log_images(values, step=step, **log_kwargs.get(tracker.name, {}))

    
    def log_optimizer(self, epoch: int = None, step: int = None, attrs: list[str] = [], group_ids: list[int] = []):
        log_type, log_progress = self._get_str_progress(epoch, step)
        assert self.optimizer is not None, 'Optimizer is not initialized'
        if not attrs:
            logger.warning('No optimizer attributes are provided, nothing will be logged')
        if not group_ids:
            logger.warning('No optimizer group ids are provided, nothing will be logged')
        for attr in attrs:
            assert attr in ['lr', 'momentum', 'weight_decay'], f'Invalid optimizer attribute {attr}'
            for group_id in group_ids:
                # self.accelerator.log({f'opt/{attr}/{group_id}': self.optimizer.param_groups[group_id][attr]}, log_progress)
                logger.info(f'{log_progress} - opt/{attr}/{group_id}: {self.optimizer.param_groups[group_id][attr]}')

    def log_inital_info(self):
        assert self.model is not None, 'Model is not initialized'
        assert self.optimizer is not None, 'Optimizer is not initialized'
        assert self.scheduler is not None, 'Scheduler is not initialized'
        # self.accelerator.log({'Config': "```\n" + OmegaConf.to_yaml(self.cfg) + "\n```"})
        # self.accelerator.log({'Model': "```\n" + str(self.model) + "\n```"})
        # self.accelerator.log({'Optimizer': "```\n" + str(self.optimizer) + "\n```"})
        # self.accelerator.log({'Scheduler': "```\n" + str(self.scheduler) + "\n```"})
        logger.info(f'Config: ```\n {OmegaConf.to_yaml(self.cfg)} + \n```')
        logger.info(f'Model: ```\n {str(self.model)} \n```')
        logger.info(f'Optimizer: ```\n {str(self.optimizer)} \n```')
        logger.info(f'Scheduler: ```\n {str(self.scheduler)} \n```')

    def run(self):
        self.train(self.args, self.cfg)
