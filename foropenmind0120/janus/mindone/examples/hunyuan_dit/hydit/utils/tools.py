import logging
import os
import random
import re
from pathlib import Path
from typing import Union

import numpy as np
from transformers.utils import is_safetensors_available

import mindspore as ms
from mindspore import nn, ops

if is_safetensors_available():
    from safetensors import safe_open

    from mindone.safetensors.mindspore import load_file as safe_load_file


def get_trainable_params(model):
    params = model.get_parameters()
    params = [p for p in params if p.requires_grad]
    return params


def set_seeds(seed_list):
    if isinstance(seed_list, (tuple, list)):
        seed = sum(seed_list)
    else:
        seed = seed_list
    random.seed(seed)
    np.random.seed(seed)

    return np.random.RandomState(seed)


def get_start_epoch(resume_path, ckpt, steps_per_epoch):
    if "epoch" in ckpt:
        start_epoch = ckpt["epoch"]
    else:
        start_epoch = 0
    if "steps" in ckpt:
        train_steps = ckpt["steps"]
    else:
        try:
            train_steps = int(Path(resume_path).stem)
        except TypeError:
            train_steps = start_epoch * steps_per_epoch

    start_epoch_step = train_steps % steps_per_epoch + 1
    return start_epoch, start_epoch_step, train_steps


def assert_shape(*args):
    if len(args) < 2:
        return
    cond = True
    fail_str = f"{args[0] if isinstance(args[0], (list, tuple)) else args[0].shape}"
    for i in range(1, len(args)):
        shape1 = args[i] if isinstance(args[i], (list, tuple)) else args[i].shape
        shape2 = args[i - 1] if isinstance(args[i - 1], (list, tuple)) else args[i - 1].shape
        cond = cond and (shape1 == shape2)
        fail_str += f" vs {args[i] if isinstance(args[i], (list, tuple)) else args[i].shape}"
    assert cond, fail_str


def create_logger(rank, logging_dir=None, logging_file=None, ddp=True):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not ddp or (ddp and rank == 0):  # real logger
        if logging_file is not None:
            file_handler = [logging.FileHandler(logging_file)]
        elif logging_dir is not None:
            file_handler = [logging.FileHandler(f"{logging_dir}/log.txt")]
        else:
            file_handler = []
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler()] + file_handler,
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def create_exp_folder(args, rank):
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
    existed_experiments = list(Path(args.results_dir).glob("*"))
    if len(existed_experiments) == 0:
        experiment_index = 1
    else:
        existed_experiments.sort()
        print("existed_experiments", existed_experiments)
        experiment_index = max([int(x.stem.split("-")[0]) for x in existed_experiments]) + 1
    model_string_name = args.task_flag if args.task_flag else args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(rank, experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(rank)
        experiment_dir = ""

    return experiment_dir, checkpoint_dir, logger


def get_pt2ms_mappings(m):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    for name, cell in m.cells_and_names():
        if isinstance(cell, (nn.Conv1d, nn.Conv1dTranspose)):
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: ms.Parameter(
                ops.expand_dims(x, axis=-2), name=x.name
            )
        elif isinstance(cell, nn.Embedding):
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
            if isinstance(cell, (nn.BatchNorm2d,)):
                mappings[f"{name}.running_mean"] = f"{name}.moving_mean", lambda x: x
                mappings[f"{name}.running_var"] = f"{name}.moving_variance", lambda x: x
                mappings[f"{name}.num_batches_tracked"] = None, lambda x: x
    return mappings


def convert_state_dict(m, state_dict_pt):
    if not state_dict_pt:
        return state_dict_pt
    pt2ms_mappings = get_pt2ms_mappings(m)
    state_dict_ms = {}
    while state_dict_pt:
        name_pt, data_pt = state_dict_pt.popitem()
        name_ms, data_mapping = pt2ms_mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = data_mapping(data_pt)
        if name_ms is not None:
            state_dict_ms[name_ms] = data_ms
    return state_dict_ms


def load_state_dict(checkpoint_file: Union[str, os.PathLike]):
    """
    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
            # Check format of the archive
            with safe_open(checkpoint_file, framework="np") as f:
                metadata = f.metadata()
            if metadata.get("format") not in ["pt", "tf", "flax", "np"]:
                raise OSError(
                    f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                    "you save your model with the `save_pretrained` method."
                )
            return safe_load_file(checkpoint_file)
        else:
            raise NotImplementedError(
                f"Only supports deserialization of weights file in safetensors format, but got {checkpoint_file}"
            )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read(7) == "version":
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' " f"at '{checkpoint_file}'. "
            )


def model_resume(args, model, ema, logger, len_loader):
    """
    Load pretrained weights.
    """
    start_epoch = 0
    start_epoch_step = 0
    train_steps = 0

    # Resume model
    if args.resume:
        resume_path = args.resume_module_root
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"    Cannot find model checkpoint from {resume_path}")
        logger.info(f"    Resume from checkpoint {resume_path}")
        if resume_path.endswith(".safetensors"):
            resume_ckpt = load_state_dict(resume_path)
            resume_ckpt = convert_state_dict(model, resume_ckpt)
            if args.use_fp16:
                resume_ckpt = {"module." + k: v for k, v in resume_ckpt.items()}
            local_state = {re.sub("_backbone.", "", k): v for k, v in model.parameters_and_names()}
            for k, v in resume_ckpt.items():
                if k in local_state:
                    v.set_dtype(local_state[k].dtype)
                else:
                    pass  # unexpect key keeps origin dtype
        elif resume_path.endswith(".ckpt"):
            resume_ckpt = ms.load_checkpoint(resume_path)
        ms.load_param_into_net(model, resume_ckpt, strict_load=True)

    # # Resume EMA model
    if args.use_ema:
        raise NotImplementedError("EMA feature is not supported.")

    if not args.reset_loader:
        start_epoch, start_epoch_step, train_steps = get_start_epoch(args.resume, resume_ckpt, len_loader)

    return model, ema, start_epoch, start_epoch_step, train_steps
