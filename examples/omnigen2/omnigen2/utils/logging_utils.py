# Adapted from https://github.com/VectorSpaceLab/OmniGen2/blob/main/omnigen2/utils/logging_utils.py
# Adapted from https://github.com/huggingface/accelerate/blob/v1.9.0-release/src/accelerate/logging.py
import logging
import os
import time
from pathlib import Path

from transformers.utils.logging import set_verbosity as set_transformers_verbosity

from mindone.diffusers.utils.logging import set_verbosity as set_diffusers_verbosity
from mindone.utils import count_params


class TqdmToLogger(object):
    """File-like object to redirect tqdm output to a logger."""

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line)

    def flush(self):
        for handler in self.logger.logger.handlers:
            handler.flush()


def get_logger(name: str, log_level: str = None):
    """
    Returns a `logging.Logger` for `name` that can handle multiprocessing.

    If a log should be called on all processes, pass `main_process_only=False` If a log should be called on all
    processes and in order, also pass `in_order=True`

    Args:
        name (`str`):
            The name for the logger, such as `__file__`
        log_level (`str`, *optional*):
            The log level to use. If not passed, will default to the `LOG_LEVEL` environment variable, or `INFO` if not
    """
    if log_level is None:
        log_level = os.environ.get("ACCELERATE_LOG_LEVEL", None)
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
    return logging.LoggerAdapter(logger, {})


def setup_logging(logger, output_dir, is_main_process: bool) -> None:
    """
    Set up logging configuration for training.

    Args:
        args: Configuration object
        is_main_process: Whether the current process is the main process.
    """

    logging_dir = Path(output_dir, "logs")
    if is_main_process:
        # Create logging directory and file handler
        os.makedirs(logging_dir, exist_ok=True)
        log_file = Path(logging_dir, f'{time.strftime("%Y%m%d-%H%M%S")}.log')

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.logger.addHandler(file_handler)

    # Configure basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )

    # Set verbosity for different processes
    log_level = logging.INFO if is_main_process else logging.ERROR
    set_transformers_verbosity(log_level)
    set_diffusers_verbosity(log_level)


def log_model_info(logger, name: str, model):
    """Logs parameter counts for a given model."""
    total_params, trainable_params = count_params(model)
    logger.info(f"--- {name} ---")
    logger.info(model)
    logger.info(f"Total parameters (M): {total_params / 1e6:.2f}")
    logger.info(f"Trainable parameters (M): {trainable_params / 1e6:.2f}")
