import logging
import math
from typing import Literal

__all__ = ["auto_scale_lr"]

logger = logging.getLogger(__name__)


def auto_scale_lr(
    effective_bs: int, lr: float, rule: Literal["linear", "sqrt"] = "linear", base_batch_size: int = 256
) -> float:
    # scale by world size
    if rule == "sqrt":
        scale_ratio = math.sqrt(effective_bs / base_batch_size)
    elif rule == "linear":
        scale_ratio = effective_bs / base_batch_size
    lr *= scale_ratio
    logger.info(f"Automatically adapt lr to {lr:.7f} (using {rule} scaling rule).")
    return lr
