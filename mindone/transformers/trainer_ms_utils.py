import os
import numpy as np
from typing import Dict
from dataclasses import dataclass

import mindspore as ms
from mindspore import nn, ops, Tensor

from .mindspore_adapter.train_onestep_wrapper import TrainOneStepWrapper


@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output: Dict, labels: Tensor, shift_labels: bool = False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :]
            labels = labels[..., 1:]

        log_probs = -ops.log_softmax(logits.to(ms.float32), axis=-1).to(logits.dtype)
        if labels.ndim == log_probs.ndim - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.equal(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = ops.clamp(labels, min=0)
        nll_loss = log_probs.gather_elements(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(axis=-1, keepdims=True, dtype=ms.float32)

        nll_loss = nll_loss.masked_fill(padding_mask, 0.0)
        smoothed_loss = smoothed_loss.masked_fill(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.to(ms.int32).sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


def get_model_param_count(model, trainable_only=False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """

    def numel(p):
        # return p.numel()
        return np.prod(p.shape)

    return sum(numel(p) for p in model.get_parameters() if not trainable_only or p.requires_grad)


def get_parameter_names(model: nn.Cell, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """

    # method 1
    # _neg_result = []
    # for name, child in model.cells_and_names():
    #     if isinstance(child, tuple(forbidden_layer_types)):
    #         _neg_result += [n for n, _ in child.parameters_and_names(expand=False)]
    #
    # result = []
    # for p_name, _ in model.parameters_and_names():
    #     if p_name not in _neg_result:
    #         result += [p_name,]
    #
    # return result

    # method 2
    result = []
    for name, child in model.name_cells().items():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += [n for n, p in model.parameters_and_names(expand=False)]
    return result


def _get_learning_rate(object, global_step):

    if isinstance(object, nn.Optimizer):
        optimizer = object
        if optimizer.dynamic_lr:
            if optimizer.is_group_lr:
                lr_cell = optimizer.learning_rate[0]
                cur_lr = lr_cell(Tensor(global_step, ms.int32)).asnumpy().item()
            else:
                cur_lr = optimizer.learning_rate(Tensor(global_step, ms.int32)).asnumpy().item()
        else:
            cur_lr = optimizer.learning_rate.asnumpy().item()
    elif isinstance(object, nn.learning_rate_schedule.LearningRateSchedule):
        lr_cell = object
        cur_lr = lr_cell(Tensor(global_step, ms.int32)).asnumpy().item()
    else:
        raise NotImplementedError

    return cur_lr


def save_state(self):
    """
    Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model

    Under distributed environment this is done only for a process with rank 0.
    """
    if not self.is_world_process_zero():
        return

    path = os.path.join(self.args.output_dir, "trainer_state.json")
    self.state.save_to_json(path)
