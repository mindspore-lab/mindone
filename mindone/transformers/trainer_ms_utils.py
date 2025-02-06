import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from transformers import BatchEncoding, logging

import mindspore as ms
from mindspore import Tensor, nn, ops

from .mindspore_adapter import Sampler

logger = logging.get_logger(__name__)


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

    def __call__(self, logits: ms.Tensor, labels: ms.Tensor, shift_labels: bool = False):
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


def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use numpy for the random part as a distributed sampler will set the random seed for numpy.
    indices = np.random.permutation(len(lengths))
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = np.argmax(np.array(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Iterable] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, ms.Tensor):
            logger.info(
                "If lengths is a ms.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size)
        return iter(indices)


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
