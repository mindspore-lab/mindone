import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings

import mindspore.dataset
import numpy as np
from collections.abc import Mapping
from packaging import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Dict, List, Optional, Tuple, Union, NamedTuple
from ezcolorlog import root_logger as logger

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.communication.management import get_group_size

from transformers import PreTrainedTokenizerBase
from transformers.utils import logging, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from transformers.integrations import get_reporting_integration_callbacks
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from ..safetensors.mindspore import save_file
from .mindspore_adapter.utils import _is_parallel
from .mindspore_adapter import (
    auto_mixed_precision,
    Sampler,
    RandomSampler
)

from .modeling_utils import MSPreTrainedModel as PreTrainedModel
from .trainer_ms_utils import (
    get_model_param_count,
    _get_learning_rate,
    TrainOneStepWrapper,
    LengthGroupedSampler
)
from .training_args import TrainingArguments, OptimizerNames
from .trainer_utils import (
    EvalPrediction,
    RemoveColumnsCollator,
    enable_full_determinism,
    set_seed,
    has_length,
    number_of_arguments,
    speed_metrics,
    get_last_checkpoint
)
from .trainer_callback import (
    TrainerState,
    TrainerCallback,
    CallbackHandler,
    DefaultFlowCallback,
    ProgressCallback,
    PrinterCallback,
    TrainerControl,
    ExportableState
)
from .data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator
)
from .optimization import get_scheduler
from .feature_extraction_sequence_utils import SequenceFeatureExtractor
from .trainer_ms_utils import LabelSmoother, get_parameter_names
from .utils import find_labels, can_return_loss, is_datasets_available
from .mindspore_utils import ALL_LAYERNORM_LAYERS
from .debug_utils import DebugOption


if is_datasets_available():
    import datasets


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


def _is_peft_model(model):
    # TODO: support PEFT Model
    return False


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


# Name of the files used for checkpointing
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.ckpt"
# SCHEDULER_NAME = "scheduler.ckpt"     # Note: lr_scheduler is already included in the optimizer on MindSpore 2.3.1
SCALER_NAME = "scaler.ckpt"


class Trainer:

    from .trainer_ms_utils import save_state

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Cell] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Iterable] = None,
            eval_dataset: Optional[Iterable] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[nn.Optimizer, nn.learning_rate_schedule.LearningRateSchedule] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        if args.batch_eval_metrics and compute_metrics is not None:
            if "compute_result" not in inspect.signature(compute_metrics).parameters.keys():
                raise ValueError(
                    "When using `batch_eval_metrics`, your `compute_metrics` function must take a `compute_result`"
                    " boolean argument which will be triggered after the last batch of the eval set to signal that the"
                    " summary statistics should be returned by the function."
                )
        self.args = args
        # Seed must be set before instantiating the model when using model
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        # self.create_accelerator_and_postprocess()

        # memory metrics - must set up as early as possible
        # self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        # self._memory_tracker.start()

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method. This will become a fatal error in the next"
                    " release.",
                    FutureWarning,
                )
            self.model_init = model_init

        if getattr(model, "is_parallelizable", False) and getattr(model, "model_parallel", False):
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        _is_quantized_and_base_model = getattr(model, "is_quantized", False) and not getattr(
            model, "_hf_peft_config_loaded", False
        )
        _quantization_method_supports_training = (
                getattr(model, "hf_quantizer", None) is not None and model.hf_quantizer.is_trainable
        )
        if _is_quantized_and_base_model or _quantization_method_supports_training:
            raise NotImplementedError

        # Filter out quantized + compiled models
        if _is_quantized_and_base_model and hasattr(model, "_orig_mod"):
            raise ValueError(
                "You cannot fine-tune quantized model with `ms.jit()` or `ms.GRAPH_MODE` make sure to pass a non-compiled model when fine-tuning a quantized model with PEFT"
            )

        # At this stage the model is already loaded
        if _is_quantized_and_base_model and not _is_peft_model(model):
            raise ValueError(
                "You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of"
                " the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft"
                " for more details"
            )
        elif _is_quantized_and_base_model and not _quantization_method_supports_training:
            raise ValueError(
                f"The model you are trying to fine-tune is quantized with {model.hf_quantizer.quantization_config.quant_method}"
                " but that quantization method do not support training. Please open an issue on GitHub: https://github.com/huggingface/transformers"
                f" to request the support for training support for {model.hf_quantizer.quantization_config.quant_method}"
            )

        default_collator = (
            DataCollatorWithPadding(tokenizer)
            if tokenizer is not None and isinstance(tokenizer, (PreTrainedTokenizerBase, SequenceFeatureExtractor))
            else lambda features, batch_info: default_data_collator(features, return_tensors='np')
        )
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.model = model

        self.neftune_noise_alpha = args.neftune_noise_alpha

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            # self.init_hf_repo()
            raise NotImplementedError
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if args.max_steps > 0 and args.num_train_epochs > 0:
            logger.warning("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        if (
                train_dataset is not None
                and args.group_by_length
        ):
            raise NotImplementedError

        self._signature_columns = None

        # Mixed precision setup
        self.use_apex = False
        self.use_cpu_amp = False

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`] and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            [`~transformers.TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self.model
            if _is_peft_model(self.model):
                raise NotImplementedError
            signature = inspect.signature(model_to_inspect.construct)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.construct` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.construct`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if len(columns) == 0:
            raise ValueError(
                "No columns in the dataset match the model's construct method signature. "
                f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "
                "Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        lr_scheduler = self.create_scheduler(num_training_steps=num_training_steps)
        self.create_optimizer(lr_scheduler)

    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def create_optimizer(self, lr_scheduler: Union[Tuple, List] = None):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.parameters_and_names() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.parameters_and_names() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            # Note: Init optimizer with lr scheduler on MindSpore 2.3.1
            # Update learning rate with lr_scheduler
            if lr_scheduler is not None:
                optimizer_kwargs.update({"learning_rate": lr_scheduler})

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                raise NotImplementedError

        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(
            args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"learning_rate": args.learning_rate}

        adam_kwargs = {
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = nn.AdaFactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_MINDSPORE:
            from .mindspore_adapter.adamw import AdamWeightDecay
            optimizer_cls = AdamWeightDecay
            optimizer_kwargs.update(adam_kwargs)
            optimizer_kwargs.update({"enable_fuse": getattr(args, "adamw_enable_fuse", True)})
        elif args.optim in (OptimizerNames.ADAMW_ZERO1_MINDSPORE, OptimizerNames.ADAMW_ZERO2_MINDSPORE):
            from .mindspore_adapter.adamw_zero import AdamWeightDecayZeRO1, AdamWeightDecayZeRO2
            optimizer_cls = \
                AdamWeightDecayZeRO1 if args.optim == OptimizerNames.ADAMW_ZERO1_MINDSPORE else AdamWeightDecayZeRO2
            optimizer_kwargs.update(adam_kwargs)
            optimizer_kwargs.update({"enable_fuse": getattr(args, "adamw_enable_fuse", True)})
            optimizer_kwargs.update({"shard_size": getattr(args, "adamw_zero_shard_size", None)})
            optimizer_kwargs.update({"momentum_dtype": getattr(args, "adamw_zero_momentum_dtype", ms.float32)})
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = nn.SGD
        elif args.optim == OptimizerNames.Momentum:
            optimizer_cls = nn.Momentum
            optimizer_kwargs.update({"momentum": getattr(args, "momentum_value", 0.9)})
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = nn.Adagrad
        elif args.optim == OptimizerNames.RMSPROP:
            optimizer_cls = nn.RMSProp
        elif args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            raise NotImplementedError
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                base_lr=self.args.learning_rate,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def get_train_dataloader(self) -> ms.dataset.Dataset:
        """
        Returns the training [`~mindspore.dataset.GeneratorDataset`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            class MSDataset:
                def __init__(self, dataset: datasets.Dataset):
                    self.dataset = dataset
                def __getitem__(self, item):
                    return self.dataset[int(item)]
                def __len__(self):
                    return len(self.dataset)

            train_dataset = self._remove_unused_columns(train_dataset, description="training")
            train_dataset = MSDataset(train_dataset)

        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if self.args.dataloader_pin_memory:
            logger.warning("Not support `dataloader_pin_memory`")
        if self.args.dataloader_persistent_workers:
            logger.warning("Not support `dataloader_persistent_workers`")

        prefetch_factor = self.args.dataloader_prefetch_factor
        if prefetch_factor is not None and prefetch_factor > 0:
            ms.dataset.config.set_prefetch_size(prefetch_factor)

        ds_init_params = {
            "num_parallel_workers": self.args.dataloader_num_workers,
            "sampler": self._get_train_sampler(),
            "python_multiprocessing": False,
            "num_shards": getattr(self.args, "rank_size", 1),
            "shard_id": getattr(self.args, "rank", 0),
            "column_names": "item"
        }

        ds_batch_params = {
            "num_parallel_workers": self.args.dataloader_num_workers,  # num workers
            "batch_size": self.args.per_device_train_batch_size,       # per device batch size
            "per_batch_map": data_collator,                            # collate function
            "drop_remainder": self.args.dataloader_drop_last,          # drop last
        }
        ds_repeat_params = {
            "count": 1  # self.args.num_train_epochs            # num_train_epochs, loop at train func
        }

        loader = ms.dataset.GeneratorDataset(train_dataset, **ds_init_params)
        loader = loader.batch(**ds_batch_params)
        loader = loader.repeat(**ds_repeat_params)

        logger.info(f"create dataloader success, \n"
                    f"\tshard_id/num_shards: {ds_init_params['shard_id']}/{ds_init_params['num_shards']}\n"
                    f"\tnum_parallel_workers: {ds_init_params['num_parallel_workers']}\n"
                    f"\tpython_multiprocessing: {ds_init_params['python_multiprocessing']}\n"
                    f"\tper_batch_size: {ds_batch_params['batch_size']}")

        return loader

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.train_dataset)

    def num_examples(self, dataloader: ms.dataset.Dataset) -> int:
        if not isinstance(dataloader, ms.dataset.Dataset):
            dataset = dataloader.dataset
            return len(dataset)
        else:  # no dataset or length, estimate by length of dataloader
            # FIXME: Consider parallel scenarios
            return len(dataloader) * self.args.per_device_train_batch_size

    def num_tokens(self, train_dl: ms.dataset.Dataset, max_steps: Optional[int] = None) -> int:
        """
        Helper to get number of tokens in a [`~mindspore.dataset.Dataset`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            # FIXME: Consider padding
            for step, batch in enumerate(train_dl):
                if isinstance(batch["input_ids"], Tensor):
                    # tokens = batch["input_ids"].numel()
                    tokens = np.prod(batch["input_ids"].shape)
                elif isinstance(batch["input_ids"], np.ndarray):
                    tokens = batch["input_ids"].size
                else:
                    tokens = None

                if max_steps is not None:
                    return tokens * max_steps
                train_tokens += tokens
            return train_tokens
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")
            return train_tokens

    def mindspore_jit_model(self, model, dataloader):
        # TODO: add pre-compile
        logger.warning(f"wrap model[{model.__class__.__name__}] to jit model.")

        class JitWarpper(nn.Cell):
            def __init__(self, model):
                super(JitWarpper, self).__init__(auto_prefix=False)
                self.jit_model = model

            @ms.jit
            def construct(self, *args, **kwargs):
                self.jit_model(*args, **kwargs)

        return JitWarpper(model)

    def _wrap_model(self, model, dataloader=None):
        if self.args.jit_mode and ms.get_context("mode") == ms.PYNATIVE_MODE:
            start_time = time.time()
            model = self.mindspore_jit_model(model, dataloader)

            # FIXME: just build model, time not included compile cost.
            self.jit_compilation_time = round(time.time() - start_time, 4)

        # enable auto mix precision
        assert not (self.args.fp16 and self.args.bf16)
        amp_level = self.args.amp_opt_level if self.args.amp_opt_level is not None else "O2"
        if self.args.fp16:
            model = auto_mixed_precision(model, amp_level, dtype=ms.float16)
        if self.args.bf16:
            model = auto_mixed_precision(model, amp_level, dtype=ms.bfloat16)

        # Note: unlike the original transformers, support label_smoother through `Trainer._wrap_model`, and origin support it at `Trainer.compute_loss`
        if self.label_smoother is not None:
            signature_columns = list(inspect.signature(self.model.construct).parameters.keys())[1:]
            input_labels_index = signature_columns.index("labels") if "labels" in signature_columns else None

            class LabelSmootherModel(nn.Cell):
                def __init__(self, model, label_smoother, labels_index):
                    super(LabelSmootherModel, self).__init__(auto_prefix=False)
                    self.model = model
                    self.label_smoother_ = label_smoother
                    self.labels_index = labels_index
                    self.shift_labels = model._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()

                def construct(self, *inputs):
                    labels = None
                    if self.labels_index is not None:
                        labels = inputs[self.labels_index]

                    outputs = self.model(*inputs)
                    loss, logits = outputs[:2]
                    if labels is not None:
                        loss = self.label_smoother_(logits, labels, self.shift_labels)

                    return loss

            model_ = LabelSmootherModel(model, self.label_smoother, input_labels_index)
        else:
            class ReturnLoss(nn.Cell):
                def __init__(self, model):
                    super(ReturnLoss, self).__init__(auto_prefix=False)
                    self.model = model

                def construct(self, *args, **kwargs):
                    outputs = self.model(*args, **kwargs)
                    loss = outputs[0]
                    return loss

            model_ = ReturnLoss(model)

        # Note: unlike the original transformers, we will define train step process
        # that include auto mix precision, forward process, loss compute and optimizer step on `train_model`
        train_model = TrainOneStepWrapper(
            model_,
            self.optimizer,
            ema=None,
            drop_overflow_step=True,
            scaler="default",
            scaler_config={"scale_value": 1024},
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            clip_grad="global_norm",
            clip_value=self.args.max_grad_norm
        )

        return model, train_model

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        # self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            raise NotImplementedError

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        # self._hp_search_setup(trial)  # TODO, level 3, Add hyper parameters search function
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            # self._load_from_checkpoint(resume_from_checkpoint)  # load weight later

            if os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
                # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
                state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
                if state.train_batch_size is not None:
                    self._train_batch_size = state.train_batch_size

        inner_training_loop = functools.partial(self._inner_training_loop, batch_size=self._train_batch_size)  # TODO: level 3, Add find_executable_batch_size function
        if args.push_to_hub:
            raise NotImplementedError
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP."
                )
            else:
                raise NotImplementedError

        # FIXME: Consider parallelism mode
        delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        else:
            raise NotImplementedError

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        self.model, self.train_model = self._wrap_model(self.model)

        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            raise NotImplementedError

        # ckpt loading
        if resume_from_checkpoint is not None:
            logger.info("Checkpoint loading...")

            self._load_from_checkpoint(resume_from_checkpoint)

            # Check if saved optimizer or scheduler states exist
            self._load_optimizer_and_scheduler(resume_from_checkpoint)
        else:
            logger.warning("No available resume checkpoint.")

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of full parameters = {get_model_param_count(self.model, trainable_only=False):,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            raise NotImplementedError

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
            raise NotImplementedError
        if trial is not None:
            raise NotImplementedError
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = 0.0
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader.create_dict_iterator(num_epochs=1, output_numpy=True)
            # FIXME: consider resume, skip the previous steps
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(train_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                # self._load_rng_state(resume_from_checkpoint)  # FIXME: load rng state
                pass

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                raise NotImplementedError

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                inputs = inputs["item"]

                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    raise NotImplementedError

                if rng_to_sync:
                    # self._load_rng_state(resume_from_checkpoint)
                    # rng_to_sync = False
                    raise NotImplementedError

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    raise NotImplementedError
                elif steps_trained_progress_bar is not None:
                    raise NotImplementedError

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                self.model.set_train(True)
                self.train_model.set_train(True)
                tr_loss_step, overflow = self.training_step(self.train_model, inputs)
                tr_loss_step = tr_loss_step.asnumpy()

                # TODO: log by callback_fn
                logger.info(f"Epoch: {epoch}, Step: {step}, tr_loss: {tr_loss_step}, overflow: {overflow}")

                if (
                    args.logging_nan_inf_filter
                    and (np.isnan(tr_loss_step) or np.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    logger.warning("tr_loss exist nan/inf, replace to average of previous")
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        logger.warning("last step not gradient_accumulation_steps, skip.")

                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, self.model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, self.model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            raise NotImplementedError

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        # TODO: level 3, Add memory tracker
        # self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            raise NotImplementedError

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if model is None:
            model = self.model

        if os.path.isfile(resume_from_checkpoint):
            s_time = time.time()
            state_dict = ms.load_checkpoint(resume_from_checkpoint)
            m, u = ms.load_param_into_net(model, state_dict)

            m = [n for n in m if ("_buffer" not in n) and (".inv_freq" not in n)]
            if len(m) > 0:
                logger.info(f"WARNING: missing keys num: {len(m)}, names (top 100): {m[:10]}")
            if len(u) > 0:
                logger.info(f"WARNING: unexpected keys num: {len(u)}, names (top 100): {u[:10]}")

            logger.info(f"load checkpoint from `{resume_from_checkpoint}` success, time cost: {time.time() - s_time:.2f}s")
        else:
            logger.warning(f"resume_from_checkpoint is not file: `{resume_from_checkpoint}`")

    def _load_optimizer_and_scheduler(self, resume_from_checkpoint):
        if resume_from_checkpoint is None:
            return

        # get path to file
        OPTIMIZER_PATH = os.path.join(resume_from_checkpoint, OPTIMIZER_NAME)

        # Note: lr_scheduler is already included in the optimizer on MindSpore 2.3.1
        # LR_PATH = os.path.join(resume_from_checkpoint, SCHEDULER_NAME)

        if os.path.isfile(OPTIMIZER_PATH):
            optimizer_state = ms.load_checkpoint(OPTIMIZER_PATH)
            optimizer_state = optimizer_state['optimizer_state']
            ms.load_param_into_net(self.optimizer, optimizer_state)
            logger.info(f"Optimizer state successfully loaded from {OPTIMIZER_PATH}")
        else:
            logger.warning(f"Not exist optimizer state checkpoint path: `{OPTIMIZER_PATH}`")

        # Note: lr_scheduler is already included in the optimizer on MindSpore 2.3.1
        # if os.path.isfile(LR_PATH):
        #     lr_scheduler_state = ms.load_checkpoint(LR_PATH)
        #     ms.load_param_into_net(self.lr_scheduler, lr_scheduler_state)
        #     logger.info(f"LR scheduler state successfully loaded from {LR_PATH}")
        # else:
        #     logger.warning(f"Not exist lr scheduler state checkpoint path: `{LR_PATH}`")

        print("Loaded optimizer and lr scheduler state done.")

    def _nested_reduce_sum(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return

        if self.args.framework == "mindspore":
            if _is_parallel():
                return ops.AllReduce()(tensors).mean()
        else:
            raise NotImplementedError

        return tensors

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:

            logs: Dict[str, float] = {}

            # FIXME: consider parallel reduce
            # get average loss over all processes
            # tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            # if _is_parallel():
            #     tr_loss_scalar = self._nested_reduce_sum(tr_loss).item() / get_group_size()
            # else:
            #     tr_loss_scalar = tr_loss.item()
            tr_loss_scalar = tr_loss.item() if isinstance(tr_loss, (Tensor, np.ndarray)) else tr_loss

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, (Tensor, np.ndarray)) else grad_norm
            # logs["learning_rate"] = _get_learning_rate(self.optimizer, self.state.global_step) # FIXME: may causl memory leak?

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            # self._save_optimizer_and_scheduler(output_dir)

            # Save RNG state
            # self._save_rng_state(output_dir)

            raise NotImplementedError

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                metric_value = metrics[metric_to_check]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            # Update the `TrainerControl` state to where we are currently
            self.state.stateful_callbacks["TrainerControl"] = self.control.state()
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            # self._push_from_checkpoint(output_dir)
            raise NotImplementedError

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _prepare_input(self, data: Union[Tensor, Any]) -> Union[Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, ms.Tensor):
            if hasattr(self.args, "input_dtype"):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                if data.dtype in (ms.int32, ms.int64, ms.bool_):
                    return data

                kwargs = {"dtype": self.args.input_dtype}
                return data.to(**kwargs)

        elif isinstance(data, np.ndarray):
            return self._prepare_input(Tensor(data))

        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[Tensor, Any]]) -> Dict[str, Union[Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def _prepare_inputs_ms(self, inputs: Dict[str, Union[Tensor, Any]]):
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        # 1. get model args
        model_to_inspect = self.model
        signature = inspect.signature(model_to_inspect.construct)
        for n, p in signature.parameters.items():
            assert p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.VAR_POSITIONAL), \
                f"construct func input not position args, check in `class {model_to_inspect.__class__.__name__}`"
        _signature_columns = list(signature.parameters.keys())
        _signature_columns = _signature_columns[1:] if _signature_columns[0] == self else _signature_columns

        input_keys = _signature_columns
        dict_inputs = inputs
        input_len = max([input_keys.index(k) for k in dict_inputs]) + 1

        # 2. to tuple
        tuple_inputs = ()
        for k in input_keys[:input_len]:
            if k not in dict_inputs:
                assert not isinstance(signature.parameters[k].default, inspect._empty)
                v = signature.parameters[k].default
            else:
                v = dict_inputs.pop(k)
            if isinstance(v, (tuple, list)):
                tuple_inputs += (*v,)
            else:
                tuple_inputs += (v,)
        if len(dict_inputs) > 0:
            logger.warning(f"input args {dict_inputs.keys()} not found in {self.model.__class__.__name__}, ignore them.")

        # 3. to tensor
        inputs = ()
        for data in tuple_inputs:
            if data is not None:
                if hasattr(self.args, "input_dtype") and data.dtype in (np.float16, np.float32, np.float64):
                    data = ms.Tensor(data, dtype=self.args.input_dtype)
                elif data.dtype in (np.uint8, np.uint16, np.uint32, np.uint64,
                                    np.int8, np.int16, np.int32, np.int64):
                    data = ms.Tensor(data, dtype=ms.int32)
                else:
                    data = ms.Tensor(data)
            inputs += (data,)

        return inputs

    def call_model_init(self, trial=None):
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model

    def training_step(self, model: nn.Cell, inputs: Dict[str, Union[ms.Tensor, Any]]) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Cell`):
                The model to train.
            inputs (`Dict[str, Union[ms.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `Tuple[ms.Tensor, ms.Tensor]`: The tensor with training loss and overflow flag on this batch.
        """
        train_model = model
        train_model.set_train()

        tuple_inputs = self._prepare_inputs_ms(inputs)

        loss, _, overflow = train_model(*tuple_inputs)

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            raise NotImplementedError

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            raise NotImplementedError

        return loss / self.args.gradient_accumulation_steps, overflow

    def compute_loss(self, model, inputs, return_outputs=False):
        raise NotImplementedError

    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        raise NotImplementedError

    def _get_output_dir(self, trial):
        if self.hp_search_backend is not None and trial is not None:
            raise NotImplementedError
        else:
            run_dir = self.args.output_dir
        return run_dir

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
        # process index.
        return self.args.process_index == 0

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            # self.push_to_hub(commit_message="Model save")
            raise NotImplementedError

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = {k: v for k, v in self.model.parameters_and_names()}

            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")

            if self.args.save_safetensors:
                save_file(
                    state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "ms"}
                )
            else:
                ms.save_checkpoint(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # TODO: save args
        # Good practice: save your training arguments together with the trained model
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model

        if _is_parallel():
            # FIXME: consider parallel reduce when dynamic size
            # self.state.total_flos += (
            #     ops.AllReduce()(Tensor(self.current_flos, ms.float32)).item()
            # )
            self.state.total_flos += self.current_flos * get_group_size()
            self.current_flos = 0
        else:
            self.state.total_flos += self.current_flos
            self.current_flos = 0

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if (
            self.state.best_model_checkpoint is not None
            and str(Path(self.state.best_model_checkpoint)) in checkpoints_sorted
        ):
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)

    def floating_point_ops(self, inputs: Dict[str, Union[Tensor, Any]]):
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[ms.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def _finish_current_push(self):
        if not hasattr(self, "push_in_progress"):
            return
        if self.push_in_progress is not None and not self.push_in_progress.is_done():
            logger.info("Waiting for the current checkpoint push to be finished, this might take a couple of minutes.")
            self.push_in_progress.wait_until_done()
