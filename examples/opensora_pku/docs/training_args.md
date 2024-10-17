# Training Arguments
This document includes the training arguments of [`opensora/train/train_t2v.py`](../opensora/train/train_t2v.py).

## Dataset and Pretrained Weights

- `image_data` (type: str, default: None): the path to the image dataset file, e.g., `scripts/train_data/image_data.txt`. If not provided: if `use_image_num` is not zero, `use_img_from_vid` must be True, and the dataset will load images from the video dataset; if `use_image_num` is zero, the model will be trained on video datasets only.
- `video_data` (type: str, default: None): the path to the video dataset file, e.g., `scripts/train_data/video_data.txt`. It must be provided, and the video dataset file should not be empty.
- `max_image_size` (type: int, default: 512): the image/frame resolution, which equals to the width and the height.
- `num_frames` (type: int, default: 17): the number of frames sampled from the video file. It should be an odd integer, e.g., 17 or 65.
- `use_img_from_vid` (type: bool, default: False): whether to use the non-consecutive frames sampled from videos as images in video-image joint training. If it is False, the images will be sampled from the given image dataset. If it is True, the images will be sampled from videos.
- `use_image_num` (type: int, default: 0): the number of images sampled from the image dataset. If it is set to an integer greater than zero, it enables video-image-joint training.
- `text_embed_cache` (type: bool, default: True): whether to use t5 text embedding cache for acceleration. If False, it will extract text embedding on-the-fly during training, and the expected paths in `video_data` and `image_data` should be two: the images/videos folder path and the annotation json file path. If True, the text embedding cache must be extracted ahead, and the expected paths in `video_data` and `image_data` should be three: the images/videos folder path, the text embedding cache folder, and the annotation json file path.
- `enable_flip` (type: bool, default: False): whether to enable random flip of video frames. This is to avoid motion direction and text mismatch.
- `model_max_length` (type: int, default: 300): the text embedding length. When `text_embed_cache` is True, `model_max_length` should match with the length of the text embedding cache.
- `filter_nonexistent` (type: bool, default: True): whether to filter non-existent samples in the dataset.
- `pretrained` (type: str, default: None): the pretrained checkpoint path to be loaded as initial weight. If not specified, LatteT2V will use random initialization.
- `text_encoder_name` (type: str, default: "DeepFloyd/t5-v1_1-xxl"): the text encoder name and the path.
- `ae` (type: str, default: "CausalVAEModel_4x8x8"): the VAE name.
- `ae_path` (type: str, default: "LanguageBind/Open-Sora-Plan-v1.1.0"): the directory name of VAE.
- `dataset` (type: str, default: "t2v"): the dataset type.

## LatteT2V Model Definition
- `use_rope` (type: bool, default: False): whether to use RoPE(Rotary Position Embedding).
- `compress_kv` (type: bool, default: False): whether to apply KV compression.
- `compress_kv_factor` (type: int, default: 1): the KV compression factor.
- `model` (type: str, default: "LatteT2V-XL/122"): the lattet2v model name.
- `multi_scale` (type: bool, default: False): whether to support multi-scale training. Multi-scale training is not supported now. Working in progress.

## Model Acceleration
- `enable_tiling` (type: bool, default: False): whether to use vae tiling to save memory. If True, it will run vae inference with less memory in a slower speed.
- `tile_overlap_factor` (type: float, default: 0.25, range: (0, 1)): the overlap factor of vae tiling.
- `use_recompute` (type: bool, default: False): whether to use recompute (gradient checkpointing) to save memory. If True, will run lattet2v training with less memory in a slower speed.
- `num_no_recompute` (type: int, default: 0, min: 0): the number of transformer blocks to be removed from the recomputation list if `use_recompute` is True.

## Mixed Precision
- `precision` (type: str, default: "bf16", choices: ["bf16", "fp16", "fp32"]): the data type for mixed precision configuration of LatteT2V model.
- `amp_level` (type: str, default: "O1"): mindspore amp level, "O1": most fp32, only layers in whitelist compute in fp16/bf16 (dense, conv, etc), "O2": most fp16 or bf16, only layers in blacklist compute in fp32 (batch norm etc).
- `text_encoder_precision` (type: str, default: "bf16"): the data type for mixed precision configuration of T5 model.
- `vae_precision` (type: str, default: "fp16"): the data type for mixed precision configuration of VAE model.
- `vae_keep_gn_fp32` (type: bool, default: False): whether to put GroupNorm in to the custom fp32 cells list when amp_level is "O2" for VAE model.
- `loss_scaler_type` (type: str, default: "dynamic"): the loss scaler type. Options: ["dynamic", "static"].
- `loss_scale_factor`: (type: int, default:2): the loss scale factor. It is only applied when `loss_scaler_type` is dynamic.
- `scale_window`: (type: int, default:2000): the loss scale change window. It is only applied when `loss_scaler_type` is dynamic.
- `init_loss_scale` (type: int, default: 65536): the initial value of loss scale. If `loss_scaler_type` is static, the initial value of loss scale will not change. If `loss_scaler_type` is dynamic: if there is no overflow within `scale_window` steps, the loss scale value will be reduced by `loss_scale_factor`; if there is overflow within `scale_window` steps, the loss scale value will be increased by `loss_scale_factor`.

## Dataloader Acceleration
- `dataset_sink_mode` (type: bool, default: False): whether the dataset should be used in "sink mode". When set to True, the dataset is loaded and sent to NPU devices in parallel which saves loading time.
- `sink_size` (type: int, default: -1): the size of the dataset sink. If set to -1, the sink size will be equal to the full dataset size.
- `dataloader_num_workers` (type: int, default: 4): the number of worker processes to use for the data loader.
- `max_rowsize` (type: int, default: 4): the maximum size of row in MB that is used for shared memory allocation to copy data between processes.
- `batch_size` (type: int, default: 32): the local training batch size. In parallel training, the global batch size is the product of the local batch size and the number of devices.
- `dataloader_prefetch_size` (type: int, default: None) : the queue capacity of the thread in pipeline. If provided, the size must be greater than 0, otherwise the queue capacity of the thread is invalid.

## Optimizer
- `optim` (type: str, default: "adamw"): the optimizer to use for training.
- `betas` (type: list[float], default: [0.9, 0.999]): the beta1 and beta2 parameters for the AdamW optimizer.
- `optim_eps` (type: float, default: 1e-8): the epsilon parameter for the AdamW optimizer.
- `group_strategy` (type: str, default: "norm_and_bias"): the grouping strategy for weight decay. If set to "norm_and_bias", the weight decay filter list includes beta, gamma, and bias parameters. If set to None, the filter list includes layernorm and bias parameters.
- `clip_grad` (type: bool, default: False): whether to apply gradient clipping during training.
- `max_grad_norm` (type: float, default: 1.0): the maximum allowed gradient norm when gradient clipping is enabled (i.e., `clip_grad` is True).
- `use_ema` (type: bool, default: True): whether to use Exponential Moving Average (EMA) of the model parameters during training.
- `ema_decay` (type: float, default: 0.9999): the decay rate for the Exponential Moving Average (EMA) of the model parameters.

## Learning Rate
- `gradient_accumulation_steps` (type: int, default: 1): the number of gradient accumulation steps.
- `weight_decay` (type: float, default: 0.01): the weight decay factor to be used during optimization.
- `lr_scheduler` (type: str, default: "cosine_decay"): the learning rate scheduler to use during training. Supported values are "constant", "cosine_decay", "polynomial_decay", and "multi_step_decay".
- `start_learning_rate` (type: float, default: 1e-5): the starting learning rate for the optimizer.
- `end_learning_rate` (type: float, default: 1e-6): the final learning rate for the optimizer. This is the smallest learning rate during the adjustment for "cosine_decay", "polynomial_decay", and "multi_step_decay" schedulers.
- `lr_warmup_steps` (type: int, default: 0): the number of warmup steps for the learning rate scheduler.
- `lr_decay_steps` (type: int, default: 0): the number of steps over which the learning rate will be decayed. This is only applicable when `lr_scheduler` is "cosine_decay" or "polynomial_decay".
- `scale_lr` (type: bool, default: False): whether to scale the learning rate based on the batch size, gradient accumulation steps, and the number of NPUs/cards used for training.

## MindSpore Envs and Modes
- `device` (type: str, default: "Ascend"):  the device to be used for training. The options are "Ascend" or "GPU".
- `max_device_memory` (type: str, default: None): the maximum available device memory, e.g., "30GB" for Ascend 910A or "59GB" for Ascend 910B.
- `mode` (type: int, default: 0): the execution mode for the model. The options are 0 for graph mode and 1 for pynative mode.
- `use_parallel` (type: bool, default: False): whether to use parallel processing during training.
- `parallel_mode` (type: str, default: "data"): the parallel mode to be used when `use_parallel` is True. The options are "data", "optim", and "semi".
- `seed` (type: int, default: 3407): the random seed for reproducibility.
- `mempool_block_size` (type: str, default: "9GB"): the size of the memory pool block in PyNative mode or GRAPH_OP_RUN=1 for devices.
- `optimizer_weight_shard_size` (type: int, default: 8): the size of the communication domain split by the optimizer weight.

## Hyper-Parameters
- `max_train_steps` (type: int, default: None): the maximum number of training steps. If `max_train_steps` is not specified, `epochs` will be the number of epochs, and `step_mode` will be False. If `max_train_steps` is specified, it will overwrite `epochs` to `max_train_steps/num_batches`, and `step_mode` will be True. `max_train_steps` should be greater than the number of batches in an epoch, if provided.
- `epochs` (type: int, default: 10): the number of training epochs. It should be a positive integer. When `max_train_steps` is not provided, the training steps will be `epochs x num_batches`. When `max_train_steps` is provided, the training steps will be `floor(max_train_steps/num_batches) x num_batches`, and `args.epoch` will be overwritten to `floor(max_train_steps/num_batches)`.
- `step_mode` (type: bool, default: False): whether save ckpt by steps. If False, save ckpt by epochs.


## Callbacks and Logging
- `resume_from_checkpoint` (type: bool or str, default: False): if it is a bool value, it means whether to resume training from `train_resume.ckpt`. If it is a string, it is the resume checkpoint path to be loaded.
- `ckpt_save_interval` (type: int, default: 1): the interval of saving checkpoints. If `step_mode` is True, it will save checkpoints every this step number.  If `step_mode` is False, it will save checkpoints every this epoch number.
- `checkpointing_steps` (type: int, default: None): Save a checkpoint of the training state every X steps. It `checkpointing_steps` is not specified, it will use `ckpt_save_interval` as the checkpoint saving interval (epochs), and set `step_mode` to False. If `checkpointing_steps` is provided, it will overwrite `ckpt_save_interval` to the same value as `checkpointing_steps`, and set `step_mode` to True.
- `ckpt_max_keep` (type: int, default: 10): the maximum number of checkpoints to keep.
- `log_level` (type: str, default: logging.INFO): log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR.
- `log_interval` (type: int, default: 1): the log interval. If dataset sink mode is False, the loss will be printed for every X steps. If dataset sink mode is True, the loss will be printed for every `sink_size` steps.
- `profile` (type: bool, default: False): whether to run profiling for time/speed analysis. The profiling data will saved under `./profile_data`.
