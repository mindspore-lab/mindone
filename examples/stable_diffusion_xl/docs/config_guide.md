# Training config guide

## 1. Script parameters

### common

- `--config`: configure file, e.g. configs/training/sd_xl_base_finetune_910b.yaml
- `--weight`: pre-train weight, e.g. checkpoints/sd_xl_base_1.0_ms.ckpt
- `--data_path`: dataset path
- `--per_batch_size`: training batch size of per device card
- `--gradient_accumulation_steps`: gradient accumulation steps
- `--clip_grad`: whether apply gradient clipping
- `--max_grad_norm`: max gradient norm for clipping, effective when `clip_grad` is enabled.
- `--use_ema`: whether use ema
- `--scale_lr`: whether scale lr with global batch size
- `--max_device_memory`: set the maximum memory usage for model training, recommended to `59GB` on 910*

- `--is_parallel`: whether to train the model in parallel, must be `True` during multi-card training

- `--data_sink`: whether sink data to npu device
- `--sink_size`: sink size, effective when `data_sink` enabled.
- `--sink_queue_size`: sink queue size on npu device, reduce it to reduce npu memory usage, effective when `data_sink` enabled.

- `--weight`: resume checkpoint
- `--resume_step`: resume step
- `--optimizer_weight`: optimizer weight
- `--save_optimizer`: whether to save optimizer weight, turn on to save last step optimizer weight for resume


### mix-precision training

- `--ms_amp_level`: setting mindspore auto-mix-precision level, e.g. default is `O2` level, keep full precision operations for cells and operators in the blacklist(norm layer/silu), and convert the rest to lower precision operations. reference to [MindSpore Document](https://www.mindspore.cn/docs/en/r2.2/api_python/amp/mindspore.amp.auto_mixed_precision.html).

- `--param_fp16`: convert weight to `fp16`

  > ⚠️: It is not recommended to turn on `--param_fp16`, which will force the conversion of the weight to `fp16` and may lead to unstable training.

  > ⚠️: If you still insist on using it, you can try replacing the [vae-fp16-fix weight](./weight_convertion.md), which can bring slight help.

- `disable_first_stage_amp`: recommended to `True`, keep vae compute with fp32

  ```shell
  # setting `disable_first_stage_amp` on configs(*.yaml)
  target: gm.models.diffusion.DiffusionEngine
  params:
    disable_first_stage_amp: True
    ...
  ```


### timestep bias weighting

- `--timestep_bias_strategy`: where (earlier vs. later) in the timestep to apply a bias, which can encourage the model to either learn low or high frequency details
- `--timestep_bias_multiplier`: the weight of the bias to apply to the timestep
- `--timestep_bias_begin`: the timestep to begin applying the bias
- `--timestep_bias_end`: the timestep to end applying the bias
- `--timestep_bias_portion`: the proportion of timesteps to apply the bias to

  ```shell
  python train.py \
    ...
    --timestep_bias_multiplier 2 \
    --timestep_bias_strategy range \
    --timestep_bias_begin 200 \
    --timestep_bias_end 500 \
    --timestep_bias_portion 0.25
  ```


### snr weighting

The [Min-SNR](https://huggingface.co/papers/2303.09556) weighting strategy can help with training by rebalancing the loss to achieve faster convergence.

Add the `--snr_gamma` parameter and set it to the recommended value of 5.0:

```shell
python train.py \
  ...
  --snr_gamma 5.0
```


### cache

- `--task = cache`: cache task
- `--cache_latent`: whether to cache vae latent
- `--cache_text_embedding`: whether to cache text embedding
- `--cache_path`: save path for cache files

  ```shell
  python train.py \
    --task cache \
    --cache_latent True \
    --cache_text_embedding True \
    --cache_path ./cache_data \
    ...
  ```


## 2. Config file

```shell
# Diffusion Engine
model:
    target: gm.models.diffusion.DiffusionEngine
    params:

        # disable vae auto-mix-precision: keep vae compute with fp32
        disable_first_stage_amp: True

        # VAE
        first_stage_config:
            target: gm.models.autoencoder.AutoencoderKLInferenceWrapper
            params:
                ...

        # UNet
        network_config:
            target: gm.modules.diffusionmodules.openaimodel.UNetModel
            params:
                ...

                # depth of transformer: if debug, you can try setting it to [1, 1, 1] to
                #   reduce the size of the unet to reduce compilation time.
                transformer_depth: [1, 2, 10]

                # attention type: flash-attention or vanilla, default is `flash-attention`,
                #   that can reduce npu/gpu memory and improve speed
                spatial_transformer_attn_type: flash-attention

                # recompute: turn on to reduce npu/gpu memory
                use_recompute: True

        # Text-Encoders and vector conditioners
        conditioner_config:
            target: gm.modules.GeneralConditioner
            params:
                emb_models:
                  # text-encoder 1, clip
                  - is_trainable: False
                    input_key: txt
                    target: gm.modules.embedders.modules.FrozenCLIPEmbedder
                    params:
                      layer: hidden
                      layer_idx: 11
                      version: openai/clip-vit-large-patch14

                  # text-encoder 2, openclip
                  - ...

                  # vector conditioner 1, original_size_as_tuple
                  - is_trainable: False
                    input_key: original_size_as_tuple
                    target: gm.modules.embedders.modules.ConcatTimestepEmbedderND
                    params:
                      outdim: 256

                  # vector conditioner 2, crop_coords_top_left
                  - ...

                  # vector conditioner 3, target_size_as_tuple
                  - ...

        # Loss Function
        loss_fn_config:
            target: gm.modules.diffusionmodules.loss.StandardDiffusionLoss
            params:
                # noise offset
                offset_noise_level: 0.05

        # Denoiser
        denoiser_config: ...

        # Sigma sampler
        sigma_sampler_config: ...


# Optimizer
optim:
    # learning rate
    base_learning_rate: 1.0e-6

    # adamw optimizer
    optimizer_config:
        target: mindspore.nn.AdamWeightDecay
        params:
            beta1: 0.9
            beta2: 0.999
            weight_decay: 0.01

    # learning rate scheduler
    scheduler_config:
        target: gm.lr_scheduler.LambdaWarmUpScheduler
        params:
            warm_up_steps: 500

# DataLoader
data:
    per_batch_size: 1               # batch size of per device card
    total_step: 100000              # train total steps
    num_parallel_workers: 2         # num of data preprocess worker
    python_multiprocessing: False   # whether to turn on `python_multiprocessing`
    shuffle: True                   # whether to shuffle dataset

    # Dataset
    dataset_config:
        target: gm.data.dataset.Text2ImageDataset
        params:
            target_size: 1024

            # dataset augment
            transforms:
                - target: gm.data.mappers.Resize
                  params:
                    size: 1024
                    interpolation: 3
                - target: gm.data.mappers.Rescaler
                  params:
                    isfloat: False
                - target: gm.data.mappers.AddOriginalImageSizeAsTupleAndCropToSquare
                - target: gm.data.mappers.Transpose
                  params:
                    type: hwc2chw
```


## 3. Long prompts training

By default, SDXL only supports the token sequence no longer than 77. Those sequences longer than 77 will be truncated to 77, which can cause information loss.

To avoid information loss for long text prompts, we add the feature of long prompts training. Long prompts training is supported by `args.lpw` in `train.py`.

```shell
python train.py \
  ...  \  # other arguments configurations
  --lpw True \
```

## 4. EDM training

> [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/pdf/2206.00364.pdf)

By default, SDXL uses DDPM for training. It can be changed to the EDM-style training by configuring the `denoiser` and other related parameters of the training.

We have provided an EDM-style-training yaml configuration file, in which parameters `denoiser_config` its associated `weighting_config,` and `scaling_config` are modified to support EDM training. You can refer to the following case to make it effective.

```shell
python train.py \
  ...  \  # other arguments configurations
  --config configs/training/sd_xl_base_finetune_910b_edm.yaml \
```
