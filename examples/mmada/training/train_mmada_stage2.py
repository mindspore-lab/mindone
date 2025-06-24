import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import glob
import json
import logging
import math
import pickle as pkl
import random
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from models import MAGVITv2, MMadaModelLM, get_mask_schedule
from models.lr_schedulers import get_scheduler
from omegaconf import OmegaConf
from parquet import RefinedWebDataset
from parquet.loader import CombinedLoader, create_dataloader
from PIL import Image
from training.data import Text2ImageDataset
from training.imagenet_dataset import ImageNetDataset
from training.prompting_utils import UniversalPrompting
from transformers import AutoTokenizer

import mindspore as ms
import mindspore.mint as mint
from mindspore.amp import DynamicLossScaler
from mindspore.experimental import optim
from mindspore.mint.distributed import get_rank, get_world_size, init_process_group

from mindone.diffusers.models.model_loading_utils import load_checkpoint_and_dispatch
from utils import TrainOneStepWrapper, init_from_ckpt, no_grad

SYSTEM_PROMPT_LEN = 28

from training.utils import AverageMeter, get_config, image_transform, mask_or_random_replace_tokens

logger = logging.getLogger(__name__)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def main():
    config = get_config()
    ms.set_context(device_target="Ascend")

    if config.experiment.get("distributed", False):
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
        init_process_group()
        rank_id = get_rank()
        device_num = get_world_size()
    else:
        rank_id = 0
        device_num = 1
    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    if rank_id == 0:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        os.makedirs(config.experiment.logging_dir, exist_ok=True)

    LOG_FILE = os.path.join(config.experiment.logging_dir, "loss.log")
    total_batch_size_per_device = (
        config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu
    )
    total_batch_size = (
        config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu
    ) * config.training.gradient_accumulation_steps

    if config.experiment.profile:
        os.environ["MS_ALLOC_CONF"] = "memory_tracker:True"
        profiler = ms.Profiler(output_path="./mem_info", profile_memory=True)
        ms.set_context(memory_optimize_level="O0")
        ms.set_context(pynative_synchronize=True)
    else:
        profiler = None

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if rank_id == 0:
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        random.seed(config.training.seed)
        np.random.seed(config.training.seed)
        ms.set_seed(config.training.seed)
    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.tokenizer_path, padding_side="left")

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    print("special tokens : \n", uni_prompting.sptids_dict)

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name, use_safetensors=True)
    vq_model.set_train(False)
    for p in vq_model.get_parameters():
        p.requires_grad = False

    # Initialize mmada in stage 2
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, mindspore_dtype=ms.bfloat16)
    model.set_train(True)
    mask_id = model.config.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight", "embeddings.embedding_table"]
    trainable_params = model.trainable_params()
    optimizer_grouped_parameters = [
        {
            "params": [p for p in trainable_params if not any(nd in p.name for nd in no_decay)],
            "weight_decay": config.optimizer.params.weight_decay,
        },
        {
            "params": [p for p in trainable_params if any(nd in p.name for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # filter empty params
    optimizer_grouped_parameters = [d for d in optimizer_grouped_parameters if len(d["params"])]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_t2i_without_accum = config.training.batch_size_t2i
    total_batch_size_t2i = config.training.batch_size_t2i * config.training.gradient_accumulation_steps

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # Data for generation
    if config.dataset.gen_type == "t2i":
        dataset = Text2ImageDataset(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            tokenizer=None,  # we want to get raw texts
            max_seq_length=preproc_config.max_seq_length,
            num_train_examples=config.experiment.max_train_examples_t2i,
            per_device_batch_size=config.training.batch_size_t2i,
            global_batch_size=total_batch_size_t2i_without_accum,
            num_workers=dataset_config.num_workers,
            resolution=preproc_config.resolution,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
            external_caption_path=dataset_config.external_caption_path,
            external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
            external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
            external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
        )
        train_dataloader_t2i = dataset.train_dataloader
        train_dataloader_t2i.dataset_size = train_dataloader_t2i.num_batches
        num_update_steps_per_epoch = math.ceil(
            train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
    elif config.dataset.gen_type == "t2i_parquet":
        # this part relies on the internal packages, which will not be released
        raise NotImplementedError()
    elif config.dataset.gen_type == "imagenet1k":
        dataset_imagenet = ImageNetDataset(
            dataset_config.train_t2i_shards_path_or_url,
            image_size=preproc_config.resolution,
        )
        sampler = None
        shuffle = True

        train_dataloader_t2i = create_dataloader(
            dataset_imagenet,
            column_names=["images", "input_ids", "class_ids"],
            batch_size=config.training.batch_size_t2i,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=dataset_config.num_workers,
            rank_id=rank_id,
            device_num=device_num,
        )
        train_dataloader_t2i = train_dataloader_t2i.create_dict_iterator(num_epochs=1, output_numpy=True)
        train_dataloader_t2i.dataset_size = len(dataset_imagenet) // config.training.batch_size_t2i

        num_update_steps_per_epoch = math.ceil(len(dataset_imagenet) / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    else:
        raise ValueError(f"Unsupported dataset type {config.dataset.type}")

    total_batch_size_mmu_without_accum = config.training.batch_size_mmu
    # Data for image captioning
    if config.dataset.und_type == "captioning":
        dataset_mmu = Text2ImageDataset(
            train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
            tokenizer=None,  # we want to get raw texts
            max_seq_length=preproc_config.max_seq_length,
            num_train_examples=config.experiment.max_train_examples_mmu,
            per_device_batch_size=config.training.batch_size_mmu,
            global_batch_size=total_batch_size_mmu_without_accum,
            num_workers=dataset_config.num_workers,
            resolution=preproc_config.resolution,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
            external_caption_path=dataset_config.external_caption_path,
            external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
            external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
            external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
            is_captioning=True,
            add_caption_prompt=dataset_config.add_caption_prompt,
        )
        train_dataloader_mmu = dataset_mmu.train_dataloader
        train_dataloader_mmu.dataset_size = train_dataloader_mmu.num_batches
    else:
        raise NotImplementedError(f"Unsupported dataset type {config.dataset.und_type}")

    # LLM pure text dataset: RefinedWeb
    dataset_lm = RefinedWebDataset(
        data_path=dataset_config.train_lm_shards_path_or_url,
        rank=rank_id,
        world_size=device_num,
        num_workers=dataset_config.num_workers,
    )

    train_dataloader_lm = create_dataloader(
        dataset_lm,
        column_names=["input_ids"],
        batch_size=config.training.batch_size_lm,
        sampler=None,
        num_workers=dataset_config.num_workers,
    )
    train_dataloader_lm = train_dataloader_lm.create_dict_iterator(num_epochs=1, output_numpy=True)
    train_dataloader_lm.dataset_size = len(dataset_lm) // config.training.batch_size_lm

    # Combine these dataloaders into a single iterable model
    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "lm_flow": train_dataloader_lm,
        "mmu_flow": train_dataloader_mmu,
    }

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        assert config.experiment.resume_from_checkpoint == "latest"
        dirs = os.listdir(config.experiment.output_dir)
        logger.info(f"dirs: {dirs}")
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        logger.info(f"path: {path}")
        # if mindspore checkpoint file exists
        if len(glob.glob(os.path.join(config.experiment.output_dir, path, "unwrapped_model", "*.ckpt"))):
            ckpt_path = sorted(glob.glob(os.path.join(path, "unwrapped_model", "*.ckpt")))[-1]
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            global_step = int(os.path.basename(ckpt_path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            init_from_ckpt(model, path)

        # if safetensors sharded checkpoint exists
        elif os.path.exists(
            os.path.join(config.experiment.output_dir, f"{path}/unwrapped_model/model.safetensors.index.json")
        ):
            index_file = os.path.join(
                config.experiment.output_dir, f"{path}/unwrapped_model/model.safetensors.index.json"
            )
            load_checkpoint_and_dispatch(model, index_file, dtype=ms.float32, strict=True)
        else:
            raise FileNotFoundError(f"Checkpoint {path}/unwrapped_model/pytorch_model.bin not found")
    else:
        logger.info("Not resuming from checkpoint")

    logger.info("Preparing model, optimizer and dataloaders")

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_device}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    def prepare_inputs_and_labels(
        pixel_values_or_image_ids: Union[ms.Tensor, ms.Tensor],
        texts: Union[str, str],
        min_masking_rate: float = 0.0,
        is_train: bool = True,
    ):
        if not isinstance(pixel_values_or_image_ids, ms.Tensor):
            pixel_values_or_image_ids = ms.Tensor(pixel_values_or_image_ids)
        if not isinstance(texts, (list, tuple)):
            texts = [str(t) for t in texts]

        image_tokens = vq_model.get_code(pixel_values_or_image_ids)
        image_tokens = image_tokens + len(uni_prompting.text_tokenizer)
        # create MLM mask and labels
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            image_tokens,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
        )
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), "t2i")
        return input_ids, labels, mask_prob, image_tokens, masks

    def prepare_inputs_and_labels_for_text(texts_lm: Union[str, str], max_seq_len, eps=1e-3):
        # create MLM mask and labels
        if not isinstance(texts_lm, (list, tuple)):
            texts_lm = [str(t) for t in texts_lm]

        input_ids_lm, prompt_mask, labels_lm = uni_prompting((texts_lm, max_seq_len), "lm")
        b, length = input_ids_lm.shape
        t = mint.rand(
            b,
        )
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, length)

        masked_indices = (
            mint.rand(
                (b, length),
            )
            < p_mask
        )
        # 126336 is used for [MASK] token
        noisy_batch = mint.where(masked_indices, mask_id, input_ids_lm)
        masked_indices = noisy_batch == mask_id

        return noisy_batch, labels_lm, p_mask

    def prepare_inputs_and_labels_for_mmu(input_ids_mmu, prompt_masks, labels_mmu, eps=1e-3):
        if not isinstance(input_ids_mmu, ms.Tensor):
            input_ids_mmu = ms.Tensor(input_ids_mmu)
        b, length = input_ids_mmu.shape
        t = mint.rand(
            b,
        )
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, length)

        masked_indices = (
            mint.rand(
                (b, length),
            )
            < p_mask
        )
        # 126336 is used for [MASK] token
        noisy_batch = mint.where(masked_indices, mask_id, input_ids_mmu)
        masked_indices = noisy_batch == mask_id
        noisy_batch[prompt_masks.bool()] = input_ids_mmu[prompt_masks.bool()]
        masked_indices = noisy_batch == mask_id

        prompt_masks = prompt_masks.to(ms.int64)
        answer_lengths = mint.sum((1 - prompt_masks), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

        return noisy_batch, labels_mmu, p_mask, answer_lengths

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    loss_scaler = DynamicLossScaler(scale_value=65356, scale_factor=2, scale_window=2000)
    train_step_model = TrainOneStepWrapper(
        model,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        drop_overflow_update=True,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        clip_grad=config.training.max_grad_norm is not None,
        clip_norm=config.training.max_grad_norm,
        ema=None,
        config=config,
    )
    if profiler is not None:
        profiler.start()
        logger.info("Memroy profiling starts!")
    for epoch in range(first_epoch, num_train_epochs):
        model.set_train(True)
        for batch in combined_dataloader:
            # for loss calculation
            batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
            batch_size_lm = len(batch["lm_flow"]["input_ids"])
            batch_size_mmu = batch["mmu_flow"]["images"].shape[0]

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for class-conditional/text-to-image generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]

            data_time_m.update(time.time() - end)
            with no_grad():
                # Encode images to image tokens, mask them and create input and labels
                (input_ids, labels, mask_prob, image_tokens_ori, t2i_masks) = prepare_inputs_and_labels(
                    pixel_values, texts, config.training.min_masking_rate
                )

                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                # Build formatted sequences for language modeling
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                max_seq_len = input_ids.shape[-1]
                texts_lm = batch["lm_flow"]["input_ids"]
                (input_ids_lm, labels_lm, p_mask_lm) = prepare_inputs_and_labels_for_text(texts_lm, max_seq_len)
                input_ids = mint.cat((input_ids.to(ms.int32), input_ids_lm.to(ms.int32)), dim=0)
                labels = mint.cat((labels.to(ms.int32), labels_lm.to(ms.int32)), dim=0)

                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                # Build formatted sequences for captioning/multimodal understanding
                # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
                if "llava" in config.dataset.und_type:
                    pixel_values_mmu, input_ids_mmu, labels_mmu = (
                        batch["mmu_flow"]["images"],
                        batch["mmu_flow"]["input_ids"],
                        batch["mmu_flow"]["labels"],
                    )
                    if not isinstance(pixel_values_mmu, ms.Tensor):
                        pixel_values_mmu = ms.Tensor(pixel_values_mmu)

                    if not isinstance(input_ids_mmu, ms.Tensor):
                        input_ids_mmu = ms.Tensor(input_ids_mmu)

                    image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                    image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)

                    input_ids_mmu = mint.cat(
                        [
                            (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.sptids_dict["<|mmu|>"]).to(
                                ms.int32
                            ),
                            (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.sptids_dict["<|soi|>"]).to(
                                ms.int32
                            ),
                            image_tokens_mmu.to(ms.int32),
                            (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.sptids_dict["<|eoi|>"]).to(
                                ms.int32
                            ),
                            input_ids_mmu.to(ms.int32),
                        ],
                        dim=1,
                    )

                    labels_mmu = mint.cat(
                        [
                            (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.ignore_id).to(ms.int32),
                            (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.ignore_id).to(ms.int32),
                            (mint.ones_like(image_tokens_mmu) * uni_prompting.ignore_id).to(ms.int32),
                            (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.ignore_id).to(ms.int32),
                            labels_mmu.to(ms.int32),
                        ],
                        dim=1,
                    )

                else:
                    pixel_values_mmu, texts_mmu = batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"]
                    if not isinstance(pixel_values_mmu, ms.Tensor):
                        pixel_values_mmu = ms.Tensor(pixel_values_mmu)

                    if not isinstance(texts_mmu, (list, tuple)):
                        texts_mmu = [str(t) for t in texts_mmu]
                    image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                    image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)

                    input_ids_mmu, prompt_masks, labels_mmu = uni_prompting((image_tokens_mmu, texts_mmu), "mmu")
                    (input_ids_mmu, labels_mmu, p_mask_mmu, answer_lengths) = prepare_inputs_and_labels_for_mmu(
                        input_ids_mmu, prompt_masks, labels_mmu
                    )

                input_ids = mint.cat((input_ids.to(ms.int32), input_ids_mmu.to(ms.int32)), dim=0)
                labels = mint.cat((labels.to(ms.int32), labels_mmu.to(ms.int32)), dim=0)

            # compute_loss and logits

            loss, logits, loss_t2i, loss_lm, loss_mmu = train_step_model.train_one_step(
                input_ids,
                labels,
                batch_size_t2i,
                batch_size_lm,
                batch_size_mmu,
                p_mask_lm,
                p_mask_mmu,
                answer_lengths,
                t2i_masks,
            )

            lr_scheduler.step()
            avg_loss_t2i = loss_t2i.mean()
            avg_loss_lm = loss_lm.mean()
            avg_loss_mmu = loss_mmu.mean()
            # avg_masking_rate = mask_prob.mean()

            # # log gradient norm before zeroing it
            # if (global_step + 1) % config.experiment.log_grad_norm_every == 0:
            #     log_grad_norm(model, global_step + 1)

            batch_time_m.update(time.time() - end)
            end = time.time()

            # Log metrics
            if (global_step + 1) % config.experiment.log_every == 0 and rank_id == 0:
                samples_per_second_per_device = (
                    config.training.gradient_accumulation_steps * total_batch_size_per_device / batch_time_m.val
                )

                logger.info(
                    f"Step: {global_step + 1} "
                    f"Loss_t2i: {avg_loss_t2i.asnumpy().item():0.4f} "
                    f"Loss_mmu: {avg_loss_mmu.asnumpy().item():0.4f} "
                    f"Loss_lm: {avg_loss_lm.asnumpy().item():0.4f} "
                    f"Loss_combined: {loss.asnumpy().item():0.4f} "
                    f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_device:0.2f}/s/device "
                    f"Batch (t): {batch_time_m.val:0.4f} "
                    f"LR: {lr_scheduler.get_last_lr()[0].asnumpy().item():0.6f}",
                    f"Loss scaler {loss_scaler.scale_value.value().asnumpy().item()}",
                )

                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
            if rank_id == 0:
                try:
                    if not os.path.exists(LOG_FILE):
                        with open(LOG_FILE, "w", encoding="utf-8") as fp:
                            fp.write("\t".join(["step", "loss", "per step time (s)"]) + "\n")
                    with open(LOG_FILE, "a", encoding="utf-8") as fp:
                        fp.write(
                            "\t".join(
                                [
                                    f"{global_step + 1:<7}",
                                    f"{loss.asnumpy().item():<10.6f}",
                                    f"{batch_time_m.val:<13.3f}",
                                ]
                            )
                            + "\n"
                        )
                except (IOError, PermissionError) as e:
                    logger.error(f"Failed to write log: {e}")
            # Save model checkpoint
            if (global_step + 1) % config.experiment.save_every == 0 and rank_id == 0:
                save_checkpoint(model, config, global_step + 1, uni_prompting)

            if (
                ((global_step + 1) % config.experiment.generate_every == 0 or global_step == 0)
                and config.experiment.eval_during_train
                and rank_id == 0
            ):
                model.set_train(False)
                with no_grad():
                    generate_images(
                        model,
                        vq_model,
                        uni_prompting,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                        force_no_cfg=False,
                    )
                    generate_images(
                        model,
                        vq_model,
                        uni_prompting,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                        force_no_cfg=True,
                    )
                    visualize_predictions(
                        model,
                        vq_model,
                        uni_prompting,
                        config,
                        global_step + 1,
                        input_ids,
                        image_tokens_ori,
                        ms.Tensor(batch["t2i_flow"]["images"]),
                        texts,
                        logits,
                    )

                    understanding_images(
                        model,
                        vq_model,
                        uni_prompting,
                        config,
                        global_step + 1,
                    )
                model.set_train(True)

            global_step += 1
            if profiler is not None and global_step == 2:
                # save first two steps
                profiler.stop()
                profiler.analyse()
                logger.info("Memroy profiling and analysis is finished! Check ./mem_info!")
            if global_step >= config.training.max_train_steps:
                break

    # Evaluate and save checkpoint at the end of training
    if rank_id == 0:
        save_checkpoint(model, config, global_step)
        # Save the final trained checkpoint
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)


def visualize_predictions(
    model, vq_model, uni_prompting, config, global_step, input_ids, image_tokens_ori, ori_images, texts, logits
):
    logger.info("Visualizing predictions...")

    recons_images = vq_model.decode_code(image_tokens_ori - len(uni_prompting.text_tokenizer))
    recons_images = mint.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
    recons_images *= 255.0
    recons_images = recons_images.permute(0, 2, 3, 1).asnumpy().astype(np.uint8)

    images = mint.clamp((ori_images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).asnumpy().astype(np.uint8)
    predictions = logits[
        : config.training.batch_size_t2i,
        -(config.model.mmada.num_vq_tokens + 1) : -1 :,
        len(uni_prompting.text_tokenizer)
        + config.model.mmada.num_new_special_tokens : len(uni_prompting.text_tokenizer)
        + config.model.mmada.num_new_special_tokens
        + config.model.mmada.codebook_size,
    ]

    predictions = predictions.argmax(axis=-1)
    mask_token_id = model.config.mask_token_id - len(uni_prompting.text_tokenizer)
    input_ids = input_ids[: config.training.batch_size_t2i, -(config.model.mmada.num_vq_tokens + 1) : -1 :] - len(
        uni_prompting.text_tokenizer
    )
    mask_ratio = list(
        (mint.where(input_ids == mask_token_id, 1, 0).sum(dim=-1) / config.model.mmada.num_vq_tokens).asnumpy()
    )
    predicted_images = mint.where(input_ids == mask_token_id, predictions, input_ids)
    predicted_images = vq_model.decode_code(predicted_images)
    predicted_images = mint.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
    predicted_images *= 255.0
    predicted_images = predicted_images.permute(0, 2, 3, 1).asnumpy().astype(np.uint8)
    predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
    pil_images = [Image.fromarray(image) for image in predicted_images]

    # save to directory
    output_dir = os.path.join(config.experiment.logging_dir, f"visualization/{global_step}")
    os.makedirs(output_dir, exist_ok=True)
    index = 0
    for image, mr in zip(pil_images, mask_ratio):
        fn = f"{index}-mask_ratio{mr:3f}.png"
        fp = os.path.join(output_dir, fn)
        image.save(fp)
    logger.info(f"Images, reconstructed images, and predicted images saved state to {output_dir}")
    return


def generate_images(model, vq_model, uni_prompting, config, global_step, mask_schedule, force_no_cfg=False):
    logger.info("Generating images...")

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    mask_token_id = model.config.mask_token_id
    image_tokens = (
        mint.ones((len(validation_prompts), config.model.mmada.num_vq_tokens), dtype=ms.int64) * mask_token_id
    )
    input_ids, attention_mask = uni_prompting((validation_prompts, image_tokens), "t2i_gen")
    if not force_no_cfg and config.training.guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(
            ([""] * len(validation_prompts), image_tokens), "t2i_gen"
        )
        cfg_scale = config.training.guidance_scale
    else:
        uncond_input_ids = None
        uncond_attention_mask = None
        cfg_scale = 0

        # Generate images
    gen_token_ids = model.t2i_generate(
        input_ids=input_ids,
        uncond_input_ids=uncond_input_ids,
        attention_mask=attention_mask,
        uncond_attention_mask=uncond_attention_mask,
        guidance_scale=cfg_scale,
        temperature=config.training.get("generation_temperature", 1.0),
        timesteps=config.training.generation_timesteps,
        noise_schedule=mask_schedule,
        noise_type=config.training.get("noise_type", "mask"),
        predict_all_tokens=config.training.get("predict_all_tokens", False),
        seq_len=config.model.mmada.num_vq_tokens,
        uni_prompting=uni_prompting,
        config=config,
    )
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = mint.clamp(gen_token_ids, max=model.config.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)

    if config.training.get("pre_encode", False):
        del vq_model

    # Convert to PIL images
    images = mint.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).asnumpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # save to directory
    output_dir = os.path.join(config.experiment.logging_dir, f"generated_images_cfg={cfg_scale}/{global_step}")
    os.makedirs(output_dir, exist_ok=True)
    for image, prompt in zip(pil_images, validation_prompts):
        fn = prompt.strip()[:50] + ".png"
        fp = os.path.join(output_dir, fn)
        image.save(fp)
    logger.info(f"Generated images saved state to {output_dir}")

    return


def understanding_images(
    model,
    vq_model,
    uni_prompting,
    config,
    global_step,
):
    logger.info("Understanding images...")

    file_list = os.listdir(config.dataset.params.mmu_image_root)
    file_list = [f for f in file_list if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    responses = ["" for i in range(len(file_list))]
    images = []

    for i, file_name in enumerate(file_list):
        image_path = os.path.join(config.dataset.params.mmu_image_root, file_name)
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution)
        image = ms.Tensor(image).unsqueeze(0)
        images.append(image)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)

        input_ids = uni_prompting.text_tokenizer(
            [
                "<|start_header_id|>user<|end_header_id|>\n"
                + "Please describe this image in detail."
                + "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
            ]
        )["input_ids"]
        input_ids = ms.tensor(input_ids)

        input_ids = mint.cat(
            [
                (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|mmu|>"]).to(ms.int32),
                (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|soi|>"]).to(ms.int32),
                image_tokens.to(ms.int32),
                (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|eoi|>"]).to(ms.int32),
                (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|sot|>"]).to(ms.int32),
                input_ids.to(ms.int32),
            ],
            dim=1,
        )

        output_ids = model.mmu_generate(input_ids)
        # output_ids = mint.stack(output_ids).squeeze()[None]

        text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :], skip_special_tokens=True)
        responses[i] += text[0]

    # images = mint.cat(images, dim=0)
    # images = mint.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # images *= 255.0
    # images = images.permute(0, 2, 3, 1).asnumpy().astype(np.uint8)
    # pil_images = [Image.fromarray(image) for image in images]

    # save to directory
    output_dir = os.path.join(config.experiment.logging_dir, f"understanding_images/{global_step}")
    os.makedirs(output_dir, exist_ok=True)
    for fp, responses in zip(file_list, responses):
        base_name = os.path.basename(fp)
        fp = os.path.join(output_dir, base_name.split(".")[0] + ".txt")
        with open(fp, "w") as f:
            f.writelines([responses])

    logger.info(f"Image understanding results saved state to {output_dir}")
    return


def save_checkpoint(model, config, global_step, uni_prompting):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = model.state_dict()
    model.save_pretrained(save_path / "unwrapped_model", state_dict=state_dict, safe_serialization=True)
    json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
    logger.info(f"Saved state to {save_path}")

    # save tokenizer
    uni_prompting.text_tokenizer.save_pretrained(save_path / "unwrapped_model")


def log_grad_norm(model, config, global_step):
    output_dir = os.path.join(config.experiment.logging_dir, f"grad_norm/{global_step}")
    os.makedirs(output_dir, exist_ok=True)

    save_gradnorm_dict = {}
    for name, param in model.name_cells().items():
        if param.grad is not None:
            grads = param.grad.data
            grad_norm = (grads.norm(p=2) / grads.numel()).asnumpy().item()
            save_gradnorm_dict[name] = grad_norm
    fp = os.path.join(output_dir, "gradients_norm_dict.pkl")
    with open(fp, "wb") as f:
        pkl.dump(save_gradnorm_dict, f)
    logger.info(f"Gradients norms at global step {global_step} saved state to {fp}")


if __name__ == "__main__":
    main()
