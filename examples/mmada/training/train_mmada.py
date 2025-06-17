import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from models import MAGVITv2, MMadaConfig, MMadaModelLM, get_mask_schedule
from models.lr_schedulers import get_scheduler
from omegaconf import OmegaConf
from parquet import RefinedWebDataset
from parquet.loader import CombinedLoader, create_dataloader
from PIL import Image
from training.data import Text2ImageDataset
from training.imagenet_dataset import ImageNetDataset
from training.prompting_utils import UniversalPrompting
from transformers import AutoConfig, AutoTokenizer

import mindspore as ms
import mindspore.mint as mint

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.trainers.utils import create_optimizer

SYSTEM_PROMPT_LEN = 28

from training.utils import AverageMeter, get_config, image_transform, mask_or_random_replace_tokens

logger = logging.getLogger(__name__)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def main():
    config = get_config()

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")

    total_batch_size_per_device = (
        config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu
    )
    total_batch_size = (
        config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu
    ) * config.training.gradient_accumulation_steps

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    os.makedirs(config.experiment.output_dir, exist_ok=True)
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

    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")

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
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.eval()
    vq_model.requires_grad = False

    # Initialize mmada in pretraining stage
    base_config = AutoConfig.from_pretrained(config.model.mmada.pretrained_model_path).to_dict()
    mmada_config_dict = {k: v for k, v in config.model.mmada.items()}
    merged_config = {**base_config, **mmada_config_dict}
    mmada_config = MMadaConfig(**merged_config)
    model = MMadaModelLM.from_pretrained(
        config.model.mmada.pretrained_model_path, mindspore_dtype=ms.bfloat16, config=mmada_config
    )
    model.resize_token_embeddings(mmada_config.new_vocab_size)
    model.config.embedding_size = model.config.vocab_size
    mask_id = model.config.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.name_cells().items() if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.name_cells().items() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = create_optimizer(
            optimizer_grouped_parameters,
            name="adamw",
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
        num_update_steps_per_epoch = math.ceil(
            train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    elif config.dataset.gen_type == "t2i_parquet":
        # this part relies on the internal packages, which will not be released
        num_update_steps_per_epoch = math.ceil(config.experiment.max_train_examples_t2i / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

        train_dataloader_t2i = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            batch_size=config.training.batch_size_t2i,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        )

    elif config.dataset.gen_type == "imagenet1k":
        dataset_imagenet = ImageNetDataset(
            dataset_config.train_t2i_shards_path_or_url,
            image_size=preproc_config.resolution,
        )
        sampler = None
        shuffle = True

        train_dataloader_t2i = create_dataloader(
            dataset_imagenet,
            column_names=["image", "input_ids", "class_ids"],
            batch_size=config.training.batch_size_t2i,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=dataset_config.num_workers,
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

    elif config.dataset.und_type == "captioning_parquet":
        train_dataloader_mmu = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
            batch_size=config.training.batch_size_mmu,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            is_captioning=True,
        )

    else:
        raise NotImplementedError(f"Unsupported dataset type {config.dataset.und_type}")

    # LLM pure text dataset: RefinedWeb
    dataset_lm = RefinedWebDataset(
        data_path=dataset_config.train_lm_shards_path_or_url,
        rank=0,
        world_size=1,
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
        dirs = os.listdir(config.experiment.output_dir)
        logger.info(f"dirs: {dirs}")
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        logger.info(f"path: {path}")
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)
            logger.info(f"Resuming from checkpoint: {path}")
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            if os.path.exists(f"{path}/unwrapped_model/pytorch_model.bin"):
                state_dict = torch.load(f"{path}/unwrapped_model/pytorch_model.bin", map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
                del state_dict
            elif os.path.exists(f"{path}/unwrapped_model/pytorch_model.bin.index.json"):
                from safetensors.torch import load_file
                from transformers.modeling_utils import load_sharded_checkpoint

                load_sharded_checkpoint(model, f"{path}/unwrapped_model/")
            # if safetensors sharded checkpoint exists
            elif os.path.exists(f"{path}/unwrapped_model/model.safetensors.index.json"):
                from transformers.modeling_utils import load_sharded_checkpoint

                load_sharded_checkpoint(
                    model,
                    f"{path}/unwrapped_model/",
                    # weight_map=None,
                    # load_state_dict_fn="safetensors"
                )
            else:
                raise FileNotFoundError(f"Checkpoint {path}/unwrapped_model/pytorch_model.bin not found")
    else:
        logger.info("Not resuming from checkpoint")

    logger.info("Preparing model, optimizer and dataloaders")

    mask_dtype = model.get_input_embeddings().weight.dtype

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

    def prepare_inputs_and_labels_for_text(texts: Union[str, str], max_seq_len, eps=1e-3):
        # create MLM mask and labels

        input_ids_lm, prompt_mask, labels_lm = uni_prompting((texts_lm, max_seq_len), "lm")
        b, l = input_ids_lm.shape
        t = mint.rand(
            b,
        )
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = (
            mint.rand(
                (b, l),
            )
            < p_mask
        )
        # 126336 is used for [MASK] token
        noisy_batch = mint.where(masked_indices, mask_id, input_ids_lm)
        masked_indices = noisy_batch == mask_id

        return noisy_batch, labels_lm, p_mask

    def prepare_inputs_and_labels_for_mmu(input_ids_mmu, prompt_masks, labels_mmu, eps=1e-3):
        b, l = input_ids_mmu.shape
        t = mint.rand(
            b,
        )
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = (
            mint.rand(
                (b, l),
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

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch, batch_idx, dataloader_idx in combined_dataloader:
            # for loss calculation
            batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
            batch_size_lm = len(batch["lm_flow"]["input_ids"])
            batch_size_mmu = batch["mmu_flow"]["images"].shape[0]

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for class-conditional/text-to-image generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]

            data_time_m.update(time.time() - end)

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
            input_ids = mint.cat((input_ids, input_ids_lm), dim=0)
            labels = mint.cat((labels, labels_lm), dim=0)

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for captioning/multimodal understanding
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            if "llava" in config.dataset.und_type:
                pixel_values_mmu, input_ids_mmu, labels_mmu = (
                    batch["mmu_flow"]["images"],
                    batch["mmu_flow"]["input_ids"],
                    batch["mmu_flow"]["labels"],
                )

                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)

                input_ids_mmu = mint.cat(
                    [
                        (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.sptids_dict["<|mmu|>"]),
                        (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.sptids_dict["<|soi|>"]),
                        image_tokens_mmu,
                        (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.sptids_dict["<|eoi|>"]),
                        input_ids_mmu,
                    ],
                    dim=1,
                ).to(ms.int32)

                labels_mmu = mint.cat(
                    [
                        (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.ignore_id),
                        (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.ignore_id),
                        mint.ones_like(image_tokens_mmu) * uni_prompting.ignore_id,
                        (mint.ones((input_ids_mmu.shape[0], 1)) * uni_prompting.ignore_id),
                        labels_mmu,
                    ],
                    dim=1,
                ).to(ms.int32)

            else:
                pixel_values_mmu, texts_mmu = batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"]

                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)

                input_ids_mmu, prompt_masks, labels_mmu = uni_prompting((image_tokens_mmu, texts_mmu), "mmu")
                (input_ids_mmu, labels_mmu, p_mask_mmu, answer_lengths) = prepare_inputs_and_labels_for_mmu(
                    input_ids_mmu, prompt_masks, labels_mmu
                )

            input_ids = mint.cat((input_ids, input_ids_mmu), dim=0)
            labels = mint.cat((labels, labels_mmu), dim=0)

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            logits, loss_t2i, loss_lm, loss_mmu = model.forward_process(
                input_ids=input_ids,
                labels=labels,
                batch_size_t2i=batch_size_t2i,
                batch_size_lm=batch_size_lm,
                batch_size_mmu=batch_size_mmu,
                max_seq_length=config.dataset.preprocessing.max_seq_length,
                p_mask_lm=p_mask_lm,
                p_mask_mmu=p_mask_mmu,
                answer_lengths=answer_lengths,
                t2i_masks=t2i_masks,
            )

            avg_loss_t2i = loss_t2i.mean()
            avg_loss_lm = loss_lm.mean()
            avg_loss_mmu = loss_mmu.mean()
            loss = (
                config.training.t2i_coeff * loss_t2i
                + config.training.lm_coeff * loss_lm
                + config.training.mmu_coeff * loss_mmu
            )

            avg_masking_rate = mask_prob.mean()

            loss.backward()

            if config.training.max_grad_norm is not None:
                mint.nn.utils.clip_grad_norm_(model.get_parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # log gradient norm before zeroing it
            if (global_step + 1) % config.experiment.log_grad_norm_every == 0:
                log_grad_norm(model, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            batch_time_m.update(time.time() - end)
            end = time.time()

            # Log metrics
            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_device = (
                    config.training.gradient_accumulation_steps * total_batch_size_per_device / batch_time_m.val
                )
                logs = {
                    "step_loss_t2i": avg_loss_t2i.item(),
                    "step_loss_mmu": avg_loss_mmu.item(),
                    "step_loss_lm": avg_loss_lm.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "avg_masking_rate": avg_masking_rate.item(),
                    "samples/sec/gpu": samples_per_second_per_device,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                }

                logger.info(
                    f"Step: {global_step + 1} "
                    f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                    f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
                    f"Loss_lm: {avg_loss_lm.item():0.4f} "
                    f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_device:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_m.val:0.4f} "
                    f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                )

                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()

            # Save model checkpoint
            if (global_step + 1) % config.experiment.save_every == 0:
                save_checkpoint(model, config, global_step + 1)

            if (global_step + 1) % config.experiment.generate_every == 0 or global_step == 0:
                generate_images(
                    model,
                    vq_model,
                    uni_prompting,
                    config,
                    global_step + 1,
                    mask_schedule=mask_schedule,
                )

                visualize_predictions(
                    model,
                    vq_model,
                    uni_prompting,
                    config,
                    global_step + 1,
                    input_ids,
                    image_tokens_ori,
                    batch["t2i_flow"]["images"],
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

            global_step += 1

            if global_step >= config.training.max_train_steps:
                break

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, global_step)

    # Save the final trained checkpoint
    model.save_pretrained(config.experiment.output_dir, safe_serialization=True)


def visualize_predictions(
    model, vq_model, uni_prompting, config, global_step, input_ids, image_tokens_ori, ori_images, texts, logits
):
    logger.info("Visualizing predictions...")
    model.eval()

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

    model.train()


def generate_images(
    model,
    vq_model,
    uni_prompting,
    config,
    global_step,
    mask_schedule,
):
    logger.info("Generating images...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    mask_dtype = model.get_input_embeddings().weight.dtype
    mask_token_id = model.config.mask_token_id
    image_tokens = (
        mint.ones((len(validation_prompts), config.model.mmada.num_vq_tokens), dtype=ms.int64) * mask_token_id
    )
    input_ids, attention_mask = uni_prompting((validation_prompts, image_tokens), "t2i_gen")
    if config.training.guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(
            ([""] * len(validation_prompts), image_tokens), "t2i_gen"
        )
    else:
        uncond_input_ids = None
        uncond_attention_mask = None

        # Generate images
    gen_token_ids = model.t2i_generate(
        input_ids=input_ids,
        uncond_input_ids=uncond_input_ids,
        attention_mask=attention_mask,
        uncond_attention_mask=uncond_attention_mask,
        guidance_scale=config.training.guidance_scale,
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

    model.train()

    if config.training.get("pre_encode", False):
        del vq_model

    # Convert to PIL images
    images = mint.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).asnumpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]


def understanding_images(
    model,
    vq_model,
    uni_prompting,
    config,
    global_step,
):
    logger.info("Understanding images...")
    model.eval()

    file_list = os.listdir(config.dataset.params.mmu_image_root)
    file_list = [f for f in file_list if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    responses = ["" for i in range(len(file_list))]
    images = []

    for i, file_name in enumerate(file_list):
        image_path = os.path.join(config.dataset.params.mmu_image_root, file_name)
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution)
        image = image.unsqueeze(0)
        images.append(image)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        batch_size = 1

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
                (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|mmu|>"]),
                (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|soi|>"]),
                image_tokens,
                (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|eoi|>"]),
                (mint.ones((input_ids.shape[0], 1)) * uni_prompting.sptids_dict["<|sot|>"]),
                input_ids,
            ],
            dim=1,
        ).to(ms.int32)

        output_ids = model.mmu_generate(input_ids)
        # output_ids = mint.stack(output_ids).squeeze()[None]

        text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :], skip_special_tokens=True)
        responses[i] += text[0]
    model.train()
    images = mint.cat(images, dim=0)
    images = mint.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).asnumpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]


def save_checkpoint(model, config, global_step):
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


def log_grad_norm(model, global_step):
    for name, param in model.name_cells().items():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()


if __name__ == "__main__":
    main()
