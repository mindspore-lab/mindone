import argparse
import importlib
import json
import math
import os
import random
import sys
import time
from multiprocessing import Value

import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
import library.train_util as train_util
from library.config_util import BlueprintGenerator, ConfigSanitizer
from library.custom_train_functions import (  # get_weighted_text_embeddings,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
)
from library.train_util import DreamBoothDataset, freeze_params, set_seed
from library.utils import add_logging_arguments, setup_logging
from tqdm import tqdm

import mindspore as ms
from mindspore import nn, ops
from mindspore.amp import StaticLossScaler
from mindspore.dataset import GeneratorDataset

from mindone.diffusers import DDPMScheduler
from mindone.diffusers.training_utils import AttrJitWrapper, TrainStep, cast_training_params


def unwrap_model(model: nn.Cell, prefix=""):
    for name, param in model.parameters_and_names(name_prefix=prefix):
        param.name = name
    return model


setup_logging()
import logging

logger = logging.getLogger(__name__)


class NetworkTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    # TODO: refactor the trainstep setting
    def set_train_step_class(self):
        if self.is_sdxl:
            self.train_step_class = TrainStepForSDXLLoRA
        else:
            NotImplementedError("SD training not yet implemented")

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        logger.warning("generate_step_logs not implement yet")
        return {}

    def assert_extra_args(self, args, train_dataset_group):
        pass

    def load_target_model(self, args, weight_dtype):
        pass

    def load_tokenizer(self, args):
        pass

    def is_text_encoder_outputs_cached(self, args):
        return False

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only and not self.is_text_encoder_outputs_cached(args)

    def cache_text_encoder_outputs_if_needed(
        self, args, unet, vae, tokenizers, text_encoders, data_loader, weight_dtype
    ):
        pass

    def get_text_cond(self, args, batch, tokenizers, text_encoders, weight_dtype):
        pass

    def call_unet(self, args, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        pass

    def sample_images(self, args, epoch, global_step, vae, tokenizer, text_encoder, unet):
        pass

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # tokenizerは単体またはリスト、tokenizersは必ずリスト：既存のコードとの互換性のため
        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

        # データセットを準備する
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                if use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            # use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) \
                    / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        self.assert_extra_args(args, train_dataset_group)

        # acceleratorを準備する
        is_main_process = train_util.is_master(args)

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = ms.float32 if args.no_half_vae else weight_dtype

        # モデルを読み込む # 1. build text_encoders/vae/unet
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype)

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # notes: add prefix to the para.name, or optimize may give a para.name duplication error.
        if len(text_encoders) > 1:
            for idx, text_encoder in enumerate(text_encoders):
                prefix = "text_encoder_" + f"{idx+1}"
                text_encoders[idx] = unwrap_model(text_encoder, prefix)

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.flash_attn)
        # if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
        #     vae.set_use_memory_efficient_attention_xformers(args.xformers)

        # notes: here we set the unet/text_encoders params dtype before lora network prepration.
        # once the lora network is created, unet/tes related module and params are replaced.

        # 差分追加学習のためにモデルを読み込む # prepare lora modules
        sys.path.append(os.path.dirname(__file__))
        logger.info(f"import network module: {args.network_module}")
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする # if base_weights of loras assigned，load and merge.
            # TODO
            NotImplementedError
            # for i, weight_path in enumerate(args.base_weights):
            #     if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
            #         multiplier = 1.0
            #     else:
            #         multiplier = args.base_weights_multiplier[i]

            #     logger.info(f"merging module: {weight_path} with multiplier {multiplier}")

            #     module, weights_sd = network_module.create_network_from_weights(
            #         multiplier, weight_path, vae, text_encoder, unet, for_inference=True
            #     )
            #     module.merge_to(
            #         text_encoder, unet, weights_sd, weight_dtype
            #     )  # , accelerator.device if args.lowram else "cpu")

            # logger.info(f"all weights merged: {', '.join(args.base_weights)}")

        # 学習を準備する
        if cache_latents:
            raise NotImplementedError

        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        self.cache_text_encoder_outputs_if_needed(
            args, unet, vae, tokenizers, text_encoders, train_dataset_group, weight_dtype
        )

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)

        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            NotImplementedError
        else:
            if "dropout" not in net_kwargs:
                # workaround for LyCORIS (;^ω^)
                net_kwargs["dropout"] = args.network_dropout

            # notes: after creation, some modules of unet or text_encs are replaced by LoRAmoudle,
            # no need to have `network.apply_to` later.
            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoders,
                unet,
                neuron_dropout=args.network_dropout,
                train_unet=train_unet,
                train_text_encoder=train_text_encoder,
                original_dtype=weight_dtype,  # make the dtypes of original modules unchanged after replacement.
                **net_kwargs,
            )
        if network is None:
            return
        network_has_multiplier = False  # FIXME:hasattr(network, "set_multiplier")

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        # notes: so we dont have the apply_to method here.
        # network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            param_not_load = network.load_weights(args.network_weights)
            if len(param_not_load) == 0:
                logger.info(f"load network weights from {args.network_weights}: all params loaded")
            else:
                logger.info(
                    f"load network weights from {args.network_weights}, {len(param_not_load)} not loaded: {param_not_load}"
                )

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            logger.warning("text encoders gradient_checkpointing not support yet.")
            # for t_enc in text_encoders:
            #     t_enc.gradient_checkpointing_enable()
            # del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # notes: set params dtype and mix preciosns here.

        # notes: just make sure agian, the trainbale params (the lora layers) are in float32 here.
        # except enabling full_fp16/full_bf, which cast the whole network(include unet/tes) params to fp16/bf16
        if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
            cast_training_params(network, dtype=ms.float32)

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            logger.info("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            logger.info("enable full bf16 training.")
            network.to(weight_dtype)

        # unet_weight_dtype = te_weight_dtype = weight_dtype
        # Experimental Feature: Put base model into fp8 to save vram
        if args.fp8_base:
            NotImplementedError

        # notes: setting amp for "fp16" or "bf16", default amp_level = "O2"
        if args.mixed_precision != "no":
            # notes: weight_dtype already set to match args.mix_precioson below (see `train_util.prepare_dtype(args)`)
            if train_unet:
                unet = ms.amp.auto_mixed_precision(unet, amp_level=args.amp_level, dtype=weight_dtype)
                logger.info(f"unet using amp_level: {args.amp_level}, mix_precision: {weight_dtype}")
            if train_text_encoder:
                for t_enc in text_encoders:
                    ms.amp.auto_mixed_precision(t_enc, amp_level=args.amp_level, dtype=weight_dtype)
                logger.info(f"text_encoders using amp_level: {args.amp_level}, mix_precision: {weight_dtype}")

        # 学習に必要なクラスを準備する
        logger.info("prepare optimizer, data loader etc.")

        # dataloaderを準備する
        # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
        n_workers = max(1, n_workers)  # notes: avoid num_parallel_workers=0 in ms
        train_dataloader = GeneratorDataset(
            train_dataset_group,
            column_names=["example"],
            shuffle=True,
            num_parallel_workers=n_workers,
            num_shards=args.world_size,
            shard_id=args.rank,
        ).batch(
            batch_size=1,  # notes: batch_size is set in train_dataset_group via dataset_conifg, don't change bs here.
            per_batch_map=lambda examples, batch_info: collator(examples),
            input_columns=["example"],
            output_columns=["example"],
            num_parallel_workers=n_workers,
        )

        # 学習ステップ数を計算する
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / args.world_size / args.gradient_accumulation_steps
            )
            logger.info(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )

        # データセット側にも学習ステップを送信
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # 後方互換性を確保するよ
        # TODO - suppuort params grouping

        # notes: temporary solution
        if args.unet_lr:
            if args.text_encoder_lr and args.text_encoder_lr != args.unet_lr:
                logger.warning("not support seperate lr for te and unet yet.")
            if args.learning_rate and args.unet_lr:
                logger.warning("args.unet_lr is assigned, args.learning_rate will be covered.")

        # lr schedulerを用意する
        lr_scheduler = train_util.get_scheduler_fix(args, num_processes=args.world_size)

        # notes: `LoRANetwork.set_required_grad()` set the lora trainable and freeze others in `__init__`
        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(
            args, learning_rate=lr_scheduler, trainable_params=network.trainable_params()
        )

        if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
            freeze_params(vae)
            vae.set_train(False)
            vae.to(dtype=vae_dtype)

        # resumeする TODO
        if args.resume:
            NotImplementedError

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * args.world_size * args.gradient_accumulation_steps

        logger.info("running training / 学習開始")
        logger.info(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        logger.info(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        logger.info(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        logger.info(f"  num epochs / epoch数: {num_train_epochs}")
        logger.info(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # logger.info(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        logger.info(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        logger.info(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": args.text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
            "ss_debiased_estimation": bool(args.debiased_estimation_loss),
            "ss_noise_offset_random_strength": args.noise_offset_random_strength,
            "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength,
            "ss_loss_type": args.loss_type,
            "ss_huber_schedule": args.huber_schedule,
            "ss_huber_c": args.huber_c,
        }

        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time
            #   or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                        "keep_tokens_separator": subset.keep_tokens_separator,
                        "secondary_separator": subset.secondary_separator,
                        "enable_wildcard": bool(subset.enable_wildcard),
                        "caption_prefix": subset.caption_prefix,
                        "caption_suffix": subset.caption_suffix,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                    # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                    # なので、ここで複数datasetの回数を合算してもあまり意味はない
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug.\
                / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),
                }
            )

        # add extra args
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not is_main_process, desc="steps")
        global_step = 0

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )
        prepare_scheduler_for_custom_training(noise_scheduler)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        # tracker TODO

        loss_recorder = train_util.LossRecorder()
        del train_dataset_group

        # function for saving/removing
        def save_model(ckpt_name, network, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            logger.info(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)
            metadata_to_save.update(sai_metadata)

            network.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                logger.warning("hugging face upload not supported yet")

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                logger.info(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # For --sample_at_first # TODO
        self.sample_images(args, 0, global_step, vae, tokenizer, text_encoder, unet)

        # TODO
        if args.weighted_captions:
            # get_text_cond_fn = get_weighted_text_embeddings
            NotImplementedError("weighted_captions not supported yet")
        else:
            get_text_cond_fn = self.get_text_cond

        # initial a train_step_class for mindspore
        self.set_train_step_class()
        train_step = self.train_step_class(
            vae=vae,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            unet=unet,
            optimizer=optimizer,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            length_of_dataloader=len(train_dataloader),
            args=args,
            vae_scale_factor=self.vae_scale_factor,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            call_unet=self.call_unet,
            get_text_cond=get_text_cond_fn,
        )
        train_step.set_train()

        # training loop
        train_dataloader_iter = train_dataloader.create_dict_iterator(num_epochs=num_train_epochs)
        for epoch in range(num_train_epochs):
            logger.info(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            for step, batch in enumerate(train_dataloader_iter):
                batch = batch["example"]
                # get multiplier for each sample
                if network_has_multiplier:
                    multipliers = batch["network_multipliers"]
                    # if all multipliers are same, use single multiplier
                    if ops.all(multipliers == multipliers[0]):
                        multipliers = multipliers[0].item()
                    else:
                        raise NotImplementedError("multipliers for each sample is not supported yet")
                    network.set_multiplier(multipliers)

                inputs = (
                    batch["loss_weights"],
                    batch["input_ids"],
                    batch["input_ids2"],
                    batch["images"],
                    batch["original_sizes_hw"],
                    batch["crop_top_lefts"],
                    batch["target_sizes_hw"],
                )

                s = time.time()
                loss, model_pred = train_step(*inputs)
                logger.info(
                    f"current step loss: {loss.numpy().item()}, current step time cost: {time.time() - s:.2f} s"
                )

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = network.apply_max_norm_regularization(
                        args.scale_weight_norms
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if train_step.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.sample_images(args, None, global_step, vae, tokenizer, text_encoder, unet)

                    # 指定ステップごとにモデルを保存
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        if is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            save_model(ckpt_name, network, global_step, epoch)

                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, network, global_step)

                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(
                                    args, "." + args.save_model_as, remove_step_no
                                )
                                remove_model(remove_ckpt_name)

                current_loss = loss.numpy().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    # it might cost 20+ s
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if args.logging_dir is not None:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, keys_scaled, mean_norm, maximum_norm
                    )
                    # accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_recorder.moving_average}

            # 指定エポックごとにモデルを保存
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, network, global_step, epoch + 1)

                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(
                            args, "." + args.save_model_as, remove_epoch_no
                        )
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(args, network, epoch + 1)

            self.sample_images(args, epoch + 1, global_step, vae, tokenizer, text_encoder, unet)

            # end of epoch

        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_util.save_state_on_train_end(args, network)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=False)
            logger.info("model saved.")


class TrainStepForSDXLLoRA(TrainStep):
    def __init__(
        self,
        vae: nn.Cell,
        text_encoders,
        tokenizers,
        unet: nn.Cell,
        optimizer: nn.Optimizer,
        noise_scheduler,
        weight_dtype,
        length_of_dataloader,
        args,
        vae_scale_factor,
        train_text_encoder,
        train_unet,
        get_text_cond,
        call_unet,
    ):
        super().__init__(
            unet,
            optimizer,
            StaticLossScaler(65536),
            args.max_grad_norm,
            args.gradient_accumulation_steps,
            gradient_accumulation_kwargs=dict(length_of_dataloader=length_of_dataloader),
        )
        self.unet = self.model
        self.vae = vae
        self.vae_scale_factor = vae_scale_factor
        self.text_encoder1 = text_encoders[0]
        self.text_encoder2 = text_encoders[1]
        self.noise_scheduler = noise_scheduler
        self.max_timestep = (
            noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep
        )
        self.min_timestep = 0 if args.min_timestep is None else args.min_timestep
        self.weight_dtype = weight_dtype
        self.args = AttrJitWrapper(**vars(args))

        # from kohya
        self.vae_dtype = ms.float32 if args.no_half_vae else weight_dtype
        self.call_unet = call_unet
        self.get_text_cond = get_text_cond
        self.tokenizers = tokenizers
        self.train_text_encoder = train_text_encoder
        self.train_unet = train_unet

        self.tokenizer1_model_max_length = tokenizers[0].model_max_length
        self.tokenizer2_model_max_length = tokenizers[1].model_max_length
        self.tokenizer2_eos_token_id = tokenizers[1].eos_token_id

    def forward(
        self,
        loss_weights,
        input_ids,
        input_ids2,
        images,
        original_sizes_hw,
        crop_top_lefts,
        target_sizes_hw,
        latents=None,
        text_encoder_outputs1_list=None,
        text_encoder_outputs2_list=None,
        text_encoder_pool2_list=None,
        conditioning_images=None,
    ):
        if latents is not None:
            latents = latents.to(dtype=self.weight_dtype)
        else:
            latents = self.vae.diag_gauss_dist.sample(self.vae.encode(images.to(dtype=self.vae_dtype))[0])
            latents = ops.stop_gradient(latents)

        # NaNが含まれていれば警告を表示し0に置き換える
        if ops.any(ops.isnan(latents)):
            logger.info("NaN found in latents, replacing with zeros")
            latents = ops.nan_to_num(latents)
        latents = latents * self.vae_scale_factor

        text_encoder_conds = self.get_text_cond(
            self.args,
            self.tokenizer1_model_max_length,
            self.tokenizer2_model_max_length,
            self.tokenizer2_eos_token_id,
            self.text_encoder1,
            self.text_encoder2,
            input_ids,
            input_ids2,
            text_encoder_outputs1_list,
            text_encoder_outputs2_list,
            text_encoder_pool2_list,
            self.weight_dtype,
        )
        if not self.train_text_encoder:
            text_encoder_conds = ops.stop_gradient(text_encoder_conds)

        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
            self.args, self.noise_scheduler, latents, self.max_timestep, self.min_timestep
        )

        noise_pred = self.call_unet(
            self.args,
            self.unet,
            noisy_latents,
            timesteps,
            text_encoder_conds,
            original_sizes_hw,
            crop_top_lefts,
            target_sizes_hw,
            self.weight_dtype,
        )
        if self.args.v_parameterization:
            # v-parameterization training
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        loss = train_util.conditional_loss(
            noise_pred.float(), target.float(), reduction="none", loss_type=self.args.loss_type, huber_c=huber_c
        )
        if self.args.masked_loss:
            loss = apply_masked_loss(loss, conditioning_images)
        loss = loss.mean([1, 2, 3])

        loss = loss * loss_weights

        if self.args.min_snr_gamma:
            loss = apply_snr_weight(
                loss, timesteps, self.noise_scheduler, self.args.min_snr_gamma, self.args.v_parameterization
            )
        if self.args.scale_v_pred_loss_like_noise_pred:
            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, self.noise_scheduler)
        if self.args.v_pred_like_loss:
            loss = add_v_prediction_like_loss(loss, timesteps, self.noise_scheduler, self.args.v_pred_like_loss)
        if self.args.debiased_estimation_loss:
            loss = apply_debiased_estimation(loss, timesteps, self.noise_scheduler)

        loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし
        loss = self.scale_loss(loss)

        return loss, noise_pred


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    # deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument("--distributed", default=False, action="store_true", help="Enable distributed training")
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument(
        "--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率"
    )

    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument("--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) \
            / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) \
            / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) \
            / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--amp_level",
        type=str,
        default="O2",
        help="amp level of the trainning nets",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
