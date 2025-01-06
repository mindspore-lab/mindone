import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))  # for mindone

import datetime
import json
import logging
import math

import yaml

import mindspore as ms
from mindspore import Model, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

logger = logging.getLogger(__name__)

from omegaconf import OmegaConf
from openlrm.models.rendering.utils.renderer import MatrixInv
from openlrm.runners import REGISTRY_RUNNERS
from openlrm.utils import seed_everything

from mindone.data import create_dataloader
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallbackEpoch
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils import count_params, init_train_env, set_logger
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import instantiate_from_config

from .base_trainer import Trainer


@REGISTRY_RUNNERS.register("train.lrm")
class LRMTrainer(Trainer):
    def __init__(self):
        super().__init__()

        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        if self.args.resume:
            self.args.output_path = self.args.resume
        elif not self.args.debug:
            self.args.output_path = os.path.join(self.args.output_path, time_str)
        else:
            print(
                "Make sure you are debugging now, as previous checkpoints will be overwritten and training could be slow."
            )
        print(f"Checkpoints and configs will be stored in {self.args.output_path}.")

        # 1. env init
        did, self.rank_id, self.device_num = init_train_env(
            self.args.mode,
            seed=self.args.seed,
            distributed=self.args.use_parallel,
            device_target=self.args.device_target,
            max_device_memory=self.args.max_device_memory,
            debug=self.args.debug,
        )
        seed_everything(self.args.seed)
        set_logger(name="", output_dir=self.args.output_path, rank=self.rank_id, log_level=eval(self.args.log_level))

        self.ckpt_dir = os.path.join(self.args.output_path, "ckpt")

        # 2. build model
        self.model_with_loss = self._build_model(self.cfg)

        # 3. create dataset
        self.train_loader, self.val_loader = self._build_dataloader(self.args, self.cfg)

        # 4. build optimizer, scheduler, trainer, etc
        dataset_size = self.train_loader.get_dataset_size()
        self._build_utils(self.args, self.cfg, dataset_size)

    # Model initialization
    def _build_model(self, cfg):
        assert (
            cfg.experiment.type == "lrm"
        ), f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"
        from openlrm.models import ModelLRMWithLoss

        lrm_model_with_loss = ModelLRMWithLoss(cfg)
        lrm_model_with_loss.set_train(True)
        # lrm_model_eval = ModelLRMWithLossEval(cfg) # TODO

        if not self.args.global_bf16:
            lrm_model_with_loss = auto_mixed_precision(
                lrm_model_with_loss, amp_level=self.args.amp_level, custom_fp32_cells=[MatrixInv]
            )

        return lrm_model_with_loss

    # Build training utils: lr, optim, callbacks, trainer
    def _build_utils(self, args, cfg, dataset_size):
        total_train_steps = cfg.train.epochs
        # build learning rate scheduler
        if not args.decay_steps:
            args.decay_steps = total_train_steps - args.warmup_steps  # fix lr scheduling
            if args.decay_steps <= 0:
                logger.warning(
                    f"decay_steps is {args.decay_steps}, please check epochs, dataset_size and warmup_steps. "
                    f"Will force decay_steps to be set to 1."
                )
                args.decay_steps = 1

        lr = create_scheduler(
            steps_per_epoch=dataset_size,
            name=args.scheduler,  # "cosine_annealing_warm_restarts_lr"
            lr=args.start_learning_rate,
            end_lr=args.end_learning_rate,
            warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps,
            num_epochs=args.epochs,
        )

        self.optimizer = create_optimizer(
            self.model_with_loss.trainable_params(),
            name=args.optim,  # "adamw"
            betas=args.betas,
            eps=args.optim_eps,
            group_strategy=args.group_strategy,
            weight_decay=args.weight_decay,
            lr=lr,
        )

        if args.loss_scaler_type == "dynamic":  # for the case when there is an overflow during training
            self.loss_scaler = DynamicLossScaleUpdateCell(
                loss_scale_value=args.init_loss_scale,
                scale_factor=args.loss_scale_factor,
                scale_window=args.scale_window,
            )
        elif args.loss_scaler_type == "static":
            self.loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
        else:
            self.loss_scaler = ms.Tensor([1.0], dtype=ms.float32)

    # Build dataset
    def _build_dataloader(self, args, cfg):
        # dataset class
        from openlrm.datasets import MixerDataset

        train_dataset = MixerDataset(
            split="train",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res_low=cfg.dataset.render_image.low,
            render_image_res_high=cfg.dataset.render_image.high,
            render_region_size=cfg.dataset.render_image.region,
            source_image_res=cfg.dataset.source_image_res,
            normalize_camera=cfg.dataset.normalize_camera,
            normed_dist_to_center=cfg.dataset.normed_dist_to_center,
        )
        val_dataset = None
        # TODO
        # val_dataset = MixerDataset(
        #     split="val",
        #     subsets=cfg.dataset.subsets,
        #     sample_side_views=cfg.dataset.sample_side_views,
        #     render_image_res_low=cfg.dataset.render_image.low,
        #     render_image_res_high=cfg.dataset.render_image.high,
        #     render_region_size=cfg.dataset.render_image.region,
        #     source_image_res=cfg.dataset.source_image_res,
        #     normalize_camera=cfg.dataset.normalize_camera,
        #     normed_dist_to_center=cfg.dataset.normed_dist_to_center,
        # )

        # build data loader
        train_loader = create_dataloader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            drop_remainder=True,
            device_num=self.device_num,
            rank_id=self.rank_id,
            num_workers_dataset=args.num_parallel_workers,  # too large may encounter RAM error
            num_workers=cfg.dataset.num_train_workers,
            python_multiprocessing=args.data_multiprocessing,
            max_rowsize=args.max_rowsize,
            debug=True,  # ms213, if False, training would get stuck
        )
        val_loader = None
        # val_loader = create_dataloader(
        #     val_dataset,
        #     batch_size=self.cfg.train.batch_size,
        #     shuffle=False,
        #     drop_remainder=False,
        #     device_num=self.device_num,
        #     rank_id=self.rank_id,
        #     num_workers=cfg.dataset.num_val_workers,
        #     python_multiprocessing=args.data_multiprocessing,
        #     max_rowsize=args.max_rowsize,
        #     debug=False,  # ms240_sept4: THIS CANNOT BE TRUE, OTHERWISE loader error
        # )

        # compute total steps and data epochs (in unit of data sink size)
        dataset_size = train_loader.get_dataset_size()

        if args.train_steps == -1:
            assert args.epochs != -1
            total_train_steps = args.epochs * dataset_size
        else:
            total_train_steps = args.train_steps

        if args.dataset_sink_mode and args.sink_size != -1:
            steps_per_sink = args.sink_size
        else:
            steps_per_sink = dataset_size
        self.sink_epochs = math.ceil(total_train_steps / steps_per_sink)

        if args.ckpt_save_steps == -1:
            self.ckpt_save_interval = args.ckpt_save_interval
            self.step_mode = False
        else:
            self.step_mode = not args.dataset_sink_mode
            if not args.dataset_sink_mode:
                self.ckpt_save_interval = args.ckpt_save_steps
            else:
                # still need to count interval in sink epochs
                self.ckpt_save_interval = max(1, args.ckpt_save_steps // steps_per_sink)
                if args.ckpt_save_steps % steps_per_sink != 0:
                    logger.warning(
                        f"'ckpt_save_steps' must be times of sink size or dataset_size under dataset sink mode."
                        f"Checkpoint will be saved every {self.ckpt_save_interval * steps_per_sink} steps."
                    )
        self.step_mode = self.step_mode if args.step_mode is None else args.step_mode

        logger.info(f"train_steps: {total_train_steps}, train_epochs: {args.epochs}, sink_size: {args.sink_size}")
        logger.info(f"total train steps: {total_train_steps}, sink epochs: {self.sink_epochs}")
        logger.info(
            "ckpt_save_interval: {} {}".format(
                self.ckpt_save_interval, "steps" if (not args.dataset_sink_mode and self.step_mode) else "sink epochs"
            )
        )

        return train_loader, val_loader

    def register_hooks(self):
        pass

    def train(self, args, cfg):
        # weight loading: load checkpoint when resume
        lrm_model = self.model_with_loss.lrm_generator
        if args.resume:
            logger.info(f"Loading train_resume.ckpt in {args.resume} to resume training")
            resume_ckpt = os.path.join(args.resume, "ckpt", "train_resume.ckpt")
            start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
                lrm_model, self.optimizer, resume_ckpt
            )  # NOTE: if total training steps is different from original resume checkpoint, optimizer has different shape and encounter error.
            self.loss_scaler.loss_scale_value = loss_scale
            self.loss_scaler.cur_iter = cur_iter
            self.loss_scaler.last_overflow_iter = last_overflow_iter
        else:
            start_epoch = 0
            # resume_param = ms.load_checkpoint(config.model.params.lrm_generator_config.openlrm_ckpt)
            # ms.load_param_into_net(lrm_model, resume_param)

        ema = (
            EMA(
                self.model_with_loss,
                ema_decay=0.9999,
            )
            if args.use_ema
            else None
        )

        net_with_grads = TrainOneStepWrapper(
            self.model_with_loss,
            optimizer=self.optimizer,
            scale_sense=self.loss_scaler,
            drop_overflow_update=args.drop_overflow_update,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad=args.clip_grad,
            clip_norm=args.max_grad_norm,
            ema=ema,
        )

        if args.global_bf16:
            self.model = Model(net_with_grads, amp_level="O0")  # TODO: metrics={}
        else:
            self.model = Model(net_with_grads)

        # 4.3 callbacks
        callback = [
            TimeMonitor(),
            OverflowMonitor(),
        ]

        if self.rank_id == 0:
            save_cb = EvalSaveCallback(
                network=self.model_with_loss,
                rank_id=self.rank_id,
                ckpt_save_dir=self.ckpt_dir,
                ema=ema,
                ckpt_save_policy="top_k",  # top_k error after training: no self.main_indicator
                ckpt_max_keep=args.ckpt_max_keep,
                step_mode=self.step_mode,
                use_step_unit=(args.ckpt_save_steps != -1),
                ckpt_save_interval=self.ckpt_save_interval,
                log_interval=args.log_interval,
                start_epoch=start_epoch,
                model_name="openlrm",
                record_lr=True,
                prefer_low_perf=True,  # prefer low loss, this for top_k recording
            )
            callback.append(save_cb)

        if args.profile:
            callback.append(ProfilerCallbackEpoch(2, 3, "./profile_data"))

        # log and save config
        if self.rank_id == 0:
            num_params_lrm, num_params_lrm_trainable = count_params(self.model_with_loss)
            key_info = "Key Settings:\n" + "=" * 50 + "\n"
            key_info += "\n".join(
                [
                    f"\tMindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                    f"\tDistributed mode: {args.use_parallel}",
                    f"\tNum params: {num_params_lrm} (lrm: {num_params_lrm})",
                    f"\tNum trainable params: {num_params_lrm_trainable}",
                    f"\tLearning rate: {args.start_learning_rate}",
                    f"\tBatch size: {cfg.train.batch_size}",
                    f"\tImage size: {cfg.dataset.source_image_res}",
                    f"\tWeight decay: {args.weight_decay}",
                    f"\tGrad accumulation steps: {args.gradient_accumulation_steps}",
                    f"\tNum epochs: {args.epochs}",
                    f"\tUse model dtype: {args.dtype}",
                    f"\tMixed precision level: {args.amp_level}",
                    f"\tLoss scaler: {args.loss_scaler_type}",
                    f"\tInit loss scale: {args.init_loss_scale}",
                    f"\tGrad clipping: {args.clip_grad}",
                    f"\tMax grad norm: {args.max_grad_norm}",
                    f"\tEMA: {args.use_ema}",
                    # f"\tUse recompute: {args.use_recompute}", #TBD
                    f"\tDataset sink: {args.dataset_sink_mode}",
                ]
            )
            key_info += "\n" + "=" * 50
            logger.info(key_info)
            logger.info("Start training...")
            with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
                yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)
            OmegaConf.save(self.cfg, os.path.join(args.output_path, "cfg.yaml"))
            with open(os.path.join(args.output_path, "config.json"), "w") as f:
                json.dump(dict(self.cfg.model), f, indent=4)
                # save in same format as released checkpoints, for later inference

        logger.info("Using the standard fitting api")
        self.model.fit(
            self.sink_epochs,
            self.train_loader,
            self.val_loader,
            valid_frequency=cfg.val.global_step_period,
            callbacks=callback,
            dataset_sink_mode=args.dataset_sink_mode,
            sink_size=args.sink_size,
            initial_epoch=start_epoch,
        )

        # starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        # # skipped_loader = self.accelerator.skip_first_batches(self.train_loader, starting_local_step_in_epoch)
        # logger.info(f"======== Skipped {starting_local_step_in_epoch} local batches ========")

        # with tqdm(
        #     range(0, self.N_max_global_steps),
        #     initial=self.global_step,
        # ) as pbar:

        # TODO add profiler
        # profiler = torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(
        #         wait=10, warmup=10, active=100,
        #     ),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
        #         self.cfg.logger.tracker_root,
        #         self.cfg.experiment.parent, self.cfg.experiment.child,
        #     )),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        # ) if self.cfg.logger.enable_profiler else DummyProfiler()

        # with profiler:

        #     self.optimizer.zero_grad()
        #     for _ in range(self.current_epoch, self.cfg.train.epochs):

        #         loader = skipped_loader or self.train_loader
        #         skipped_loader = None
        #         self.train_epoch(pbar=pbar, loader=loader, profiler=profiler)
        #         if self.accelerator.check_trigger():
        #             break

        # logger.info(f"======== Training finished at global step {self.global_step} ========")

        # final checkpoint and evaluation
        # self.save_checkpoint()
        # self.evaluate()

    # TODO
    # @no_grad()
    # def evaluate(self, epoch: int = None):
    #     self.model.lrm_generator.set_Train(False)

    #     max_val_batches = self.cfg.val.debug_batches or len(self.val_loader)
    #     running_losses = []
    #     sample_data, sample_outs = None, None

    #     for data in tqdm(self.val_loader, total=max_val_batches):

    #         if len(running_losses) >= max_val_batches:
    #             logger.info(f"======== Early stop validation at {len(running_losses)} batches ========")
    #             break

    #         outs, loss, loss_pixel, loss_perceptual, loss_tv = self.forward_loss_local_step(data)
    #         sample_data, sample_outs = data, outs

    #         running_losses.append(torch.stack([
    #             _loss if _loss is not None else ms.Tensor(float('nan'), device=self.device)
    #             for _loss in [loss, loss_pixel, loss_perceptual, loss_tv]
    #         ]))

    #     total_losses = self.accelerator.gather(torch.stack(running_losses)).mean(dim=0).cpu()
    #     total_loss, total_loss_pixel, total_loss_perceptual, total_loss_tv = total_losses.unbind()
    #     total_loss_dict = {
    #         'loss': total_loss.item(),
    #         'loss_pixel': total_loss_pixel.item(),
    #         'loss_perceptual': total_loss_perceptual.item(),
    #         'loss_tv': total_loss_tv.item(),
    #     }

    #     if epoch is not None:
    #         self.log_scalar_kwargs(
    #             epoch=epoch, split='val',
    #             **total_loss_dict,
    #         )
    #         logger.info(
    #             f'[VAL EPOCH] {epoch}/{self.cfg.train.epochs}: ' + \
    #                 ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
    #         )
    #         self.log_image_monitor(
    #             epoch=epoch, split='val',
    #             renders=sample_outs['images_rgb'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
    #             gts=sample_data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
    #         )
    #     else:
    #         self.log_scalar_kwargs(
    #             step=self.global_step, split='val',
    #             **total_loss_dict,
    #         )
    #         logger.info(
    #             f'[VAL STEP] {self.global_step}/{self.N_max_global_steps}: ' + \
    #                 ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
    #         )
    #         self.log_image_monitor(
    #             step=self.global_step, split='val',
    #             renders=sample_outs['images_rgb'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
    #             gts=sample_data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
    #         )
