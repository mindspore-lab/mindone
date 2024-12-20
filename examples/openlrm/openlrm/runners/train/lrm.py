
import os, sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))  # for mindone

import argparse
import datetime
import logging 
import math

import yaml
from utils.train_util import str2bool

import mindspore as ms
from mindspore import Model, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

logger = logging.getLogger(__name__)

from tqdm.auto import tqdm
from .base_trainer import Trainer
from openlrm.runners import REGISTRY_RUNNERS

# from model_stage1 import InstantMeshStage1WithLoss
from omegaconf import OmegaConf

from mindone.data import create_dataloader
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallbackEpoch
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import instantiate_from_config
from mindone.utils.env import init_train_env
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed

@REGISTRY_RUNNERS.register('train.lrm')
class LRMTrainer(Trainer):
    def __init__(self):
        super().__init__()
        
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        if self.args.resume:
            self.args.output_path = self.args.resume
        elif not args.debug:
            self.args.output_path = os.path.join(self.args.output_path, time_str)
        else:
            print("make sure you are debugging now, as no ckpt will be saved.")

        # 1. env init
        did, rank_id, device_num = init_train_env(
            self.args.mode,
            seed=self.args.seed,
            distributed=self.args.use_parallel,
            device_target=self.args.device_target,
            max_device_memory=self.args.max_device_memory,
            debug=self.args.debug,
            )
        seed_everything(self.cfg.experiment.seed)
        set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

        self.ckpt_dir = os.path.join(args.output_path, "ckpt")

        # 2. build model
        # self.cfg.model.dtype = self.args.dtype
        self.model_with_loss = self._build_model(self.cfg)

        # 3. create dataset
        self.train_loader, self.val_loader = self._build_dataloader(self.cfg)

        # 4. build optimizer, scheduler, trainer, etc 
        self._build_utils(self.args, self.cfg)

        # self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        # self.pixel_loss_fn, self.perceptual_loss_fn, self.tv_loss_fn = self._build_loss_fn(self.cfg)

    # Model initialization
    def _build_model(self, cfg):
        assert cfg.experiment.type == 'lrm', \
            f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"
        from openlrm.models import ModelLRMWithLoss
        lrm_model_with_loss = ModelLRMWithLoss(cfg) #TODO

        # TBD
        if not self.args.global_bf16:
            lrm_model_with_loss = auto_mixed_precision(
                lrm_model_with_loss,
                amp_level=args.amp_level,
            )

        return lrm_model_with_loss

    # Build training utils: lr, optim, callbacks, trainer
    def _build_utils(self, args, cfg):
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
            name=args.scheduler, # cfg_scheduler.type "cosine_decay"
            lr=args.start_learning_rate, #cfg.train.optim.lr,
            end_lr=args.end_learning_rate,
            warmup_steps= args.warmup_steps, #cfg.scheduler.warmup_real_iters,
            decay_steps=args.decay_steps,
            num_epochs=args.epochs,
        )
        
        self.optimizer = create_optimizer(
            self.model_with_loss.trainable_params(),
            name=args.optim, # "adamw"
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2), #args.betas
            eps=args.optim_eps,
            group_strategy=args.group_strategy,
            weight_decay=args.weight_decay,
            lr=lr,
        )

        if args.loss_scaler_type == "dynamic":  # for the case when there is an overflow during training
            loss_scaler = DynamicLossScaleUpdateCell(
                loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
            )
        elif args.loss_scaler_type == "static":
            loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
        else:
            loss_scaler = ms.Tensor([1.0], dtype=ms.float32)

        ############################################
        # TBD
        # decay_params, no_decay_params = [], []

        # add all bias and LayerNorm params to no_decay_params
        # for name, module in model.name_cells().items():
        #     if isinstance(module, nn.LayerNorm):
        #         no_decay_params.extend([p for p in module.get_parameters()])
        #     elif hasattr(module, 'beta') and module.beta is not None:
        #         no_decay_params.append(module.beta)

        # add remaining parameters to decay_params
        # _no_decay_ids = set(map(id, no_decay_params))
        # decay_params = [p for p in model.get_parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        # decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        # no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # monitor this to make sure we don't miss any parameters
        # logger.info("======== Weight Decay Parameters ========")
        # logger.info(f"Total: {len(decay_params)}")
        # logger.info("======== No Weight Decay Parameters ========")
        # logger.info(f"Total: {len(no_decay_params)}")

        # Optimizer
        # opt_groups = [
        #     {'params': decay_params, 'weight_decay': cfg.train.optim.weight_decay},
        #     {'params': no_decay_params, 'weight_decay': 0.0},
        # ]
        # optimizer = torch.optim.AdamW(
        #     opt_groups,
        #     lr=cfg.train.optim.lr,
        #     betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),
        # )

        # return optimizer

        # ############################################
        

    # TBD - customized scheduler
    # def _build_scheduler(self, optimizer, cfg):
    #     local_batches_per_epoch = math.floor(len(self.train_loader) / self.accelerator.num_processes)
    #     total_global_batches = cfg.train.epochs * math.ceil(local_batches_per_epoch / self.cfg.train.accum_steps)
    #     effective_warmup_iters = cfg.train.scheduler.warmup_real_iters
    #     logger.debug(f"======== Scheduler effective max iters: {total_global_batches} ========")
    #     logger.debug(f"======== Scheduler effective warmup iters: {effective_warmup_iters} ========")
    #     if cfg.train.scheduler.type == 'cosine':
    #         from openlrm.utils.scheduler import CosineWarmupScheduler
    #         scheduler = CosineWarmupScheduler(
    #             optimizer=optimizer,
    #             warmup_iters=effective_warmup_iters,
    #             max_iters=total_global_batches,
    #         )
    #     else:
    #         raise NotImplementedError(f"Scheduler type {cfg.train.scheduler.type} not implemented")
    #     return scheduler

    # Build dataset
    def _build_dataloader(self, cfg):
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
        val_dataset = MixerDataset(
            split="val",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res_low=cfg.dataset.render_image.low,
            render_image_res_high=cfg.dataset.render_image.high,
            render_region_size=cfg.dataset.render_image.region,
            source_image_res=cfg.dataset.source_image_res,
            normalize_camera=cfg.dataset.normalize_camera,
            normed_dist_to_center=cfg.dataset.normed_dist_to_center,
        )

        # build data loader
        train_loader = create_dataloader(
            train_dataset,
            batch_size=self.train.batch_size,
            shuffle=True,
            drop_remainder=True,
            device_num=device_num,
            rank_id=rank_id,
            num_workers=cfg.dataset.num_train_workers,
            python_multiprocessing=args.data_multiprocessing,
            max_rowsize=args.max_rowsize,
            debug=False,  # ms240_sept4: THIS CANNOT BE TRUE, OTHERWISE loader error
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=self.train.batch_size,
            shuffle=False,
            drop_remainder=False,
            device_num=device_num,
            rank_id=rank_id,
            num_workers=cfg.dataset.num_val_workers,
            python_multiprocessing=args.data_multiprocessing,
            max_rowsize=args.max_rowsize,
            debug=False,  # ms240_sept4: THIS CANNOT BE TRUE, OTHERWISE loader error
        )

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
        sink_epochs = math.ceil(total_train_steps / steps_per_sink)

        if args.ckpt_save_steps == -1:
            ckpt_save_interval = args.ckpt_save_interval
            step_mode = False
        else:
            step_mode = not args.dataset_sink_mode
            if not args.dataset_sink_mode:
                ckpt_save_interval = args.ckpt_save_steps
            else:
                # still need to count interval in sink epochs
                ckpt_save_interval = max(1, args.ckpt_save_steps // steps_per_sink)
                if args.ckpt_save_steps % steps_per_sink != 0:
                    logger.warning(
                        f"'ckpt_save_steps' must be times of sink size or dataset_size under dataset sink mode."
                        f"Checkpoint will be saved every {ckpt_save_interval * steps_per_sink} steps."
                    )
        step_mode = step_mode if args.step_mode is None else args.step_mode

        logger.info(f"train_steps: {total_train_steps}, train_epochs: {args.epochs}, sink_size: {args.sink_size}")
        logger.info(f"total train steps: {total_train_steps}, sink epochs: {sink_epochs}")
        logger.info(
            "ckpt_save_interval: {} {}".format(
                ckpt_save_interval, "steps" if (not args.dataset_sink_mode and step_mode) else "sink epochs"
            )
        )

        return train_loader, val_loader

    # def _build_loss_fn(self, cfg):
    #     from openlrm.losses import PixelLoss, LPIPSLoss, TVLoss
    #     pixel_loss_fn = PixelLoss()
    #     perceptual_loss_fn = LPIPSLoss(prefech=True)
    #     tv_loss_fn = TVLoss()
    #     return pixel_loss_fn, perceptual_loss_fn, tv_loss_fn

    def register_hooks(self):
        pass
    
    # def forward_loss_local_step(self, data):

    #     source_camera = data['source_camera']
    #     render_camera = data['render_camera']
    #     source_image = data['source_image']
    #     render_image = data['render_image']
    #     render_anchors = data['render_anchors']
    #     render_full_resolutions = data['render_full_resolutions']
    #     render_bg_colors = data['render_bg_colors']

    #     N, M, C, H, W = render_image.shape

    #     # forward
    #     outputs = self.model(
    #         image=source_image,
    #         source_camera=source_camera,
    #         render_cameras=render_camera,
    #         render_anchors=render_anchors,
    #         render_resolutions=render_full_resolutions,
    #         render_bg_colors=render_bg_colors,
    #         render_region_size=self.cfg.dataset.render_image.region,
    #     )

    #     # loss calculation
    #     loss = 0.
    #     loss_pixel = None
    #     loss_perceptual = None
    #     loss_tv = None

    #     if self.cfg.train.loss.pixel_weight > 0.:
    #         loss_pixel = self.pixel_loss_fn(outputs['images_rgb'], render_image)
    #         loss += loss_pixel * self.cfg.train.loss.pixel_weight
    #     if self.cfg.train.loss.perceptual_weight > 0.:
    #         loss_perceptual = self.perceptual_loss_fn(outputs['images_rgb'], render_image)
    #         loss += loss_perceptual * self.cfg.train.loss.perceptual_weight
    #     if self.cfg.train.loss.tv_weight > 0.: 
    #         loss_tv = self.tv_loss_fn(outputs['planes'])
    #         loss += loss_tv * self.cfg.train.loss.tv_weight

    #     return outputs, loss, loss_pixel, loss_perceptual, loss_tv

    # # TODO
    # def train_epoch(self, pbar: tqdm, loader: torch.utils.data.DataLoader, profiler: torch.profiler.profile):
    #     self.model.train()

    #     local_step_losses = []
    #     global_step_losses = []

    #     logger.debug(f"======== Starting epoch {self.current_epoch} ========")
    #     for data in loader:

    #         logger.debug(f"======== Starting global step {self.global_step} ========")
    #         with self.accelerator.accumulate(self.model):

    #             # forward to loss
    #             outs, loss, loss_pixel, loss_perceptual, loss_tv = self.forward_loss_local_step(data)
                
    #             # backward
    #             self.accelerator.backward(loss)
    #             if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
    #                 self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()

    #             # track local losses
    #             local_step_losses.append(torch.stack([
    #                 _loss.detach() if _loss is not None else ms.Tensor(float('nan'), device=self.device)
    #                 for _loss in [loss, loss_pixel, loss_perceptual, loss_tv]
    #             ]))

    #         # track global step
    #         if self.accelerator.sync_gradients:
    #             profiler.step()
    #             self.scheduler.step()
    #             logger.debug(f"======== Scheduler step ========")
    #             self.global_step += 1
    #             global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
    #             loss, loss_pixel, loss_perceptual, loss_tv = global_step_loss.unbind()
    #             loss_kwargs = {
    #                 'loss': loss.item(),
    #                 'loss_pixel': loss_pixel.item(),
    #                 'loss_perceptual': loss_perceptual.item(),
    #                 'loss_tv': loss_tv.item(),
    #             }
    #             self.log_scalar_kwargs(
    #                 step=self.global_step, split='train',
    #                 **loss_kwargs
    #             )
    #             self.log_optimizer(step=self.global_step, attrs=['lr'], group_ids=[0, 1])
    #             local_step_losses = []
    #             global_step_losses.append(global_step_loss)

    #             # manage display
    #             pbar.update(1)
    #             description = {
    #                 **loss_kwargs,
    #                 'lr': self.optimizer.param_groups[0]['lr'],
    #             }
    #             description = '[TRAIN STEP]' + \
    #                 ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in description.items() if not math.isnan(v))
    #             pbar.set_description(description)

    #             # periodic actions
    #             if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
    #                 self.save_checkpoint()
    #             if self.global_step % self.cfg.val.global_step_period == 0:
    #                 self.evaluate()
    #                 self.model.train()
    #             if self.global_step % self.cfg.logger.image_monitor.train_global_steps == 0:
    #                 self.log_image_monitor(
    #                     step=self.global_step, split='train',
    #                     renders=outs['images_rgb'].detach()[:self.cfg.logger.image_monitor.samples_per_log].cpu(),
    #                     gts=data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
    #                 )

    #             # progress control
    #             if self.global_step >= self.N_max_global_steps:
    #                 self.accelerator.set_trigger()
    #                 break

    #     # track epoch
    #     self.current_epoch += 1
    #     epoch_losses = torch.stack(global_step_losses).mean(dim=0)
    #     epoch_loss, epoch_loss_pixel, epoch_loss_perceptual, epoch_loss_tv = epoch_losses.unbind()
    #     epoch_loss_dict = {
    #         'loss': epoch_loss.item(),
    #         'loss_pixel': epoch_loss_pixel.item(),
    #         'loss_perceptual': epoch_loss_perceptual.item(),
    #         'loss_tv': epoch_loss_tv.item(),
    #     }
    #     self.log_scalar_kwargs(
    #         epoch=self.current_epoch, split='train',
    #         **epoch_loss_dict,
    #     )
    #     logger.info(
    #         f'[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: ' + \
    #             ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in epoch_loss_dict.items() if not math.isnan(v))
    #     )

    def train(self, args, cfg):

        # weight loading: load checkpoint when resume
        lrm_model = self.model_with_loss.lrm_generator
        if args.resume:
            logger.info(f"Loading latest.ckpt in {args.resume} to resume training")
            resume_ckpt = os.path.join(args.resume, "latest.ckpt")
            start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
                lrm_model, optimizer, resume_ckpt
            )  # refer to hpcai train script about the input usage of this func
            loss_scaler.loss_scale_value = loss_scale
            loss_scaler.cur_iter = cur_iter
            loss_scaler.last_overflow_iter = last_overflow_iter
        else:
            start_epoch = 0
            # resume_param = ms.load_checkpoint(config.model.params.lrm_generator_config.openlrm_ckpt)
            # ms.load_param_into_net(lrm_model, resume_param)
            # logger.info("Use random initialization for lrm, NO ckpt loading")  # NOT converge

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
            optimizer=optimizer,
            scale_sense=loss_scaler,
            drop_overflow_update=args.drop_overflow_update,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad=args.clip_grad,
            clip_norm=args.max_grad_norm, # cfg.train.optim.clip_grad_norm
            ema=ema,
        )

        if args.global_bf16:
            model = Model(net_with_grads, amp_level="O0")
        else:
            model = Model(net_with_grads)

        # 4.3 callbacks
        callback = [
            TimeMonitor(),
            OverflowMonitor(),
        ]

        if rank_id == 0:
            save_cb = EvalSaveCallback(
                network=self.model_with_loss,
                rank_id=rank_id,
                ckpt_save_dir=ckpt_dir,
                ema=ema,
                ckpt_save_policy="top_k",
                ckpt_max_keep=args.ckpt_max_keep,
                step_mode=step_mode,
                use_step_unit=(args.ckpt_save_steps != -1),
                ckpt_save_interval=ckpt_save_interval,
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
        if rank_id == 0:
            num_params_lrm, num_params_lrm_trainable = count_params(lrm_model_with_loss)
            key_info = "Key Settings:\n" + "=" * 50 + "\n"
            key_info += "\n".join(
                [
                    f"\tMindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                    f"\tDistributed mode: {args.use_parallel}",
                    f"\tNum params: {num_params_lrm} (lrm: {num_params_lrm})",
                    f"\tNum trainable params: {num_params_lrm_trainable}",
                    f"\tLearning rate: {args.start_learning_rate}",
                    f"\tBatch size: {config.data.batch_size}",
                    f"\tImage size: {img_size}",
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
                    f"\tUse recompute: {config.model.params.lrm_generator_config.params.use_recompute}",
                    f"\tDataset sink: {args.dataset_sink_mode}",
                ]
            )
            key_info += "\n" + "=" * 50
            logger.info(key_info)
            logger.info("Start training...")
            with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
                yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)
            OmegaConf.save(config, os.path.join(args.output_path, "cfg.yaml"))

        logger.info("using the standard fitting api")
        self.model.fit(
            sink_epochs,
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

    #TODO
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



