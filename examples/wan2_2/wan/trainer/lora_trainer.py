import math
import os
import time
from typing import Tuple

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist
import mindspore.mint.nn.functional as functional
import mindspore.nn as nn
from mindspore.communication import GlobalComm
from mindspore.mint.optim import AdamW

from mindone.diffusers.utils import get_logger
from mindone.peft import LoraConfig, PeftModel, get_peft_model
from mindone.trainers.zero import ZeroHelper, prepare_network

from ..distributed.util import all_gather, get_rank
from ..utils.fm_solvers import FlowDPMSolverMultistepScheduler
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.utils import save_video
from .utils import clip_by_global_norm, map_, save_checkpoint

logger = get_logger(__name__)


class LoRATrainer:
    def __init__(self, pipeline, train_loader, training_config, generation_config):
        self.pipeline = pipeline
        self.train_loader = train_loader.create_dict_iterator()
        self.training_config = training_config
        self.generation_config = generation_config

        self.pipeline.model, self.lora_config = self._configure_peft(self.pipeline.model)
        self.optimizer = self.create_optimizer(self.pipeline.model)
        self.pipeline.model, self.zero_helper = self._configure_zero3(self.pipeline.model, self.optimizer)
        self.grad_fn = ms.value_and_grad(self.calculate_loss, grad_position=None, weights=self.optimizer.parameters)

        sample_solver = generation_config.get("sample_solver", "unipc")
        self.num_train_timesteps = self.training_config.get("num_train_timesteps", 1000)
        if sample_solver == "unipc":
            self.sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
            )
        elif sample_solver == "dpm++":
            self.sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
            )
        else:
            raise ValueError(f"Unsupported sample solver: {sample_solver}")

        self.output_dir = self.generation_config.pop("output_dir", "./output/visual")
        self.fps = self.generation_config.pop("sample_fps", 24)

        self.global_step = 0

    def _configure_peft(self, model: nn.Cell) -> Tuple[PeftModel, LoraConfig]:
        target_modules = ["q", "k", "v", "o"]
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model: PeftModel = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, lora_config

    def _configure_zero3(self, model: nn.Cell, optimizer: nn.Cell) -> nn.Cell:
        model = prepare_network(model, 3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
        zero_helper = ZeroHelper(optimizer, 3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
        zero_helper.split_params()
        return model, zero_helper

    def create_optimizer(self, model: nn.Cell):
        optimizer = AdamW(
            model.trainable_params(),
            lr=self.training_config.get("learning_rate", 1e-4),
            weight_decay=self.training_config.get("weight_decay", 0.01),
        )
        return optimizer

    def train_epoch(self):
        self.pipeline.model.set_train(True)

        for _, batch in enumerate(self.train_loader):
            if self.global_step % self.training_config.get("validation_interval", 100) == 0:
                self.validate()

            if self.global_step % self.training_config.get("save_interval", 100) == 0 and self.global_step > 0:
                self.combine_and_save_checkpoint()

            self.global_step += 1
            start_time = time.time()
            video_inputs = batch["video"]
            text_inputs = batch["prompt"].asnumpy().tolist()

            bsz, _, F, H, W = video_inputs.shape

            target_shape = (
                self.pipeline.vae.model.z_dim,
                (F - 1) // self.training_config["vae_stride"][0] + 1,
                W // self.training_config["vae_stride"][1],
                H // self.training_config["vae_stride"][2],
            )

            seq_len = math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.training_config["patch_size"][1] * self.training_config["patch_size"][2])
                * target_shape[1]
            )

            # Convert video to latent space
            latents_mean, latents_log_var = self.pipeline.vae.model.encode(
                video_inputs, self.pipeline.vae.scale, return_log_var=True
            )
            latents_log_var = mint.clamp(latents_log_var, min=-30.0, max=20.0)
            latents_std = mint.exp(0.5 * latents_log_var)
            latents = latents_mean + latents_std * mint.randn_like(latents_mean)

            # Sample noise that we'll add to the latents
            noise = mint.randn_like(latents)

            # Sample a random timestep for each image
            timesteps = mint.randint(0, self.num_train_timesteps, (bsz,))

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.sample_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.pipeline.text_encoder(text_inputs)

            # Update timesteps
            mask = mint.ones((bsz, 1, noise.shape[2], noise.shape[3] // 2, noise.shape[4] // 2), dtype=ms.float32)
            temp_ts = mask * timesteps[:, None, None, None, None]
            temp_ts = temp_ts.flatten(start_dim=1)
            timesteps = mint.cat(
                [temp_ts, temp_ts.new_ones((bsz, seq_len - temp_ts.shape[1])) * timesteps[:, None]], dim=1
            )

            loss, grads = self.grad_fn(noise, latents, noisy_latents, timesteps, encoder_hidden_states, seq_len)
            grads = self.zero_helper.cal_gradients(grads)
            clip_by_global_norm(grads, self.training_config.get("max_grad_norm", 1.0))

            self.zero_helper.run_optimizer(grads)
            duration = time.time() - start_time
            logger.info(f"Step {self.global_step}, Loss: {loss.asnumpy():.4f}, Step time: {duration:.4f} sec")

    def calculate_loss(self, noise, latents, noisy_latents, timesteps, encoder_hidden_states, seq_len):
        model_pred = self.pipeline.model(noisy_latents, t=timesteps, context=encoder_hidden_states, seq_len=seq_len)
        model_pred = mint.stack(model_pred)
        loss = functional.mse_loss(model_pred, noise - latents, reduction="mean")
        return loss

    def validate(self):
        logger.info("Running validation...")
        dist.barrier()
        self.pipeline.model.set_train(False)
        video = self.pipeline.generate(**self.generation_config)
        if get_rank() == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            save_file = os.path.join(self.output_dir, f"step_{self.global_step}.mp4")
            logger.info(f"Saving generated video to {save_file}")
            save_video(
                tensor=video[None],
                save_file=save_file,
                fps=self.fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        self.pipeline.model.set_train(True)
        dist.barrier()

    def combine_and_save_checkpoint(self):
        dist.barrier()
        output_dir = self.training_config.get("output_dir", "./output/ckpt")
        lora_parameters = self.pipeline.model.trainable_params()
        lora_parameters = map_(lambda x: ms.Parameter(mint.cat(all_gather(x)), name=x.name), lora_parameters)
        if get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)
            save_checkpoint(lora_parameters, output_dir, fname=f"step_{self.global_step}_lora.ckpt")
            self.lora_config.save_pretrained(output_dir)
            logger.info(f"Saved LoRA checkpoint at step {self.global_step} to {output_dir}")
        dist.barrier()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            self.train_epoch()
