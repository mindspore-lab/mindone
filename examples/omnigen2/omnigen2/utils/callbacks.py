import logging
import os
from time import perf_counter

import numpy as np
from omnigen2.utils.img_util import to_pil_image

from mindspore import Callback, RunContext, dtype, mint, ops, tensor
from mindspore.communication import GlobalComm, get_group_size

_logger = logging.getLogger(__name__)


class BinLossCallback(Callback):
    def __init__(
        self, batch_size: int, n_loss_bins=10, grad_steps: int = 1, latent_channels: int = 16, spat_comp: int = 8
    ):
        super().__init__()
        self.batch_size = batch_size
        self.grad_steps = grad_steps
        self.latent_channels = latent_channels
        self.spat_comp = spat_comp

        # Number of bins, for loss recording
        self.n_loss_bins = n_loss_bins
        # Create bins for t
        self.loss_bins = np.linspace(0.0, 1.0, self.n_loss_bins + 1, dtype=np.float32).tolist()
        # Initialize occurrence and sum tensors
        self.bin_occurrence, self.bin_sum_loss = None, None
        self.step_time = None

        # Initialize for distributed training
        self.reduce, self.rank_size = None, 1
        if GlobalComm.INITED:
            self.reduce = ops.AllReduce(op=ops.ReduceOp.SUM)
            self.rank_size = get_group_size()

    def on_train_begin(self, run_context: RunContext):
        self.step_time = perf_counter()

    def on_train_epoch_begin(self, run_context: RunContext):
        self.bin_occurrence = tensor(np.zeros(self.n_loss_bins, dtype=np.int32))
        self.bin_sum_loss = tensor(np.zeros(self.n_loss_bins, dtype=np.float32))

    def on_train_step_end(self, run_context: RunContext):
        step_time = perf_counter() - self.step_time
        self.step_time = perf_counter()

        cb_params = run_context.original_args()
        loss = cb_params.net_outputs[0]
        t = cb_params.network.network.network.transport.t
        lr = cb_params.network.optimizer.get_lr().item()
        global_step = cb_params.network.optimizer.global_step.item()

        bin_indices = ops.bucketize(t.to(dtype.float32), self.loss_bins, right=True) - 1  # MS doesn't support BF16

        # Iterate through each bin index to update occurrence and sum
        self.bin_occurrence[bin_indices] += 1
        self.bin_sum_loss[bin_indices] = self.bin_sum_loss[bin_indices] + loss  # Sum loss values in the i-th bin.

        if self.reduce is not None:
            loss = self.reduce(loss) / self.rank_size
            self.bin_occurrence = self.reduce(self.bin_occurrence)
            self.bin_sum_loss = self.reduce(self.bin_sum_loss)

        logs = {"Global step": global_step, "Step time": step_time, "lr": lr, "loss": loss.item()}
        for i in range(self.n_loss_bins):
            if self.bin_occurrence[i] > 0:
                logs[f"loss_bin#{i + 1}"] = self.bin_sum_loss[i].item()

        _logger.info(
            f"Step logs: {', '.join(f'{k}={v}'if isinstance(v, int) else f'{k}={v:.4f}' for k, v in logs.items())}"
        )


class VisualizationCallback(Callback):
    def __init__(self, output_dir: str, vae, text_tokenizer, frequency: int = 10):
        super().__init__()
        self._output_dir = output_dir
        self._vae = vae
        self._text_tokenizer = text_tokenizer
        self._frequency = frequency

    def on_train_step_end(self, run_context: RunContext):
        cb_params = run_context.original_args()
        input_images, output_images, text_input_ids, *_ = cb_params.train_dataset_element
        pred_images = cb_params.network.network.network.transport.pred_images
        t = cb_params.network.network.network.transport.t.to(dtype.float32).asnumpy()
        global_step = cb_params.network.optimizer.global_step.item()

        if global_step % self._frequency != 0:
            return

        if self._vae.config.scaling_factor is not None:
            pred_images = pred_images / self._vae.config.scaling_factor
        if self._vae.config.shift_factor is not None:
            pred_images = pred_images + self._vae.config.shift_factor
        pred_images = self._vae.decode(pred_images.to(dtype=self._vae.dtype), return_dict=False)

        pred_images = [pi.clamp(-1, 1).to(dtype.float32)[0] for pi in pred_images]
        output_images = output_images[:, 0].to(dtype.float32)
        if len(input_images[0]):
            input_images = input_images[:, :, 0].to(dtype.float32)

        for i in range(len(pred_images)):
            vis_images = [output_images[i], pred_images[i]]
            if len(input_images[0]):
                vis_images = [im for im in input_images[i]] + vis_images

            # Concatenate input images of different sizes horizontally
            max_height = max(img.shape[-2] for img in vis_images)
            total_width = sum(img.shape[-1] for img in vis_images)
            canvas = mint.zeros((3, max_height, total_width), dtype=dtype.float32)

            current_x = 0
            for img in vis_images:
                h, w = img.shape[-2:]
                # Place image at the top of canvas
                canvas[:, :h, current_x : current_x + w] = img
                current_x += w
            canvas = canvas * 0.5 + 0.5

            to_pil_image(canvas).save(
                os.path.join(self._output_dir, f"input_visualization_{global_step}_{i}_t{t[i]}.png")
            )

            input_ids = text_input_ids[i]
            instruction = self._text_tokenizer.decode(input_ids, skip_special_tokens=False)

            with open(os.path.join(self._output_dir, f"instruction_{global_step}_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(f"token len: {len(input_ids)}\ntext: {instruction}")
