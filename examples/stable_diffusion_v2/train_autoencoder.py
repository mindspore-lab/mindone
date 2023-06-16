import os
import numpy as np
import argparse
import ast
import time
from PIL import Image

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.train.callback import Callback
from mindspore.communication import init

from ldm.modules.train.tools import set_random_seed
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.discriminator.ae_kl_loss import *
from ldm.modules.discriminator import LPIPSWithDiscriminator

from ldm.data.dataset_openimage import create_openimage_dataset
from configs.autoencoder.hparams import load_hparams


config, config_dict = load_hparams('configs/autoencoder/autoencoder_kl_32x32x4.yaml')
hps_dict = config_dict['model']['params']['first_stage_config']
hps = config.model.params.first_stage_config
hps.batch_size = config_dict['data']['params']['batch_size']
hps.num_epochs = config_dict['data']['params']['num_epochs']
hps.save_every = config_dict['data']['params']['save_every']
hps.max_grad_norm = config.model.max_grad_norm
hps.grad_clip = config.model.grad_clip
hps.decay_epoch = config.model.decay_epoch

def parse_args():
    parser = argparse.ArgumentParser(description='first_stage_model training')
    parser.add_argument('--is_distributed', type=ast.literal_eval, default=False)
    parser.add_argument('--device_target', type=str, default="Ascend", choices=("GPU", "CPU", 'Ascend'))
    parser.add_argument('--device_id', '-i', type=int, default=0)
    parser.add_argument('--context_mode', type=str, default='py', choices=['py', 'graph'])
    parser.add_argument('--restore', '-r', type=str, default='saved.ckpt')
    args = parser.parse_args()
    return args


class SaveCallBack(Callback):
    def __init__(
        self,
        model,
        save_step,
        save_dir,
        global_step=None,
        optimiser=None,
        checkpoint_path=None,
        train_url=''
    ):
        super().__init__()
        self.save_step = save_step
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.optimiser = optimiser
        self.save_dir = save_dir
        self.global_step = global_step
        self.train_url = train_url
        os.makedirs(save_dir, exist_ok=True)

    def step_end(self, step):
        cur_step = step + self.global_step
        if cur_step % self.save_step != 0:
            return
        model_save_name = 'model'
        optimiser_save_name = 'optimiser'
        for module, name in zip([self.model, self.optimiser], [model_save_name, optimiser_save_name]):
            name = os.path.join(self.save_dir, name)
            ms.save_checkpoint(module, name + '_%d.ckpt' % cur_step, append_dict={'cur_step': cur_step})


_grad_scale = ops.MultitypeFuncGraph("grad_scale")

@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class AEKLTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    def __init__(self,
                 network,
                 optimizer,
                 max_grad_norm=1., 
                 scale_sense=ms.Tensor(1.),
                 grad_clip=True
    ):
        super().__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip
        self.max_grad_norm = max_grad_norm

    def construct(self, x, global_step=1, w=1, is_generator=True):
        loss = self.network(x, global_step, w, is_generator, True, False, False)

        status, scaling_sens = self.start_overflow_check(loss, self.scale_sense)

        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, self.weights)(x, global_step, w, is_generator, True, False, False, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        grads = self.grad_reducer(grads)
        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)

        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)

        return loss


def main():
    args = parse_args()

    mode = ms.context.PYNATIVE_MODE if args.context_mode == 'py' else ms.context.GRAPH_MODE
    if args.is_distributed:
        init()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    rank = int(os.getenv('DEVICE_ID', '0'))
    group = int(os.getenv('RANK_SIZE', '1')) if args.is_distributed else 1
    print('[info] rank: %d group: %d batch: %d' % (rank, group, hps.batch_size // group))
    ms.context.set_context(mode=mode, device_target=args.device_target, device_id=rank)

    set_random_seed(0)

    ds = create_openimage_dataset(
        metadata=config_dict['data']['params']['metadata'],
        data_folder=config_dict['data']['params']['data_folder'],
        use_fp16=hps.params.use_fp16, 
        resolution=hps.params.ddconfig.resolution,
        batch_size=hps.batch_size // group,
        rank=rank,
        group_size=group
    )

    lr = nn.exponential_decay_lr(
        config.model.base_learning_rate,
        0.96,
        int(ds.get_dataset_size() * hps.num_epochs),
        int(ds.get_dataset_size()),
        decay_epoch=hps.decay_epoch,
        is_stair=True
    )

    discriminator = LPIPSWithDiscriminator(
        use_fp16=config_dict['model']['params']['use_fp16'],
        **config_dict['model']['params']['lossconfig']['params']
    )

    first_stage_model = AutoencoderKL(
        ddconfig=hps_dict['params']['ddconfig'],
        embed_dim=hps.params.embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        use_fp16=hps.params.use_fp16,
    )
    net_g = AEKLLoss(first_stage_model, discriminator=discriminator)

    optimiser_g = nn.Adam(first_stage_model.trainable_params(), learning_rate=lr)
    optimiser_d = nn.Adam(discriminator.trainable_params(), learning_rate=lr)

    global_step = 0
    if os.path.exists(args.restore):
        print('[info] restore model from', args.restore)
        ms.load_checkpoint(args.restore, net_g, strict_load=True)
        print('[info] restore optimiser from', args.restore.replace('model', 'optimiser'))
        global_step = int(ms.load_checkpoint(args.restore.replace('model', 'optimiser'), optimiser_g, filter_prefix='learning_rate')['global_step'].asnumpy())

    if rank == 0:
        saver_g = SaveCallBack(
            net_g,
            save_step=hps.save_every,
            save_dir='saved_g',
            global_step=global_step,
            optimiser=optimiser_g,
        )
        saver_d = SaveCallBack(
            discriminator,
            save_step=hps.save_every,
            save_dir='saved_d',
            global_step=global_step,
            optimiser=optimiser_d,
        )

    scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
    train_g = AEKLTrainOneStepCell(net_g, optimiser_g, max_grad_norm=hps.max_grad_norm, scale_sense=scale_sense)
    train_d = AEKLTrainOneStepCell(net_g, optimiser_d, max_grad_norm=hps.max_grad_norm, scale_sense=scale_sense)

    total_loss = []
    total_recons = []
    total_perceptual = []
    total_kl = []
    total_adv_d_real = []
    total_adv_d_fake = []
    N = ds.get_dataset_size()
    for e in range(hps.num_epochs):
        for i, data in enumerate(ds.create_dict_iterator()):
            x = data['image']
            t = time.time()
            gs = e * N + i + 1
            w = 1

            loss = train_g(x, global_step=gs, w=w, is_generator=True).asnumpy()
            if gs >= config_dict['model']['params']['lossconfig']['params']['disc_start']:
                adv_d_fake = train_d(x, global_step=gs, w=w, is_generator=False).asnumpy()
            if not args.is_distributed or rank == 0:
                recons = net_g.recons.asnumpy()
                kl = net_g.kl.asnumpy()
                perceptual = net_g.perceptual.asnumpy()
                adv_d_real = net_g.adv_d_real.asnumpy()
                adv_d_fake = net_g.adv_d_fake.asnumpy()
                t = time.time() - t
                print(f'[RN {rank}] [EP {e} ST {gs}] [T {t:.2f}s] [L {loss:.3f}] [REC {recons:.3f}] [KL {kl:.3f}] [PER {perceptual:.3f}] [ADV-R {adv_d_real:.3f}] [ADV-F {adv_d_fake:.3f}]')

                total_loss.append(loss)
                total_recons.append(recons)
                total_perceptual.append(perceptual)
                total_kl.append(kl)
                total_adv_d_real.append(adv_d_real)
                total_adv_d_fake.append(adv_d_fake)

                saver_g.step_end(gs)
                saver_d.step_end(gs)

            if gs % hps.save_every == 0:
                yh = first_stage_model(x)
                yh = ms.ops.clip_by_value((yh + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
                yh = (yh.asnumpy() * 255).astype(np.uint8)
                x = ms.ops.clip_by_value((x + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
                x = (x.asnumpy() * 255).astype(np.uint8)
                yh = Image.fromarray(np.concatenate([yh[0].transpose([1,2,0]), x[0].transpose([1,2,0])], axis=0))
                yh.save(f'ae_yh_{e}_{i}.png')

                np.save(f'{gs}_total_loss_ms.npy', np.array(total_loss))
                np.save(f'{gs}_total_recons_ms.npy', np.array(total_recons))
                np.save(f'{gs}_total_perceptual_ms.npy', np.array(total_perceptual))
                np.save(f'{gs}_total_kl_ms.npy', np.array(total_kl))
                np.save(f'{gs}_total_adv_d_real_ms.npy', np.array(total_adv_d_real))
                np.save(f'{gs}_total_adv_d_fake_ms.npy', np.array(total_adv_d_fake))


if __name__ == '__main__':
    main()
