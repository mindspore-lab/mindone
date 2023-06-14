'''
Train AutoEncoder-KL
'''
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
from mindspore.amp import StaticLossScaler, DynamicLossScaler, all_finite
from mindspore import SummaryCollector

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
hps.save_per_step = config_dict['data']['params']['save_per_step']
hps.max_grad_norm = 2.
hps.grad_clip = False

def parse_args():
    parser = argparse.ArgumentParser(description='first_stage_model training')
    parser.add_argument('--is_distributed', type=ast.literal_eval, default=False)
    parser.add_argument('--device_target', type=str, default="Ascend", choices=("GPU", "CPU", 'Ascend'))
    parser.add_argument('--device_id', '-i', type=int, default=0)
    parser.add_argument('--context_mode', type=str, default='py', choices=['py', 'graph'])
    parser.add_argument('--restore', '-r', type=str, default='saved.ckpt')
    parser.add_argument('--data_url', default='')
    parser.add_argument('--train_url', default='')
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


def calculate_adaptive_weight(nll_grads, g_grads):
    axis = (0, 1, 2, 3) # conv kernel
    d_weight = ops.norm(nll_grads, axis) / (ops.norm(g_grads, axis) + 1e-4)
    d_weight = (ms.ops.clip_by_value(d_weight, 0.0, 1e4))
    return d_weight


class MyTrainOneStepCell(nn.Cell):
    def __init__(
        self,
        network,
        optimizer,
        max_grad_norm=1., 
        grad_clip=True,
        is_distributed=False,
        adapt=False,
        scale_sense=ms.Tensor(1.),
    ):
        super().__init__()
        self.grad_clip = ms.Parameter(ms.Tensor(grad_clip))
        self.max_grad_norm = max_grad_norm
        self.network = network
        self.optimizer = optimizer
        self.grad = ops.grad(network, grad_position=None, weights=optimizer.parameters, has_aux=False)
        w = ms.ParameterTuple([network.first_stage_model.get_last_layer()])
        self.grad_nll = ops.grad(network, grad_position=None, weights=w, has_aux=False)

        self.grad_reducer = None
        self.is_distributed = is_distributed
        if is_distributed:
            mean = ms.context.get_auto_parallel_context('gradients_mean')
            degree = ms.context.get_auto_parallel_context('device_num')
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
            self.nll_grad_reducer = nn.DistributedGradReducer(w, mean, degree)
        # self.loss_scale = StaticLossScaler(1024)
        # self.loss_scale = DynamicLossScaler(4, scale_factor=2, scale_window=1000)

    def construct(self, x, global_step, G):
        nll_grads = self.grad_nll(x, global_step, 1., G, False, True, False)
        g_grads = self.grad_nll(x, global_step, 1., G, False, False, True)
        w = calculate_adaptive_weight(nll_grads[0], g_grads[0])
        grads = self.grad(x, global_step, w, G, True, False, False)
        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
        if self.is_distributed:
            grads = self.grad_reducer(grads)

        self.optimizer(grads)
        return self.network.loss


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

    np.random.seed(0)
    ms.set_seed(0)

    ds = create_openimage_dataset(
        split='test',
        folder='openimage_test',
        use_fp16=hps.params.use_fp16, 
        resolution=hps.params.ddconfig.resolution,
        batch_size=hps.batch_size // group,
        is_train=True,
        rank=rank,
        group_size=group
    )
    print('ds.get_dataset_size():', ds.get_dataset_size())
    lr = nn.exponential_decay_lr(
        config.model.base_learning_rate,
        0.96,
        int(ds.get_dataset_size() * hps.num_epochs),
        int(ds.get_dataset_size()),
        1000,
        is_stair=True
    )
    lr = config.model.base_learning_rate

    D = LPIPSWithDiscriminator(
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
    net_g = Loss_G(first_stage_model, D)

    optim_g = nn.Adam(first_stage_model.trainable_params(), learning_rate=lr)
    optim_d = nn.Adam(D.trainable_params(), learning_rate=lr)
    global_step = 0
    # args.restore = '/home/zhudongyao/stablediffusionv2_512.ckpt'
    if os.path.exists(args.restore):
        print('restore:', args.restore)
        t = ms.load_checkpoint(args.restore, first_stage_model, strict_load=False, specify_prefix='first_stage_model')
        # global_step = int(ms.load_checkpoint(args.restore.replace('model', 'optimiser'), optimiser, filter_prefix='learning_rate')['global_step'].asnumpy())

    if rank == 0:
        saver_g = SaveCallBack(
            first_stage_model,
            save_step=hps.save_per_step,
            save_dir='saved_g',
            global_step=global_step,
            optimiser=optim_g,
        )
        saver_d = SaveCallBack(
            D,
            save_step=hps.save_per_step,
            save_dir='saved_d',
            global_step=global_step,
            optimiser=optim_d,
        )

    train_g = MyTrainOneStepCell(
        net_g,
        optim_g,
        grad_clip=hps.grad_clip,
        max_grad_norm=hps.max_grad_norm,
        is_distributed=args.is_distributed,
        adapt=True
    )
    train_d = MyTrainOneStepCell(
        net_g,
        optim_d,
        grad_clip=hps.grad_clip,
        max_grad_norm=hps.max_grad_norm,
        is_distributed=args.is_distributed,
        adapt=False
    )
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
            loss = train_g(x, global_step=gs, G=True).asnumpy()
            if gs >= config_dict['model']['params']['lossconfig']['params']['disc_start']:
                adv_d_fake = train_d(x, global_step=gs, G=False).asnumpy()
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
        yh = (first_stage_model(x).asnumpy() * 255).astype(np.uint8)
        y = (x.asnumpy() * 255).astype(np.uint8)
        yh = Image.fromarray(np.concatenate([yh[0].transpose([1,2,0]), y[0].transpose([1,2,0])], axis=0))
        yh.save(f'ae_yh_{e}_{i}.png')

    np.save('total_loss_ms.npy', np.array(total_loss))
    np.save('total_recons_ms.npy', np.array(total_recons))
    np.save('total_perceptual_ms.npy', np.array(total_perceptual))
    np.save('total_kl_ms.npy', np.array(total_kl))
    np.save('total_adv_d_real_ms.npy', np.array(total_adv_d_real))
    np.save('total_adv_d_fake_ms.npy', np.array(total_adv_d_fake))


if __name__ == '__main__':
    main()
