import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore import Model, nn
from mindspore.train.callback import LossMonitor 

import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.trainers.train_step import TrainOneStepWrapper

from opensora.models.layers.blocks import Mlp

net = nn.Dense(4, 8, has_bias=False) # Mlp(4, 16, 8)

bs = 1
train_x = np.random.randn(bs, 4) 
train_y = np.ones(shape=[bs, 8])
train_y1 = -1 * np.ones(shape=[bs, 4])  
train_y2 = np.ones(shape=[bs, 4])


class DumpyDataset():
    def __init__(self, d=4):
        self.d = d
        self.length = 100
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # train_x = np.random.randn(4).astype(np.float32) 
        # train_y = np.ones(shape=[8]).astype(np.float32)
        return train_x[0].astype(np.float32), train_y[0].astype(np.float32)

def get_dataloader():
    ds =  DumpyDataset()
    dataloader = ms.dataset.GeneratorDataset(
        source=ds,
        column_names=[
            "x",
            "y",
        ],
    )
    dl = dataloader.batch(
        bs,
    )
    return dl


class NetWithLoss(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
    '''
    def construct_1(self, x, y1, y2):
        pred = self.net(x) 
        pred1, pred2 = ops.split(pred, 4, axis=1)
        loss1 = ((ops.stop_gradient(pred1) + pred2 - y1)**2).mean()
        loss2 = ((pred1 - y2)**2).mean()

        return loss1 + loss2
    '''

    def construct(self, x, y):
        pred = self.net(x) 
        loss1 = ((ops.stop_gradient(pred) - y)**2).mean()
        loss2 =  (pred ** 2).mean()
        # print('loss1: ', loss)
        return loss1 + loss2

def test():
    ms.set_context(mode=0)

    net_with_loss = NetWithLoss(net)
    optimizer = nn.SGD(net.trainable_params(), 1e-2)

    train_step = TrainOneStepWrapper(
        net_with_loss,
        optimizer=optimizer,
        scale_sense=Tensor(1.0),
    )

    dl = get_dataloader()
    model = Model(train_step)
    model.train(
        10,
        dl,
        callbacks=[LossMonitor()],
    )

'''
    for i in range(100):
        ms_x = Tensor(train_x, dtype=ms.float32)
        ms_y = Tensor(train_y, dtype=ms.float32)

        loss, overflow, scaling_sens = train_step(ms_x, ms_y)

        print('loss: ', loss)
'''

if __name__ == '__main__':
    test()
