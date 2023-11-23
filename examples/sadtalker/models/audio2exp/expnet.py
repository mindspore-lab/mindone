from mindspore import nn, ops


class Conv2d(nn.Cell):
    def __init__(self, cin, cout, kernel_size, stride, padding, use_residual=False, use_act=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.SequentialCell(
            nn.Conv2d(cin, cout, kernel_size, stride, pad_mode='pad',
                      padding=padding, has_bias=True),
            nn.BatchNorm2d(cout, momentum=0.9)
        )
        self.act = nn.ReLU()
        self.use_residual = use_residual
        self.use_act = use_act

    def construct(self, x):
        out = self.conv_block(x)
        if self.use_residual:
            out += x

        if self.use_act:
            return self.act(out)
        else:
            return out


class ExpNet(nn.Cell):
    """ ExpNet implementation (inference)
    """

    def __init__(self, wav2lip=None):
        super().__init__()
        self.audio_encoder = nn.SequentialCell(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1,
                 padding=1, use_residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1,
                 padding=1, use_residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1,
                 padding=1, use_residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1,
                 padding=1, use_residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1,
                 padding=1, use_residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                 padding=1, use_residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                 padding=1, use_residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.wav2lip = wav2lip
        self.mapping1 = nn.Dense(512+64+1, 64, bias_init="zeros")

    def construct(self, x, ref, ratio):

        x = self.audio_encoder(x).view(x.shape[0], -1)
        ref_reshape = ref.reshape(x.shape[0], -1)
        ratio = ratio.reshape(x.shape[0], -1)

        y = self.mapping1(ops.cat([x, ref_reshape, ratio], axis=1))
        out = y.reshape(ref.shape[0], ref.shape[1], -1)  # + ref # resudial
        return out
