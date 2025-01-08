import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class FlowHead(nn.Cell):
    def __init__(
        self,
        input_dim=128,
        hidden_dim=256,
        out_dim=2,
    ):
        super(FlowHead, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=1,
            has_bias=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=1,
            has_bias=True
        )
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        return out

class SepConvGRU(nn.Cell):
    def __init__(
        self,
        hidden_dim=128,
        input_dim=192 + 128,
        kernel_size=5,
    ):
        super(SepConvGRU, self).__init__()

        padding = (kernel_size - 1) // 2

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.padding = padding

        # Initialize convolution layers without padding
        self.convz1 = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, kernel_size),
            stride=1,
            pad_mode='pad',
            padding=0,
            has_bias=True
        )
        self.convr1 = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, kernel_size),
            stride=1,
            pad_mode='pad',
            padding=0,
            has_bias=True
        )
        self.convq1 = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, kernel_size),
            stride=1,
            pad_mode='pad',
            padding=0,
            has_bias=True
        )

        self.convz2 = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=(kernel_size, 1),
            stride=1,
            pad_mode='pad',
            padding=0,
            has_bias=True
        )
        self.convr2 = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=(kernel_size, 1),
            stride=1,
            pad_mode='pad',
            padding=0,
            has_bias=True
        )
        self.convq2 = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=(kernel_size, 1),
            stride=1,
            pad_mode='pad',
            padding=0,
            has_bias=True
        )

        # padding operations
        self.pad_horizontal = ops.Pad(((0, 0), (0, 0), (0, 0), (padding, padding)))  # padding on width
        self.pad_vertical = ops.Pad(((0, 0), (0, 0), (padding, padding), (0, 0)))  # padding on height

        # Activation functions
        self.sigmoid = ops.Sigmoid()
        self.tanh = ops.Tanh()

    def construct(self, h, x):
        # Horizontal pass
        hx = ops.concat((h, x), axis=1)
        hx_padded = self.pad_horizontal(hx)
        z = self.sigmoid(self.convz1(hx_padded))
        r = self.sigmoid(self.convr1(hx_padded))
        q = self.tanh(self.convq1(self.pad_horizontal(ops.concat((r * h, x), axis=1))))
        h = (1 - z) * h + z * q

        # Vertical pass
        hx = ops.concat((h, x), axis=1)
        hx_padded = self.pad_vertical(hx)
        z = self.sigmoid(self.convz2(hx_padded))
        r = self.sigmoid(self.convr2(hx_padded))
        q = self.tanh(self.convq2(self.pad_vertical(ops.concat((r * h, x), axis=1))))
        h = (1 - z) * h + z * q

        return h

class BasicMotionEncoder(nn.Cell):
    def __init__(
        self,
        corr_channels=324,
        flow_channels=2,
    ):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(
            in_channels=corr_channels,
            out_channels=256,
            kernel_size=1,
            stride=1,
            pad_mode='valid',  # No padding
            has_bias=True
        )
        self.convc2 = nn.Conv2d(
            in_channels=256,
            out_channels=192,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=1,
            has_bias=True
        )
        self.convf1 = nn.Conv2d(
            in_channels=flow_channels,
            out_channels=128,
            kernel_size=7,
            stride=1,
            pad_mode='pad',
            padding=3,
            has_bias=True
        )
        self.convf2 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=1,
            has_bias=True
        )
        self.conv = nn.Conv2d(
            in_channels=64 + 192,
            out_channels=128 - flow_channels,
            kernel_size=3,
            stride=1,
            pad_mode='pad',
            padding=1,
            has_bias=True
        )
        self.relu = nn.ReLU()

    def construct(self, flow, corr):
        cor = self.relu(self.convc1(corr))
        cor = self.relu(self.convc2(cor))
        flo = self.relu(self.convf1(flow))
        flo = self.relu(self.convf2(flo))

        cor_flo = ops.concat((cor, flo), axis=1)
        out = self.relu(self.conv(cor_flo))
        return ops.concat((out, flow), axis=1)

class BasicUpdateBlock(nn.Cell):
    def __init__(
            self,
            corr_channels=324,
            hidden_dim=128,
            context_dim=128,
            downsample_factor=8,
            flow_dim=2,
            bilinear_up=False,
    ):
        super(BasicUpdateBlock, self).__init__()

        self.encoder = BasicMotionEncoder(
            corr_channels=corr_channels,
            flow_channels=flow_dim,
        )

        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=context_dim + hidden_dim)

        self.flow_head = FlowHead(
            input_dim=hidden_dim,
            hidden_dim=256,
            out_dim=flow_dim,
        )

        if bilinear_up:
            self.mask = None
        else:
            self.mask = nn.SequentialCell(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    pad_mode='pad',
                    padding=1,
                    has_bias=True
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=256,
                    out_channels=downsample_factor ** 2 * 9,
                    kernel_size=1,
                    stride=1,
                    pad_mode='valid',
                    has_bias=True
                ),
            )

    def construct(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)

        inp = ops.concat((inp, motion_features), axis=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        if self.mask is not None:
            mask = self.mask(net)
        else:
            mask = None

        return net, mask, delta_flow