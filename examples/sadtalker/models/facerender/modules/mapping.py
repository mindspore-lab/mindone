from mindspore import nn


class MappingNet(nn.Cell):
    def __init__(self, coeff_nc, descriptor_nc, layer, num_kp, num_bins):
        super(MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.SequentialCell(
            nn.Conv1d(
                coeff_nc,
                descriptor_nc,
                kernel_size=7,
                pad_mode="pad",
                padding=0,
                has_bias=True,
            )
        )

        for i in range(layer):
            net = nn.SequentialCell(
                nonlinearity,
                nn.Conv1d(
                    descriptor_nc,
                    descriptor_nc,
                    kernel_size=3,
                    pad_mode="pad",
                    padding=0,
                    dilation=3,
                    has_bias=True,
                ),
            )
            setattr(self, "encoder" + str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

        self.fc_roll = nn.Dense(descriptor_nc, num_bins)
        self.fc_pitch = nn.Dense(descriptor_nc, num_bins)
        self.fc_yaw = nn.Dense(descriptor_nc, num_bins)
        self.fc_t = nn.Dense(descriptor_nc, 3)
        self.fc_exp = nn.Dense(descriptor_nc, 3 * num_kp)

    def construct(self, input_3dmm):
        out = self.first(input_3dmm)

        # for i in range(self.layer):
        #     model = getattr(self, "encoder" + str(i))
        #     out = model(out) + out[:, :, 3:-3]

        out = self.encoder0(out) + out[:, :, 3:-3]
        out = self.encoder1(out) + out[:, :, 3:-3]
        out = self.encoder2(out) + out[:, :, 3:-3]

        out = self.pooling(out)
        out = out.view(out.shape[0], -1)

        yaw = self.fc_yaw(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_roll(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)

        # return {"yaw": yaw, "pitch": pitch, "roll": roll, "t": t, "exp": exp}
        return (yaw, pitch, roll, t, exp)
