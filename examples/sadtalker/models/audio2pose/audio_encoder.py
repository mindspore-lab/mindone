from mindspore import nn, ops


class Conv2d(nn.Cell):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.SequentialCell(
            nn.Conv2d(cin, cout, kernel_size, stride,
                      pad_mode='pad', padding=padding, has_bias=True),
            nn.BatchNorm2d(cout, momentum=0.9)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def construct(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Cell):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.SequentialCell(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def construct(self, audio_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.shape[0]

        audio_sequences = ops.cat([audio_sequences[:, i]
                                  for i in range(audio_sequences.shape[1])], axis=0)
        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1
        dim = audio_embedding.shape[1]
        audio_embedding = audio_embedding.reshape((B, -1, dim, 1, 1))

        return audio_embedding.squeeze(-1).squeeze(-1)  # B seq_len+1 512
