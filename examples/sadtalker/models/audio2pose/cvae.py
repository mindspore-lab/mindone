import numpy as np
import mindspore as ms
from mindspore import nn, ops
from models.audio2pose.resunet import ResUNet


class CVAE(nn.Cell):
    def __init__(self, cfg):
        super().__init__()
        encoder_layer_sizes = cfg.model.cvae.encoder_layer_sizes
        decoder_layer_sizes = cfg.model.cvae.decoder_layer_sizes
        latent_size = cfg.model.cvae.latent_size
        num_classes = cfg.dataset.num_classes
        audio_emb_in_size = cfg.model.cvae.audio_emb_in_size
        audio_emb_out_size = cfg.model.cvae.audio_emb_out_size
        seq_len = cfg.model.cvae.seq_len

        self.latent_size = latent_size

        self.encoder = Encoder(encoder_layer_sizes, latent_size, num_classes,
                               audio_emb_in_size, audio_emb_out_size, seq_len)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, num_classes,
                               audio_emb_in_size, audio_emb_out_size, seq_len)

    def reparameterize(self, mu, logvar):
        std = ops.Exp()(0.5 * logvar)
        eps = ops.randn_like(std)
        return mu + eps * std

    def construct(self, batch):
        batch = self.encoder(batch)
        mu = batch['mu']
        logvar = batch['logvar']
        z = self.reparameterize(mu, logvar)
        batch['z'] = z
        return self.decoder(batch)

    def test(self, batch):
        '''
        class_id = batch['class']
        z = np.random.randn([class_id.size(0), self.latent_size])
        batch['z'] = z
        '''
        return self.decoder(batch)


class Encoder(nn.Cell):
    def __init__(self, layer_sizes, latent_size, num_classes,
                 audio_emb_in_size, audio_emb_out_size, seq_len):
        super().__init__()

        self.resunet = ResUNet()
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.mlp = nn.SequentialCell()
        layer_sizes[0] += latent_size + seq_len * audio_emb_out_size + 6
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.mlp.append(nn.Dense(in_size, out_size))
            self.mlp.append(nn.ReLU())

        self.linear_means = nn.Dense(layer_sizes[-1], latent_size)
        self.linear_logvar = nn.Dense(layer_sizes[-1], latent_size)
        self.linear_audio = nn.Dense(audio_emb_in_size, audio_emb_out_size)

        self.classbias = ms.Parameter(
            np.random.randn(self.num_classes, latent_size))

    def construct(self, batch):
        class_id = batch['class']
        pose_motion_gt = batch['pose_motion_gt']  # bs seq_len 6
        ref = batch['ref']  # bs 6
        bs = pose_motion_gt.shape[0]
        # bs seq_len audio_emb_in_size
        audio_in = batch['audio_emb']

        # pose encode
        pose_emb = self.resunet(pose_motion_gt.unsqueeze(1))  # bs 1 seq_len 6
        pose_emb = pose_emb.reshape(bs, -1)  # bs seq_len*6

        # audio mapping
        # bs seq_len audio_emb_out_size
        audio_out = self.linear_audio(audio_in)
        audio_out = audio_out.reshape(bs, -1)

        class_bias = self.classbias[class_id]  # bs latent_size
        # bs seq_len*(audio_emb_out_size+6)+latent_size
        x_in = ops.cat([ref, pose_emb, audio_out, class_bias], axis=-1)
        x_out = self.mlp(x_in)

        mu = self.linear_means(x_out)
        logvar = self.linear_means(x_out)  # bs latent_size

        batch.update({'mu': mu, 'logvar': logvar})
        return batch


class Decoder(nn.Cell):
    def __init__(self, layer_sizes, latent_size, num_classes,
                 audio_emb_in_size, audio_emb_out_size, seq_len):
        super().__init__()

        self.resunet = ResUNet()
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.mlp = nn.SequentialCell()
        input_size = latent_size + seq_len * audio_emb_out_size + 6
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.mlp.append(nn.Dense(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.mlp.append(nn.ReLU())
            else:
                self.mlp.append(nn.Sigmoid())

        self.pose_linear = nn.Dense(6, 6)
        self.linear_audio = nn.Dense(audio_emb_in_size, audio_emb_out_size)

        self.classbias = ms.Parameter(
            np.random.randn(self.num_classes, latent_size))

    def construct(self, batch):

        z = batch['z']  # bs latent_size
        bs = z.shape[0]
        class_id = batch['class']
        ref = batch['ref']  # bs 6
        # bs seq_len audio_emb_in_size
        audio_in = batch['audio_emb']
        # print('audio_in: ', audio_in[:, :, :10])

        # bs seq_len audio_emb_out_size
        audio_out = self.linear_audio(audio_in)
        # print('audio_out: ', audio_out[:, :, :10])
        # bs seq_len*audio_emb_out_size
        audio_out = audio_out.reshape([bs, -1])
        class_bias = self.classbias[class_id]  # bs latent_size

        z = z + class_bias
        z = ops.Cast()(z, ref.dtype)
        x_in = ops.cat([ref, z, audio_out], axis=-1)
        # bs layer_sizes[-1]
        x_out = self.mlp(x_in)
        x_out = x_out.reshape((bs, self.seq_len, -1))

        pose_emb = self.resunet(x_out.unsqueeze(1))  # bs 1 seq_len 6

        pose_motion_pred = self.pose_linear(
            pose_emb.squeeze(1))  # bs seq_len 6

        batch.update({'pose_motion_pred': pose_motion_pred})
        return batch
