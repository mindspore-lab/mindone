import mindspore as ms
from mindspore import nn, ops
from models.audio2pose.cvae import CVAE
from models.audio2pose.discriminator import PoseSequenceDiscriminator
from models.audio2pose.audio_encoder import AudioEncoder


class Audio2Pose(nn.Cell):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.model.cvae.seq_len
        self.latent_dim = cfg.model.cvae.latent_size

        self.audio_encoder = AudioEncoder()
        self.audio_encoder.set_train(False)
        for param in self.audio_encoder.get_parameters():
            param.requires_grad = False

        self.netG = CVAE(cfg)
        self.netD_motion = PoseSequenceDiscriminator(cfg)

    def construct(self, x):

        batch = {}
        coeff_gt = x['gt'].squeeze(0)  # bs frame_len+1 73
        batch['pose_motion_gt'] = coeff_gt[:, 1:, 64:70] - \
            coeff_gt[:, :1, 64:70]  # bs frame_len 6
        batch['ref'] = coeff_gt[:, 0, 64:70]  # bs  6
        batch['class'] = x['class'].squeeze(0)  # bs
        indiv_mels = x['indiv_mels'].squeeze(0)  # bs seq_len+1 80 16

        # forward
        audio_emb = self.audio_encoder(
            indiv_mels[:, 1:, :, :].unsqueeze(2))  # bs seq_len 512
        batch['audio_emb'] = audio_emb
        batch = self.netG(batch)

        pose_motion_pred = batch['pose_motion_pred']           # bs frame_len 6
        pose_gt = coeff_gt[:, 1:, 64:70].clone()               # bs frame_len 6
        pose_pred = coeff_gt[:, :1, 64:70] + pose_motion_pred  # bs frame_len 6

        batch['pose_pred'] = pose_pred
        batch['pose_gt'] = pose_gt

        return batch

    def test(self, x):

        batch = {}
        ref = x['ref']  # bs 1 70
        batch['ref'] = x['ref'][:, 0, -6:]
        batch['class'] = x['class']
        bs = ref.shape[0]

        indiv_mels = x['indiv_mels']               # bs T 1 80 16
        # we regard the ref as the first frame
        indiv_mels_use = indiv_mels[:, 1:]
        num_frames = x['num_frames']
        num_frames = int(num_frames) - 1

        #
        div = num_frames // self.seq_len
        re = num_frames % self.seq_len

        pose_motion_pred_list = [
            ops.zeros(batch['ref'].unsqueeze(1).shape, dtype=batch['ref'].dtype)]

        for i in range(div):
            z = ops.randn(bs, self.latent_dim)
            # z = ops.zeros((bs, self.latent_dim), ms.float32) # for debug
            batch['z'] = z
            audio_emb = self.audio_encoder(
                indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len, :, :, :])  # bs seq_len 512
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(
                batch['pose_motion_pred'])  # list of bs seq_len 6

        if re != 0:
            z = ops.randn(bs, self.latent_dim)
            # z = ops.zeros((bs, self.latent_dim), ms.float32) # for debug
            batch['z'] = z
            audio_emb = self.audio_encoder(
                indiv_mels_use[:, -1*self.seq_len:, :, :, :])  # bs seq_len  512
            if audio_emb.shape[1] != self.seq_len:
                pad_dim = self.seq_len-audio_emb.shape[1]
                pad_audio_emb = audio_emb[:, :1].repeat(pad_dim, axis=1)
                audio_emb = ops.cat([pad_audio_emb, audio_emb], 1)
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(
                batch['pose_motion_pred'][:, -1*re:, :])

        pose_motion_pred = ops.cat(pose_motion_pred_list, axis=1)
        batch['pose_motion_pred'] = pose_motion_pred

        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # bs T 6

        batch['pose_pred'] = pose_pred

        return batch
