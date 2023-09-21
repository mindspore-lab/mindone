import audioldm.hifigan as hifigan

HIFIGAN_16K_64 = {
    "resblock": "1",
    "num_gpus": 6,
    "batch_size": 16,
    "learning_rate": 0.0002,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "seed": 1234,
    "upsample_rates": [5, 4, 2, 2, 2],
    "upsample_kernel_sizes": [16, 16, 8, 4, 4],
    "upsample_initial_channel": 1024,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "segment_size": 8192,
    "num_mels": 64,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 160,
    "win_size": 1024,
    "sampling_rate": 16000,
    "fmin": 0,
    "fmax": 8000,
    "fmax_for_loss": None,
    "num_workers": 4,
    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "world_size": 1,
    },
}


def get_vocoder(config=None, trainable=False):
    config = hifigan.AttrDict(HIFIGAN_16K_64)
    vocoder = hifigan.Generator(config)
    if not trainable:
        vocoder.set_train(False)
        for param in vocoder.get_parameters():
            param.requires_grad = False
    return vocoder


def vocoder_infer(mels, vocoder, lengths=None):
    wavs = vocoder(mels).squeeze(1)
    wavs = (wavs.asnumpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    return wavs
