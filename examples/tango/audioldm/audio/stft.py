import numpy as np
from audioldm.audio.audio_processing import dynamic_range_compression, dynamic_range_decompression, window_sumsquare
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center, tiny
from scipy.signal import get_window

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class STFT(nn.Cell):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        dtype = ms.float32
        forward_basis = ms.Tensor(fourier_basis[:, None, :], dtype=dtype)
        inverse_basis = ms.Tensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :], dtype=dtype)

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = ms.Tensor.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.forward_basis = forward_basis.float()
        self.inverse_basis = inverse_basis.float()

    def transform(self, input_data):
        num_batches = input_data.shape(0)
        num_samples = input_data.shape(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, 1, num_samples)
        input_data = ops.pad(
            input_data,
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = ops.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0,
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = ops.sqrt(real_part**2 + imag_part**2)
        phase = ops.atan2(imag_part, real_part)

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = ops.cat([magnitude * ops.cos(phase), magnitude * ops.sin(phase)], dim=1)

        inverse_transform = ops.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = ms.Tensor.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = ms.Tensor.from_numpy(window_sum)
            window_sum = window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(nn.Cell):
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = ms.Tensor.from_numpy(mel_basis).float()
        self.mel_basis = mel_basis

    def spectral_normalize(self, magnitudes, normalize_fun):
        output = dynamic_range_compression(magnitudes, normalize_fun)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y, normalize_fun=ops.log):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(Float, Tensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: Float, Tensor of shape (B, n_mel_channels, T)
        """
        assert ops.min(y) >= -1, ops.min(y)
        assert ops.max(y) <= 1, ops.max(y)

        magnitudes, phases = self.stft_fn.transform(y)
        mel_output = ops.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output, normalize_fun)
        energy = ops.norm(magnitudes, dim=1)

        log_magnitudes = self.spectral_normalize(magnitudes, normalize_fun)

        return mel_output, log_magnitudes, energy
