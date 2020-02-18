import torch
import torch.nn as nn
import torch.nn.functional as F


class Spectrogram(nn.Module):
    """
    Calculate the mel spectrogram as an additional input for the encoder
    """
    def __init__(self, n_fft, hop, mels, sr):
        """
        Arguments:
            n_fft {int} -- The number fo frequency bins
            hop {int} -- Hop size (stride)
            mels {int} -- The number of mel filters
            sr {int} -- Sampling rate of the signal
        """
        super(Spectrogram, self).__init__()

        self.n_fft = n_fft
        self.hop = hop
        self.mels = mels
        self.sr = sr

        # Hann window for STFT
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

        # learnable mel transform
        stft_size = n_fft // 2 + 1
        self.mel_transform = nn.Conv1d(stft_size, mels, kernel_size=1, stride=1, padding=0, bias=True)

        # statistics for normalization
        self.mean = nn.Parameter(torch.empty(1, stft_size, 1), requires_grad=False)
        self.std = nn.Parameter(torch.empty(1, stft_size, 1), requires_grad=False)

        # affine transform after normalization
        self.affine_bias = nn.Parameter(torch.zeros(1, stft_size, 1), requires_grad=True)
        self.affine_scale = nn.Parameter(torch.ones(1, stft_size, 1), requires_grad=True)

    def forward(self, audio_signal, target_length=None):
        """
        Arguments:
            audio_signal {torch.tensor} -- input tensor of shape (B, 1, T)

        Keyword Arguments:
            target_length {int, None} -- Optional argument for interpolating the time dimension of the result to $target_length (default: {None})

        Returns:
            torch.tensor -- mel spectrogram of shape (B, mels, T')
        """
        mag = self.calculate_mag(audio_signal, db_conversion=True)  # shape: (B, N', T')
        mag = (mag - self.mean) / self.std  # shape: (B, N', T')
        mag = mag*self.affine_scale + self.affine_bias  # shape: (B, N', T')
        mag = self.mel_transform(mag)  # shape: (B, N, T')

        if target_length is not None:
            mag = F.interpolate(mag, size=target_length, mode='linear', align_corners=True)  # shape: (B, N, T'')

        return mag  # shape: (B, N, T'')

    def calculate_mag(self, signal, db_conversion=True):
        """
        Calculate the dB magnitude of the STFT of the input signal

        Arguments:
            audio_signal {torch.tensor} -- input tensor of shape (B, 1, T)

        Keyword Arguments:
            db_conversion {bool} -- True if the method should logaritmically transform the result to dB (default: {True})

        Returns:
            torch.tensor -- output tensor of shape (B, N', T')
        """
        signal = signal.view(-1, signal.shape[-1])  # shape (B, T)

        stft = torch.stft(
            signal,
            n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True,
            normalized=False, onesided=True, pad_mode='reflect'
        )  # shape: (B, N', T', 2)

        mag = (stft ** 2).sum(-1)  # shape: (B, N', T')
        if db_conversion:
            mag = torch.log10(mag + 1e-8)  # shape: (B, N', T')

        return mag  # shape: (B, N', T')

    def compute_stats(self, dataset, portion):
        """
        Calculate the mean and std statistics of the dataset

        Arguments:
            dataset {MusicDataset} -- MusicDataset class
            portion {int from {0,1,2}} -- Used to select data with only one value of the sampling rate
        """
        with torch.no_grad():
            specgrams = []
            samples = 5000  # randomly sample the statistics from the dataset
            for i_batch, (mix, _, _) in enumerate(dataset):
                mix = mix[portion]

                spec = self.calculate_mag(mix.to(self.window.device), db_conversion=True)
                specgrams.append(spec)

                if (i_batch + 1) * mix.shape[0] > samples:
                    break

            specgrams = torch.cat(specgrams, 0)

            self.mean.data = specgrams.mean(dim=(0, 2), keepdim=True)
            self.std.data = specgrams.std(dim=(0, 2), keepdim=True)

        print("Mean and std for spectrogram transform computed")
