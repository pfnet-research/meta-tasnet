import torch
import torch.nn as nn

from model.spectrogram import Spectrogram


class Encoder(nn.Module):
    """
    Encodes the waveforms into the latent representation
    """
    def __init__(self, N, kernel_size, stride, layers, num_mels, sampling_rate):
        """
        Arguments:
            N {int} -- Dimension of the output latent representation
            kernel_size {int} -- Base convolutional kernel size
            stride {int} -- Stride of the convolutions
            layers {int} -- Number of parallel convolutions with different kernel sizes
            num_mels {int} -- Number of mel filters in the mel spectrogram
            sampling_rate {int} -- Sampling rate of the input
        """
        super(Encoder, self).__init__()

        K = sampling_rate//8000
        self.spectrogram = Spectrogram(n_fft=1024*K, hop=256*K, mels=num_mels, sr=sampling_rate)

        self.filters = nn.ModuleList([])
        filter_width = num_mels
        for l in range(layers):
            n = N // 4
            k = kernel_size * (2**l)
            self.filters.append(nn.Conv1d(1, n, kernel_size=k, stride=stride, bias=False, padding=(k-stride)//2))
            filter_width += n

        self.nonlinearity = nn.ReLU()
        self.bottleneck = nn.Sequential(
            nn.Conv1d(filter_width, N, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(N, N, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, signal):
        """
        Arguments:
            signal {torch.tensor} -- mixed signal of shape (B, 1, T)

        Returns:
            torch.tensor -- latent representation of shape (B, N, T)
        """
        convoluted_x = []
        for filter in self.filters:
            x = filter(signal).unsqueeze(-2)  # shape: (B, N^, 1, T')
            convoluted_x.append(x)

        x = torch.cat(convoluted_x, dim=-2)  # shape: (B, N^, L, T')
        x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])  # shape: (B, N', T')
        x = self.nonlinearity(x)  # shape: (B, N', T')

        spectrogram = self.spectrogram(signal, x.shape[-1])  # shape: (B, mel, T')
        x = torch.cat([x, spectrogram], dim=1)  # shape: (B, N*, T')

        return self.bottleneck(x)  # shape: (B, N, T)
