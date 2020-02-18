import torch.nn as nn
import numpy as np


class Decoder(nn.Module):
    """
    Decodes the latent representation back to waveforms
    """
    def __init__(self, N, kernel_size, stride, layers):
        """
        Arguments:
            N {int} -- Dimension of the input latent representation
            kernel_size {int} -- Base convolutional kernel size
            stride {int} -- Stride of the transposed covolutions
            layers {int} -- Number of parallel convolutions with different kernel sizes
        """
        super(Decoder, self).__init__()

        self.filter_widths = [N // (2**(l+1)) for l in range(layers)]
        total_input_width = np.array(self.filter_widths).sum()

        self.bottleneck = nn.Sequential(
            nn.ConvTranspose1d(N, total_input_width, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )
        self.filters = nn.ModuleList([])
        for l in range(layers):
            n = N // (2**(l+1))
            k = kernel_size * (2**l)
            self.filters.append(nn.ConvTranspose1d(n, 1, kernel_size=k, stride=stride, bias=False, padding=(k-stride)//2))

    def forward(self, x):
        """
        Arguments:
            x {torch.tensor} -- Latent representation of the four instrument with shape (B*4, N, T')

        Returns:
            torch.tensor -- Signal of the four instruments with shape (B*4, 1, T)
        """
        x = self.bottleneck(x)  # shape: (B*4, N', T')

        output = 0.0
        x = x.split(self.filter_widths, dim=1)
        for i in range(len(x)):
            output += self.filters[i](x[i])  # shape: (B*4, 1, T)

        return output  # shape: (B*4, 1, T)
