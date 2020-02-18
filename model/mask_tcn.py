import torch
import torch.nn as nn
import torch.nn.functional as F

from model.conv1d import Conv1dWrapper
from model.group_norm import GroupNormWrapper


class TCNLayer(nn.Module):
    """
    One layer of the dilated temporal convolution with bottleneck
    """
    def __init__(self, generated, E_1, E_2, B, H, Sc, kernel, residual_bias, padding, dilation=1):
        """
        Arguments:
            generated {bool} -- True if you want to use the generated weights
            E_1 {int} -- Dimension of the instrument embedding
            E_2 {int} -- Dimension of the instrument embedding bottleneck
            B {int} -- Dimension of the bottleneck convolution
            H {int} -- Hidden dimension
            Sc {int} -- Skip-connection dimension
            kernel {int} -- Kernel size of the dilated convolution
            residual_bias {bool} -- True if you want to apply bias to the residual and skip connections
            padding {int} -- Padding of the dilated convolution

        Keyword Arguments:
            dilation {int} -- Dilation of the dilated convolution (default: {1})
        """
        super(TCNLayer, self).__init__()

        self.norm_1 = GroupNormWrapper(generated, E_1, E_2, 8, H, eps=1e-08)
        self.prelu_1 = nn.PReLU()
        self.conv1d = Conv1dWrapper(generated, E_1, E_2, B, H, 1, bias=False)

        self.norm_2 = GroupNormWrapper(generated, E_1, E_2, 8, H, eps=1e-08)
        self.prelu_2 = nn.PReLU()
        self.dconv1d = Conv1dWrapper(generated, E_1, E_2, H, H, kernel, dilation=dilation, groups=H, padding=padding, bias=False)

        self.res_out = Conv1dWrapper(generated, E_1, E_2, H, B, 1, bias=residual_bias)
        self.skip_out = Conv1dWrapper(generated, E_1, E_2, H, Sc, 1, bias=residual_bias)

    def forward(self, instrument, x):
        """
        Arguments:
            instrument {torch.tensor} -- Instrument embedding of shape (4, E_1)
            x {torch.tensor} -- Input of the module of shape (B, 4, B, T)

        Returns:
            (torch.tensor, torch.tensor) -- Output of the module of shape [(B, 4, B, T), (B, 4, Sc, T)]
        """
        x = self.norm_1(instrument, self.prelu_1(self.conv1d(instrument, x)))  # shape: (B, 4, H, T)
        x = self.norm_2(instrument, self.prelu_2(self.dconv1d(instrument, x)))  # shape: (B, 4, H, T)

        residual = self.res_out(instrument, x)  # shape: (B, 4, B, T)
        skip = self.skip_out(instrument, x)  # shape: (B, 4, Sc, T)

        return residual, skip  # shape: [(B, 4, B, T), (B, 4, Sc, T)]


class MaskingModule(nn.Module):
    """
    Creates a [0,1] mask of the four instruments on the latent matrix
    """
    def __init__(self, generated, E_1, E_2, N, B, H, layer, stack, kernel=3, residual_bias=False, partial_input=False):
        """
        Arguments:
            generated {bool} -- True if you want to use the generated weights
            E_1 {int} -- Dimension of the instrument embedding
            E_2 {int} -- Dimension of the instrument embedding bottleneck
            N {int} -- Dimension of the latent matrix
            B {int} -- Dimension of the bottleneck convolution
            H {int} -- Hidden dimension
            layer {[type]} -- Number of temporal convolution layers in a stack
            stack {[type]} -- Number of stacks

        Keyword Arguments:
            kernel {int} -- Kernel size of the dilated convolution (default: {3})
            residual_bias {bool} -- True if you want to apply bias to the residual and skip connections (default: {False})
            partial_input {bool} -- True if the module expects input from the preceding masking module (default: {False})
        """
        super(MaskingModule, self).__init__()

        self.N = N
        self.in_N = (N + N // 2) if partial_input else N

        self.norm_1 = GroupNormWrapper(generated, E_1, E_2, 8, self.in_N, eps=1e-8)
        self.prelu_1 = nn.PReLU()
        self.in_conv = Conv1dWrapper(generated, E_1, E_2, self.in_N, B, 1, bias=False)
        self.norm_2 = GroupNormWrapper(generated, E_1, E_2, 8, B, eps=1e-8)
        self.prelu_2 = nn.PReLU()

        self.tcn = nn.ModuleList([TCNLayer(generated, E_1, E_2, B, H, B, kernel, residual_bias, dilation=2**i, padding=2**i) for _ in range(stack) for i in range(layer)])

        self.norm_3 = GroupNormWrapper(generated, E_1, E_2, 8, B, eps=1e-8)
        self.prelu_3 = nn.PReLU()
        self.mask_output = Conv1dWrapper(generated, E_1, E_2, B, N, 1, bias=False)
        self.norm_4 = GroupNormWrapper(generated, E_1, E_2, 8, N, eps=1e-8)
        self.prelu_4 = nn.PReLU()

    def forward(self, instrument, x, partial_input=None):
        """
        Arguments:
            instrument {torch.tensor} -- Instrument embedding of shape (4, E_1)
            x {torch.tensor} -- Latent representation of the mix of shape (B, 4, N, T) (expanded in the 2nd dimension)
            partial_input {torch.tensor, None} -- Optional input from the preceding masking module of shape (B, 4, N/2, T)

        Returns:
            torch.tensor -- [0,1] mask of shape (B, 4, N, T)
        """
        if partial_input is not None:
            x = torch.cat([x, partial_input], 2)  # shape: (B, 4, N+N/2, T)

        x = self.in_conv(instrument, self.norm_1(instrument, self.prelu_1(x)))  # shape: (B, 4, B, T)
        x = self.norm_2(instrument, self.prelu_2(x))  # shape: (B, 4, B, T)

        skip_connection = 0.0
        for layer in self.tcn:
            residual, skip = layer(instrument, x)  # shape: [(B, 4, B, T), (B, 4, B, T)]
            x = x + residual  # shape: (B, 4, B, T)
            skip_connection = skip_connection + skip  # shape: (B, 4, B, T)

        mask = self.mask_output(instrument, self.norm_3(instrument, self.prelu_3(skip_connection)))  # shape: (B, 4, N, T)
        mask = self.norm_4(instrument, self.prelu_4(mask))  # shape: (B, 4, N, T)
        return F.softmax(mask, dim=1)  # shape: (B, 4, N, T)
