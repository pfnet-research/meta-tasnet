import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.mask_tcn import MaskingModule
from utility.loss import calculate_loss


class TasNet(nn.Module):
    """
    One stage of encoder->mask->decoder for a single sampling rate
    """
    def __init__(self, independent_params, N, L, W, B, H, sr, partial_input, args):
        """
        Arguments:
            independent_params {bool} -- False if you want to use the generated weights
            N {int} -- Dimension of the latent matrix
            L {int} -- Dimension of the latent representation
            W {int} -- Kernel size of the en/decoder transfomation
            B {int} -- Dimension of the bottleneck convolution in the masking subnetwork
            H {int} -- Hidden dimension of the masking subnetwork
            sr {int} -- Sampling rate of the processed signal
            partial_input {bool} -- True if the module should expect input from preceding stage
            args {dict} -- Other argparse hyperparameters
        """
        super(TasNet, self).__init__()

        assert sr*4 % L == 0
        self.N = N
        self.stride = W
        self.out_channels = 1
        self.C = 4

        self.encoder = Encoder(self.N, L, W, args.filters, args.num_mels, sr)
        self.decoder = Decoder(self.N, L, W, args.filters)

        self.dropout = nn.Dropout2d(args.dropout)
        self.mask = MaskingModule(not independent_params, args.E_1, args.E_2, N, B, H, args.layers, args.stack, args.kernel, args.residual_bias, partial_input=partial_input)
        self.instrument_embedding = nn.Embedding(self.C, args.E_1) if not independent_params and args.E_1 is not None else None

        self.args = args

    def forward(self, input_mix, separated_inputs, mask, partial_input=None):
        """
        Forward pass for training; returns the loss and hidden state to be passed to the next stage

        Arguments:
            input_mix {torch.tensor} -- Mixed signal of shape (B, 1, T)
            separated_inputs {torch.tensor} -- Ground truth separated mixed signal of shape (B, 4, 1, T)
            mask {torch.tensor} -- Boolean mask: True when $separated_inputs is 0.0; shape: (B, 4, 1)

        Keyword Arguments:
            partial_input {torch.tensor, None} -- Optional input from the preceding masking module of shape (B, 4, N/2, T') (default: {None})

        Returns:
            (torch.tensor, torch.tensor, torch.tensor) -- (
                the total loss of shape (),
                list of statistics with partial losses and metrics of shape (7),
                partial input to be passed to the next stage of shape (B, 4, N, T')
            )
        """
        batch_size = input_mix.shape[0]

        # waveform encoder
        mix_latent = self.encoder(input_mix)  # shape: (B, N, T')
        mix_latents = mix_latent.unsqueeze(1)  # shape: (B, 1, N, T')
        mix_latents = mix_latents.expand(-1, self.C, -1, -1).contiguous()  # shape: (B, 4, N, T')

        if self.args.similarity_loss_weight > 0.0 or self.args.dissimilarity_loss_weight > 0.0:
            separated_gold_latents = self.encoder(separated_inputs.view(self.C*batch_size, input_mix.shape[1], -1))  # shape: (B*4, N, T')
            separated_gold_latents = separated_gold_latents.view(batch_size, self.C, self.N, -1).permute(0, 1, 3, 2).contiguous()  # shape: (B, 1, T', N)
        else:
            separated_gold_latents = None

        instruments = torch.arange(0, self.C, device=mix_latent.device)  # shape: (4)
        if self.instrument_embedding is not None:
            instruments = self.instrument_embedding(instruments)  # shape: (4, E)

        # generate masks
        mask_input = self.dropout(mix_latents.view(batch_size*self.C, self.N, -1).unsqueeze(-1)).squeeze(-1).view(batch_size, self.C, self.N, -1)  # shape: (B, 4, N, T')
        masks = self.mask(instruments, mask_input, partial_input)  # shape: (B, 4, N, T')

        separated_latents = mix_latents * masks  # shape: (B, 4, N, T')

        # waveform decoder
        decoder_input = separated_latents.view(batch_size * self.C, self.N, -1)  # shape: (B*4, N, T')
        output_signal = self.decoder(decoder_input)  # shape: (B*4, channels, T)
        output_signal = output_signal.view(batch_size, self.C, self.out_channels, -1)  # shape: (B, 4, 1, T) [drums, bass, other, vocals]

        if self.args.reconstruction_loss_weight > 0:
            reconstruction = self.decoder(mix_latent)  # shape: (B, 1, T)
        else:
            reconstruction = None

        loss, stats = calculate_loss(output_signal, separated_inputs, mask, separated_gold_latents, reconstruction, input_mix, self.args)
        return loss, stats, separated_latents

    def inference(self, x, partial_input=None):
        """
        Forward pass for inference; returns the separated signal and hidden state to be passed to the next stage

        Arguments:
            x {torch.tensor} -- mixed signal of shape (1, 1, T)

        Keyword Arguments:
            partial_input {torch.tensor, None} -- Optional input from the preceding masking module of shape (B, 4, N/2, T') (default: {None})

        Returns:
            (torch.tensor, torch.tensor) -- (
                separated signal of shape (1, 4, 1, T),
                hidden state to be passed to the next stage of shape (1, 4, N, T')
            )
        """

        x = self.encoder(x)  # shape: (1, N, T')
        x = x.expand(self.C, -1, -1).unsqueeze_(0)  # shape: (1, 4, N, T')

        if partial_input is not None:
            mask_input = torch.cat([x, partial_input], 2)  # shape: (1, 4, N+N/2, T')
        else:
            mask_input = x  # shape: (1, 4, N, T')
        del partial_input

        instruments = torch.arange(0, self.C, device=x.device)  # shape: (4)
        if self.instrument_embedding is not None:
            instruments = self.instrument_embedding(instruments)  # shape: (4, E)

        masks = self.mask(instruments, mask_input)  # shape: (1, 4, N, T')
        del mask_input

        x = x * masks  # shape: (1, 4, N, T')
        del masks

        x.squeeze_(0)  # shape: (4, N, T')
        hidden = x

        x = self.decoder(x)  # shape: (4, 1, T)

        return x.unsqueeze_(0), hidden.unsqueeze_(0)  # shape: [(1, 4, 1, T), (1, 4, N, T')]


class MultiTasNet(nn.Module):
    """
    Multiple stages of Tasnet stacked sequentially
    """
    def __init__(self, args):
        """
        Arguments:
            args {dict} -- Other argparse hyperparameters
        """
        super(MultiTasNet, self).__init__()

        self.args = args
        self.W = args.W
        self.base_sr = args.sampling_rate
        self.stages_num = args.stages_num
        self.stages = nn.ModuleList([])
        for stage_i in range(self.stages_num):
            m = 2 ** stage_i
            stage = TasNet(args.independent_params, m*args.N, m*args.L, m*args.W, args.B, args.H, m*args.sampling_rate, partial_input=stage_i != 0, args=args)
            self.stages.append(stage)

    def forward(self, input_mixes, separated_inputs, masks):
        """
        Forward pass for training

        Arguments:
            input_mixes {[torch.tensor]} -- List of mixed signals for all stages of shape (B, 1, T)
            separated_inputs {[torch.tensor]} -- List of ground truth separated mixed signal of shape (B, 4, 1, T)
            masks {[torch.tensor]} -- List of boolean mask: True when $separated_inputs is 0.0; shape: (B, 4, 1)

        Returns:
            (torch.tensor, torch.tensor) -- (
                the total loss of shape (1),
                list of statistics with partial losses and metrics (15)
            )
        """
        assert len(input_mixes) == self.stages_num
        assert len(separated_inputs) == self.stages_num
        assert len(masks) == self.stages_num

        loss, stats, hidden = None, None, None
        for i, stage in enumerate(self.stages):
            _loss, _stats, hidden = stage(input_mixes[i], separated_inputs[i], masks[i], hidden)

            loss = _loss if loss is None else loss + _loss
            stats = _stats if stats is None else torch.cat([stats[:i*4], _stats], 0)

        stats.unsqueeze_(0)
        loss.unsqueeze_(0)

        return loss, stats

    def inference(self, input_audio, n_chunks=4):
        """
        Forward pass for inference; returns the separated signal

        Arguments:
            input_audio {torch.tensor} -- List of mixed signals for all stages of shape (B, 1, T)

        Keyword Arguments:
            n_chunks {int} -- Divide the $input_audio to chunks to trade speed for memory (default: {4})

        Returns:
            torch.tensor -- Separated signal of shape (1, 4, 1, T)
        """
        assert len(input_audio) == self.stages_num

        # split the input audio to $n_chunks and make sure they overlap to not lose the accuracy
        # $chunk_intervals contain the (start, end) times of all chunks in a list
        chunks = [int(input_audio[0].shape[-1] / n_chunks * c + 0.5) for c in range(n_chunks)]
        chunks.append(input_audio[0].shape[-1])
        chunk_intervals = [(max(0, chunks[n] - self.base_sr*8), min(chunks[n+1] + self.base_sr*8, input_audio[0].shape[-1])) for n in range(n_chunks)]
        chunk_intervals = [(s, e - ((e-s) % self.W)) if s == 0 else (s + (e-s) % self.W, e) for s, e in chunk_intervals]

        full_outputs = None
        for c in range(n_chunks):
            outputs, hidden = [], None
            for i, stage in enumerate(self.stages):
                m = 2**i
                output, hidden = stage.inference(input_audio[i][:, :, m*chunk_intervals[c][0]: m*chunk_intervals[c][1]], hidden)

                output = output[:, :, :, m*(chunks[c] - chunk_intervals[c][0]): output.shape[-1] - m*((chunk_intervals[c][1] - chunks[c+1]))]
                outputs.append(output)

            del hidden

            # concatenate the chunks togerther
            if full_outputs is None:
                full_outputs = outputs
            else:
                full_outputs = [torch.cat([f, o], -1) for f, o in zip(full_outputs, outputs)]

        return full_outputs

    def compute_stats(self, train_loader):
        """
        Calculate the mean and std statistics of the dataset for the spectrogram modules

        Arguments:
            train_loader {MusicDataset}
        """
        for i, stage in enumerate(self.stages):
            stage.encoder.spectrogram.compute_stats(train_loader, i)
