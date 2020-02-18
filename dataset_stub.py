import numpy as np
import torch
from torch.utils.data import Dataset
from functools import partial
from torch.utils.data.dataloader import default_collate
from random import shuffle


class MusicDataset(Dataset):

    def __init__(self, base_path, base_sr_kHz, num_stages, sample_length, is_train: bool, shuffle_p=0.0, verbose=False):
        super(MusicDataset, self).__init__()

        self.base_sr_kHz = base_sr_kHz
        self.num_stages = num_stages
        self.base_length = sample_length*1000*self.base_sr_kHz

        self._shuffle_p = shuffle_p if is_train else 0.0
        self._sample_length = sample_length
        self._len = 10 if is_train else 10
        self._train = is_train

    def __getitem__(self, index):
        if self._train:
            return self.get_train_sample()
        else:
            return self.get_validation_sample(index)

    def get_train_sample(self):
        tracks = [np.random.randn(2, 4, 2 * self.base_length * (2**s)).astype('float32') for s in range(self.num_stages)]

        start_t = torch.randint(0, tracks[0].shape[2] - self.base_length, (1,)).item()
        tracks = [track[:, :, (2**i)*start_t: (2**i)*(start_t + self.base_length)].copy() for i, track in enumerate(tracks)]

        tracks = [self.random_channels(track).transpose(1, 0, 2) for track in tracks]

        tracks = [self.random_amp(track) for track in tracks]
        tracks = [(mix, separated, (separated != 0).any(-1).astype('float32')) for mix, separated in tracks]
        tracks = [(torch.from_numpy(mix), torch.from_numpy(separated), torch.from_numpy(mask)) for mix, separated, mask in tracks]

        mix, separated, mask = tuple(zip(*tracks))
        return mix, separated, mask

    def get_validation_sample(self, index):
        tracks = [np.random.randn(1, 5, (13 * self.base_length + 111) * (2**s)).astype('float32') for s in range(self.num_stages)]
        tracks = [(track[:, 0, :], track[:, 1:, :].transpose(1, 0, 2)) for track in tracks]
        tracks = [(torch.from_numpy(mix), torch.from_numpy(separated)) for mix, separated in tracks]

        mix, separated = tuple(zip(*tracks))
        return mix, separated

    def random_channel_swap(self, track):
        if self.random_uniform(0.0, 1.0, (1,))[0] > 0.5:
            track = np.flip(track, axis=1).copy()
        return track

    def random_amp(self, separated):
        separated *= self.random_uniform(0.75, 1.25, (4, separated.shape[1], 1)).astype('float32')
        mix = separated.sum(0)  # shape: (2, T) or (1, T)

        return mix, separated

    def random_channels(self, track):
        channels = torch.randint(0, 2, (4,)).numpy()
        separated = [track[c:c+1, i:i+1, :] for i, c in enumerate(channels)]  # drums, bass, other, vocals

        return np.concatenate(separated, 1)  # shape: (1, C, T)

    def random_uniform(self, low, high, size):
        r = torch.rand(size).numpy()
        return low + r * (high - low)

    def __len__(self):
        return self._len

    def get_collate_fn(self):
        if self._shuffle_p == 0.0:
            return default_collate
        return partial(MusicDataset._shuffle_collate_fn, self._shuffle_p)

    @staticmethod
    def _shuffle_collate_fn(shuffle_p, batch):
        num_stages = len(batch[0][0])
        portion = int(len(batch) * shuffle_p + 0.5)

        alternative_batch = []
        for i in range(portion):
            separated = [
                torch.cat([
                    batch[i][1][s][0:1, :, :],
                    batch[(i+1) % portion][1][s][1:2, :, :],
                    batch[(i+2) % portion][1][s][2:3, :, :],
                    batch[(i+3) % portion][1][s][3:4, :, :]
                ], 0) for s in range(num_stages)
            ]
            mix = [s.sum(0) for s in separated]
            mask = [(s != 0.0).any(-1).float() for s in separated]

            alternative_batch.append((mix, separated, mask))

        batch = alternative_batch + batch[portion:]
        shuffle(batch)

        return default_collate(batch)
