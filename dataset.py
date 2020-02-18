import os
from functools import partial
from random import shuffle

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class MusicDataset(Dataset):

    def __init__(self, base_path: str, base_sr_kHz: int, num_stages: int, sample_length: int, is_train: bool, shuffle_p=0.0, verbose=False):
        super(MusicDataset, self).__init__()

        filenames = os.listdir(f"{base_path}_{base_sr_kHz}")  # list of files in the base-sampling-rate folder
        self._tracks = []  # the actual data loaded into RAM

        for i, f in enumerate(filenames):
            tracks = []

            # load data for each stage, data should be located in the corresponding folder -- like ".../train_16/" for 16kHz data
            for stage_i in range(num_stages):
                tracks.append(np.load(f"{base_path}_{(2**stage_i) * base_sr_kHz}/{f}")['arr_0'].astype('float32'))

            # shouldn't happen, but just for safety: check if the total size is larger than size of one training sample
            if is_train and any(track.shape[-1] < (2**stage_i)*1000*base_sr_kHz*sample_length for stage_i, track in enumerate(tracks)):
                continue

            # crop the data so that the lengths are multiples of each other for each song across stages
            # actually, this should only get rid of the odd samples
            min_len = min(t.shape[-1] // (2**i) for i, t in enumerate(tracks))

            if is_train:
                # normalize
                tracks = [track / np.std(track[:2, 1:-1, :].mean(0).sum(0), axis=-1) for track in tracks]
                # we only need the left/right channel and separated instruments for training
                tracks = [track[:2, 1:-1, :(2 ** i) * min_len].copy() for i, track in enumerate(tracks)]
            else:
                tracks = [track / np.std(track[2, 0, :], axis=-1) for track in tracks]
                tracks = [track[2:, :-1, :(2 ** i) * min_len].copy() for i, track in enumerate(tracks)]

            self._tracks.append(tracks)

            if verbose: print(f"\r{i + 1}/{len(filenames)} files loaded", end='')
        if verbose: print(f"\n{len(self._tracks)} tracks cached")

        self._shuffle_p = shuffle_p if is_train else 0.0
        self._sample_length = sample_length
        self._train = is_train
        self._base_sr_kHz = base_sr_kHz
        self._num_stages = num_stages
        self._base_length = sample_length * 1000 * self._base_sr_kHz  # length of one training sample in the base module

        # batches are sampled randomly so the length is very arbitrary for training
        self._len = 40 * 3000 // self._sample_length if is_train else len(self._tracks)

    def __getitem__(self, index):
        if self._train:
            return self.get_train_sample()
        else:
            return self.get_validation_sample(index)

    def get_train_sample(self):
        track_id = torch.randint(0, len(self._tracks), (1,)).item()
        tracks = self._tracks[track_id]

        start_t = torch.randint(0, tracks[0].shape[2] - self._base_length, (1,)).item()
        tracks = [track[:, :, (2**i)*start_t: (2**i)*(start_t + self._base_length)].copy() for i, track in enumerate(tracks)]
        tracks = [self.random_channels(track).transpose(1, 0, 2) for track in tracks]

        tracks = [self.random_amp(track) for track in tracks]
        tracks = [(mix, separated, (separated != 0).any(-1).astype('float32')) for mix, separated in tracks]
        tracks = [(torch.from_numpy(mix), torch.from_numpy(separated), torch.from_numpy(mask)) for mix, separated, mask in tracks]

        mix, separated, mask = tuple(zip(*tracks))
        return mix, separated, mask

    def get_validation_sample(self, index):
        tracks = self._tracks[index]
        tracks = [(track[:, 0, :], track[:, 1:, :].transpose(1, 0, 2)) for track in tracks]
        tracks = [(torch.from_numpy(mix), torch.from_numpy(separated)) for mix, separated in tracks]

        mix, separated = tuple(zip(*tracks))
        return mix, separated

    # randomly amplify each channel
    def random_amp(self, separated):
        separated *= self.random_uniform(0.75, 1.25, (4, separated.shape[1], 1)).astype('float32')
        mix = separated.sum(0)  # shape: (1, T)

        return mix, separated

    # randomly select left/right channel
    def random_channels(self, track):
        channels = torch.randint(0, 2, (4,)).numpy()
        separated = [track[c:c+1, i:i+1, :] for i, c in enumerate(channels)]  # drums, bass, other, vocals

        return np.concatenate(separated, 1)  # shape: (1, C, T)

    # randomly swap left/right channel
    def random_channel_swap(self, track):
        if self.random_uniform(0.0, 1.0, (1,))[0] > 0.5:
            track = np.flip(track, axis=1).copy()
        return track

    def random_uniform(self, low, high, size):
        r = torch.rand(size).numpy()
        return low + r * (high - low)

    def __len__(self):
        return self._len

    def get_collate_fn(self):
        if self._shuffle_p == 0.0:
            return default_collate
        return partial(MusicDataset._shuffle_collate_fn, self._shuffle_p)

    # shuffle channels (across tracks) in the $shuffle_p portion of the batch
    @staticmethod
    def _shuffle_collate_fn(shuffle_p: float, batch):
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
