import musdb
import librosa
import numpy as np
import os
import argparse


"""
Creates the dataset used for training by converting the MUSDB18 stems to the right sampling rates and
saves it as numpy arrays
"""

def process(directory, sources, target_sr, save_only_mono=False):
    for track_i, track in enumerate(sources):
        original_sr = track.rate

        mix = librosa.core.resample(track.audio.T, original_sr, target_sr)
        drums = librosa.core.resample(track.targets['drums'].audio.T, original_sr, target_sr)
        bass = librosa.core.resample(track.targets['bass'].audio.T, original_sr, target_sr)
        other = librosa.core.resample(track.targets['other'].audio.T, original_sr, target_sr)
        vocal = librosa.core.resample(track.targets['vocals'].audio.T, original_sr, target_sr)
        acc = librosa.core.resample(track.targets['accompaniment'].audio.T, original_sr, target_sr)

        stereo = [mix, drums, bass, other, vocal, acc]
        length = min([t.shape[1] for t in stereo])
        if length <= 1: continue

        left = np.array([t[0, :length] for t in stereo])
        right = np.array([t[1, :length] for t in stereo])
        mono = np.array([librosa.to_mono(t[:, :length]) for t in stereo])

        if save_only_mono:
            together = mono
        else:
            together = np.array([left, right, mono])

        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savez_compressed(f'{directory}/{track_i:04d}', together.astype('float32'))

        print(f"Track: {track_i}, sampling rate: {target_sr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--musdb_path", required=True, type=str, help="Path to the MUSDB18 dataset.")
    args = parser.parse_args()

    mus_train = musdb.DB(root=args.musdb_path, subsets="train", split="train")
    mus_val = musdb.DB(root=args.musdb_path, subsets="train", split="valid")

    print(f"The training set size: {len(mus_train)}")
    print(f"The validation set size: {len(mus_val)}\n")

    for sample_rate in [8, 16, 32]:
        print("Converting the training set...")
        process(f"../data/train_{sample_rate}", mus_train, sample_rate*1000, save_only_mono=False)
        print("converting the validation set...")
        process(f"../data/validation_{sample_rate}", mus_val, sample_rate*1000, save_only_mono=False)
        print()
