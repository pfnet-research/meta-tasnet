import argparse
import multiprocessing
import os
from pathlib import Path

import librosa
import musdb
import museval
import numpy as np
import pandas as pd
import simplejson
import torch
from pandas.io.json import json_normalize

from model.tasnet import MultiTasNet


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="Directory of the model to evaluate (in the './checkpoints' folder).")
    parser.add_argument("--musdb_path", required=True, type=str, help="Path to the MUSDB18 dataset.")
    parser.add_argument("--threads", default=4, type=int, help="Parallelize the evaluation to more threads.")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    checkpoint = torch.load(f"checkpoints/{args.model_dir}/best_checkpoint")
    model_args = checkpoint["args"]

    network = MultiTasNet(model_args).to(device)
    network.load_state_dict(checkpoint["state_dict"])

    mus_test = musdb.DB(root=args.musdb_path, subsets="test")


    def separate_sample(network, track, verbose=False):

        audio = track.audio.astype('float32').transpose(1, 0)
        mix = [librosa.core.resample(audio, 44100, s, res_type='kaiser_best', fix=False) for s in[8000, 16000, 32000]]
        mix = [librosa.util.fix_length(m, (mix[0].shape[-1]+1)*(2**i)) for i, m in enumerate(mix)]
        mix = [torch.from_numpy(s).float().to(device).unsqueeze_(1) for s in mix]
        mix = [s / s.std(dim=-1, keepdim=True) for s in mix]

        mix_left = [s[0:1, :, :] for s in mix]
        mix_right = [s[1:2, :, :] for s in mix]
        del mix

        network.eval()
        with torch.no_grad():
            separation_left = network.inference(mix_left, n_chunks=8)[-1].cpu().squeeze_(2)  # shape: (5, T)
            separation_right = network.inference(mix_right, n_chunks=8)[-1].cpu().squeeze_(2)  # shape: (5, T)

            separation = torch.cat([separation_left, separation_right], 0).numpy()

        if verbose: print(separation.shape)

        estimates = {
            'drums': librosa.core.resample(separation[:, 0, :], 32000, 44100, res_type='kaiser_best', fix=True)[:, :track.audio.shape[0]].T,
            'bass': librosa.core.resample(separation[:, 1, :], 32000, 44100, res_type='kaiser_best', fix=True)[:, :track.audio.shape[0]].T,
            'other': librosa.core.resample(separation[:, 2, :], 32000, 44100, res_type='kaiser_best', fix=True)[:, :track.audio.shape[0]].T,
            'vocals': librosa.core.resample(separation[:, 3, :], 32000, 44100, res_type='kaiser_best', fix=True)[:, :track.audio.shape[0]].T,
        }

        a_l = np.array([estimates['drums'][:, 0], estimates['bass'][:, 0], estimates['other'][:, 0], estimates['vocals'][:, 0]]).T
        a_r = np.array([estimates['drums'][:, 1], estimates['bass'][:, 1], estimates['other'][:, 1], estimates['vocals'][:, 1]]).T

        b_l = track.audio[:, 0]
        b_r = track.audio[:, 1]

        if verbose: print(a_l.shape, b_l.shape)

        sol_l = np.linalg.lstsq(a_l, b_l, rcond=None)[0]
        sol_r = np.linalg.lstsq(a_r, b_r, rcond=None)[0]

        e_l = a_l * sol_l
        e_r = a_r * sol_r

        separation = np.array([e_l, e_r])  # shape: (channel, time, instrument)

        if verbose: print(separation.shape)

        estimates = {
            'drums': separation[:, :, 0].T,
            'bass': separation[:, :, 1].T,
            'other': separation[:, :, 2].T,
            'vocals': separation[:, :, 3].T,
        }

        return estimates


    print("separating...")
    track_estimates_pairs = []
    for i, track in enumerate(mus_test.tracks):
        estimates = separate_sample(network, track)
        track_estimates_pairs.append((track, estimates))

        print(f"{int((i + 1) / len(mus_test.tracks) * 100)} %")

    print("\nall tracks are separated, evaluation starts")


    output_dir = f"checkpoints/{args.model_dir}/scores"
    if not os.path.exists(output_dir): os.mkdir(output_dir)

    def evaluate(track_estimates):
        track, estimates = track_estimates
        museval.eval_mus_track(track, estimates, output_dir=output_dir)

    pool = multiprocessing.Pool(args.threads)
    scores_list = list(
        pool.imap_unordered(
            func=evaluate,
            iterable=track_estimates_pairs,
            chunksize=1
        )
    )
    pool.close()
    pool.join()

    print("Everything is evaluated")

    def json2df(json_string, track_name):
        df = json_normalize(json_string['targets'], ['frames'], ['name'])
        df.columns = [col.replace('metrics.', '') for col in df.columns]
        df = pd.melt(
            df,
            var_name='metric',
            value_name='score',
            id_vars=['time', 'name'],
            value_vars=['SDR', 'SAR', 'ISR', 'SIR']
        )
        df['track'] = track_name
        df = df.rename(index=str, columns={"name": "target"})
        return df

    scores = museval.EvalStore(frames_agg='median')
    p = Path(output_dir)
    json_paths = p.glob('test/**/*.json')
    for json_path in json_paths:
        with open(json_path) as json_file:
            json_string = simplejson.loads(json_file.read())
        track_df = json2df(json_string, json_path.stem)
        scores.add_track(track_df)

    print(scores)
