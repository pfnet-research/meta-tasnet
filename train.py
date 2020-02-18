import argparse
import os
import re
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.tasnet import MultiTasNet
from utility.ranger import Ranger
from utility.logger import Logger
from utility.loss import sdr_objective
from utility.sgdr_learning_rate import SGDRLearningRate


def train_step(network, batch, s):
    batch = tuple([s.to(device) for s in b] for b in batch)
    loss, stats = network(*batch)

    loss = loss.mean()
    stats = stats.mean(0).cpu().detach().numpy()

    return loss, stats


def eval_step(network, batch, device):
    batch = tuple([s.to(device) for s in b] for b in batch)
    mix, separated = batch

    outputs = network.inference(mix, n_chunks=4)  # shape: (1, 4, 1, T)

    objectives = [sdr_objective(o, s) for o, s in zip(outputs, separated)]
    objectives = torch.cat(objectives, 0).cpu().numpy()
    return objectives


if __name__ == "__main__":

    def optional(type): return lambda x: None if x == 'None' else type(x)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", default=160, type=int, help="Mask input dimension.")
    parser.add_argument("--base_directory", default=".", type=str, help="Directory for the log and checkpoints.")
    parser.add_argument("--batch_size", default=12, type=int, help=".")
    parser.add_argument("--checkpoint", default=None, type=str, help="Saved checkpoint to continue in training")
    parser.add_argument("--clip_gradient", default=5.0, type=float, help=".")
    parser.add_argument("--debug", dest="debug", action="store_true", default=False, help="Use fake dataset for debugging purposes.")
    parser.add_argument("--dissimilarity_loss_weight", default=3.0, type=float, help="Weight of the cosine dissimilarity of the latent spectrograms.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout whole timesteps before masking.")
    parser.add_argument("--E_1", default=8, type=optional(int), help="Embedding size of instruments for parameter generation.")
    parser.add_argument("--E_2", default=5, type=int, help="Bottlenecked embedding size of instruments for parameter generation.")
    parser.add_argument("--epochs", default=250, type=int, help=".")
    parser.add_argument("--filters", default=3, type=int, help="Number of encoder/decoder filters with different kernel size.")
    parser.add_argument("--H", default=160, type=int, help="Mask hidden dimension.")
    parser.add_argument("--independent_params", dest="independent_params", action="store_true", default=False, help="Don't generate weights and use independent weights for each masking head.")
    parser.add_argument("--kernel", default=3, type=int, help="Kernel size in convolutional blocks.")
    parser.add_argument("--L", default=20, type=int, help="Encoder base kernel size.")
    parser.add_argument("--layers", default=6, type=int, help="Number of layers in each masking TCN stack.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Base learning rate.")
    parser.add_argument("--N", default=440, type=int, help="Encoder latent dimension.")
    parser.add_argument("--num_mels", default=256, type=int, help="The output dimension of spectrogram mel transform.")
    parser.add_argument("--reconstruction_loss_weight", default=0.05, type=float, help="Weight of the SDR reconstruction of the original signal.")
    parser.add_argument("--residual_bias", dest="residual_bias", action="store_true", default=False, help="Add bias before adding to residual/skip connection.")
    parser.add_argument("--sampling_rate", default=8000, type=int, help="Base sampling rate.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--sgdr_period", default=200000, type=int, help="Period of the SGDR decay.")
    parser.add_argument("--shuffle_p", default=0.5, type=float, help="Portion of shuffled tracks in the training data.")
    parser.add_argument("--similarity_loss_weight", default=2.0, type=float, help="Weight of the cosine similarity of the latent spectrograms.")
    parser.add_argument("--stack", default=3, type=int, help="Number of stack in the masking TCN.")
    parser.add_argument("--stages_num", default=3, type=int, help="Number of stages with different sampling rates.")
    parser.add_argument("--threads", default=10, type=int, help=".")
    parser.add_argument("--time_length", default=8, type=int, help="Length of one training sample (in seconds)")
    parser.add_argument("--train_data", default="data/train", type=str, help="Absolute path to the training dataset.")
    parser.add_argument("--validation_data", default="data/validation", type=str, help="Absolute path to the validation dataset.")
    parser.add_argument("--W", default=20, type=int, help="Encoder stride.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help=".")
    args = parser.parse_args()

    architecture_description = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if key not in ["validation_data", "train_data", "directory", "base_directory", "epochs", "seed", "sgdr_period", "clip_gradient", "threads", "checkpoint", "decay_epochs"]))
    args.directory = f"{args.base_directory}/checkpoints/{architecture_description}"

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load data
    if args.debug: from dataset_stub import MusicDataset
    else: from dataset import MusicDataset

    train_data = MusicDataset(args.train_data, args.sampling_rate // 1000, args.stages_num, sample_length=args.time_length, shuffle_p=args.shuffle_p, is_train=True, verbose=True)
    eval_data = MusicDataset(args.validation_data, args.sampling_rate // 1000, args.stages_num, sample_length=args.time_length, is_train=False, verbose=True)

    print("Data loaded successfully")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.threads, collate_fn=train_data.get_collate_fn())
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=args.threads)

    # Create the model
    network = MultiTasNet(args).to(device)

    # Check the model
    network.compute_stats(train_loader)
    optimizer = Ranger(filter(lambda p: p.requires_grad, network.parameters()), weight_decay=args.weight_decay)
    decay = SGDRLearningRate(optimizer, args.learning_rate, t_0=args.sgdr_period, mul=0.85)
    logger = Logger()

    # Optionally load from a checkpoint
    if args.checkpoint is not None:
        state = torch.load(f"{args.directory}/{args.checkpoint}")
        optimizer.load_state_dict(state['optimizer'])
        network.load_state_dict(state['state_dict'])
        initial_epoch = state['epoch'] + 1
        steps = state['steps']
    else:
        initial_epoch, steps = 0, 0

    # Optionally distribute the model across more GPUs
    raw_network = network
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        network = nn.DataParallel(network)
        network = network.to(device)

    # Start training
    best_validation_objective = float('-inf')

    for epoch in range(initial_epoch, args.epochs):
        with open(f"{args.directory}/log.txt", "a", encoding="utf-8") as log_file:

            #
            # TRAIN EPOCH
            #

            network.train()

            running_objectives, batches_done = np.zeros(args.stages_num*4 + 3), 0
            start_time = time.time()

            for i_batch, batch in enumerate(train_loader):

                steps += 1
                if decay(steps):
                    break  # evaluate after decay cycle resets

                loss, stats = train_step(network, batch, device)

                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), args.clip_gradient)
                optimizer.step()
                optimizer.zero_grad()
                del loss

                with torch.no_grad():
                    batches_done += 1
                    running_objectives += stats

                    progress = 100 * i_batch//len(train_loader)
                    average_objectives = running_objectives / batches_done
                    logger.log_train_progress(epoch, average_objectives, args.stages_num, decay.learning_rate, progress)

            average_objectives = running_objectives / batches_done
            logger.log_train(epoch, average_objectives, args.stages_num, int(time.time() - start_time), log_file)


            #
            # EVALUATE EPOCH
            #

            network.eval()
            running_stats, batches_done = np.zeros(args.stages_num*4), 0

            with torch.no_grad():
                for i_batch, batch in enumerate(eval_loader):
                    stats = eval_step(raw_network, batch, device)
                    running_stats += stats
                    batches_done += 1

                average_stats = running_stats / batches_done
                logger.log_dev(average_stats, args.stages_num, decay.learning_rate, log_file)

                state = {
                    'epoch': epoch,
                    'state_dict': raw_network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'steps': steps,
                    'args': args
                }
                objective = average_stats[-5:-1].mean()

                if objective > best_validation_objective:
                    best_validation_objective = objective
                    torch.save(state, f'{args.directory}/best_checkpoint')
                torch.save(state, f'{args.directory}/last_checkpoint')
