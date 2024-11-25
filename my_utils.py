import numpy as np
import torch
import torch.nn.functional as F
import os
import random
import math
import gymnasium as gym
import matplotlib.pyplot as plt
import h5py
import time
from matplotlib.animation import FuncAnimation

from torch import nn


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


# all the above are from B-PREFs


def initialize_h5_file(h5_file_path):
    with h5py.File(h5_file_path, "w") as _:
        pass


def save_frames(h5_file_path, key, frames):
    with h5py.File(h5_file_path, "a") as hf:
        hf.create_dataset(str(key), data=frames)


def load_frames(h5_file_path, key, start, end):
    with h5py.File(h5_file_path, "r") as hf:
        frames = hf[str(key)][start:end]
    return frames


def label_preference(
    frames1: np.ndarray, frames2: np.ndarray, interval: float = 1 / 60
):
    """
    Play two videos side by side and return user preference:
    - Return 0 if the left video is clicked
    - Return 1 if the right video is clicked
    - Return -1 if the bottom block is clicked
    """
    fig, ax = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [10, 1], "hspace": 0.1, "wspace": 0.01},
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    clicked_result = {"choice": None}

    # Show the videos
    im1 = ax[0, 0].imshow(frames1[0])
    im2 = ax[0, 1].imshow(frames2[0])
    ax[0, 0].set_title("Left Video", fontsize=12, pad=10)
    ax[0, 1].set_title("Right Video", fontsize=12, pad=10)
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")

    # Create a single clickable block at the bottom
    ax[1, 0].axis("off")
    ax[1, 0].set_facecolor("lightgray")
    ax[1, 0].text(
        0.5,
        0.5,
        "no preference",
        fontsize=14,
        ha="center",
        va="center",
        color="black",
    )
    ax[1, 1].axis("off")
    ax[1, 1].set_facecolor("lightgray")
    ax[1, 1].text(
        0.5,
        0.5,
        "no preference",
        fontsize=14,
        ha="center",
        va="center",
        color="black",
    )

    def update(i):
        im1.set_data(frames1[i])
        im2.set_data(frames2[i])
        return [im1, im2]

    def on_click(event):
        if event.inaxes == ax[0, 0]:
            clicked_result["choice"] = 0
            plt.close(fig)
        elif event.inaxes == ax[0, 1]:
            clicked_result["choice"] = 1
            plt.close(fig)
        elif event.inaxes in [ax[1, 0], ax[1, 1]]:
            clicked_result["choice"] = -1
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)

    _ = FuncAnimation(
        fig, update, frames=len(frames1), interval=1000 * interval, blit=True
    )
    plt.show()

    return clicked_result["choice"]
