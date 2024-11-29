import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
from matplotlib.animation import FuncAnimation


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
    - Return -1 if the no preference area is clicked
    - Return None if the skip area is clicked
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
    ax[0, 0].set_title("query 0", fontsize=12, pad=10)
    ax[0, 1].set_title("query 1", fontsize=12, pad=10)
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")

    # Bottom-left: "Skip"
    ax[1, 0].axis("off")
    ax[1, 0].set_facecolor("lightgray")
    ax[1, 0].text(
        0.5, 0.5, "Skip", fontsize=14, ha="center", va="center", color="black"
    )

    # Bottom-right: "No Preference"
    ax[1, 1].axis("off")
    ax[1, 1].set_facecolor("lightgray")
    ax[1, 1].text(
        0.5,
        0.5,
        "No Preference",
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
            print("choosed query 0 ", end="\r", flush=True)
            plt.close(fig)
        elif event.inaxes == ax[0, 1]:
            clicked_result["choice"] = 1
            print("choosed query 1 ", end="\r", flush=True)
            plt.close(fig)
        elif event.inaxes == ax[1, 0]:  # Skip area
            clicked_result["choice"] = None
            print("skipped this one", end="\r", flush=True)
            plt.close(fig)
        elif event.inaxes == ax[1, 1]:  # No preference area
            clicked_result["choice"] = -1
            print("tied            ", end="\r", flush=True)
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)

    _ = FuncAnimation(
        fig, update, frames=len(frames1), interval=1000 * interval, blit=True
    )
    plt.show()

    return clicked_result["choice"]
