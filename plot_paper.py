# Hardcoded plotting for ViZDoom results
from argparse import ArgumentParser
import os
import glob
import re

import matplotlib
from matplotlib import pyplot
import numpy as np

LABEL_FONTSIZE = "large"
TITLE_FONTSIZE = "x-large"
TICK_FONTSIZE = "large"

# Stackoverflow #4931376
matplotlib.use('Agg')

FIGURES_DIR = "./figures"

def rolling_average(x, window_length):
    """
    Do rolling average on vector x with window
    length window_length.

    Values in the beginning of the array are smoothed
    over only the valid number of samples
    """
    new_x = np.convolve(np.ones((window_length,))/window_length, x, mode="valid")
    # Add missing points to the beginning
    num_missing = len(x) - len(new_x)
    new_x = np.concatenate(
        (np.cumsum(x[:num_missing])/(np.arange(num_missing) + 1), new_x)
    )
    return new_x


def interpolate_and_average(xs, ys, interp_points=None):
    """
    Average bunch of repetitions (xs, ys)
    into one curve. This is done by linearly interpolating
    y values to same basis (same xs). Maximum x of returned
    curve is smallest x of repetitions.

    If interp_points is None, use maximum number of points
    in xs as number of points to interpolate. If int, use this
    many interpolation points.

    Returns [new_x, mean_y, std_y]
    """
    # Get the xs of shortest curve
    max_min_x = max(x.min() for x in xs)
    min_max_x = min(x.max() for x in xs)
    if interp_points is None:
        # Interop points according to curve with "max resolution"
        interp_points = max(x.shape[0] for x in xs)

    new_x = np.linspace(max_min_x, min_max_x, interp_points)
    new_ys = []

    for old_x, old_y in zip(xs, ys):
        new_ys.append(np.interp(new_x, old_x, old_y))

    # Average out
    # atleast_2d for case when we only have one reptition
    new_ys = np.atleast_2d(np.array(new_ys))
    new_y = np.mean(new_ys, axis=0)
    std_y = np.std(new_ys, axis=0)

    return new_x, new_y, std_y


def plot_vizdoom(ax):
    EXPERIMENTS_DIR = "./vizdoom-runs"

    LOG_FILE = "stdout.txt"

    # Number of repetitions for each experiment
    NUM_REPETITIONS = 5
    # Different experiments that we should plot for
    EXPERIMENT_DIRS = [
        "vizdoom_vanilla-dqn",
        "vizdoom_reward-shaping",
        "vizdoom_manual-actions",
        "vizdoom_manual-actions-reward-shaping",
        "vizdoom_manual-hierarchy",
    ]
    EXPERIMENT_NAMES = [
        "RL",
        "RL + RS",
        "RL + MA",
        "RL + RS + MA",
        "RL + RS + MA + MH",
    ]

    FRAMESKIP = 4

    for experiment_name, experiment_dir_template in zip(EXPERIMENT_NAMES, EXPERIMENT_DIRS):
        experiment_repetition_paths = glob.glob(os.path.join(EXPERIMENTS_DIR, experiment_dir_template + "_*"))
        assert len(experiment_repetition_paths) == NUM_REPETITIONS
        experiment_xs = []
        experiment_ys = []
        for experiment_path in experiment_repetition_paths:
            logfile = open(os.path.join(experiment_path, LOG_FILE)).read()
            time_steps = np.array(list(map(float, re.findall("train-steps ([0-9]+)", logfile))))
            average_rewards = np.array(list(map(float, re.findall(r"average-reward ([\.\-0-9]+)", logfile))))
            experiment_xs.append(time_steps)
            experiment_ys.append(rolling_average(average_rewards, 20))
        mean_xs, mean_ys, mean_std = interpolate_and_average(experiment_xs, experiment_ys)

        ax.plot(mean_xs * FRAMESKIP, mean_ys, label=experiment_name)
        ax.fill_between(
            mean_xs * FRAMESKIP,
            mean_ys - mean_std,
            mean_ys + mean_std,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.2
        )

    ax.set_title("ViZDoom, Deathmatch", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Environment steps", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Average episodic reward", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='both', labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.2)

    ax.legend()


def plot_minerl(ax):
    EXPERIMENTS_DIR = "./minerl-runs"

    LOG_FILE_GLOB = "*.train"

    # Number of repetitions for each experiment
    NUM_REPETITIONS = 5
    EXPERIMENT_DIRS = [
        "RL",
        "RL+RS+MH",
        "RL+RS+MH+MA",
    ]
    EXPERIMENT_NAMES = [
        "RL",
        "RL + RS + MH",
        "RL + RS + MH + SA",
    ]

    for experiment_name, experiment_dir_template in zip(EXPERIMENT_NAMES, EXPERIMENT_DIRS):
        experiment_repetition_paths = glob.glob(os.path.join(EXPERIMENTS_DIR, experiment_dir_template, LOG_FILE_GLOB))
        assert len(experiment_repetition_paths) == NUM_REPETITIONS
        experiment_xs = []
        experiment_ys = []
        for experiment_path in experiment_repetition_paths:
            log_data = np.genfromtxt(experiment_path, dtype=float, skip_header=1, delimiter="\t")
            environment_frames = log_data[:, 2]
            rewards = log_data[:, 3]
            experiment_xs.append(environment_frames)
            experiment_ys.append(rolling_average(rewards, 20))
        mean_xs, mean_ys, mean_std = interpolate_and_average(experiment_xs, experiment_ys)

        ax.plot(mean_xs, mean_ys, label=experiment_name)
        ax.fill_between(
            mean_xs,
            mean_ys - mean_std,
            mean_ys + mean_std,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.2
        )
    ax.set_title("MineRL, ObtainDiamond", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Environment steps", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='both', labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.2)

    ax.legend()


def plot_football(ax):
    EXPERIMENTS_DIR = "./football-runs"

    # Number of repetitions for each experiment
    NUM_REPETITIONS = 3
    EXPERIMENT_DIRS = [
        "full_game_ASFalse_CLFalse_HLFalse",
        "full_game_ASFalse_CLFalse_HLTrue",
        "full_game_ASFalse_CLTrue_HLFalse",
        "full_game_ASFalse_CLTrue_HLTrue",
        #"full_game_ASTrue_CLFalse_HLFalse",
        #"full_game_ASTrue_CLFalse_HLTrue",
        #"full_game_ASTrue_CLTrue_HLFalse",
    ]
    EXPERIMENT_NAMES = [
        "RL",
        "RL + MH",
        "RL + CL",
        "RL + CL + MH",
        #"Action shaping",
        #"Hierarchy + Action shaping",
        #"Hierarchy + Action shaping + Curriculum Learning",
    ]

    for experiment_name, experiment_dir_template in zip(EXPERIMENT_NAMES, EXPERIMENT_DIRS):
        experiment_repetition_paths = glob.glob(os.path.join(EXPERIMENTS_DIR, experiment_dir_template, "*", "*.txt"))
        assert len(experiment_repetition_paths) == NUM_REPETITIONS
        experiment_xs = []
        experiment_ys = []
        for experiment_path in experiment_repetition_paths:
            log_data = np.loadtxt(experiment_path, delimiter=";")
            environment_frames = log_data[:3333, 0] * 3000
            rewards = rolling_average(log_data[:3333, 1], 20)
            experiment_xs.append(environment_frames)
            experiment_ys.append(rewards)
        mean_xs, mean_ys, mean_std = interpolate_and_average(experiment_xs, experiment_ys)

        ax.plot(mean_xs, mean_ys, label=experiment_name)
        ax.fill_between(
            mean_xs,
            mean_ys - mean_std,
            mean_ys + mean_std,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.2
        )
    ax.set_title("GFootball, 11 vs 11 Easy Stochastic", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Environment steps", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='both', labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.2)

    ax.legend()


def main():
    fig, axs = pyplot.subplots(nrows=1, ncols=3, figsize=[4.8 * 3, 4.8 * 0.75])
    plot_vizdoom(axs[0])
    plot_minerl(axs[1])
    plot_football(axs[2])

    pyplot.tight_layout()
    pyplot.savefig(os.path.join(FIGURES_DIR, "learning_curves.pdf"), bbox_inches='tight', pad_inches=0.0)


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    main()
