# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import seaborn as sns

from pathlib import Path
from pnec.plotting import Trajectories, YPR, Metrics, Errors
from pnec.plotting.FigureSize import FigureSize
from pnec.visual_odometry.io.evaluate_run import read_poses
from typing import List

from pnec.visual_odometry.io.evaluate_run import evaluate_run


def sequence_evaluation(sequence_dir: Path, method: str):
    gt_path = Path(
        "default location" + sequence_dir.parts[-1] + "/poses.txt")
    ground_truth = read_poses(gt_path)

    iterations = []
    trajectories = {}
    for iteration_dir in sorted(sequence_dir.iterdir()):
        if not iteration_dir.is_dir():
            continue
        if iteration_dir.parts[-1][0] == 's':
            continue

        if not iteration_dir.joinpath(method + '.txt').is_file():
            continue

        iterations.append(iteration_dir.parts[-1])

        _, trajectory = evaluate_run(method, iteration_dir, gt_path)
        trajectories[iteration_dir.parts[-1]] = trajectory

    if len(iterations) == 0:
        return

    # Plots
    colors = sns.color_palette('bright')[:len(iterations)]
    linestyles = ['--', '-.', ':',
                  (0, (1, 10)), (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10))][:len(iterations)]
    fig_size = FigureSize(
        'thesis', subplots=(1, 1), fraction=1/2)

    if not sequence_dir.joinpath('method_evaluation', method).is_dir():
        sequence_dir.joinpath('method_evaluation', method).mkdir(parents=True)

    fig, ax = Trajectories.Trajectory2D(
        ground_truth, ('x', 'z'), trajectories, colors, linestyles, fig_size, trajectory_args={'linewidth': 1.0}, axis_args={'xlabel': 'x [meters]', 'ylabel': 'y [meters]'})
    fig.savefig(sequence_dir.joinpath(
        'method_evaluation', method, 'trajectories.pdf'), bbox_inches='tight')
    fig, axs = YPR.YPR(ground_truth, trajectories, colors,
                       linestyles, line_args={'linewidth': 1.0}, axis_args={'xlabel': 't [sec]', 'title': 'Yaw, pitch, roll'})
    fig.savefig(sequence_dir.joinpath('method_evaluation',
                method, 'ypr.pdf'), bbox_inches='tight')
    fig, axs = Errors.Frame2FrameError(
        ground_truth, trajectories, colors, linestyles, 15, fig_sizescatter_args={'s': 0.5, 'edgecolors': 'none'}, line_args={'linewidth': 0.5})
    fig.savefig(sequence_dir.joinpath(
        'method_evaluation', method, 'f2f_error.pdf'), bbox_inches='tight')
    fig, ax = Errors.CummulativeError(
        ground_truth, trajectories, colors, linestyles, 15, fig_size, line_args={'linewidth': 1.0})
    fig.savefig(sequence_dir.joinpath(
        'method_evaluation', method, 'cummulative_error.pdf'), bbox_inches='tight')


def main(dataset_dir: Path, method: str, sequences: List[str]):
    for sequence in sequences:
        sequence_dir = dataset_dir.joinpath(sequence)

        if not sequence_dir.is_dir():
            continue

        sequence_evaluation(sequence_dir, method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate performance of an algorithm on a dataset sequence')
    parser.add_argument("-d", "--dir", help="Base directory of all results",
                        type=str, default="default path")
    parser.add_argument("-m", "--method", help="Method",
                        type=str, default="PNEC")
    parser.add_argument("-s", '--sequences', nargs='+',
                        default=['02', '03'])

    args = parser.parse_args()

    main(Path(args.dir), args.method, args.sequences)
