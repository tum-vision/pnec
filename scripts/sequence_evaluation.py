# BSD 3-Clause License
#
# This file is part of the PNEC project.
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


def main(sequence_dir, sequence, methods: List[str]):
    colors = sns.color_palette('bright')[:len(methods)]
    linestyles = ['--', '-.', ':',
                  (0, (1, 10)), (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10))][:len(methods)]
    fig_size = FigureSize(
        'thesis', subplots=(1, 1), fraction=1/2)

    gt_path = Path(
        "/Volumes/Samsung_T5/results/repeat/KITTI/" + sequence + "/poses.txt")
    ground_truth = read_poses(gt_path)

    for iteration_dir in sorted(sequence_dir.iterdir()):
        if not iteration_dir.is_dir():
            continue
        if iteration_dir.parts[-1][0] == 's':
            continue
        if iteration_dir.parts[-1] == 'method_evaluation':
            continue

        trajectories = {}
        for method in methods:
            _, trajectory = evaluate_run(
                method, iteration_dir.joinpath("ablation"), gt_path)
            trajectories[method] = trajectory

        # Plots
        if not iteration_dir.joinpath('plots').is_dir():
            iteration_dir.joinpath('plots').mkdir()

        fig, ax = Trajectories.Trajectory2D(
            ground_truth, ('x', 'z'), trajectories, colors, linestyles, fig_size, trajectory_args={'linewidth': 1.0}, axis_args={'xlabel': 'x [meters]', 'ylabel': 'y [meters]'})
        # plt.tight_layout()
        fig.savefig(iteration_dir.joinpath(
            'plots', 'trajectories.pdf'), bbox_inches='tight')
        fig, axs = YPR.YPR(ground_truth, trajectories, colors,
                           linestyles, line_args={'linewidth': 1.0}, axis_args={'xlabel': 't [sec]', 'title': 'Yaw, pitch, roll'})
        fig.savefig(iteration_dir.joinpath(
            'plots', 'ypr.pdf'), bbox_inches='tight')
        fig, axs = Errors.Frame2FrameError(
            ground_truth, trajectories, colors, linestyles, 15, fig_size, scatter_args={'s': 0.5, 'edgecolors': 'none'}, line_args={'linewidth': 0.5})
        fig.savefig(iteration_dir.joinpath(
            'plots', 'f2f_error.pdf'), bbox_inches='tight')
        fig, ax = Errors.CummulativeError(
            ground_truth, trajectories, colors, linestyles, 15, fig_size, line_args={'linewidth': 1.0})
        fig.savefig(iteration_dir.joinpath(
            'plots', 'cummulative_error.pdf'), bbox_inches='tight')

        if len(methods) == 2:
            if not iteration_dir.joinpath('plots', 'comp').is_dir():
                iteration_dir.joinpath('plots', 'comp').mkdir()
            fig, axs = Errors.CompF2Ferror(
                ground_truth, trajectories, 15, fig_size, scatter_args={'s': 0.5, 'edgecolors': 'none'}, line_args={'linewidth': 0.5})
            fig.savefig(iteration_dir.joinpath(
                'plots', 'comp', '{}vs{}.pdf'.format(list(trajectories.keys())[0], list(trajectories.keys())[1])), bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate performance of an algorithm on a dataset sequence')
    parser.add_argument("-d", "--dir", help="Base directory of all results",
                        type=str, default="default path")
    parser.add_argument("-s", "--sequence", help="Sequence",
                        type=str, default="03")
    parser.add_argument("-m", '--methods', nargs='+',
                        default=['NEC', 'PNEC'])

    args = parser.parse_args()

    main(Path(args.dir).joinpath(args.sequence), args.sequence, args.methods)
