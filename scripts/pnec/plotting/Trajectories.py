# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pnec.helper import set_axes_equal
from pnec.plotting.FigureSize import FigureSize


def Trajectory2D(ground_truth: pd.DataFrame, axes: Tuple, estimated: Dict[str, pd.DataFrame], colors: List, linestyles: List, figsize=FigureSize(
        'thesis'), trajectory_args: Dict = {}, axis_args: Dict = {}):
    if len(estimated) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[:, len(estimated.columns)]

    assert len(colors) == len(linestyles)
    plt.style.use('seaborn')
    sns.set_context("talk")
    sns.set_style("white")
    plt.style.use('tex')

    helper = {'x': 0, 'y': 1, 'z': 2}
    ax1 = helper[axes[0][-1]]
    ax1_sign = -1 if axes[0][0] == '-' else 1
    ax2 = helper[axes[1][-1]]
    ax2_sign = -1 if axes[1][0] == '-' else 1

    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    color_gt = [0, 0, 0]
    gt_translation = np.stack(ground_truth['poses'].apply(
        lambda pose: pose.translation()).to_list())
    ax.plot(ax1_sign * gt_translation[:, ax1], ax2_sign * gt_translation[:, ax2],
            c=color_gt, label='ground truth', **trajectory_args)

    for (name, result), color, linestyle in zip(estimated.items(), colors, linestyles):
        translation = np.stack(result['poses'].apply(
            lambda pose: pose.translation()).to_list())
        ax.plot(ax1_sign * translation[:, ax1], ax2_sign * translation[:, ax2],
                c=color, label=name, linestyle=linestyle, **trajectory_args)

    ax.axis('equal')
    ax.legend(prop={'size': 6}, handlelength=1.5)
    ax.set(**axis_args)

    return fig, ax


def Trajectory3D(ground_truth, estimated, directory, step_angle=5, line_of_sight=True, step_arrow=10,
                 arrow_length=0.02, colors=None):
    """plot 3D trajectory with optional line of sight arrows of the given ground truth and estimated poses

    Args:
        ground_truth (pandas Dataframe): ground truth poses
        estimated (pandas Dataframe): estimated poses
        directory (os.path): outputs are saved to this directory
        step_angle (int, optional): step size for the different angles at which the 3D is generated. Defaults to 5.
        line_of_sight (bool, optional): if True plot line of sight arrows. Defaults to True.
        step_arrow (int, optional): save line of sight arrow every x steps. Defaults to 10.
        arrow_length (float, optional): length of line of sight arrows. Defaults to 0.02.
    """
    if (colors is None) or len(colors) != len(estimated) + 1:
        colors = sns.color_palette(n_colors=len(estimated) + 1)

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot groundtruth
    gt_translation = np.stack(ground_truth['poses'].apply(
        lambda pose: pose.translation()).to_list())
    ax.plot(gt_translation[:, 0], gt_translation[:, 1],
            gt_translation[:, 2], c=colors[0], label='ground truth')

    for i, (name, result) in enumerate(estimated.items()):
        translation = np.stack(result['poses'].apply(
            lambda pose: pose.translation()).to_list())
        ax.plot(translation[:, 0], translation[:, 1],
                translation[:, 2], c=colors[i + 1], label=name)

    ax.set_xlabel('x in m')
    ax.set_ylabel('y in m')
    ax.set_zlabel('z in m')

    set_axes_equal(ax)

    # generate plots at different view points
    directory.mkdir(parents=True, exist_ok=True)
    for azimut in range(0, 360, step_angle):
        ax.view_init(elev=30., azim=azimut)
        plt.savefig(directory.joinpath("3dtrajectory{}.png".format(azimut)))

    plt.clf()
