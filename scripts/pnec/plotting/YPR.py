# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sophus as sp
from typing import Dict, List
from datetime import datetime
from pnec.plotting.FigureSize import FigureSize
from pnec.helper import getYPR


def YPR(ground_truth: pd.DataFrame, estimated: Dict[str, pd.DataFrame], colors: List, linestyles: List, figsize=FigureSize('thesis'), rotation_offset: sp.SE3 = None, line_args: Dict = {}, axis_args: Dict = {}):

    if len(estimated) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[:, len(estimated.columns)]

    assert len(colors) == len(linestyles)

    if not rotation_offset is None:
        ground_truth['poses'] = ground_truth['poses'].apply(
            lambda pose: rotation_offset * pose)
        for key, _ in estimated.items():
            estimated[key]['poses'] = estimated[key]['poses'].apply(
                lambda pose: rotation_offset * pose)

    plt.style.use('seaborn')
    sns.set_context("talk")
    sns.set_style("white")
    plt.style.use('tex')

    start = datetime.strptime(ground_truth['timestamp'].to_list()[
                              0], '%Y-%m-%d %H:%M:%S.%f')

    plt.close()
    fig, axs = plt.subplots(nrows=3, figsize=figsize)
    times = [(datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f') - start).total_seconds() for time
             in ground_truth['timestamp'].to_list()]
    min_x = min(times)
    max_x = max(times)

    ypr = getYPR(ground_truth['poses'])
    gt_color = [0, 0, 0]
    axs[0].plot(times, np.unwrap(ypr['yaw'].to_list(), 180),
                color=gt_color, label="ground truth", **line_args)
    axs[1].plot(times, np.unwrap(ypr['pitch'].to_list(), 180),
                color=gt_color, label="ground truth", **line_args)
    axs[2].plot(times, np.unwrap(ypr['roll'].to_list(), 180),
                color=gt_color, label="ground truth", **line_args)

    for (name, result), color, linestyle in zip(estimated.items(), colors, linestyles):
        times = [(datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f') - start).total_seconds() for time
                 in result['timestamp'].to_list()]
        ypr = getYPR(result['poses'])

        axs[0].plot(times, np.unwrap(ypr['yaw'].to_list(), 180),
                    color=color, label=name, linestyle=linestyle, **line_args)
        axs[1].plot(times, np.unwrap(ypr['pitch'].to_list(), 180),
                    color=color, label=name, linestyle=linestyle, **line_args)
        axs[2].plot(times, np.unwrap(ypr['roll'].to_list(), 180),
                    color=color, label=name, linestyle=linestyle, **line_args)

    axs[0].legend(prop={'size': 6}, handlelength=1.5)
    axs[0].set_ylabel('yaw')
    axs[1].set_ylabel('pitch')
    axs[2].set_ylabel('roll')
    axs[0].set_title(axis_args.pop('title', 'Yaw, pitch, roll'))
    axs[2].set_xlabel(axis_args.pop('xlabel', 't [sec]'))

    for ax in axs:
        ax.set_xlim((min_x, max_x))
        ax.set(**axis_args)

    return fig, axs
