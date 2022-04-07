# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from math import pi
import seaborn as sns
from datetime import datetime
from pnec.plotting.FigureSize import FigureSize
from typing import Dict, List


def groundTruthTimestamps(ground_truth: pd.DataFrame):
    start = datetime.strptime(ground_truth['timestamp'].to_list()[
        0], '%Y-%m-%d %H:%M:%S.%f')
    gt_timestamps = np.zeros((len(ground_truth) - 1, 1))
    for index, _ in ground_truth.iterrows():
        loc = ground_truth.index.get_loc(index)
        if loc + 1 >= ground_truth.shape[0]:
            break
        poses_1 = ground_truth.iloc[loc + 1]
        gt_timestamps[loc, 0] = (datetime.strptime(
            poses_1['timestamp'], '%Y-%m-%d %H:%M:%S.%f') - start).total_seconds()
    return gt_timestamps


def getErrors(ground_truth: pd.DataFrame, estimated: pd.DataFrame, absolut: bool = False):
    start = datetime.strptime(ground_truth['timestamp'].to_list()[
        0], '%Y-%m-%d %H:%M:%S.%f')

    def extractTimeandError(row: pd.DataFrame):
        timestamp = (datetime.strptime(
            row['timestamp'], '%Y-%m-%d %H:%M:%S.%f') - start).total_seconds()
        est_rel_rotation = row["prev_poses_est"].so3(
        ).inverse() * row["poses_est"].so3()
        gt_rel_rotation = row["prev_poses_gt"].so3().inverse(
        ) * row["poses_gt"].so3()

        rotational_error = est_rel_rotation.inverse() * gt_rel_rotation
        error = np.linalg.norm(
            rotational_error.log()) * 180 / pi

        return pd.Series([timestamp, error], index=['times', 'error'])

    matches = ground_truth.join(
        estimated.set_index('timestamp'), on='timestamp', lsuffix='_gt', rsuffix='_est').dropna()

    if absolut:
        matches['prev_poses_est'] = matches['poses_est'].iloc[0]
        matches['prev_poses_gt'] = matches['poses_gt'].iloc[0]
    else:
        matches['prev_poses_est'] = matches['poses_est'].shift(periods=1)
        matches['prev_poses_gt'] = matches['poses_gt'].shift(periods=1)

    return matches.iloc[1:].apply(extractTimeandError, axis=1)


def TandRError(ground_truth, estimated, path, prefix=''):
    t_errors = np.zeros((len(ground_truth) - 1, 1))
    r_errors = np.zeros((len(ground_truth) - 1, 1))
    matches = ground_truth.join(
        estimated.set_index('timestamp'), on='timestamp', lsuffix='_gt', rsuffix='_est').dropna()
    for index, host_poses in matches.iterrows():
        loc = matches.index.get_loc(index)
        # break if next pose doesn't exist
        if loc + 1 >= matches.shape[0]:
            break
        target_poses = matches.iloc[loc + 1]
        est_rel = host_poses["poses_est"].inverse() * target_poses["poses_est"]
        gt_rel = host_poses["poses_gt"].inverse(
        ) * target_poses["poses_gt"]

        gt_rel_t_norm = gt_rel.translation() / np.linalg.norm(gt_rel.translation())
        est_rel_t_norm = est_rel.translation() / np.linalg.norm(est_rel.translation())

        pos_error = np.arccos(
            np.clip(np.dot(gt_rel_t_norm, est_rel_t_norm), -1.0, 1.0)) * 180.0 / np.pi
        neg_error = np.arccos(
            np.clip(np.dot(gt_rel_t_norm, -est_rel_t_norm), -1.0, 1.0)) * 180.0 / np.pi

        t_errors[loc, :] = min(pos_error, neg_error)
        rotational_error = est_rel.so3().inverse() * gt_rel.so3()
        r_errors[loc, 0] = np.linalg.norm(
            rotational_error.log()) * 180 / pi

    errors = pd.DataFrame(
        {'t_error': t_errors[:, 0], 'r_error': r_errors[:, 0]})

    figsize = FigureSize('beamer')
    sns_plot = sns.jointplot(data=errors, x='t_error',
                             y='r_error', height=figsize[0], kind="reg")
    sns_plot.savefig(path.joinpath(prefix + 'error_comp.png'))
    sns_plot.savefig(path.joinpath(prefix + 'error_comp.pdf'))

    plt.close('all')


def Frame2FrameError(ground_truth: pd.DataFrame, estimated: Dict[str, pd.DataFrame], colors: List, linestyles: List, window_size: int = 15, figsize=FigureSize('thesis'), scatter_args: Dict = {}, line_args: Dict = {}):
    if len(estimated) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[:, len(estimated.columns)]

    assert len(colors) == len(linestyles)
    plt.style.use('seaborn')
    sns.set_context("talk")
    sns.set_style("white")
    plt.style.use('tex')

    plt.close()
    fig, axs = plt.subplots(2, 1, figsize=figsize)

    # gt_timestamps = groundTruthTimestamps(ground_truth)

    for (name, result), color, linestyle in zip(estimated.items(), colors, linestyles):
        errors = getErrors(ground_truth, result)

        axs[0].scatter(errors['times'], errors['error'],
                       color=color, label=name, **scatter_args)
        axs[1].plot(errors['times'], np.log10(np.convolve(
            errors['error'].to_numpy(), np.ones(window_size)/window_size, mode='same')), c=color, label=name, linestyle=linestyle, **line_args)

    axs[0].set_ylabel(r'$e_R$')
    axs[1].set_ylabel(r'log$_{10}$(avg $e_R$)')
    axs[1].set_xlabel('t [sec]')

    axs[1].legend(prop={'size': 6}, handlelength=1.6, handleheight=1.4)

    return fig, axs


def CummulativeError(ground_truth: pd.DataFrame, estimated: Dict[str, pd.DataFrame], colors: List, linestyles: List, window_size: int = 15, figsize=FigureSize('thesis'), line_args: Dict = {}):
    if len(estimated) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[:, len(estimated.columns)]

    assert len(colors) == len(linestyles)
    plt.style.use('seaborn')
    sns.set_context("talk")
    sns.set_style("white")
    plt.style.use('tex')

    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for (name, result), color, linestyle in zip(estimated.items(), colors, linestyles):
        errors = getErrors(ground_truth, result, True)

        ax.plot(errors['times'], errors['error'],
                c=color, label=name, linestyle=linestyle, **line_args)

    ax.set_ylabel(r'$e_R$')
    ax.set_xlabel('t [sec]')

    ax.legend(prop={'size': 6}, handlelength=1.5)

    return fig, ax


def CompF2Ferror(ground_truth: pd.DataFrame, estimated: Dict[str, pd.DataFrame], window_size: int = 15, figsize=FigureSize('thesis'), scatter_args: Dict = {}, line_args: Dict = {}):
    assert len(estimated) == 2
    plt.style.use('seaborn')
    sns.set_context("talk")
    sns.set_style("white")
    plt.style.use('tex')

    plt.close()
    fig, axs = plt.subplots(2, 1, figsize=figsize)

    gt_timestamps = groundTruthTimestamps(ground_truth)
    errors = []
    timestamps = []
    names = []

    for (name, result) in estimated.items():
        error = getErrors(ground_truth, result)
        errors.append(error['error'].to_numpy())
        timestamps.append(error['times'].to_numpy())
        names.append(name)

    diff = errors[0] - errors[1]
    pos_diff = diff[diff > 1.0e-4]
    neg_diff = diff[diff < -1.0e-4]
    neutral_diff = diff[abs(diff) <= 1.0e-4]
    pos_time = gt_timestamps[:len(diff)][diff > 1.0e-4]
    neg_time = gt_timestamps[:len(diff)][diff < -1.0e-4]
    neutral_time = gt_timestamps[:len(diff)][abs(diff) <= 1.0e-4]

    axs[0].scatter(pos_time, pos_diff, c='r', **scatter_args)
    axs[0].scatter(neutral_time, neutral_diff, c='y',
                   **scatter_args)
    axs[0].scatter(neg_time, neg_diff, c='g', **scatter_args)
    axs[0].set_ylabel(r'$\Delta e_{rot}$')
    axs[0].set_xlim(min(gt_timestamps), max(gt_timestamps))

    y_range = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]
    axs[0].text(min(gt_timestamps), axs[0].get_ylim()[0] + 0.9 * y_range,
                f'{names[1]} better', size=5)
    axs[0].text(min(gt_timestamps), axs[0].get_ylim()[0] + 0.05 * y_range,
                f'{names[0]} better', size=5)

    axs[1].plot(gt_timestamps[:len(diff)], np.convolve(diff, np.ones(window_size)/window_size, mode='same'),
                c='k', label=f'{names[0]} - {names[1]}', **line_args)
    axs[1].plot(gt_timestamps[:len(diff)], np.zeros((len(gt_timestamps[:len(diff)]), 1)),
                c='r', linewidth=0.2)
    axs[1].set_xlabel(r'$t$ in sec')
    axs[1].set_ylabel(r'smoothed $\Delta e_{rot}$')
    axs[1].set_xlim(min(gt_timestamps), max(gt_timestamps))

    fig.suptitle(f'Frame2Frame difference {names[0]} - {names[1]}')

    return fig, axs
