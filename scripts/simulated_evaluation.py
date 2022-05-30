# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axis

from pnec.plotting.FigureSize import FigureSize


def plot(results: Dict[str, pd.DataFrame], colors: List, linestyles: List, figsize=FigureSize(
        'thesis'), line_args: Dict = {}, axis_args: Dict = {}):
    if len(results) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[:, len(results.columns)]

    assert len(colors) == len(linestyles)
    plt.style.use('seaborn')
    sns.set_context("talk")
    sns.set_style("white")
    plt.style.use("scripts/tex.mplstyle")

    plt.close()
    constraint = False
    fig, ax = plt.subplots(1, 1, figsize=figsize,
                           constrained_layout=constraint)
    for (name, result), color, linestyle in zip(results.items(), colors, linestyles):
        ax.plot(result.index, result, color=color,
                linestyle=linestyle, label=name, **line_args)

    ax.legend(prop={'size': 6}, handlelength=1.5)
    ax.set(**axis_args)

    return fig, ax


def load_results(sub_dir: Path, name: str, methods: List[str]):
    method_results = {}
    for method in methods:
        levels = []
        for level in sub_dir.iterdir():
            result_path = level.joinpath(name)
            if not result_path.is_dir():
                continue

            metrics = []
            for metric in ['cost', 'r_error', 't_error']:

                results = pd.read_csv(result_path.joinpath(metric + '.csv'))

                mean = results[method].mean()
                median = results[method].median()
                rmse = np.sqrt(np.mean(results[method] ** 2))

                metrics.append(
                    pd.DataFrame([mean, median, rmse], columns=[float(level.parts[-1])], index=['mean', 'median', 'rmse']).transpose().add_suffix('_' + metric))

            levels.append(pd.concat(metrics, axis=1))

        if len(levels) == 0:
            continue

        method_results[method] = pd.concat(levels)

    if len(method_results) == 0:
        return None

    return method_results


def standard_experiment_evaluation(experiment_dir: Path, name: str, methods: List[str]):
    axis_args = {'xlabel': r'noise scale [pix]', 'ylim': (0, None)}
    line_args = {'linewidth': 1.0}
    colors = sns.color_palette('bright')[:len(methods)]
    linestyles = ['--', '-.', ':',
                  (0, (1, 10)), (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10))][:len(methods)]
    figsize = FigureSize('thesis', subplots=(1, 1), fraction=1.0/2.0)

    for camera_dir in experiment_dir.iterdir():
        if not camera_dir.is_dir():
            continue

        for pose_type_dir in camera_dir.iterdir():
            if not pose_type_dir.is_dir():
                continue

            for noise_type_dir in pose_type_dir.iterdir():
                if not noise_type_dir.is_dir():
                    continue

                results = load_results(noise_type_dir, name, methods)
                if results is None:
                    continue

                if not noise_type_dir.joinpath(name).is_dir():
                    noise_type_dir.joinpath(name).mkdir()

                for metric, (y_label, title) in zip(['cost', 'r_error', 't_error'], [(r'$E_P$', r'energy function'), (r'$e_{rot}$ [deg]', r'rotation'), (r'$e_{t}$ [deg]', r'translation')]):
                    axis_args['ylabel'] = y_label
                    axis_args['title'] = title

                    for aggregation in ['mean', 'median', 'rmse']:
                        plot_results = {}
                        for method in methods:
                            plot_results[method] = results[method][aggregation + '_' + metric]

                        fig, ax = plot(plot_results, colors, linestyles,
                                       figsize, line_args, axis_args)

                        fig.savefig(noise_type_dir.joinpath(
                            name, aggregation + '_' + metric + '.pdf'), bbox_inches='tight')


def anisotropy_experiment_evaluation(experiment_dir: Path, name: str, methods: List[str]):
    axis_args = {'xlabel': r'$\beta$', 'ylim': (0, None)}
    line_args = {'linewidth': 1.0}
    colors = sns.color_palette('bright')[:len(methods)]
    linestyles = ['--', '-.', ':',
                  (0, (1, 10)), (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10))][:len(methods)]
    figsize = FigureSize('thesis', subplots=(1, 1), fraction=1.0/2.0)

    for camera_dir in experiment_dir.iterdir():
        if not camera_dir.is_dir():
            continue

        for pose_type_dir in camera_dir.iterdir():
            if not pose_type_dir.is_dir():
                continue

            results = load_results(pose_type_dir, name, methods)
            if results is None:
                continue

            if not pose_type_dir.joinpath(name).is_dir():
                pose_type_dir.joinpath(name).mkdir()

            for metric, (y_label, title) in zip(['cost', 'r_error', 't_error'], [(r'$E_P$', r'energy function'), (r'$e_{rot}$ [deg]', r'rotation'), (r'$e_{t}$ [deg]', r'translation')]):
                axis_args['ylabel'] = y_label
                axis_args['title'] = title

                for aggregation in ['mean', 'median', 'rmse']:
                    plot_results = {}
                    for method in methods:
                        plot_results[method] = results[method][aggregation + '_' + metric]

                    fig, ax = plot(plot_results, colors, linestyles,
                                   figsize, line_args, axis_args)

                    fig.savefig(pose_type_dir.joinpath(
                        name, aggregation + '_' + metric + '.pdf'), bbox_inches='tight')


def offset_experiment_evaluation(experiment_dir: Path, name: str, methods: List[str]):
    axis_args = {'xlabel': r'parameter offset [\%]', 'ylim': (0, None)}
    line_args = {'linewidth': 1.0}
    colors = sns.color_palette('bright')[:len(methods)]
    linestyles = ['--', '-.', ':',
                  (0, (1, 10)), (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10))][:len(methods)]
    figsize = FigureSize('thesis', subplots=(1, 1), fraction=1.0/2.0)

    for camera_dir in experiment_dir.iterdir():
        if not camera_dir.is_dir():
            continue

        for pose_type_dir in camera_dir.iterdir():
            if not pose_type_dir.is_dir():
                continue

            for noise_type_dir in pose_type_dir.iterdir():
                if not noise_type_dir.is_dir():
                    continue

                results = load_results(noise_type_dir, name, methods)
                if results is None:
                    continue

                if not noise_type_dir.joinpath(name).is_dir():
                    noise_type_dir.joinpath(name).mkdir()

                for metric, (y_label, title) in zip(['cost', 'r_error', 't_error'], [(r'$E_P$', r'energy function'), (r'$e_{rot}$ [deg]', r'rotation'), (r'$e_{t}$ [deg]', r'translation')]):
                    axis_args['ylabel'] = y_label
                    axis_args['title'] = title

                    for aggregation in ['mean', 'median', 'rmse']:
                        plot_results = {}
                        for method in methods:
                            plot_results[method] = results[method][aggregation + '_' + metric]

                        fig, ax = plot(plot_results, colors, linestyles,
                                       figsize, line_args, axis_args)

                        fig.savefig(noise_type_dir.joinpath(
                            name, aggregation + '_' + metric + '.pdf'), bbox_inches='tight')


def main(experiment_dir: Path, experiment: str, name: str, methods: List[str]):
    evaluation_funcs = {'anisotropy': anisotropy_experiment_evaluation,
                        'offset': offset_experiment_evaluation}

    evaluation_func = evaluation_funcs.get(
        experiment, standard_experiment_evaluation)

    evaluation_func(experiment_dir, name, methods)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the results from synthethic different types of synthetic experiments")
    parser.add_argument("-p", "--path", help="Base directory of all results",
                        type=str, default="default path")
    parser.add_argument(
        "-e", "--experiment", help="Name of the experiment, needed to parse the directory structure", type=str, default="standard")
    parser.add_argument(
        "-n", "--name", help="Name of the directory of where the experiment is stored in - e.g. normal, ablation", type=str, default="default")
    parser.add_argument("-m", "--methods", nargs="+", default=['NEC', 'PNEC'])

    args = parser.parse_args()

    main(Path(args.path), args.experiment, args.name, args.methods)
