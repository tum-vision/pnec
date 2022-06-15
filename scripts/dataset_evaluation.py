#
# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.
#

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import seaborn as sns

from pnec.latex.Tables import TableHighlighting
from pnec.plotting.FigureSize import FigureSize
from pnec.plotting.Metrics import DatasetMetric
from pnec.visual_odometry.io.evaluate_run import evaluate_run


def method_evaluation(sequence_dir: Path, gt_path: Path, method: str):
    metric_names = ['RPE_1', 'RPE_n', 'L1RPE_1', 'L1RPE_n', 'r_t']
    results = {}

    method_metrics = []
    for iteration_dir in sorted(sequence_dir.iterdir()):
        if not iteration_dir.is_dir():
            continue
        if sequence_dir.parts[-1].startswith('s'):
            continue

        metrics, _ = evaluate_run(method, iteration_dir, gt_path)

        method_metrics.append(metrics)

    method_metrics = pd.concat(method_metrics)

    for name in metric_names:
        results[name] = {}
        results[name]['mean'] = method_metrics[name].mean()
        results[name]['var'] = method_metrics[name].var()
        results[name]['median'] = method_metrics[name].median()

    return results


def sequence_evaluation(sequence_dir: Path, gt_path: Path, methods: list[str]):
    metric_names = ['RPE_1', 'RPE_n', 'L1RPE_1', 'L1RPE_n', 'r_t']
    metrics = {}
    for name in metric_names:
        metrics[name] = {}
        metrics[name]['mean'] = pd.DataFrame(
            0, index=[sequence_dir.parts[-1]], columns=methods)
        metrics[name]['var'] = pd.DataFrame(
            0, index=[sequence_dir.parts[-1]], columns=methods)
        metrics[name]['median'] = pd.DataFrame(
            0, index=[sequence_dir.parts[-1]], columns=methods)

    for method in methods:
        method_metrics = method_evaluation(sequence_dir, gt_path, method)

        for name in metric_names:
            metrics[name]['mean'][method] = method_metrics[name]['mean']
            metrics[name]['var'][method] = method_metrics[name]['var']
            metrics[name]['median'][method] = method_metrics[name]['median']

    return metrics


def main(dataset_dir: Path, methods: list[str]):
    metric_names = ['RPE_1', 'RPE_n', 'L1RPE_1', 'L1RPE_n', 'r_t']
    boundaries = [(0.0, 0.2), (0.0, 15), (0.0, 0.2), (0.0, 15), (0.0, 6.0)]
    names = [r'RPE$_1$ [deg]', r'RPE$_n$ [deg]',
             r'l1 RPE$_1$ [deg]', r'l1 RPE$_n$ [deg]', r'$e_t$ [deg]']
    metrics = {}
    for name in metric_names:
        metrics[name] = {}
        metrics[name]['mean'] = []
        metrics[name]['var'] = []
        metrics[name]['median'] = []

    for sequence_dir in sorted(dataset_dir.iterdir()):
        if not sequence_dir.is_dir() or sequence_dir.parts[-1] == 'metrics':
            continue
        if sequence_dir.parts[-1].startswith('s'):
            continue

        gt_path = Path(
            "default location" + sequence_dir.parts[-1] + "/poses.txt")

        if not gt_path.is_file():
            continue

        seq_metrics = sequence_evaluation(
            sequence_dir, gt_path, methods)
        for name in metric_names:
            metrics[name]['mean'].append(seq_metrics[name]['mean'])
            metrics[name]['var'].append(seq_metrics[name]['var'])
            metrics[name]['median'].append(seq_metrics[name]['median'])

    # rewrite as pandas Dataframe and get avg
    for name in metric_names:
        metrics[name]['mean'] = pd.concat(
            metrics[name]['mean'])
        metrics[name]['mean'].loc['Avg'] = metrics[name]['mean'].mean()
        metrics[name]['var'] = pd.concat(
            metrics[name]['var'])
        metrics[name]['var'].loc['Avg'] = 0.0
        metrics[name]['median'] = pd.concat(
            metrics[name]['median'])
        metrics[name]['median'].loc['Avg'] = metrics[name]['median'].mean()

    # Plots
    if not dataset_dir.joinpath('metrics').is_dir():
        dataset_dir.joinpath('metrics').mkdir()

    colors = sns.color_palette('bright')[:len(
        metrics[metric_names[0]]['mean'].columns)]
    fig_size = FigureSize(
        'thesis', subplots=(1, 1), fraction=1/2)

    for metric, name, boundary in zip(metric_names, names, boundaries):
        (fig, ax) = DatasetMetric(
            metrics[metric]['mean'], colors, std_dev=metrics[metric]['var'] ** 0.5, legend=methods, figsize=fig_size, title='mean error', xlabel='sequences', ylabel=name, ylim=boundary)
        fig.savefig(dataset_dir.joinpath('metrics', 'mean_' + metric + '.pdf'),
                    format='pdf', bbox_inches='tight', pad_inches=0.05)

        (fig, ax) = DatasetMetric(
            metrics[metric]['median'], colors, legend=methods, figsize=fig_size, title='median error', xlabel='sequences', ylabel=name, ylim=boundary)
        fig.savefig(dataset_dir.joinpath('metrics', 'median_' + metric + '.pdf'),
                    format='pdf', bbox_inches='tight', pad_inches=0.05)

    # Table for direct comparison of RPE_1 and RPE_n
    mean_metrics = pd.concat(
        [TableHighlighting(metrics['RPE_1']['mean'].add_prefix('RPE_1_'), True, 0), TableHighlighting(metrics['RPE_n']['mean'].add_prefix('RPE_n_'), True, 0)], axis=1)

    mean_metrics = pd.concat([mean_metrics.iloc[:, i::len(methods)]
                              for i in range(len(methods))], axis=1)

    with open(dataset_dir.joinpath('metrics', 'mean_table.txt'), 'w') as text_file:
        text_file.write("=========== Mean L2 Table ===========\n")
        text_file.write(mean_metrics.to_latex(index=True, escape=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate performance of an algorithm on a dataset sequence')
    parser.add_argument("-d", "--dir", help="Base directory of all results",
                        type=str, default="default path")
    parser.add_argument("-m", '--methods', nargs='+',
                        default=['NEC', 'PNEC'])

    args = parser.parse_args()

    main(Path(args.dir), args.methods)
