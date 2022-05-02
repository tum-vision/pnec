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
from pnec.plotting.FigureSize import FigureSize
from matplotlib.lines import Line2D
from typing import Tuple, List


def MetricScatter(metrics, name1, name2, path):
    plt.close()
    colors = sns.color_palette("hls", n_colors=len(metrics))
    fig, ax = plt.subplots(1, 1, figsize=FigureSize('beamer'))
    metrics.plot.scatter(x=0, y=1, s=2.0, c=colors, ax=ax)

    ax.set_xlabel(name1)
    ax.set_ylabel(name2)

    for k, v in metrics.iterrows():
        ax.annotate(k, v,
                    xytext=(5, -5), textcoords='offset points',
                    family='sans-serif', fontsize=6, color='darkslategrey')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("Error over different methods")
    fig.savefig(path,
                format='pdf', bbox_inches='tight')


def SequenceMetric(metrics, path):
    plt.close()
    # colors = sns.color_palette(n_colors=len(metrics))
    colors = sns.color_palette("hls", n_colors=len(metrics))
    fig, ax = plt.subplots(1, 1, figsize=FigureSize('beamer'))
    for i, (name, (rpe_1, rpe_n)) in enumerate(metrics.items()):
        ax.scatter(rpe_1, rpe_n, color=colors[i], label=name, s=2.0)
        # if not isinstance(rpe_1, float):
        #     ax.scatter(np.mean(rpe_1), np.mean(rpe_n),
        #                marker='x', color=colors[i])

    ax.set_xlabel(r'$RPE_1$ in degree')
    ax.set_ylabel(r'$RPE_n$ in degree')

    # ax.set_xlim(left=0.0)
    # ax.set_ylim(bottom=0.0)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("Error over different methods")
    fig.savefig(path.joinpath("metrics.pdf"),
                format='pdf', bbox_inches='tight')
    fig.savefig(path.joinpath("metrics.png"),
                format='png', bbox_inches='tight')


def L1SequenceMetric(metrics, path):
    plt.close()
    # colors = sns.color_palette(n_colors=len(metrics))
    colors = sns.color_palette("hls", n_colors=len(metrics))
    fig, ax = plt.subplots(1, 1, figsize=FigureSize('beamer'))
    for i, (name, (rpe_1, rpe_n)) in enumerate(metrics.items()):
        ax.scatter(rpe_1, rpe_n, color=colors[i], label=name, s=2.0)
        # if not isinstance(rpe_1, float):
        #     ax.scatter(np.mean(rpe_1), np.mean(rpe_n),
        #                marker='x', color=colors[i])

    ax.set_xlabel(r'$RPE_1$ in degree')
    ax.set_ylabel(r'$RPE_n$ in degree')

    # ax.set_xlim(left=0.0)
    # ax.set_ylim(bottom=0.0)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("Error over different methods")
    fig.savefig(path.joinpath("metrics_l1.pdf"),
                format='pdf', bbox_inches='tight')
    fig.savefig(path.joinpath("metrics_l1.png"),
                format='png', bbox_inches='tight')


def VerboseSequenceMetric(metrics, path):
    plt.close()
    print(len(list(metrics.values())[0][0]))
    colors = sns.color_palette("hls", n_colors=len(
        list(metrics.values())[0][0]) + 1)

    fig, ax = plt.subplots(1, 1, figsize=FigureSize('beamer'))
    for j, (name, (rpe_1, rpe_n)) in enumerate(metrics.items()):
        for i, (rpe1, rpen) in enumerate(zip(rpe_1, rpe_n)):
            ax.scatter(rpe1, rpen, s=5.0,
                       marker=list(Line2D.markers.keys())[3 + j], color=colors[i])
        if not isinstance(rpe_1, float):
            ax.scatter(np.mean(rpe_1), np.mean(rpe_n), s=12,
                       marker=list(Line2D.markers.keys())[3 + j], color=colors[-1], label=name)
            ax.scatter(np.mean(rpe_1), np.mean(rpe_n), s=12, color='k')

    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(list(metrics.values())[0][0])):
        label_line = Line2D([0], [0], color=colors[i],
                            linestyle='-', label=i + 1)
        handles.append(label_line)

    ax.set_xlabel(r'$RPE_1$ in degree')
    ax.set_ylabel(r'$RPE_n$ in degree')

    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("Error over different methods")
    fig.savefig(path.joinpath("metrics.pdf"),
                format='pdf', bbox_inches='tight')
    fig.savefig(path.joinpath("metrics.png"),
                format='png', bbox_inches='tight')


def DatasetScatter(dataset_metrics, algorithms, path, file_prefix='', normalized=False):
    plt.style.use('tex')
    plt.style.use('seaborn')
    sns.set_context("talk")
    sns.set_style("white")

    beamer_size = FigureSize('beamer', fraction=2.0)
    thesis_size = FigureSize('thesis', fraction=2.0)

    size = 25.0
    plt.close()
    colors = sns.color_palette(n_colors=len(algorithms))
    fig1, ax1 = plt.subplots(1, 1, figsize=beamer_size)
    fig2, ax2 = plt.subplots(1, 1, figsize=beamer_size)

    for i, algorithm in enumerate(algorithms):
        sequences = []
        rpe_1 = []
        rpe_n = []
        for sequence, sequence_metrics in dataset_metrics.items():
            if algorithm in sequence_metrics:
                sequences.append(sequence)
                rpe_1.append(sequence_metrics[algorithm][0])
                rpe_n.append(sequence_metrics[algorithm][1])
        if sequences:
            rpe_1 = [rpe for _, rpe in sorted(zip(sequences, rpe_1))]
            rpe_n = [rpe for _, rpe in sorted(zip(sequences, rpe_n))]
            sequences = sorted(sequences)

            ax1.scatter(sequences, rpe_n, s=size,
                        color=colors[i], label='$' + algorithm + '$', marker=list(Line2D.markers.keys())[i])
            ax2.scatter(sequences, rpe_1, s=size,
                        color=colors[i], label='$' + algorithm + '$', marker=list(Line2D.markers.keys())[i])

    ax1.set_xlabel(r'Sequence')
    if normalized:
        ax1.set_ylabel(r'normalized $RPE_n$')
    else:
        ax1.set_ylabel(r'$RPE_n$ in degree')

    ax1.set_ylim(bottom=0.0)

    ax1.legend()
    # ax1.set_title("Error over different methods")
    fig1.savefig(path.joinpath(file_prefix + "rpe_n_beamer.pdf"),
                 format='pdf', bbox_inches='tight')
    fig1.savefig(path.joinpath(file_prefix + "rpe_n_beamer.png"),
                 format='png', bbox_inches='tight')
    fig1.set_size_inches(thesis_size[0], thesis_size[1])
    fig1.savefig(path.joinpath(file_prefix + "rpe_n_thesis.pdf"),
                 format='pdf', bbox_inches='tight')
    fig1.savefig(path.joinpath(file_prefix + "rpe_n_thesis.png"),
                 format='png', bbox_inches='tight')

    ax2.set_xlabel(r'Sequence')
    if normalized:
        ax2.set_ylabel(r'normalized $RPE_1$')
    else:
        ax2.set_ylabel(r'$RPE_1$ in degree')

    ax2.set_ylim(bottom=0.0)

    ax2.legend()
    # ax2.set_title("Error over different methods")
    fig2.savefig(path.joinpath(file_prefix + "rpe_1_beamer.pdf"),
                 format='pdf', bbox_inches='tight')
    fig2.savefig(path.joinpath(file_prefix + "rpe_1_beamer.png"),
                 format='png', bbox_inches='tight')
    fig2.set_size_inches(thesis_size[0], thesis_size[1])
    fig2.savefig(path.joinpath(file_prefix + "rpe_1_thesis.pdf"),
                 format='pdf', bbox_inches='tight')
    fig2.savefig(path.joinpath(file_prefix + "rpe_1_thesis.png"),
                 format='png', bbox_inches='tight')


def DatasetMetric(results: pd.DataFrame, colors: List, std_dev: pd.DataFrame = None, legend: List = None, figsize: Tuple = FigureSize('thesis'), **kwargs):
    if len(results.columns) != len(colors):
        print("Didn't provide the correct number of colors. Will revert to use the seaborn bright color palette")
        colors = sns.color_palette("bright")[:, len(results.columns)]

    if legend is None:
        legend = results.columns

    plt.style.use('seaborn')
    sns.set_context("talk")
    sns.set_style("white")
    plt.style.use('tex')

    fig, ax = plt.subplots(1, 1, figsize=figsize,
                           constrained_layout=False)

    results.plot.bar(ax=ax, width=0.8, color=colors, **kwargs)

    if isinstance(std_dev, pd.DataFrame):
        y_lims = ax.get_ylim()
        # Offset by a 1/100th of the y_lim_max + y_lim_min
        offset_results = results - ((y_lims[0] + y_lims[1]) / 100)
        offset_results.plot.bar(yerr=std_dev, error_kw=dict(
            ecolor='black', elinewidth=1.5), ax=ax, width=0.8, alpha=0.0, label='_nolegend_')

    ax.tick_params(axis='x', rotation=0)
    ax.legend(labels=legend,
              loc='upper right', handlelength=1.5, prop={'size': 4.5})

    plt.tight_layout()
    return (fig, ax)
