# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import matplotlib.pyplot as plt
import numpy as np


def CDFDistribution(data: np.ndarray):
    dist = (np.arange(np.size(data)) + 1) / np.size(data)
    xx = np.sort(data)
    return xx, dist

    # (ax, data: np.ndarray, color='k', linewidth=1.0, linestyle='-'):
    # size = data.size

    # yy = np.arange(size) / size

    # data.sort

    # ax.plot(data, yy, color=color, linewidth=linewidth, linestyle=linestyle)
