# BSD 3-Clause License
#
# This file is part of the PNEC project.
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
