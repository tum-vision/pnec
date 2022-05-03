# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
from pnec.metrics.rmse import rmse
from pnec.common import print_progress_bar


def rpe_n(poses):
    """Calculate the RPEn error metric from given transformations

    Args:
        poses (pandas Dataframe): dataframe holding all the frames with estimated and ground truth rotation

    Returns:
        double: RPEn error metric for the given transformations
    """
    poses_gt = poses['poses_gt'].to_list()
    poses_est = poses['poses_est'].to_list()

    poses_gt_np = np.asarray([pose.rotationMatrix() for pose in poses_gt])
    poses_est_np = np.asarray([pose.rotationMatrix() for pose in poses_est])

    rmp_n_sum = 0
    length = poses.shape[0]

    for distance in range(1, length):
        rmp_n_sum = rmp_n_sum + rmse(poses_gt_np, poses_est_np, distance)
        print_progress_bar(
            (2 * length + 1 - distance) * distance, length * (length + 1))

    print_progress_bar(
        length * (length + 1), length * (length + 1))
    return rmp_n_sum / length
