# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
from pnec.metrics.rmse import rmse


def rpe_1(poses):
    """Calculate the RPE1 error metric from given transformations

    Args:
        poses (pandas Dataframe): dataframe holding all the frames with estimated and ground truth rotation

    Returns:
        double: RPE1 error metric for the given transformations
    """
    poses_gt = poses['poses_gt'].to_list()
    poses_est = poses['poses_est'].to_list()

    poses_gt_np = np.asarray([pose.rotationMatrix() for pose in poses_gt])
    poses_est_np = np.asarray([pose.rotationMatrix() for pose in poses_est])

    return rmse(poses_gt_np, poses_est_np, 1)
