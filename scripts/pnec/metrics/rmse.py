# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np


def rmse(poses_gt: np.ndarray, poses_est: np.ndarray, distance):
    poses_gt_1 = np.transpose(poses_gt[:-distance, :, :], (0, 2, 1))
    poses_gt_2 = poses_gt[distance:, :, :]

    poses_est_1 = poses_est[:-distance, :, :]
    poses_est_2 = np.transpose(poses_est[distance:, :, :], (0, 2, 1))

    rel_gt = np.matmul(poses_gt_1, poses_gt_2)
    rel_est_inv = np.matmul(poses_est_2, poses_est_1)
    rel = np.matmul(rel_est_inv, rel_gt)

    traces = np.trace(rel, axis1=1, axis2=2)

    angles = np.arccos((traces - 1.0) / 2.0)

    return np.sqrt(np.square(np.linalg.norm(angles)) / poses_est_2.shape[0]).tolist()
