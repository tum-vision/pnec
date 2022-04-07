# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
from pnec.metrics.rmse import rmse


def r_t(poses):
    """Calculate the RPE1 error metric from given transformations

    Args:
        poses (pandas Dataframe): dataframe holding all the frames with estimated and ground truth rotation

    Returns:
        double: RPE1 error metric for the given transformations
    """
    poses_gt = poses['poses_gt'].to_list()
    poses_est = poses['poses_est'].to_list()

    poses_gt_rel = []
    for poses_gt_1, poses_gt_2 in zip(poses_gt[:-1], poses_gt[1:]):
        poses_gt_rel.append(poses_gt_1.inverse() * poses_gt_2)

    poses_est_rel = []
    for poses_est_1, poses_est_2 in zip(poses_est[:-1], poses_est[1:]):
        poses_est_rel.append(poses_est_1.inverse() * poses_est_2)

    r_trans = []
    for pose_gt_rel, pose_est_rel in zip(poses_gt_rel, poses_est_rel):
        e_pos = np.arccos(np.dot(pose_gt_rel.translation().transpose(), pose_est_rel.translation(
        )) / (np.linalg.norm(pose_gt_rel.translation()) * np.linalg.norm(pose_est_rel.translation())))
        e_neg = np.arccos(np.dot(- pose_gt_rel.translation().transpose(), pose_est_rel.translation(
        )) / (np.linalg.norm(pose_gt_rel.translation()) * np.linalg.norm(pose_est_rel.translation())))
        r_trans.append(min(e_pos, e_neg))

    return np.mean(np.array(r_trans))
