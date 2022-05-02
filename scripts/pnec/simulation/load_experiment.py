# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

from pathlib import Path
import numpy as np
import sophus as sp
from scipy.spatial.transform import Rotation as R
from pnec.math import unscented_transform


def load_problem(path: Path, omnidirectional: bool, problem_num: int, load_prediction: bool = False, prediction_path: Path = None):
    # Load the variables for a synthetic problem from the selected folder
    # Do the unscented transform to get the correct variables
    pose_1 = np.genfromtxt(path.joinpath(
        'poses_1.csv'), delimiter=',')[problem_num, :]
    sp_pose_1 = sp.SE3(R.from_quat(pose_1[:4]).as_matrix(), pose_1[4:])
    pose_2 = np.genfromtxt(path.joinpath(
        'poses_2.csv'), delimiter=',')[problem_num, :]
    sp_pose_2 = sp.SE3(R.from_quat(pose_2[:4]).as_matrix(), pose_2[4:])
    points_1 = np.genfromtxt(path.joinpath(
        'points_1.csv'), delimiter=',')[problem_num, :-1].reshape((-1, 3))
    points_2 = np.genfromtxt(path.joinpath(
        'points_2.csv'), delimiter=',')[problem_num, :-1].reshape((-1, 3))
    covs_2 = np.genfromtxt(path.joinpath(
        'covs_2.csv'), delimiter=',')[problem_num, :-1].reshape((-1, 3, 3))

    gt_pose = sp_pose_1.inverse() * sp_pose_2

    # unscented transform and projection of points onto the unit sphere
    sigmas = np.zeros(covs_2.shape)
    for i, (cov, point) in enumerate(zip(covs_2, points_2)):
        sigmas[i, :, :] = unscented_transform(
            point=point, covariance=cov, omnidirectional=omnidirectional)
    bvs_1 = points_1 / np.linalg.norm(points_1, axis=1)[:, None]
    bvs_2 = points_2 / np.linalg.norm(points_2, axis=1)[:, None]

    if load_prediction:
        pred = np.genfromtxt(prediction_path, delimiter=',')[problem_num, :]
        pred_pose = sp.SE3(R.from_quat(pred[:4]).as_matrix(), pred[4:])
        return gt_pose, bvs_1, bvs_2, sigmas, pred_pose

    return gt_pose, bvs_1, bvs_2, sigmas
