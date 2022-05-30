# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import sophus as sp
from pnec.metrics import l1_rpe_1, l1_rpe_n, r_t, rpe_1, rpe_n
from pnec.visual_odometry.trajectory.correction import correct_position
from scipy.spatial.transform import Rotation as R

PoseFromText = Callable[[np.ndarray], sp.SE3]


def pose_from_quat(quaternion: np.ndarray, format: str = "xyzabcw") -> sp.SE3:
    w = quaternion[format.index('w') - 1]
    a = quaternion[format.index('a') - 1]
    b = quaternion[format.index('b') - 1]
    c = quaternion[format.index('c') - 1]
    x = quaternion[format.index('x') - 1]
    y = quaternion[format.index('y') - 1]
    z = quaternion[format.index('z') - 1]

    rotation = R.from_quat([a, b, c, w])

    return sp.SE3(rotation.as_matrix(), np.array([x, y, z]))


def pose_from_matrix(transformation_matrix: np.ndarray, format: Tuple[int, int] = (3, 4)) -> sp.SE3:
    transformation_matrix = transformation_matrix.reshape(format)
    return sp.SE3(sp.to_orthogonal(transformation_matrix[:3, :3]), transformation_matrix[:3, 3])


def read_poses(poses_path: Path, pose_from_text: PoseFromText, timing_path: Union[Path, None] = None, delimter: str = ' ', header: bool = False):
    def format_time(timestamp: float):
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # def pose_from_matrix(transformation_matrix):
    #     return sp.SE3(sp.to_orthogonal(transformation_matrix[:3, :3]), transformation_matrix[:3, 3])

    # def pose_from_quat(quad_and_t, front_w):
    #     if front_w:
    #         w = quad_and_t[3]
    #         vec = quad_and_t[4:7]
    #     else:
    #         w = quad_and_t[6]
    #         vec = quad_and_t[3:6]
    #     rotation = R.from_quat(np.append(vec, w))
    #     # if w != 1.0:
    #     #     vec = vec / sqrt(1 - w**2)
    #     return sp.SE3(rotation.as_matrix(), quad_and_t[0:3, None])

    if timing_path == None:
        try:
            transforms: np.ndarray = np.genfromtxt(
                poses_path, delimiter=delimter, skip_header=header)
            transforms = transforms[~np.isnan(transforms).any(axis=1)]
        except OSError:
            return None
    else:
        try:
            transforms: np.ndarray = np.genfromtxt(
                poses_path, delimiter=delimter, skip_header=header)
            timings = np.genfromtxt(
                timing_path, delimiter=delimter, skip_header=header)
            transforms = np.hstack([timings[:, None], transforms, ])
            transforms = transforms[~np.isnan(transforms).any(axis=1)]
        except OSError:
            return None
    return pd.DataFrame([(format_time(transform[0]), pose_from_text(transform[1:])) for transform in transforms], columns=['timestamp', 'poses'])


def read_metrics(method: str, path_est: Path, path_gt: Path, skip_if_present=False):
    # returns metrics in a dict if exist, else returns none

    # check if exists
    if path_est.joinpath('metrics.yaml').is_file():
        metrics = pd.read_csv(path_est.joinpath(
            'metrics.yaml'), sep=',')
    else:
        metrics = pd.DataFrame(
            columns=['name', 'RPE_1', 'RPE_n', 'L1RPE_1', 'L1RPE_n', 'r_t'])

    # check if is in metrics
    if not method in metrics['name'].unique() or not skip_if_present:
        metrics.drop(metrics[metrics['name'] == method].index, inplace=True)
        # read in poses estimated and gt
        matches = matches_from_poses(method, path_est, path_gt)

        error_t = r_t.r_t(matches) * 180.0 / np.pi
        rpe1 = rpe_1.rpe_1(matches) * 180.0 / np.pi
        rpen = rpe_n.rpe_n(matches) * 180.0 / np.pi

        l1_rpe1 = l1_rpe_1.rpe_1(matches) * 180.0 / np.pi
        l1_rpen = l1_rpe_n.rpe_n(matches) * 180.0 / np.pi

        method_metrics = pd.DataFrame([[method, rpe1, rpen, l1_rpe1, l1_rpen, error_t]],
                                      columns=['name', 'RPE_1', 'RPE_n', 'L1RPE_1', 'L1RPE_n', 'r_t'])
        metrics = pd.concat([metrics, method_metrics])

        metrics.to_csv(path_est.joinpath(
            'metrics.yaml'), sep=',', float_format='%.3f', index=False)

    return metrics.loc[metrics['name'] == method]


def matches_from_poses(method: str, path_est: Path, path_gt: Path):
    pose_from_text = partial(pose_from_matrix, format=(3, 4))
    ground_truth = read_poses(
        path_gt, pose_from_text, timing_path=path_gt.parents[0].joinpath('times.txt'))

    first_timestamp = ground_truth['timestamp'][0]
    first_row = pd.DataFrame([[first_timestamp, sp.SE3()]], columns=[
        'timestamp', 'poses'])

    pose_from_text = partial(pose_from_quat, format="txyzabcw")
    estimated = read_poses(path_est.joinpath(method + '.txt'), pose_from_text)

    if estimated is None:
        print("Something wrong with {}".format(method.stem))
        return None

    for i in range(1, len(estimated)):
        estimated['poses'][i] = estimated['poses'][i - 1] * \
            estimated['poses'][i]

    first_match = ground_truth.join(
        estimated, lsuffix='_gt', rsuffix='_est').dropna().iloc[0, :]

    pose_correction = first_match["poses_gt"] * \
        first_match["poses_est"].inverse()

    estimated["poses"] = estimated["poses"].apply(
        lambda pose: pose_correction * pose)

    return ground_truth.join(
        estimated.set_index('timestamp'), on='timestamp', lsuffix='_gt', rsuffix='_est').dropna()


def read_matched_poses(path: Path):
    if not path.is_file():
        return None
    pose_from_text = partial(pose_from_quat, format="txyzabcw")
    return read_poses(path, pose_from_text, delimter=',', header=True)


def write_matched_poses(path: Path, poses: pd.DataFrame):
    def to_output_row(row: pd.DataFrame):
        timestamp = datetime.strptime(
            row['timestamp'], '%Y-%m-%d %H:%M:%S.%f').timestamp()

        pose = row['poses']
        quat = R.from_matrix(pose.rotationMatrix()).as_quat()  # x, y, z, w
        translation = pose.translation()

        data = np.concatenate(
            (np.array(timestamp)[None], translation, quat), axis=0)

        return data

    if not path.parent.is_dir():
        path.parent.mkdir(parents=True)

    output = poses.apply(to_output_row, axis=1)

    output = pd.DataFrame(output.values.tolist(), index=output.index, columns=[
                          'timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

    output.to_csv(path, sep=',', float_format='%.6f', index=False)


def evaluate_run(method: str, path_est: Path, path_gt: Path, skip_if_present=False):
    # return metrics and matched poses
    # reads metrics and matched poses if exist, otherwise creates them from the poses

    print(f'Evaluating {path_est} with ground truth at {path_gt}')
    metrics = read_metrics(method, path_est, path_gt, skip_if_present)

    corrected_poses = read_matched_poses(
        path_est.joinpath('corrected_poses', method + '.txt'))
    if corrected_poses is None or not skip_if_present:
        matches = matches_from_poses(method, path_est, path_gt)

        # redo so that only the estimated pos are returned
        corrected_matches = correct_position(matches)

        corrected_poses = corrected_matches[["timestamp", "poses_est"]].rename(
            columns={"poses_est": "poses"})

        write_matched_poses(path_est.joinpath(
            'corrected_poses', method + '.txt'), corrected_poses)

    return metrics, corrected_poses
