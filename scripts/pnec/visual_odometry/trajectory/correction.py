# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
import sophus as sp
from itertools import tee
import pandas as pd
from math import pow, sqrt
from scipy.spatial.transform import Rotation as R
from datetime import datetime


# def get_correction(match):
#     """compute correction transformation so that the estimated pose will be aligned with the ground truth pose

#     Args:
#         match (pandas Dataframe): single row of a pandas dataframe that holds single matched poses

#     Returns:
#         scipy Rotation, numpy array: relative orientation, relative translation
#     """
#     orientation_correction = match["orientation_gt"] * \
#         match["orientation_est"].inv()
#     position_correction = match["position_gt"] - \
#         orientation_correction.apply(match["position_est"])
#     return orientation_correction, position_correction


def correct_coordinate_system(poses, correction_transform):
    """Transform all poses by the corrected pose

    Args:
        orientation_correction (scipy Rotation): correct the poses by this orientation
        position_correction (numpy array): correct the poses by this position
        poses (pandas Dataframe): dataframe of the poses that should be corrected

    Returns:
        pandas Dataframe: dataframe with corrected poses
    """
    return poses.apply(lambda pose: correction_transform * pose, axis=1)

    # for index, pose in poses.iterrows():
    #     new_orientation: R = orientation_correction * pose["orientation"]
    #     new_position = orientation_correction.apply(
    #         pose["position"]) + position_correction
    #     poses.at[index, "orientation"] = new_orientation
    #     poses.at[index, "position"] = new_position

    # return poses


def correct_matched_poses(matched_poses):
    """align the estimated and ground truth poses so that the first ones have the same position and orientation

    Args:
        matched_poses (pandas Dataframe): dataframe of matched poses

    Returns:
        pandas Dataframe, pandas Dataframe: dataframe of the aligned ground truth poses, dataframe of the aligned poses
    """
    correction_transform = matched_poses["poses_gt"][0].inverse(
    ) * matched_poses["poses_est"][0]

    corrected_poses = correct_coordinate_system(
        matched_poses["poses_est"], correction_transform)

    orientation_correction, position_correction = get_correction(
        matched_poses.iloc[0, :])

    estimated = matched_poses[["orientation_est", "position_est"]].rename(
        columns={"orientation_est": "orientation", "position_est": "position"})

    corrected_estimated = correct_coordinate_system(
        orientation_correction, position_correction, estimated)

    return matched_poses[["orientation_gt", "position_gt"]].rename(
        columns={"orientation_gt": "orientation", "position_gt": "position"}), corrected_estimated


# def flip_estimated(estimated, flip_yaw=False, flip_pitch=False, flip_roll=False):
#     """flip orientation at yaw, pitch and roll if necessary

#     Args:
#         estimated (pandas Dataframe): dataframe that holds the poses that need to be flipped
#         flip_yaw (bool, optional): set true if you want to flip yaw. Defaults to False.
#         flip_pitch (bool, optional): set true if you want to flip pitch. Defaults to False.
#         flip_roll (bool, optional): set true if you want to flip roll. Defaults to False.

#     Returns:
#         pandas Dataframe: dataframe with flipped orientations
#     """
#     for index, pose in estimated.iterrows():
#         yaw, pitch, roll = pose["orientation"].as_euler(
#             'ZYX', degrees=True)
#         if flip_yaw:
#             yaw = - yaw
#         if flip_pitch:
#             pitch = - pitch
#         if flip_roll:
#             roll = -roll
#         estimated.at[index, 'orientation'] = R.from_euler(
#             'ZYX', [yaw, pitch, roll], degrees=True)
#     return estimated


def correct_position(matched_poses):
    """Estimate the trajectory using the estimated orientations and the ground truth relative translations between
    poses.

    Args:
        matched_poses (pandas.Dataframe): Pandas dataframe, that holds the matched poses of the ground truth and
        estimated poses with timestamps.

    Returns:
        pandas.Dataframe: timestamp positions for estimated poses
    """
    # iterate over pairwise ground truth and estimated poses
    for index, curr_poses in matched_poses.iterrows():
        loc = matched_poses.index.get_loc(index)
        if loc == 0:
            continue
        prev_poses = matched_poses.iloc[loc - 1]

        gt_translation = (prev_poses["poses_gt"].inverse(
        ) * curr_poses["poses_gt"]).translation()

        curr_poses["poses_est"].setTranslation(
            prev_poses["poses_est"] * gt_translation)

        # relative_ground_truth_translation = prev_poses['orientation_gt'].inv().apply(
        #     curr_poses['position_gt'] - prev_poses['position_gt'])

        # new_position = prev_poses['orientation_est'].apply(
        #     relative_ground_truth_translation) + prev_poses['position_est']
        # matched_poses.at[index, "position_est"] = new_position

    return matched_poses


def rotation_difference(prev_pose, curr_pose):
    """Returns the error of the estimated relative rotation compared to the true relative rotation.
    R_error = (R_{gt,i}^-1 * R_{gt,j})^-1 * (R_{est,i}^-1 * R_{est,j}))

    Args:
        prev_pose (single row of pandas dataframe): holds the orientations of the ground truth and estimated poses of
        the previous frame
        curr_pose (single row of pandas dataframe): holds the orientations of the ground truth and estimated poses of
        the current frame

    Returns:
        sp.SO3: the rotational error as a SO3 object
    """
    return (prev_pose["poses_gt"].so3().inverse() * curr_pose["poses_gt"].so3()).inverse() * (prev_pose["poses_est"].so3().inverse() * curr_pose["poses_est"].so3())
    # return (prev_pose["orientation_gt"].inv() * curr_pose["orientation_gt"]).inv() * (prev_pose["orientation_est"].inv()
    #                                                                                   * curr_pose["orientation_est"])


def rotation_mse(poses, distance):
    """RMSE for poses.

    Args:
        poses (pandas Dataframe): dataframe that hold the estimated and ground truth poses
        distance (int): distance between poses

    Returns:
        float: rmse for the given poses and distance
    """
    rmse_sum = 0
    rmse_counter = 0
    for index, prev_poses in poses.iterrows():
        loc = poses.index.get_loc(index)
        # break if next pose doesn't exist
        if loc + distance >= poses.shape[0]:
            break
        curr_poses = poses.iloc[loc + distance]
        rotation_error = rotation_difference(prev_poses, curr_poses)
        rmse_sum = rmse_sum + pow(rotation_error.magnitude(), 2)
        rmse_counter = rmse_counter + 1

    if rmse_counter == 0:
        return 0.0
    else:
        return sqrt(rmse_sum / rmse_counter)


def pairwise(iterable):
    """Return a pairwise iterable object that pairs an element with its following
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    Args:
        iterable (iterable): iterable object like a list

    Returns:
        iterable: zipped object of the iterable
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def gen_line_of_sight(orientation):
    """Generate vectors, that represent the line of sight for a given orientation

    Args:
        orientation (scipy Rotation): holds orientation

    Returns:
        numpy array: line of sight vector
    """
    return orientation.apply(np.array([0.0, 0.0, 1.0]))


def biggest_errors(matched_poses, min_error=2.0, n=5, surrounding=20):
    """returns the nth poses and their surroundings with the highest rotational error if the error exceeds the
    min_error threshold. The surrounding is the previous and following frames (clipped at beginning and end if
    necessary)

    Args:
        matched_poses (pandas Dataframe): dataframe that holds the matched poses and the rotation error for each pose
        min_error (float, optional): threshold for minimum error in degrees needed. Defaults to 2.0.
        n (int, optional): max number of errors returned. Defaults to 5.
        surrounding (int, optional): number of previous and following frames that are returned with the error.
            Defaults to 10.

    Returns:
        list of pandas Dataframe: list of dataframes that hold the error frames and their surrounding frames
    """
    indices = [matched_poses.index.get_loc(
        value) for value in matched_poses["rotation_error"].nlargest(n).index.values]

    indices = [
        index for index in indices if matched_poses["rotation_error"][index] > min_error]

    for index in indices:
        print(datetime.fromisoformat(
            matched_poses.index.values[index]).timestamp())

    biggest_err = []
    for index in indices:
        min_idx = max(index - surrounding, 0)
        max_idx = min(index + surrounding, matched_poses.shape[0])
        biggest_err.append(matched_poses.iloc[min_idx:max_idx, :])

    return biggest_err
