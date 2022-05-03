# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def skew(vector: np.ndarray) -> np.ndarray:
    """ Return skew of a single/multple vectors
    The skew of a vector is defined by 
                0  -v_3  v_2
    \hat{v} =  v_3    0 -v_1
              -v_2  v_1    0

    Args:
        vector (np.ndarray): vector/vectors of size [3] or [n x 3]

    Returns:
        np.ndarray: array of size [3 x 3] or [n x 3 x 3]
    """
    def single_skew(vector):
        return np.array([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]])

    assert vector.shape[-1] == 3

    if vector.ndim == 1:
        return single_skew(vector)

    result = np.zeros((vector.shape[0], 3, 3))
    for i in range(vector.shape[0]):
        v = vector[i, :]
        result[i, :, :] = single_skew(v)
    return result


def rotation_between_points(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """Returns a rotation matrix R such that
    p2 = R * p1
    where p1, p2 are unit length vectors

    R = I + \hat{v} + \hat{v}^2 / (1 + c)
    with
    v = p1 x p2
    c = p1.T * p2

    Args:
        point1 (np.ndarray): first point of size [3], doesn't need to be unit length
        point2 (np.ndarray): second point of size [3], doesn't need to be unit length

    Returns:
        np.ndarray: rotation matrix of size [3 x 3]
    """
    point1 = point1 / np.linalg.norm(point1)
    point2 = point2 / np.linalg.norm(point2)
    v_hat = skew(np.cross(point1, point2))
    c = np.dot(point1.transpose(), point2)

    return np.eye(3) + v_hat + (np.dot(v_hat, v_hat) / (1 + c))


def projection_jacobian(feature_vector: np.ndarray) -> np.ndarray:
    feature_vector = feature_vector[:, None]
    norm = np.linalg.norm(feature_vector)
    return (np.eye(3) * norm**2 - feature_vector * feature_vector.transpose()) / norm**3


def unscented_transform(point: np.ndarray, covariance: np.ndarray, omnidirectional: bool = False, kappa: float = 1.0) -> np.ndarray:
    """Perfoms the unscented transformation of a covariance from the image plane of a pinhole/omnidirectional camera onto the unit sphere in 3D. Accounts for the rank deficiency of the covariance (max rank = 2).

    Args:
        point (np.ndarray): point in the image plane of size [3]
        covariance (np.ndarray): covariance corresponding to the point of size [3 x 3]
        omnidirectional (bool, optional): Set to True if a omnidirectional camera is used. Defaults to False.
        kappa (float, optional): Parameter for the unscented transform. Defaults to 1.0.

    Returns:
        np.ndarray: Covariance after the unscented transform of size [3 x 3]
    """
    n = 2

    # Calculate rotation matrix for omnidirectional cameras
    rotation = np.eye(3)
    if omnidirectional:
        rotation = rotation = rotation_between_points(
            np.array([0.0, 0.0, 1.0]), point / np.linalg.norm(point))

    cov_2d = np.linalg.multi_dot(
        [rotation.transpose(), covariance, rotation])[:2, :2]
    chol_cov = np.linalg.cholesky(cov_2d)

    weights = [kappa / (n + kappa)]
    points = [point]
    for i in range(n):
        points.append(
            point + np.dot(rotation, np.array([chol_cov[i, 0], chol_cov[i, 1], 0.0])))
        weights.append(0.5 / (n + kappa))
        points.append(
            point - np.dot(rotation,
                           np.array([chol_cov[i, 0], chol_cov[i, 1], 0.0])))
        weights.append(0.5 / (n + kappa))

    transformed_points = []
    for p, w in zip(points, weights):

        t_p = p / np.linalg.norm(p)
        transformed_points.append(t_p)

    mean = np.zeros((3, 1))
    for t_p, w in zip(transformed_points, weights):
        mean += w * t_p[None].transpose()

    cov = np.zeros((3, 3))
    for t_p, w in zip(transformed_points, weights):
        cov += w * np.dot((t_p[None].transpose() - mean),
                          (t_p[None].transpose() - mean).transpose())

    return cov


def getYPR(poses: pd.DataFrame) -> pd.DataFrame:
    """Calculate the roll, pitch, yaw angles from the poses

    Args:
        poses (pandas.Dataframe): pandas dataframe that holds timestamped poses with orientation and position

    Returns:
        pandas.Dataframe: timestamped yaw pitch and roll
    """
    names = ["yaw", "pitch", "roll"]

    return pd.DataFrame([R.from_matrix(pose.rotationMatrix()).as_euler('ZYX', degrees=True) for pose in poses.to_list()],
                        columns=names, index=poses.index)
