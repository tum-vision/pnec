# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
import pandas as pd
import sophus as sp
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation as R
from math import pi, sin, cos


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def getYPR(poses):
    """Calculate the roll, pitch, yaw angles from the poses

    Args:
        poses (pandas.Dataframe): pandas dataframe that holds timestamped poses with orientation and position

    Returns:
        pandas.Dataframe: timestamped yaw pitch and roll
    """
    names = ["yaw", "pitch", "roll"]

    return pd.DataFrame([R.from_matrix(pose.rotationMatrix()).as_euler('ZYX', degrees=True) for pose in poses.to_list()],
                        columns=names, index=poses.index)


def flip_estimated(estimated, flip_yaw=False, flip_pitch=False, flip_roll=False):
    """flip orientation at yaw, pitch and roll if necessary

    Args:
        estimated (pandas Dataframe): dataframe that holds the poses that need to be flipped
        flip_yaw (bool, optional): set true if you want to flip yaw. Defaults to False.
        flip_pitch (bool, optional): set true if you want to flip pitch. Defaults to False.
        flip_roll (bool, optional): set true if you want to flip roll. Defaults to False.

    Returns:
        pandas Dataframe: dataframe with flipped orientations
    """
    for index, pose in estimated.iterrows():
        yaw, pitch, roll = R.from_matrix(
            pose["poses"].rotationMatrix()).as_euler('ZYX')
        translation = pose["poses"].translation()
        if flip_yaw:
            yaw = - yaw
        if flip_pitch:
            pitch = - pitch
        if flip_roll:
            roll = -roll
        estimated.at[index, 'poses'] = sp.SE3(R.from_euler(
            'ZYX', [yaw, pitch, roll]).as_matrix(), translation)
    return estimated


def get_cov_ellipsoid(cov, mu=np.zeros(3), nstd=1):
    """
        Return the 3d points representing the covariance matrix
        cov centred at mu and scaled by the factor nstd.
        Plot on your favourite 3d axis.
        Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
        Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
        """
    assert cov.shape == (3, 3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov, axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(np.absolute(eigvals))

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    x = rx * np.outer(np.cos(theta), np.sin(phi))
    y = ry * np.outer(np.sin(theta), np.sin(phi))
    z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = x.shape
    # Flatten to vectorise rotation
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    x, y, z = np.matmul(eigvecs, np.array([x, y, z]))
    x, y, z = x.reshape(old_shape), y.reshape(
        old_shape), z.reshape(old_shape)

    # Add in offsets for the mean
    x = x + mu[0]
    y = y + mu[1]
    z = z + mu[2]

    return x, y, z


def create_points(distribution=multivariate_normal(np.array([2.0, 0.0, 0.0]), np.eye(3) * 1.0), num_samples=3):
    if num_samples == 1:
        points = distribution.rvs(size=num_samples, random_state=1234)
    else:
        points = distribution.rvs(size=num_samples, random_state=1234)
    return points


def sample_covariances(sigma_type, scale, num_covariances):
    offset = 0.5
    covariances = np.zeros((num_covariances, 3, 3))
    if sigma_type == "isotropic homogeneous":
        for i in range(num_covariances):
            covariance2d = np.eye(2) * scale
            covariance = np.zeros((3, 3))
            covariance[0:2, 0:2] = covariance2d
            covariances[i, :, :] = covariance
    elif sigma_type == "isotropic inhomogeneous":
        for i in range(num_covariances):
            covariance2d = np.eye(2) * (np.random.rand(1) + offset) * scale
            covariance = np.zeros((3, 3))
            covariance[0:2, 0:2] = covariance2d
            covariances[i, :, :] = covariance
    elif sigma_type == "anisotropic homogeneous":
        beta = (np.random.rand(1) + 1) / 2
        for i in range(num_covariances):
            alpha = np.random.rand(1) * pi
            R = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
            covariance2d = scale * \
                np.linalg.multi_dot(
                    [R, np.array([[beta, 0.0], [0.0, 1.0 - beta]]), R.transpose()])
            covariance = np.zeros((3, 3))
            covariance[0:2, 0:2] = covariance2d
            covariances[i, :, :] = covariance
    elif sigma_type == "anisotropic inhomogeneous":
        for i in range(num_covariances):
            beta = (np.random.rand(1) + 1) / 2
            alpha = np.random.rand(1) * pi
            R = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
            covariance2d = scale * \
                np.linalg.multi_dot(
                    [R, np.array([[beta, 0.0], [0.0, 1.0 - beta]]), R.transpose()])
            covariance = np.zeros((3, 3))
            covariance[0:2, 0:2] = covariance2d
            covariances[i, :, :] = covariance
    else:
        print("{} not implemented".format(sigma_type))

    return covariances
