# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
import sophus as sp
from scipy.stats import multivariate_normal
from typing import List

from pnec_common.helper import projection_jacobian


class SimCamera:
    def __init__(self, pose: sp.SE3 = sp.SE3(), focal_length: float = 1.0):
        self.pose: sp.SE3 = pose
        self.focal_length: float = focal_length

        self.points: np.ndarray = np.empty([0, 3])
        self.image_points: np.ndarray = np.array([])
        self.feature_vectors: np.ndarray = np.array([])
        self.bearing_vectors: np.ndarray = np.array([])
        self.covariances: np.ndarray = np.empty([0, 3, 3])
        self.bv_covariances: np.ndarray = np.array([])

    def points_from_wc(self, points: np.ndarray):
        a = self.pose.inverse() * points
        self.points = np.append(
            self.points, self.pose.inverse() * points, axis=0)
        covariances = np.ones((points.shape[0], 3, 3)) * -1
        self.covariances = np.append(self.covariances, covariances, axis=0)

    def process_points(self):
        self.project_points()
        self.create_bearing_vectors()

    def project_points(self):
        self.image_points = (
            self.points[:, 0:2] / self.points[:, 2][:, None]) * self.focal_length

    def create_bearing_vectors(self):
        self.feature_vectors = np.ones(
            (self.image_points.shape[0], 3)) * self.focal_length
        self.feature_vectors[:, :-1] = self.image_points
        self.bearing_vectors = self.feature_vectors / \
            np.linalg.norm(self.feature_vectors, axis=1)[:, None]

        self.bv_covariances = np.zeros(self.covariances.shape)
        for index, covariance in enumerate(self.covariances):
            bv_hat = projection_jacobian(self.feature_vectors[index, :])
            self.bv_covariances[index, :, :] = np.linalg.multi_dot(
                [bv_hat, covariance, bv_hat.transpose()])

    def add_noise(self, index: int, image_covariance: np.ndarray):
        self.covariances[index, :, :] = image_covariance
        self.image_points[index, :] = self.image_points[index, :] + \
            np.random.multivariate_normal((0, 0), image_covariance[:2, :2], 1)
