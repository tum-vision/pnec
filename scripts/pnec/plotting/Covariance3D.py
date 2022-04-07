# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
import sophus as sp
from pnec_common.helper import get_cov_ellipsoid


def Covariance3D(ax, covariance: np.ndarray, mu: np.ndarray, camera_pose: sp.SE3 = sp.SE3(), color: str = 'k', alpha: float = 1.0):
    mu = camera_pose * mu
    covariance = np.linalg.multi_dot(
        [camera_pose.rotationMatrix(), covariance, camera_pose.rotationMatrix().transpose()])

    x, y, z = get_cov_ellipsoid(covariance, mu)
    ax.plot_surface(x, y, z, rstride=3, cstride=3,
                    linewidth=0.1, alpha=alpha, shade=True, color=color)
