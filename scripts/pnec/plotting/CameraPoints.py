# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
import sophus as sp


def CameraPoints(ax, points: np.ndarray, camera_pose: sp.SE3 = sp.SE3(), color: str = 'k', arrow_length_ratio: float = 0.0, linewidths: float = 1.0):
    translation = camera_pose.translation()
    for point in points:
        point_direction = np.matmul(camera_pose.rotationMatrix(), point)
        ax.quiver(translation[0], translation[1], translation[2], point_direction[0], point_direction[1],
                  point_direction[2], arrow_length_ratio=arrow_length_ratio, linewidths=linewidths, color=color)
