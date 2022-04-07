# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import sophus as sp


def CameraBox(camera_pose: sp.SE3, width=2.0, height=1.0, depth=1.0, scale=1.0, linewidth=0.5, color='k') -> Poly3DCollection:
    # Creates a box object for a camera pose
    # get the box through self.box
    bl = camera_pose * (np.array([-width / 2, -height / 2, depth]) * scale)
    br = camera_pose * (np.array([width / 2, -height / 2, depth]) * scale)
    ul = camera_pose * (np.array([-width / 2, height / 2, depth]) * scale)
    ur = camera_pose * (np.array([width / 2, height / 2, depth]) * scale)
    base = camera_pose.translation()

    vertices = [[bl, br, base], [br, ur, base], [
        ur, ul, base], [ul, bl, base], [bl, br, ur, ul]]
    box = Poly3DCollection(
        verts=vertices, linewidth=linewidth, edgecolors=color, alpha=0.0, zorder=30)
    return box
