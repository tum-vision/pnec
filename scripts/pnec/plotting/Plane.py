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
from pnec.math import rotation_between_points

def Plane(normal: np.ndarray, center: np.ndarray, width: float = 1.0, height: float = 1.0, color: str = 'k', alpha: float = 0.5) -> Poly3DCollection:
    # Creates a box object for a camera pose
    # get the box through self.box
    rotation = rotation_between_points(np.array([0.0, 0.0, 1.0]), normal)
    coordinate_frame = sp.SE3(rotation, center)

    bl = coordinate_frame * np.array([-width / 2, -height / 2, 0.0])
    br = coordinate_frame * np.array([width / 2, -height / 2, 0.0])
    ul = coordinate_frame * np.array([-width / 2, height / 2, 0.0])
    ur = coordinate_frame * np.array([width / 2, height / 2, 0.0])

    vertices = [[bl, br, ur, ul]]
    box = Poly3DCollection(
        verts=vertices, facecolors=color, alpha=alpha)
    return box
