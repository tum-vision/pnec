# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
import math
from mpmath import mp, mpf, matrix, norm
mpf("inf")


def fibonacci_sphere(samples=1):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)


def obj_fun(X, Ai, Bi, n=3, k=10):
    assert Ai.ndim == 3 and Bi.ndim == 3
    assert X.ndim == 2 and X.shape[-1] == n
    B = X.shape[0]

    nums = Ai[None] * X[:, None, None, :] * X[:, None, :, None]  # (B, k, n, n)
    assert list(nums.shape) == [B, k, n, n]

    dens = Bi[None] * X[:, None, None, :] * X[:, None, :, None]  # (B, k, n, n)
    assert list(dens.shape) == [B, k, n, n]

    nums, dens = np.sum(nums, (-1, -2)), np.sum(dens, (-1, -2))  # (B, k)
    assert np.all(dens > 0)
    fracs = nums/dens  # (B, k)

    return np.sum(fracs, -1)


def cross(a, b):
    c = matrix([a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]])

    return c


def rotate_vector(v, angle, axis):
    axis = axis/norm(axis)
    angle = mpf(angle)

    v_rot = mp.cos(angle)*v + mp.sin(angle)*cross(axis, v) + \
        (mpf(1) - mp.cos(angle))*(axis.T*v)[0, 0]*axis

    assert norm(v_rot)-norm(v) < mpf(f'1e-{mp.dps//2}')

    return v_rot


def tonp(x, dtype):
    if not isinstance(x, (list, tuple)):
        l = x.tolist()
    else:
        l = x
    l = [[dtype(x) for x in ll] for ll in l]
    return np.array(l)


# SCF Algo
# See: https://rc.library.uta.edu/uta-ir/bitstream/handle/10106/28093/BINBUHAER-DISSERTATION-2019.pdf?sequence=1&isAllowed=y


def phi_G(G, X, n=3, k=10):
    assert G.ndim == 3
    assert X.ndim == 2 and X.shape[-1] == n
    B = X.shape[0]

    vals = G[None] * X[:, None, None, :] * X[:, None, :, None]  # (B, k, n, n)
    assert list(vals.shape) == [B, k, n, n]
    vals = np.sum(vals, (-1, -2))

    return vals


def construct_E(X, Ai, Bi, n=3, k=10):
    assert Ai.ndim == 3 and Bi.ndim == 3
    assert X.ndim == 2 and X.shape[-1] == n
    B = X.shape[0]

    phi_A = phi_G(G=Ai, X=X)  # (B, k)
    phi_B = phi_G(G=Bi, X=X)  # (B, k)

    prod_phi_B = np.prod(phi_B, -1)  # (B, )
    coeffs = prod_phi_B[:, None] / phi_B  # (B, k)

    frac = phi_A / phi_B  # (B, k)

    Mi = (Ai[None] - frac[:, :, None, None]*Bi[None])  # (B, k, n, n)
#     Mi_power_km1 = np.linalg.matrix_power(Mi, n=k-1)

#     Es = coeffs[:, :, None, None] * Mi_power_km1  # (B, k, n, n)
    Es = coeffs[:, :, None, None] * Mi  # (B, k, n, n)
    assert list(Es.shape) == [B, k, n, n]
    Es = np.sum(Es, 1)  # (B, n, n)

    return Es


def scf(X0, Ai, Bi, steps=10, n=3):
    assert Ai.ndim == 3 and Bi.ndim == 3
    assert X0.ndim == 2 and X0.shape[-1] == n
    B = X0.shape[0]

    assert np.allclose((X0**2).sum(-1), 1)

    def comp_res(X):
        assert np.allclose((X**2).sum(-1), 1)

        Es = construct_E(X, Ai=Ai, Bi=Bi)  # (B, n, n)

        EsX = np.sum(Es*X[:, None, :], -1)  # (B, n)
        n_EsX = np.sqrt(np.sum(EsX**2, -1))  # (B)
        n_Es = np.linalg.norm(Es, ord=2, axis=(-1, -2))  # (B)

        # we know that norm of X is 1
        return n_EsX / n_Es

    Xs = [X0]
    res = [comp_res(X0)]
    funs = [obj_fun(X=Xs[-1], Ai=Ai, Bi=Bi)]

    for step in range(steps):
        Es = construct_E(X=Xs[-1], Ai=Ai, Bi=Bi)

        eigvals, eigvecs = np.linalg.eigh(Es)
#         idx = np.argsort(np.abs(eigvals), -1)
        # Seems they mean largest by dominant
#         idx = np.argsort(eigvals, -1)
#         dom_eigvecs = np.stack([eigvecs[i, :, idx[i, -1]] for i in range(X.shape[0])])
        dom_eigvecs = eigvecs[:, :, -1]
        assert np.allclose((dom_eigvecs**2).sum(-1), 1)

#         new = Xs[-1] + dom_eigvecs
#         new = new / np.sqrt(np.sum(new**2, -1, keepdims=True))

        Xs.append(dom_eigvecs)
        res.append(comp_res(X=Xs[-1]))
        funs.append(obj_fun(X=Xs[-1], Ai=Ai, Bi=Bi))

    return Xs, funs, res
