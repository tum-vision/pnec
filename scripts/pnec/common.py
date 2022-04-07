# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

import numpy as np
import pnec.math


def pnec_energy_rotations(rotations: np.ndarray, t: np.ndarray, fi: np.ndarray, fi_prime: np.ndarray, sigmas: np.ndarray, reg: float = 1e-10) -> np.ndarray:
    """Calculate the PNEC Energy function for different rotations.
                           (t^T (fi x R fi_prime))^2
    E_P = sum -----------------------------------------------
              (t^T \hat(fi) R sigma_i R^T \hat(fi)^T t) + reg
    Args:
        rotations (np.ndarray): rotations for which the PNEC should be calculated of size [n, m, 3, 3]
        t (np.ndarray): translation for the energy function of size [3]
        fi (np.ndarray): bearing vectors in the first frame of size [k x 3]
        fi_prime (np.ndarray): bearing vectors in the second frame of size [k x 3]
        sigmas (np.ndarray): covariance matrices of the second bearing vectors of size [k x 3 x 3]
        reg (float, optional): regularization for the PNEC energy function. Defaults to 1e-10.

    Returns:
        np.ndarray: Energy function values for the different rotations of size [n, m]
    """
    fi_skew = pnec.math.skew(fi)  # [k x 3 x 3]
    ni = np.einsum('i,jik,nmkl,jl->nmj', t,
                   fi_skew, rotations, fi_prime)  # [n x m x k]
    ni_2 = np.square(ni)  # [n, m, k]

    sigma = np.einsum('i,jik,nmkl,jlo,nmpo,jqp,q->nmj',
                      t, fi_skew, rotations, sigmas, rotations, fi_skew, t) + reg  # [n x m x k]

    return np.divide(ni_2, sigma).sum(-1)  # [n x m]


def nec_energy_rotations(rotations: np.ndarray, t: np.ndarray, fi: np.ndarray, fi_prime: np.ndarray) -> np.ndarray:
    """Calculate the NEC Energy function for different rotations.

    E = sum (t^T (fi x R fi_prime))^2

    Args:
        rotations (np.ndarray): rotations for which the NEC should be calculated of size [n, m, 3, 3]
        t (np.ndarray): translation for the energy function of size [3]
        fi (np.ndarray): bearing vectors in the first frame of size [k x 3]
        fi_prime (np.ndarray): bearing vectors in the second frame of size [k x 3]

    Returns:
        np.ndarray: Energy function values for the different rotations of size [n, m]
    """
    fi_skew = pnec.math.skew(fi)  # [k x 3]
    ni = np.einsum('i,jik,nmkl,jl->nmj', t,
                   fi_skew, rotations, fi_prime)  # [n x m x k]
    ni_2 = np.square(ni)  # [n x m x k]

    return ni_2.sum(-1)  # [n x m]


def pnec_energy_translations(ts: np.ndarray, r: np.ndarray, fi: np.ndarray, fi_prime: np.ndarray, sigmas: np.ndarray, reg: float = 10e-10) -> np.ndarray:
    """Calculate the PNEC Energy function for different rotations.
                         (t ^ T(fi x R fi_prime)) ^ 2
    E_P = sum -----------------------------------------------------
              (t ^ T \hat(fi) R sigma_i R ^ T \hat(fi) ^ T t) + reg
    Args:
        ts (np.ndarray): translations for which the PNEC should be calculated of size [n, m, 3]
        r (np.ndarray): rotation for the energy function of size [3 x 3]
        fi (np.ndarray): bearing vectors in the first frame of size [k x 3]
        fi_prime (np.ndarray): bearing vectors in the second frame of size [k x 3]
        sigmas (np.ndarray): covariance matrices of the second bearing vectors of size [k x 3 x 3]
        reg (float, optional): regularization for the PNEC energy function. Defaults to 1e-10.

    Returns:
        np.ndarray: Energy function values for the different translations of size [n, m]
    """
    fi_skew = pnec.math.skew(fi)  # [k x 3 x 3]
    ni = np.einsum('nmi,jik,kl,jl->nmj', ts, fi_skew,
                   r, fi_prime)  # [n x m x k]
    ni_2 = np.square(ni)  # [n x m x k]

    sigma = np.einsum('nmi,jik,kl,jlo,po,jqp,nmq->nmj',
                      ts, fi_skew, r, sigmas, r, fi_skew, ts) + reg  # [n x m x k]

    return np.divide(ni_2, sigma).sum(-1)  # [n x m]


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
