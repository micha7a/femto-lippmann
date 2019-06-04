# -*- coding: utf-8 -*-
"""

This file implements basic matrix theory that can be used by different classes.
Each propagation can be described in two representations:
 - a forward in space model, we call it a forward model (incoming and outgoing wave known on the one end of the
 boundary)
 - a forward in time model, we call it a backward model (incoming waves known on both ends of the boundary)

 In order to switch between the models use `swap_waves`
"""

from wave import *


def swap_waves(matrix: np.ndarray) -> np.ndarray:
    """ Change basis between forward and backward representations.

    Args:
        matrix: a 2x2 matrix in any representation (forward or backward)

    Returns:
        a 2x2 matrix in the different representation than matrix

    """
    result = np.empty_like(matrix, dtype=complex)
    result[0, 0] = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    result[0, 1] = matrix[0, 1]
    result[1, 0] = -matrix[1, 0]
    result[1, 1] = 1.0
    return result / matrix[1, 1]


def propagation_matrix(n: complex, dz: float, k: float, backward: bool = False) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=complex)
    phi = n * dz * k
    matrix[0, 0] = np.exp(1j * phi)
    if not backward:
        matrix[1, 1] = np.exp(-1j * phi)  # TODO should this be conjugate or not?
    else:
        matrix[1, 1] = np.exp(1j * phi)
    return matrix


def boundary_matrix(n1: complex, n2: complex, backward: bool = False) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=complex)
    if not backward:
        matrix[0, 0] = n1 + n2
        matrix[0, 1] = n2 - n1
        matrix[1, 0] = n2 - n1
        matrix[1, 1] = n1 + n2
        matrix = matrix / (2 * n2)
    else:
        matrix[0, 0] = 2 * n1
        matrix[0, 1] = n2 - n1
        matrix[1, 0] = n1 - n2
        matrix[1, 1] = 2 * n2
        matrix = matrix / (n1 + n2)
    return matrix
