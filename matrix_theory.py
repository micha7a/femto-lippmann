# -*- coding: utf-8 -*-
"""
1D matrix theory porting from Gilles Lippmann.

TODO (michalina) describe matrix theory

We need:
 - a forward in space model, call it forward model (incoming and outgoing wave known on the one end of the boundary)
 - a forward in time model, call it backward model (incoming waves known on both ends of the boundary)
 - does it make sense to consider a mixed option?
 - propagation + boundary matrix, that can be used just as:
    - boundary
    - propagation

We could consider writing it in the tensor form hoping it would be faster, but who knows...
"""

from wave import *


def swap_waves(matrix: np.ndarray) -> np.ndarray:
    result = np.empty_like(matrix, dtype=complex)
    result[0, 0] = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    result[0, 1] = matrix[0, 1]
    result[1, 0] = - matrix[1, 0]
    result[1, 1] = 1.0
    return result/matrix[1, 1]


def propagation_matrix(n: complex, dz: float, k: float, backward: bool = False) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=complex)
    phi = n*dz*k
    matrix[0, 0] = np.exp(1j*phi)
    if not backward:
        matrix[1, 1] = np.exp(1j * phi.conjugate())
    else:
        matrix[1, 1] = np.exp(-1j * phi.conjugate())
    return matrix


def boundary_matrix(n1:complex, n2: complex, backward: bool = False) -> np.ndarray:
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
