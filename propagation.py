# -*- coding: utf-8 -*-
"""
Created on 23/08/2021

@author: Michalina Pacholska

This module is designed to be expanded with different wave models, as long as
their propagation can be described via matrix theory. Thus, the Planar Wave
also describes its propagation through the empty space and its refraction at
the boundary via matrix theory.

Each propagation can be described in two representations:
 - a forward in space model, we call it a forward model (incoming and outgoing
 wave known on the one end of the boundary)
 - a forward in time model, we call it a backward model (incoming waves known
 on both ends of the boundary)

 In order to switch between the propagation models use `swap_waves`

"""

import numpy as np
import wave as w


def change_basis(matrix: np.ndarray) -> np.ndarray:
    """ Change basis between forward and backward representations of propagation matrices.

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


def identity_matrix(wave: w.PlanarWave = None):
    if wave is not None:
        matrix = np.zeros((2, 2, len(wave.k)), dtype=complex)
    else:
        matrix = np.zeros((2, 2, len(w.PlanarWave.k)), dtype=complex)
    matrix[0, 0] = 1
    matrix[1, 1] = 1
    return matrix


def single_propagation_matrix(k: float, n: complex, dz: float, backward: bool = False) -> np.ndarray:
    """Create a propagation matrix for a single wavenumber k

    Args:
        k: wavenumber for which the matrix is created
        n: index of refraction of the layer
        dz: depth of the layer
        backward: if True use backward model (see the module documentation)

    Returns:
        3D tensor of propagation matrices for all wavenumbers if dimensions (2, 2, # wavenumbers)"""
    matrix = np.zeros((2, 2), dtype=complex)
    phi = n * dz * k
    matrix[0, 0] = np.exp(1j * phi)
    if not backward:
        matrix[1, 1] = np.exp(-1j * phi)
    else:
        matrix[1, 1] = np.exp(1j * phi)
    return matrix


def propagation_matrix(
    n: complex,
    dz: float,
    backward: bool = False,
    wave: w.PlanarWave = None,
) -> np.ndarray:
    """
    Create 3D tensor of propagation matrices for all wavenumbers

    Args:
        n: index of refraction of the layer
        dz: depth of the layer
        backward: if True use backward model (see the module documentation)

    Returns:
        3D tensor of propagation matrices for all wavenumbers if dimensions (2, 2, # wavenumbers)
    """

    ks = w.PlanarWave.k
    if wave is not None:
        ks = wave.k
    matrix = np.zeros((2, 2, len(ks)), dtype=complex)
    phi = n * dz * ks
    matrix[0, 0, :] = np.exp(1j * phi)
    if not backward:
        matrix[1, 1, :] = np.exp(-1j * phi)
    else:
        matrix[1, 1, :] = np.exp(1j * phi)
    return matrix


def boundary_matrix(n1: complex, n2: complex, backward: bool = False) -> np.ndarray:
    """
    Create a matrix of reflection on the boundary (wavenumber independent)

    Args:
        n1: index of refraction of the first layer
        n2: index of refraction of the second layer
        backward: if True use backward model (see the module documentation)

    Returns:
         matrix of reflection on the boundary, of dimensions (2, 2)
    """

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


def reflection_matrix(r: float, backward: bool = False) -> np.ndarray:
    """
    Create a matrix describing reflection given the reflectivity;
    does not include propagation through the material

    Args:
        r: reflectivity
        backward: if True use backward model (see the module documentation)

    Returns:

    """

    matrix = np.zeros((2, 2), dtype=complex)
    if not backward:
        matrix[0, 0] = 1 - 2 * r
        matrix[0, 1] = r
        matrix[1, 0] = -r
        matrix[1, 1] = 1
        matrix = matrix / (1 - r)
    else:
        matrix[0, 0] = 1 - r
        matrix[0, 1] = r
        matrix[1, 0] = r
        matrix[1, 1] = 1 - r
    return matrix


def swap_waves(matrix, forward_wave, backward_wave):
    incoming = np.array([forward_wave.s, backward_wave.s])
    outgoing = np.einsum('ijk,jk->ik', matrix, incoming)
    return forward_wave, w.PlanarWave(outgoing[1])
