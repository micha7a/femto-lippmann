import math
import numpy as np
from typing import Tuple

import submodules.lippmann.multilayer_optics_matrix_theory as mt

from scipy import signal


# TODO (michalina) switch to using tmm & scipy.stats


def wavelengths_omega_spaced(lambda_low=400e-9, lambda_high=700e-9, n=300, c0=299792458):
    omega_low = 2 * np.pi * c0 / lambda_high
    omega_high = 2 * np.pi * c0 / lambda_low
    omegas = np.linspace(omega_high, omega_low, n)
    return 2 * np.pi * c0 / omegas


def sigmoid(x, rate=6.0, center=0.5):
    return 1 / (1 + np.exp(-rate * (x - center)))


def sigmoid_inverse(y, min_margin=0.05, max_margin=0.95):
    y = y - np.min(y)
    y = y * (max_margin - min_margin) / (np.max(y))
    y = y + min_margin
    return np.log((y / (1 - y)))


def simulate_printing(blocks, block_height, block_length, max_index_change, z, base_index, lambdas):
    """scale might be a vector of different heights, delta_n is a scalar"""

    # transform to low level description
    distances, index_in_blocks = mt.blobs_to_matrices(
        begs=blocks,
        delta_n=block_height,
        ends=(blocks + block_length),
        n0=0)
    index_in_blocks = sigmoid(index_in_blocks) * max_index_change + base_index

    index_unif_spaced = mt.blobs_to_ref_index(blob_z0=blocks,
                                              blob_delta_z=block_length,
                                              n0=0,
                                              delta_n=block_height,
                                              depths=z)
    index_unif_spaced = sigmoid(index_unif_spaced) * max_index_change + base_index

    # calculate reflection:
    reflection, _ = mt.propagation_arbitrary_layers_Born_spectrum(index_in_blocks,
                                                                  d=distances,
                                                                  lambdas=lambdas,
                                                                  plot=False)

    return reflection, index_unif_spaced


def block_approximate(depths, values, block_size, scale, mass=True):
    blocks = []
    intensities_ = []
    for idx in range(len(values) - block_size):
        if mass:
            multiple = math.floor(np.sum(values[idx:idx + block_size]) / (scale * block_size))
        else:
            multiple = math.floor(np.min(values[idx:idx + block_size]) / scale)
        if multiple > 0:
            tmp_f = values[idx:idx + block_size] - multiple * scale
            tmp_f[tmp_f < 0.0] = 0.0
            values[idx:idx + block_size] = tmp_f
            blocks.append(depths[idx])
            intensities_.append(multiple * scale)
    if len(blocks) < 1:
        raise Warning('No blocks were created during approximation!')
    return np.array(blocks), np.array(intensities_)


def shift_domain(array: np.ndarray, x_min: float, x_max: float, new_x_min: float,
                 new_x_max: float) -> Tuple[np.ndarray, np.ndarray]:
    dx = (x_max - x_min) / (len(array) - 1)
    final_length = int(np.round((new_x_max - new_x_min)/dx) + 1)
    return shift_domain_const_length(array, x_min, dx, new_x_min, final_length)

def shift_domain_const_length(array, x_min, dx, new_x_min, new_length):

    left_margin = int(np.round((x_min - new_x_min) / dx))
    new_array = np.pad(array, pad_width=(max(left_margin, 0), max(new_length - left_margin - len(array), 0)),
                       mode='constant')
    new_array = new_array[max(0, -left_margin): max(0, -left_margin) + new_length]
    new_x = np.arange(new_length)
    new_x = new_x*dx + x_min-left_margin*dx
    return new_x, new_array


def front_interference_time(wave1: np.ndarray, z1: float, wave2: np.ndarray, z2: float, dt: float, c: float,
                            z_min: float, z_max: float, pulses1: int = 1, pulses2: int = 1, period1: float = 0,
                            period2: float = 0) -> Tuple[np.array, np.array, float]:
    corr = np.real(signal.correlate(wave1, wave2) * dt) # TODO (michalina) why there is no complex conjugate?
    dz = dt * c / 2
    corr_z_range = dz * np.floor((len(corr) - 1) / 2)
    final_length = int(np.round((z_max - z_min) / dz) + 1)
    z, result = shift_domain_const_length(np.zeros_like(corr), - corr_z_range, dz, z_min, final_length)
    for p1 in range(pulses1):
        for p2 in range(pulses2):
            center = (z1 + z2) / 2 + p1 * period1 * c - p2 * period2 * c
            z_tmp, contribution = shift_domain_const_length(corr, center - corr_z_range,  dz, z_min, final_length)
            result += contribution
    return z, result, ((np.linalg.norm(wave1) * pulses1) ** 2 + (np.linalg.norm(wave2) * pulses2) ** 2) * dt


def front_interference_mirror(wave, z0, dt, c, depth, pulses=1, period=0) -> Tuple[np.array, np.array, float]:
    return front_interference_time(wave, -z0, wave, z0, dt, c, 0, depth, pulses1=pulses, pulses2=pulses,
                                   period1=period, period2=period)


def propagate(omegas, spectrum, z, t, c):
    return spectrum * np.exp(1j * omegas * (z / c - t))


def interference_with_phase(lambdas, spectrum, depths, shift, phase):
    """"Compute the Lippmann transform

        lambdas     - vector of wavelengths
        spectrum    - spectrum of light
        depths      - vector of depths
        shift       - shift in meters
        phase       - phase change in radians

        Returns intensity       - computed intensity of the interfering waves
                delta_intensity - the intensity without the baseline term"""""

    two_k = 4 * np.pi / lambdas

    one_minus_cosines = 0.5 * (1 - np.cos(two_k[None, :] * (depths[:, None]-shift/2) - phase))
    cosines = 0.5 * np.cos(two_k[None, :] * (depths[:, None]-shift/2) - phase)

    intensity = -np.trapz(one_minus_cosines * spectrum[None, :], two_k, axis=1)
    delta_intensity = np.trapz(cosines * spectrum[None, :], two_k, axis=1)
    return intensity, delta_intensity
