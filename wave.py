# -*- coding: utf-8 -*-
"""
A 1D Wave module for Femto-Lippmann project.

Currently represents only a planar wave, with arbitrary spectrum, including
specific examples of single frequency, gaussian and gaussian chirped spectrum.

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
from __future__ import annotations

import numpy as np

import constants as c


def sigmoid(x, x_min, x_max, percentile=0.01):
    """Function that calculates sigmoid saturating around given values"""
    rate = -2 * np.log(percentile) / (x_max - x_min)
    center = (x_min + x_max) / 2
    return 1 / (1 + np.exp(-rate * (x - center)))


class PlanarWave(object):
    """A class representing a planar wave propagating along z axis

    A single frequency of a PlanarWave is parametrized by positive wavenumber:
    k = omega / C where C is the speed of light in order to avoid unnecessary
    multiplication and division by large number C. The fact that k is always
    positive means that the wave is described in it's local frame, and that
    there can't be constant (frequency 0) component.

    Attributes:
        s: values of spectrum
        k: static, (absolute) wavenumbers for which the spectra are defined
    """
    k = c.DEFAULT_K

    @classmethod
    def dk(cls):
        return cls.k[1] - cls.k[0]

    def __init__(self, spectrum_array: np.ndarray = np.empty(0), **kwargs):
        if spectrum_array.size != 0 and spectrum_array.size != self.k.size:
            raise ValueError("The spectrum_array must be of len(Spectrum.k), default {}".format(c.OMEGA_STEPS))
        self.s = np.array(spectrum_array, dtype=complex)

    def __mul__(self, other) -> PlanarWave:
        if isinstance(other, (complex, float, int, np.ndarray)):
            return PlanarWave(other * self.s)
        if isinstance(other, PlanarWave):
            return PlanarWave(self.s * other.s.conj())
        raise NotImplementedError

    __rmul__ = __mul__

    def __add__(self, other) -> PlanarWave:
        if not np.allclose(self.k, other.k):
            raise NotImplementedError
        return PlanarWave(spectrum_array=self.s + other.s)

    def __eq__(self, other):
        """Overrides the default implementation using almost equal"""
        if isinstance(other, PlanarWave):
            return np.allclose(self.k, other.k) and np.allclose(self.s, other.s)
        return False

    def amplitude(self, z, time=0) -> np.array:
        transform = np.exp(1j * (z - c.C * time)[:, None] @ self.k[None, :])
        return self.dk() * transform @ self.s[:, None]

    def power_spectrum(self):
        return np.abs(self.s)**2

    def total_energy(self):
        return np.sum(self.power_spectrum()) * self.dk()

    def set_energy(self, energy):
        self.s = self.s / np.sqrt(self.total_energy()) * np.sqrt(energy)

    def from_amplitude(self, amplitude, z, time=0):
        transform = np.exp(1j * self.k[:, None] @ (-z - c.C * time)[None, :])
        self.s = (z[1] - z[0]) * transform @ amplitude[:, None]

    def delay(self, time):
        self.s = self.s * np.exp(1j * c.C * time * self.k)

    def shift(self, z):
        self.s = self.s * np.exp(1j * z * self.k)

    def wavelength(self):
        return 2 * np.pi / self.k

    def plot(self, ax, wavelength=False, spectrum_axis=None, **kwargs):
        if self.s.size == 0:
            raise ValueError("Can't plot empty spectrum")
        x = self.wavelength() if wavelength else self.k
        ax.set_xlabel(r"$\lambda$ [m]" if wavelength else r"k [1/m]")
        ax.xaxis.set_major_formatter(c.FORMATTER)
        ax.plot(x, self.power_spectrum(), **kwargs)
        color = ax.get_lines()[-1].get_color()
        ax.set_ylabel("power spectrum", color=color)
        ax.tick_params(axis='y', labelcolor=color)
        if spectrum_axis is not None:
            if "color" not in kwargs:
                color = ax._get_lines.get_next_color()
                kwargs["color"] = color
            else:
                color = kwargs["color"]
            spectrum_axis.plot(x, np.real(self.s), **kwargs)
            spectrum_axis.plot(x, np.real(self.s), **kwargs)
            spectrum_axis.tick_params(axis='y', labelcolor=color)
            spectrum_axis.set_ylabel("spectrum", color=color)

    def plot_amplitude(self, z, ax, label="", **kwargs):
        if self.s.size == 0:
            raise ValueError("Can't plot empty spectrum")
        amplitude = self.amplitude(z)
        ax.plot(z, np.real(amplitude), label="amplitude {}".format(label), **kwargs)
        ax.plot(z, np.abs(amplitude), label="envelope {}".format(label), **kwargs)
        ax.set_xlabel("z[m]")
        ax.xaxis.set_major_formatter(c.FORMATTER)

    @staticmethod
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

    @classmethod
    def single_propagation_matrix(cls: PlanarWave, n: complex, dz: float, backward: bool = False) -> np.ndarray:
        matrix = np.zeros((2, 2), dtype=complex)
        phi = n * dz * cls.k
        matrix[0, 0] = np.exp(1j * phi)
        if not backward:
            matrix[1, 1] = np.exp(-1j * phi)
        else:
            matrix[1, 1] = np.exp(1j * phi)
        return matrix

    @classmethod
    def propagation_matrix(cls: PlanarWave, n: complex, dz: float, backward: bool = False) -> np.ndarray:
        matrix = np.zeros((2, 2, len(cls.k)), dtype=complex)
        phi = n * dz * cls.k
        matrix[0, 0, :] = np.exp(1j * phi)
        if not backward:
            matrix[1, 1, :] = np.exp(-1j * phi)
        else:
            matrix[1, 1, :] = np.exp(1j * phi)
        return matrix

    @staticmethod
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


class DeltaPlanarWave(PlanarWave):
    """A class representing a single frequency wave."""
    def __init__(self,
                 energy: float = c.SINGLE_PULSE_ENERGY,
                 wavelength: float = None,
                 wavenumber: float = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.s = np.zeros_like(self.k)
        if wavenumber is None:
            if wavelength is None:
                raise ValueError("One of wavelength, wavenumber hast o be provided")
            wavenumber = 2 * np.pi / wavelength
        idx = int((wavenumber - self.k[0]) / self.dk())
        if idx > len(self.k) or idx < 0:
            raise ValueError("Wavenumber {} effectively outside the spectrum: {}-{}".format(
                wavenumber, self.k[0], self.k[-1]))
        self.s[idx] = 1
        self.set_energy(energy)


class GaussianPlanarWave(PlanarWave):
    """A class representing gaussian spectrum. Note that, since it is in 1D,
    it is not a gaussian beam."""
    def __init__(self,
                 energy: float = c.SINGLE_PULSE_ENERGY,
                 mean: float = c.GREEN,
                 std: float = 10 * c.NANO,
                 wavenumber: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.s = np.zeros_like(self.k)
        if not wavenumber:
            mean = 2 * np.pi / mean
            std = 2 * np.pi / std
        self.s = np.exp(-(self.k - mean)**2 / (2 * std**2))
        self.set_energy(energy)


class ChirpedPlanarWave(GaussianPlanarWave):
    """A class representing a chirped gaussian spectrum."""
    def __init__(self, skew: float = 0, **kwargs):
        super().__init__(**kwargs)
        if "wavenumber" in kwargs and not kwargs["wavenumber"] and skew != 0:
            skew = 2 * np.pi / skew
        self.s = self.s * np.exp(1j * skew * ((self.k - np.median(self.k))**2))
