# -*- coding: utf-8 -*-
"""
A 1D Wave module for Femto-Lippmann project
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import Union


C = 299792458
NANO = 1e-9
MICRO = 1e-6
VIOLET = 380 * NANO
GREEN = 500 * NANO
RED = 740 * NANO
OMEGA_STEPS = 100


def sigmoid(x, x_min, x_max, percentile=0.95):
    """Function that calculates sigmoid saturating around given values"""
    rate = - 2 * np.log(percentile) / (x_max - x_min)
    center = (x_min + x_max) / 2
    return 1 / (1 + np.exp(-rate * (x - center)))


class Spectrum(object):
    """A class representing a planar wave propagating along z axis

    Spectrum is parametrized by positive wavenumber: k = omega / C
    where C is the speed of light in order to avoid unnecessary multiplication
    and division by large number C. The fact that k is always positive means
    that the wave is described in it's local frame, and that there can't be
    constant (frequency 0) component.

    Attributes:
        s: values of spectrum
        k: static, (absolute) wavenumbers for which the spectra are defined
    """
    k = np.linspace(2 * np.pi / RED, 2 * np.pi / VIOLET, OMEGA_STEPS)

    def __init__(self,
                 spectrum_array: np.ndarray = np.empty(0),
                 **kwargs):
        if spectrum_array.size != 0 and spectrum_array.size != self.k.size:
            raise ValueError(
                "The spectrum_array must be of len(Spectrum.k), default {}".format(OMEGA_STEPS))
        self.s = spectrum_array

    def __mul__(self, other) -> Spectrum:
        if isinstance(other, (complex, float, int)):
            return Spectrum(other * self.s)
        if isinstance(other, Spectrum):
            return Spectrum(self.s * other.s.conj())
        raise NotImplementedError

    __rmul__ = __mul__

    def __add__(self, other) -> Spectrum:
        if not np.allclose(self.k, other.k):
            raise NotImplementedError
        return Spectrum(spectrum_array=self.s + other.s)

    def __eq__(self, other):
        """Overrides the default implementation using almost equal"""
        if isinstance(other, Spectrum):
            return np.allclose(self.k, other.k) and np.allclose(self.s, other.s)
        return False

    def amplitude(self, z, time=0) -> np.array:
        transform = np.exp(1j * (z - C * time)[:, None] @ self.k[None, :])
        return self.dk() * transform @ self.s[:, None]

    def power_spectrum(self):
        return np.abs(self.s) ** 2

    def power(self):
        return np.sum(self.power_spectrum()) * self.dk()

    def from_amplitude(self, amplitude, z, time=0):
        transform = np.exp(1j * self.k[:, None] @ (- z - C * time)[None, :])
        self.s = (z[1] - z[0]) * transform @ amplitude[:, None]

    def dk(self):
        return self.k[1] - self.k[0]

    def delay(self, time):
        self.s = self.s * np.exp(1j * C * time * self.k)

    def shift(self, z):
        self.s = self.s * np.exp(1j * z * self.k)

    def wavelength(self):
        return 2 * np.pi / self.k

    def plot(self, wavelength=False):  #  TODO(michalina) return the plot AND make pretty plots like gilles?
        if self.s.size == 0:
            raise ValueError("Can't plot empty spectrum")
        x = self.wavelength() if wavelength else self.k
        plt.plot(x, np.real(self.s), label="spectrum")
        plt.plot(x, self.power_spectrum(), label="power spectrum")
        plt.xlabel(r"$\lambda$" if wavelength else r"k")
        plt.legend()

    def plot_amplitude(self, z):
        if self.s.size == 0:
            raise ValueError("Can't plot empty spectrum")
        amplitude = self.amplitude(z)
        plt.plot(z, amplitude, label="amplitude")
        plt.plot(z, np.abs(amplitude), label="envelope")
        plt.legend()


class DeltaSpectrum(Spectrum):
    """A class representing a single frequency wave."""

    def __init__(self,
                 amplitude: complex = 1,
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
            raise ValueError(
                "Wavenumber {} effectively outside the spectrum: {}-{}".format(
                    wavenumber, self.k[0], self.k[-1]))
        self.s[idx] = amplitude


class GaussianSpectrum(Spectrum):
    """A class representing gaussian spectrum. Note that, since it is in 1D,
    it is not a gaussian beam."""

    def __init__(self,
                 amplitude: complex = 1,
                 mean: float = GREEN,
                 std: float = 10 * NANO,
                 wavenumber: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.s = np.zeros_like(self.k)
        if not wavenumber:
            mean = 2 * np.pi / mean
            std = 2 * np.pi / std
        self.s = amplitude * np.exp(- (self.k - mean) ** 2 / (2 * std ** 2))


class ChirpedSpectrum(GaussianSpectrum):
    """A class representing a chirped gaussian spectrum."""
    def __init__(self,
                 skew: float = 0,
                 **kwargs):
        super().__init__(**kwargs)
        if "wavenumber" in kwargs:
            print("we have wavenumber")
            if not kwargs["wavenumber"] and skew != 0:
                skew = 2 * np.pi / skew
        self.s = self.s * np.exp(1j * skew * self.k ** 2)


class Interference(object):
    """A class describing superposition of two waves going in opposite
    directions

    Attributes:
        forward: a wave propagating towards positive z (left to right)
        backward: a wave propagating towards negative z (right to left)
    """

    def __init__(self,
                 forward: Spectrum = Spectrum(np.empty(0)),
                 backward: Spectrum = Spectrum(np.empty(0))):
        assert np.allclose(forward.k, backward.k)
        self.forward = copy.deepcopy(forward)
        self.backward = copy.deepcopy(backward)

    def __mul__(self, alpha: complex) -> Interference:
        return Interference(self.forward * alpha, self.backward * alpha)

    __rmul__ = __mul__

    def amplitude(self, **kwargs):
        return self.backward.amplitude(**kwargs) + self.forward.amplitude(**kwargs)

    def __add__(self, other: Interference) -> Interference:
        return Interference(self.forward + other.forward,
                            self.backward + other.backward)

    __radd__ = __add__

    def power_spectrum(self):
        return self.forward.power_spectrum(), self.backward.power_spectrum()

    def intensity(self, z, **kwargs):
        correlation = 2 * np.real(
            (self.forward * self.backward).amplitude(z=2 * z, **kwargs))
        return np.squeeze(
            self.forward.power() + self.backward.power() + correlation)

    def dk(self):
        return self.forward.dk()

    # TODO(michalina) implement better getters? def __get__(self, instance, owner)


class Material(object):
    """An abstract class describing a material in which the waves can propagate

    Attributes:
        z: array of positions at which the material properties are described
        n0: basic index of refraction
        deposited_energy: energy currently deposited in the material,
        initialised to 0
        length: total length of the material
    """

    def __init__(self,
                 z: np.ndarray,
                 n0: Union[complex, np.ndarray] = 1,
                 **kwargs):
        assert np.imag(n0) >= 0
        self.z = z
        self.n0 = n0
        self.deposited_energy = np.zeros_like(z)
        self.length = z[-1] - z[0]

    def index_of_refraction(self) -> np.ndarray:
        raise NotImplementedError

    def energy_response(self, energy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transmitted(self, spectrum: Spectrum) -> Spectrum:
        raise NotImplementedError

    def reflected(self, spectrum: Spectrum) -> Spectrum:
        raise NotImplementedError

    def energy_distribution(self, itf: Interference) -> np.ndarray:
        total_amplitude = self.propagate(itf.forward, sign=1)\
                          + self.propagate(itf.backward, sign=-1)
        return np.sum((np.abs(total_amplitude) ** 2), axis=0) * itf.dk()

    def record(self, interference: Interference):
        energy = self.energy_distribution(interference)
        self.deposited_energy += self.energy_response(energy)

    def propagate(self, spectrum: Spectrum, sign: int = 1) -> np.array:
        """Calculate energy profile at each point of dielectric at each frequency"""
        raise NotImplementedError


class FixedDielectric(Material):
    """A class describing a dielectric with constant index of refraction n0
    that cannot be modified"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.n0, (complex, float, int)):
            raise ValueError("Fixed dielectric must have constant index n0")

    def index_of_refraction(self):
        return self.n0 * np.ones_like(self.z)

    def energy_response(self, energy):
        return np.zeros_like(energy)

    def transmitted(self, spectrum):
        spectrum = copy.deepcopy(spectrum)
        spectrum.shift(self.length * self.n0)
        return spectrum

    def reflected(self, spectrum):
        return 0 * spectrum

    def propagate(self, spectrum, sign=1):  # TODO make it an enum?
        if sign == 1:
            z = self.z - self.z[0]
        elif sign == -1:
            z = self.z[-1] - self.z
        else:
            raise ValueError("Sign has to have value 1 or -1")
        transform = np.exp(1j * self.n0 * spectrum.k[:, None] @ z[None, :])
        return np.repeat(spectrum.s[:, None], len(self.z), axis=1) * transform


class Dielectric(Material):
    """A class describing a dielectric with index of refraction that can
    be modified by a set of heuristics"""

    def __init__(self,
                 max_dn,
                 max_visible_energy,
                 min_energy,
                 max_energy,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_dn = max_dn
        self.max_visible_energy = max_visible_energy
        self.min_energy = min_energy
        self.max_d_energy = max_energy - min_energy

    def index_of_refraction(self):
        return self.n0 + self.max_dn * sigmoid(self.deposited_energy, 0, self.max_visible_energy)

    def energy_response(self, energy):
        factor = self.min_energy + self.max_d_energy * sigmoid(
            self.deposited_energy,
            0,
            self.max_visible_energy)
        return energy * factor / (self.min_energy + self.max_d_energy)


class LayeredMaterial(object):
    """A placeholder for a class describing a stack of materials"""

    def __init__(self, material_list):
        self.materials = material_list
