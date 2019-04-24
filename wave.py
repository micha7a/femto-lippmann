# -*- coding: utf-8 -*-
"""
Created on 17/04/19

@author: Michalina Pacholska
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import copy

C = 1.0  # TODO Make some kid of gloal k or global environment?


def sigmoid(x, x_min, x_max, percentile=0.95):
    rate = - 2 * np.log(percentile) / (x_max - x_min)
    center = (x_min + x_max) / 2
    return 1 / (1 + np.exp(-rate * (x - center)))


def sigmoid_inverse(y, min_margin=0.05, max_margin=0.95):
    y = y - np.min(y)
    y = y * (max_margin - min_margin) / (np.max(y))
    y = y + min_margin
    return np.log((y / (1 - y)))


class Spectrum(object):

    def __init__(self,
                 spectrum_array: np.ndarray = np.empty(0),
                 k: np.ndarray = np.linspace(0, np.pi, 100),
                 negative: bool = False,
                 **kwargs):
        if spectrum_array.size != 0:
            assert spectrum_array.size == k.size  # TODO(michalina) should this be an error?
        self.s = spectrum_array
        self.k = k
        self.negative = negative

    def __mul__(self, other) -> Spectrum:
        if isinstance(other, (complex, float, int)):
            return Spectrum(other * self.s, k=self.k, negative=self.negative)
        if isinstance(other, Spectrum):
            return Spectrum(self.s * other.s.conj(), k=self.k, negative=self.negative)
        raise NotImplementedError

    __rmul__ = __mul__

    def __add__(self, other) -> Spectrum:
        if self.negative != other.negative:
            return Spectrum(self.s, self.k, self.negative)
        if not np.allclose(self.k, other.k):
            raise NotImplementedError
        return Spectrum(spectrum_array=self.s + other.s,
                        k=self.k,
                        negative=self.negative)

    def amplitude(self, z, time=0) -> np.array:
        sign = -1 if self.negative else 1
        transform = np.exp(1j * (sign * z - C * time)[:, None] @ self.k[None, :])  # TODO add time
        return self.dk() * transform @ self.s[:, None]

    def power_spectrum(self):
        return np.abs(self.s) ** 2

    def power(self):
        return np.sum(self.power_spectrum()) * self.dk()

    def from_amplitude(self, amplitude, z, time=0):
        sign = 1 if self.negative else -1
        transform = np.exp(1j * self.k[:, None] @ (sign * z - C * time)[None, :])
        self.s = (z[1] - z[0]) * transform @ amplitude[:, None]

    def dk(self):
        return self.k[1] - self.k[0]

    def delay(self, time):
        self.s = self.s * np.exp(1j * self.sign() * C * time * self.k)

    def shift(self, z):
        self.shift_forward(self.sign() * z)

    def shift_forward(self, z):
        self.s = self.s * np.exp(1j * z * self.k)

    def wavelength(self):
        return 2 * np.pi / self.k

    def plot(self, wavelength=False):  #  TODO(michalina) return the plot
        if self.s.size == 0:
            raise ValueError
        x = self.wavelength() if wavelength else self.k
        plt.plot(x, np.real(self.s), label="spectrum")
        plt.plot(x, self.power_spectrum(), label="power spectrum")
        plt.xlabel(r"$\lambda$" if wavelength else r"k")
        plt.legend()

    def plot_amplitude(self, z):
        if self.s.size == 0:
            raise ValueError
        amplitude = self.amplitude(z)
        plt.plot(z, amplitude, label="amplitude")
        plt.plot(z, np.abs(amplitude), label="envelope")
        plt.legend()

    def sign(self):
        return -1 if self.negative else 1


class DeltaSpectrum(Spectrum):
    """TODO"""

    def __init__(self, amplitude: complex = 1, wavelength: float = None, wavenumber: float = None, **kwargs):
        super().__init__(**kwargs)
        self.s = np.zeros_like(self.k)
        if wavenumber is None:
            if wavelength is None:
                raise ValueError
            wavenumber = 2 * np.pi / wavelength
        idx = int((wavenumber - self.k[0]) / self.dk())
        if idx > len(self.k):
            raise ValueError
        self.s[idx] = amplitude


class GaussianSpectrum(Spectrum):
    def __init__(self,
                 amplitude: complex = 1,
                 mean: float = 0.5*np.pi,
                 std: float = 0.1,
                 wavenumber: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.s = np.zeros_like(self.k)
        if not wavenumber:
            mean = 2 * np.pi / mean
            std = 2 * np.pi / std
        self.s = amplitude / (np.sqrt(2*np.pi) * std) * np.exp(- (self.k - mean) ** 2 / (2 * std ** 2))


class ChirpedSpectrum(GaussianSpectrum):
    def __init__(self,
                 skew: float = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.s = self.s * np.exp(1j * skew * self.k ** 2)


class Interference(object):
    """TODO"""

    def __init__(self,
                 positive: Spectrum = Spectrum(np.empty(0)),
                 negative: Spectrum = Spectrum(np.empty(0))):
        assert np.allclose(positive.k, negative.k)
        self.positive = copy.deepcopy(positive)
        self.positive.negative = False
        self.negative = copy.deepcopy(negative)
        self.negative.negative = True
        self.k = positive.k

    def __mul__(self, alpha: complex) -> Interference:
        return Interference(self.positive * alpha, self.negative * alpha)

    __rmul__ = __mul__

    def amplitude(self, **kwargs):
        return self.negative.amplitude(**kwargs) + self.positive.amplitude(**kwargs)

    def __add__(self, other: Interference) -> Interference:
        return Interference(self.positive + other.positive,
                            self.negative + other.negative)

    __radd__ = __add__

    def power_spectrum(self):
        return self.positive.power_spectrum(), self.negative.power_spectrum()

    def intensity(self, z, **kwargs):
        correlation = 2 * np.real((self.positive * self.negative).amplitude(z=2*z, **kwargs))
        return self.positive.power() + self.negative.power() + correlation

    def dk(self):
        return self.positive.dk()

    # TODO(michalina) implement better getters? def __get__(self, instance, owner)


class Material(object):
    """TODO"""

    def __init__(self,
                 z: np.ndarray,
                 n0: complex = 1,
                 **kwargs):
        assert np.imag(n0) >= 0
        self.z = z
        self.z = z
        self.n0 = n0
        self.deposited_energy = np.zeros_like(z)
        self.length = z[-1] - z[0]

    def index_of_refraction(self) -> np.ndarray:
        raise NotImplementedError

    def energy_response(self, energy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transmitted(self, spectrum: Spectrum) -> Spectrum:  # TODO Do I need this?
        raise NotImplementedError

    def reflected(self, spectrum: Spectrum) -> Spectrum:
        raise NotImplementedError

    def energy_distribution(self, interference: Interference) -> np.ndarray:
        total_amplitude = self.propagate(interference.positive) + self.propagate(interference.negative)
        return np.sum((np.abs(total_amplitude) ** 2), axis=0) * interference.dk()

    def record(self, interference: Interference):
        energy = self.energy_distribution(interference)
        self.deposited_energy += self.energy_response(energy)

    def propagate(self, spectrum: Spectrum) -> np.array:
        """Calculate energy profile at each point of dielectric at each frequency"""
        raise NotImplementedError


class FixedDielectric(Material):
    """TODO"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def index_of_refraction(self):
        return self.n0 * np.ones_like(self.z)

    def energy_response(self, energy):
        return np.zeros_like(energy)

    def transmitted(self, spectrum):
        spectrum = copy.deepcopy(spectrum)
        spectrum.shift_forward(self.length * self.n0)
        return spectrum

    def reflected(self, spectrum):
        return 0 * spectrum

    def propagate(self, spectrum: Spectrum):
        if spectrum.negative:
            z = self.z - self.z[-1]
        else:
            z = self.z - self.z[0]
        transform = np.exp(1j * spectrum.sign() * self.n0 * spectrum.k[:, None] @ z[None, :])
        return np.repeat(spectrum.s[:, None], len(self.z), axis=1) * transform


class Dielectric(Material):
    """TODO"""

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

    def energy_response(self, energy):  # TODO (michalina) what is the right model for that
        factor = self.min_energy + self.max_d_energy * sigmoid(self.deposited_energy, 0, self.max_visible_energy)
        return energy * factor / (self.min_energy + self.max_d_energy)

    def propagate(self, spectrum: Spectrum):
        pass


class LayeredMaterial(object):  # Yet another TODO (michalina)

    def __init__(self, material_list):
        self.materials = material_list
