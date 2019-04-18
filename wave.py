# -*- coding: utf-8 -*-
"""
Created on 17/04/19

@author: Michalina Pacholska
"""
from __future__ import annotations

from abc import ABC

import numpy as np
import matplotlib.pyplot as plt

C = 1.0


class Spectrum(object):
    """TODO"""

    def __init__(self,
                 spectrum_array: np.ndarray = np.empty(0),
                 k: np.ndarray = np.linspace(0, np.pi, 100),
                 negative: bool = False,
                 *args):
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
        if self.k != other.k:
            raise NotImplementedError
        return Spectrum(self.s + other.s, self.k, self.negative)

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
        raise NotImplementedError

    def shift(self, z):
        raise NotImplementedError

    def lambdas(self):
        return 2*np.pi/self.k

    def plot(self):
        if self.s.size == 0:
            raise ValueError
        plt.plot(self.k, np.real(self.s), label="spectrum")
        plt.plot(self.k, self.power_spectrum(), label="power spectrum")
        plt.legend()
        plt.show()

    def plot_amplitude(self, z):
        if self.s.size == 0:
            raise ValueError
        amplitude = self.amplitude(z)
        plt.plot(z, amplitude, label="amplitude")
        plt.plot(z, np.abs(amplitude), label="envelope")
        plt.legend()
        plt.show()


class DeltaSpectrum(Spectrum):
    """TODO"""

    def __init__(self, *args):
        super().__init__(*args)


class Interference(object):
    """TODO"""

    def __init__(self,
                 positve_spectrum: Spectrum = Spectrum(np.empty(0)),
                 negative_spectrum: Spectrum = Spectrum(np.empty(0))):
        assert positve_spectrum.k == negative_spectrum.k
        self.positive_spectrum = positve_spectrum
        self.negative_spectrum = negative_spectrum
        self.k = positve_spectrum.k

    def __mul__(self, alpha: complex) -> Interference:
        return Interference(self.positive_spectrum * alpha, self.negative_spectrum * alpha)

    __rmul__ = __mul__

    def amplitude(self, params):
        return self.negative_spectrum.amplitude(params) + self.positive_spectrum.amplitude(params)

    def __add__(self, other: Interference) -> Interference:
        return Interference(self.positive_spectrum + other.positive_spectrum,
                            self.negative_spectrum + other.negative_spectrum)

    __radd__ = __add__

    def power_spectrum(self):
        return self.positive_spectrum.power_spectrum(), self.negative_spectrum.power_spectrum()

    def intensity(self, z):
        correlation = 2 * np.real((self.positive_spectrum * self.negative_spectrum).amplitude())
        return self.positive_spectrum.power() + self.positive_spectrum.power() + correlation

    # TODO(michalina) implement better getters? def __get__(self, instance, owner)
