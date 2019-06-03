# -*- coding: utf-8 -*-
"""
A 1D Wave module for Femto-Lippmann project
"""
from __future__ import annotations

import numpy as np
import copy
from typing import Union
import matrix_theory as mt

C = 299792458
NANO = 1e-9
MICRO = 1e-6
VIOLET = 380 * NANO
GREEN = 500 * NANO
RED = 740 * NANO
OMEGA_STEPS = 100
SINGLE_PULSE_ENERGY = 50 * NANO  # nanoJules


def sigmoid(x, x_min, x_max, percentile=0.01):
    """Function that calculates sigmoid saturating around given values"""
    rate = -2 * np.log(percentile) / (x_max - x_min)
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

    @classmethod
    def dk(cls):
        return cls.k[1] - cls.k[0]

    def __init__(self,
                 spectrum_array: np.ndarray = np.empty(0),
                 name="",
                 **kwargs):
        if spectrum_array.size != 0 and spectrum_array.size != self.k.size:
            raise ValueError(
                "The spectrum_array must be of len(Spectrum.k), default {}".
                format(OMEGA_STEPS))
        self.s = np.array(spectrum_array, dtype=complex)
        self.name = name

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
            return np.allclose(self.k, other.k) and np.allclose(
                self.s, other.s)
        return False

    def __copy__(self):
        return Spectrum(self.s.copy())

    def amplitude(self, z, time=0) -> np.array:
        transform = np.exp(1j * (z - C * time)[:, None] @ self.k[None, :])
        return self.dk(
        ) * transform @ self.s[:,
                               None]  # TODO why the resulting amplitudes are so big?

    def power_spectrum(self):
        return np.abs(self.s)**2

    def total_energy(self):
        return np.sum(self.power_spectrum()) * self.dk()

    def set_energy(self, energy):
        self.s = self.s / np.sqrt(self.total_energy()) * np.sqrt(energy)

    def from_amplitude(self, amplitude, z, time=0):
        transform = np.exp(1j * self.k[:, None] @ (-z - C * time)[None, :])
        self.s = (z[1] - z[0]) * transform @ amplitude[:, None]

    def delay(self, time):
        self.s = self.s * np.exp(1j * C * time * self.k)

    def shift(self, z):
        self.s = self.s * np.exp(1j * z * self.k)

    def wavelength(self):
        return 2 * np.pi / self.k

    def plot(self, ax, wavelength=False):
        if self.s.size == 0:
            raise ValueError("Can't plot empty spectrum")
        x = self.wavelength() if wavelength else self.k
        ax.plot(x, np.real(self.s), label="spectrum {}".format(self.name))
        ax.plot(
            x,
            self.power_spectrum(),
            label="power spectrum {}".format(self.name))
        ax.set_xlabel(r"$\lambda$" if wavelength else r"k")
        ax.legend()

    def plot_amplitude(self, z, ax):
        if self.s.size == 0:
            raise ValueError("Can't plot empty spectrum")
        amplitude = self.amplitude(z)
        ax.plot(z, amplitude, label="amplitude {}".format(self.name))
        ax.plot(z, np.abs(amplitude), label="envelope {}".format(self.name))
        ax.set_xlabel("z")
        ax.legend()


class DeltaSpectrum(Spectrum):
    """A class representing a single frequency wave."""

    def __init__(self,
                 energy: float = SINGLE_PULSE_ENERGY,
                 wavelength: float = None,
                 wavenumber: float = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.s = np.zeros_like(self.k)
        if wavenumber is None:
            if wavelength is None:
                raise ValueError(
                    "One of wavelength, wavenumber hast o be provided")
            wavenumber = 2 * np.pi / wavelength
        idx = int((wavenumber - self.k[0]) / self.dk())
        if idx > len(self.k) or idx < 0:
            raise ValueError(
                "Wavenumber {} effectively outside the spectrum: {}-{}".format(
                    wavenumber, self.k[0], self.k[-1]))
        self.s[idx] = 1
        self.set_energy(energy)


class GaussianSpectrum(Spectrum):
    """A class representing gaussian spectrum. Note that, since it is in 1D,
    it is not a gaussian beam."""

    def __init__(self,
                 energy: float = SINGLE_PULSE_ENERGY,
                 mean: float = GREEN,
                 std: float = 10 * NANO,
                 wavenumber: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.s = np.zeros_like(self.k)
        if not wavenumber:
            mean = 2 * np.pi / mean
            std = 2 * np.pi / std
        self.s = np.exp(-(self.k - mean)**2 / (2 * std**2))
        self.set_energy(energy)


class ChirpedSpectrum(GaussianSpectrum):
    """A class representing a chirped gaussian spectrum."""

    def __init__(self, skew: float = 0, **kwargs):
        super().__init__(**kwargs)
        if "wavenumber" in kwargs:
            print("we have wavenumber")
            if not kwargs["wavenumber"] and skew != 0:
                skew = 2 * np.pi / skew
        self.s = self.s * np.exp(1j * skew * self.k**2)


class Material(object):
    """An abstract class describing a material in which the waves can propagate

    Attributes:
        z: array of positions at which there are material or index of refraction
        boundaries, and at which the wave is evaluated if propagated through
        the dielectric
        n0: basic index of refraction,
        deposited_energy: energy currently deposited in the material,
        initialised to np.array of zeros, of length len(z_boundary).
        Note, that because the energy and changes in the index of refraction
        happen between the boundaries, one index of refraction is not needed,
        and thus the last value index of refraction is never used.
        length: total length of the material.
    """

    def __init__(self,
                 z: np.ndarray,
                 n0: Union[complex, np.ndarray] = 1,
                 name: str = "",
                 **kwargs):
        assert np.imag(n0) >= 0
        self.z = z
        self.n0 = n0
        self.deposited_energy = np.zeros(len(z))
        self.length = z[-1] - z[0]
        self.recent_energy = np.zeros(len(z))
        self.name = name

    def index_of_refraction(self) -> np.ndarray:
        """Function calculating index of refraction between each two boundaries

        Returns:
            np.array of length(self.boundary_z) of complex values of index of
            refraction. Since index is defined between boundaries, the last
            value is a dummy index, set to default value. It is set like this
            for plotting purposes, and shouldn't be used.
        """
        raise NotImplementedError

    def energy_response(self, energy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def material_matrix(self, k: float, backward: bool = False) -> np.ndarray:
        raise NotImplementedError

    def energy_distribution(self, forward_wave: Spectrum,
                            backward_wave: Spectrum) -> np.ndarray:
        raise NotImplementedError

    def record(self, forward_wave: Spectrum, backward_wave: Spectrum):
        self.recent_energy = self.energy_distribution(forward_wave,
                                                      backward_wave)
        self.deposited_energy += self.energy_response(self.recent_energy)

    def plot(self, ax, plot_imaginary=True, alpha=1):
        index = self.index_of_refraction()
        ax.step(
            self.z,
            np.real(index),
            where='post',
            label="real part {}".format(self.name),
            alpha=alpha)
        if plot_imaginary:
            ax.step(
                self.z,
                np.imag(index),
                where="post",
                label="imaginary part {}".format(self.name),
                alpha=alpha)
        ax.set_xlabel("z")
        ax.set_title("index of refraction")
        ax.legend()

    # TODO(michalina) add reflection and transmission calculated from the matrix


class ConstantMaterial(Material):
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

    def energy_distribution(self, forward_wave, backward_wave):
        """Semi-analytic interference calculation, where the amplitude at each
        point is calculated analytically using _propagate method,
        but then intensity is calculated numerically point-wise"""

        total_amplitude = self._propagate(
            forward_wave, sign=1) + self._propagate(
                backward_wave, sign=-1)
        return np.sum((np.abs(total_amplitude)**2), axis=0) * Spectrum.dk()

    def _propagate(self, spectrum, sign=1):
        """Returns amplitude inside material, for each point for each frequency

        The first dimension corresponds to different frequencies, and second
        to different depths in the material. Thus dimensions of the returned
        matrix are len(spectrum) x len(z)
        """
        if sign == 1:
            z = self.z - self.z[0]
        elif sign == -1:
            z = self.z[-1] - self.z
        else:
            raise ValueError("Sign has to have value 1 or -1")
        transform = np.exp(1j * self.n0 * spectrum.k[:, None] @ z[None, :])
        return np.repeat(spectrum.s[:, None], len(self.z), axis=1) * transform

    def material_matrix(self, k, backward=False):
        return mt.propagation_matrix(
            self.n0, self.length, k, backward=backward)


class EmptySpace(ConstantMaterial):
    """A class describing empty space with index of refraction 1,
    that cannot be modified. Used mostly for testing"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n0 = 1

    def energy_distribution(self, forward_wave: Spectrum,
                            backward_wave: Spectrum) -> np.ndarray:
        """Different implementation of interference based on correlation rather
        than matrix theory"""

        correlation = 2 * np.real(
            (forward_wave * backward_wave).amplitude(z=2 * self.z))
        return np.squeeze(forward_wave.total_energy() +
                          backward_wave.total_energy() + correlation)


class SimpleDielectric(Material):
    """A class describing a dielectric with index of refraction that can
    be modified by a set of heuristics.

    Current assumptions:
        - energy deposited in the material is linear with the intensity,
        if it's above a min_energy threshold, else it's zero
        - change of the index of refraction in the material is linear
        with deposited energy, up to max_energy, where it is constantly max_dn
        - maximal possible index pf refraction is 10^-3, and that's known quite well
        - energy of the threshold is somewhere below 150nJ, but it's not well known
        - the max energy can be 100 or 1000 times bigger than the threshold energy"""

    def __init__(self,
                 min_energy=100 * NANO,
                 max_energy=10 * MICRO,
                 max_dn=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_dn = max_dn
        self.min_energy = min_energy
        self.max_energy = max_energy

    def index_of_refraction(self):
        """Hard threshold"""
        index = self.n0 + self.max_dn * np.clip(
            self.deposited_energy / self.max_energy, a_max=1, a_min=None)
        # fill dummy index of refraction
        index[-1] = index[-2]
        return index

    def energy_response(self, energy: np.ndarray):
        """Hard threshold"""
        return np.clip(energy - self.min_energy, a_min=0, a_max=None)

    def material_matrix(self, k, backward=False):
        ns = self.index_of_refraction()
        matrix = np.eye(2)
        for idx in range(len(self.z) - 1):
            matrix = mt.propagation_matrix(
                ns[idx], self.z[idx + 1] - self.z[idx], k) @ matrix
            # add boundary:
            # the last element of ns does not correspond to any physical space
            # thus the last boundary is between second to last and last index
            if idx < len(self.z) - 2:
                matrix = mt.boundary_matrix(
                    ns[idx], ns[idx + 1]) @ matrix  # TODO is this consistent?
        if backward:
            matrix = mt.swap_waves(matrix)
        return matrix

    def energy_distribution(self, forward_wave, backward_wave):
        left_backward = Spectrum(Spectrum.k)
        right_forward = Spectrum(Spectrum.k)
        # Calculate the outgoing waves
        for idx_k, k in enumerate(Spectrum.k):
            incoming = np.array(
                [forward_wave.s[idx_k], backward_wave.s[idx_k]])[:, None]
            outgoing = self.material_matrix(k, backward=True) @ incoming
            outgoing = outgoing.squeeze()
            right_forward.s[idx_k] = outgoing[0]
            left_backward.s[idx_k] = outgoing[1]
        # Calculate the wave at each step, moving forward
        energy = np.zeros_like(self.z)
        previous_backward = copy.deepcopy(left_backward)
        previous_forward = copy.deepcopy(forward_wave)
        next_forward = copy.deepcopy(left_backward)
        next_backward = copy.deepcopy(left_backward)
        ns = self.index_of_refraction()
        for idx_z in range(len(self.z) - 1):
            for idx_k, k in enumerate(Spectrum.k):
                left = np.array(
                    [previous_forward.s[idx_k],
                     previous_backward.s[idx_k]])[:, None]
                matrix = mt.propagation_matrix(
                    ns[idx_z], self.z[idx_z + 1] - self.z[idx_z], k)
                if idx_z > 0:
                    matrix = matrix @ mt.boundary_matrix(
                        ns[idx_z - 1], ns[idx_z])
                right = (matrix @ left).squeeze()
                next_forward.s[idx_k] = right[0]
                next_backward.s[idx_k] = right[1]
                energy[idx_z] += np.abs(np.sum(right))**2 * Spectrum.dk()
            previous_backward = copy.deepcopy(next_backward)
            previous_forward = copy.deepcopy(next_forward)
        return energy


class LayeredMaterial(SimpleDielectric):
    """A placeholder for a class describing a layered material"""


class CompositeMaterial(object):
    """A placeholder for a class describing a stack of materials"""

    def __init__(self, material_list):
        self.materials = material_list
