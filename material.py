# -*- coding: utf-8 -*-
"""
Created on 17.08.2019
@author: Michalina Pacholska

A material module for Femto-Lippmann project.
"""

import numpy as np
from typing import Union

import constants as c
import wave as w


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
    def __init__(self, z: np.ndarray, n0: Union[complex, np.ndarray] = 1, **kwargs):
        if isinstance(n0, np.ndarray):
            assert (np.imag(n0) >= 0).all()
        else:
            assert np.imag(n0) >= 0
        self.z = z
        self.n0 = n0
        self.deposited_energy = np.zeros(len(z))
        self.length = z[-1] - z[0]
        self.recent_energy = np.zeros(len(z))
        self.matrix = self.material_matrix(w.PlanarWave, backward=True)

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

    def material_matrix(self, wave: w.PlanarWave, backward: bool = False) -> np.ndarray:
        raise NotImplementedError

    def energy_distribution(self, forward_wave: w.PlanarWave, backward_wave: w.PlanarWave) -> np.ndarray:
        raise NotImplementedError

    def record(self, forward_wave: w.PlanarWave, backward_wave: w.PlanarWave):
        self.recent_energy = self.energy_distribution(forward_wave, backward_wave)
        self.deposited_energy += self.energy_response(self.recent_energy)
        self.matrix = self.material_matrix(forward_wave, backward=True)

    def plot(self, ax, imaginary_axis=None, **kwargs):
        index = self.index_of_refraction()
        ax.step(self.z, np.real(index), where='post', **kwargs)
        color = ax.get_lines()[-1].get_color()
        ax.set_ylabel("real part", color=color)
        ax.tick_params(axis='y', labelcolor=color)
        color = ax._get_lines.get_next_color()
        if imaginary_axis is not None:
            imaginary_axis.step(self.z, np.imag(index), where="post", color=color, **kwargs)
            imaginary_axis.set_ylabel("imaginary part", color=color)
            imaginary_axis.tick_params(axis='y', labelcolor=color)
        ax.set_xlabel("z [m]")
        ax.xaxis.set_major_formatter(c.FORMATTER)
        ax.set_title("index of refraction")

    def plot_recent_energy(self, ax, **kwargs):
        ax.plot(self.z, self.recent_energy, **kwargs)
        ax.set_xlabel("z [m]")
        ax.xaxis.set_major_formatter(c.FORMATTER)
        ax.set_title("propagated energy")
        ax.set_ylabel("intensity")

    def reflect(self, spectrum: w.PlanarWave) -> w.PlanarWave:
        return spectrum * self.matrix[0, 1, :]

    def transmit(self, spectrum: w.PlanarWave) -> w.PlanarWave:
        return spectrum * self.matrix[0, 0, :]


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

        total_amplitude = self._propagate(forward_wave, sign=1) + self._propagate(backward_wave, sign=-1)
        return np.sum((np.abs(total_amplitude)**2), axis=0) * w.PlanarWave.dk()

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

    def material_matrix(self, wave: w.PlanarWave, backward=False):
        return wave.propagation_matrix(self.n0, self.length, backward=backward)


class EmptySpace(ConstantMaterial):
    """A class describing empty space with index of refraction 1,
    that cannot be modified. Used mostly for testing"""
    def __init__(self, **kwargs):
        super().__init__(n0=1.0, **kwargs)

    def energy_distribution(self, forward_wave: w.PlanarWave, backward_wave: w.PlanarWave) -> np.ndarray:
        """Different implementation of interference based on correlation rather
        than matrix theory"""

        correlation = 2 * np.real((forward_wave * backward_wave).amplitude(z=2 * self.z))
        return np.squeeze(forward_wave.total_energy() + backward_wave.total_energy() + correlation)


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
    def __init__(self, min_energy=100 * c.NANO, max_energy=1 * c.MICRO, max_dn=1e-3, **kwargs):
        self.max_dn = max_dn
        self.min_energy = min_energy
        self.max_energy = max_energy
        super().__init__(**kwargs)

    def index_of_refraction(self):
        """Hard threshold"""
        index = self.n0 + self.max_dn * np.clip(self.deposited_energy / self.max_energy, a_max=1, a_min=None)
        # fill dummy index of refraction
        index[-1] = index[-2]
        return index

    def energy_response(self, energy: np.ndarray):
        """Hard threshold"""
        return np.clip(energy - self.min_energy, a_min=0, a_max=None)

    def material_matrix(self, wave: w.PlanarWave, backward=False):
        ns = self.index_of_refraction()
        matrix = np.zeros((2, 2, len(wave.k)), dtype=complex)
        matrix[0, 0] = 1
        matrix[1, 1] = 1
        for idx in range(len(self.z) - 1):
            propagation = wave.propagation_matrix(ns[idx], self.z[idx + 1] - self.z[idx])
            matrix = np.einsum('ijk,jlk->ilk', propagation, matrix)
            # add boundary:
            # the last element of ns does not correspond to any physical space
            # thus the last boundary is between second to last and last index
            if idx < len(self.z) - 2:
                boundary = wave.boundary_matrix(ns[idx], ns[idx + 1])
                matrix = np.tensordot(boundary, matrix, axes=(1, 0))
        if backward:
            matrix = w.PlanarWave.swap_waves(matrix)
        return matrix

    def energy_distribution(self, forward_wave, backward_wave):
        # Calculate the outgoing waves
        incoming = np.array([forward_wave.s, backward_wave.s])
        outgoing = np.einsum('ijk,jk->ik', self.matrix, incoming)
        # Calculate the wave at each step, moving forward
        energy = np.zeros_like(self.z)
        left = np.array([incoming[0], outgoing[1]])
        ns = self.index_of_refraction()
        for idx_z in range(len(self.z) - 1):
            matrix = w.PlanarWave.propagation_matrix(ns[idx_z], self.z[idx_z + 1] - self.z[idx_z], forward_wave.k)
            if idx_z > 0:
                boundary = w.PlanarWave.boundary_matrix(ns[idx_z - 1], ns[idx_z])
                matrix = np.einsum('ijk,jl->ilk', matrix, boundary)
            right = np.einsum('ijk,jk->ik', matrix, left)
            energy[idx_z] = np.sum(np.abs(np.sum(right, axis=0))**2) * w.PlanarWave.dk()
            left = right
        energy[-1] = energy[-2]
        return energy


class LayeredMaterial(SimpleDielectric):
    """A layered dielectric, a slow implementation"""
    def __init__(self, z, n_layers, layer_width, n0, n1, **kwargs):
        period = (z[-1] - z[0]) / n_layers
        if period < layer_width:
            raise ValueError("Layers can't overlap")
        n = np.ones_like(z, dtype=complex) * n0
        for idx, pos in enumerate(z):
            relative_pos = (pos - z[0]) % period
            if relative_pos > (period - layer_width) / 2:
                if relative_pos < (period + layer_width) / 2:
                    n[idx] = n1
        super().__init__(z=np.array(z), n0=np.array(n), **kwargs)
        self.left_boundaries = z[0] + (period - layer_width) / 2 + period * np.arange(0, n_layers)
        self.layer_width = layer_width

    def shade_plot(self, ax, alpha=0.1, **kwargs):
        for boundary in self.left_boundaries:
            ax.axvspan(boundary, boundary + self.layer_width, alpha=alpha, **kwargs)


class CompositeMaterial(SimpleDielectric):
    """A very simple implementation of a composite material.

    Behaves the same way as layered material, but can be made of any stack of
    previously created materials."""
    def __init__(self, material_list):
        """
        Create material from a list of composites.

        Args:
            material_list: list of composites. Note that materials might have
            internal state (e.g. recorded pattern) that is going to be flattened
            and internal material properties lost.
        """
        self.materials = material_list
        z = np.array([])
        boundary_z = 0
        n0 = np.array([])
        for material in self.materials:
            z = np.concatenate([z, material.z[:-1] - np.min(material.z) + boundary_z])
            boundary_z = material.z[-1]
            n0 = np.concatenate([n0, material.index_of_refraction()[:-1]])
        z = np.concatenate([z, [boundary_z]])
        n0 = np.concatenate([n0, [n0[-1]]])
        super().__init__(z=z, n0=n0)