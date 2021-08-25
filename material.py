# -*- coding: utf-8 -*-
"""
Created on 17.08.2019
@author: Michalina Pacholska

A material module for Femto-Lippmann project.
"""

import copy
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
from typing import Union

import constants as c
import wave as w
import propagation as p


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
        recent_energy: energy from the last pulse
        matrix: material propagation matrix
    """
    def __init__(self, z: np.ndarray, n0: Union[complex, np.ndarray] = 1, **kwargs):
        """
        Create generic material

        Args:
            z: vector of depths of the material
            n0: scalar or vector, basic index of refraction
            **kwargs: potential other arguments that might be used by a subclass
        """
        if isinstance(n0, np.ndarray):
            assert (np.imag(n0) >= 0).all()
        else:
            assert np.imag(n0) >= 0
        self.z = z
        self.n0 = n0
        self._deposited_energy = np.zeros(len(z))
        self.length = z[-1] - z[0]
        self._recent_energy = np.zeros(len(z))
        if "matrix" in kwargs:
            self.matrix = kwargs["matrix"]
        self.matrix = self.material_matrix(backward=True)
        self._tmp_matrix = None
        self._ns = None
        self._r = np.zeros(len(z))

    @property
    def recent_energy(self):
        """Get the from the last response."""
        return self._recent_energy

    @property
    def deposited_energy(self):
        """I'm the 'deposited_energy' property."""
        return self._deposited_energy

    @deposited_energy.setter
    def deposited_energy(self, value):
        self._deposited_energy = value
        self.matrix = self.material_matrix(backward=True)

    @deposited_energy.deleter
    def deposited_energy(self):
        del self._deposited_energy

    def index_of_refraction(self) -> np.ndarray:
        """Function calculating index of refraction between each two boundaries,
        based on the basic material properties and deposited energy

        Returns:
            np.array of length(self.boundary_z) of complex values of index of
            refraction. Since index is defined between boundaries, the last
            value is a dummy index, set to default value. It is set like this
            for plotting purposes, and shouldn't be used.
        """
        raise NotImplementedError

    def energy_response(self, energy: np.ndarray) -> np.ndarray:
        """
        Response to the material to a given energy. Models energy absorbed by
        the material due to material modification. Can describe things like
        modification threshold and activation.

        Args:
            energy: vector of energies to which material was exposed

        Returns:
            vector of deposited energies

        """
        raise NotImplementedError

    def material_matrix(self, backward: bool = False) -> np.ndarray:
        """
        Calculate propagation matrix through the material

        Args:
            backward: if true, use backward propagation model (see wave documentation for details)

        Returns:
            propagation matrix through the material
        """
        raise NotImplementedError

    def energy_distribution(self,
                            forward_wave: w.PlanarWave,
                            backward_wave: w.PlanarWave = None,
                            reflected_wave: w.PlanarWave = None) -> np.ndarray:
        """
        Calculate energy distribution of two interfering pulses propagating through the material.
        """
        raise NotImplementedError

    def record(self,
               forward_wave: w.PlanarWave,
               backward_wave: w.PlanarWave = None,
               reflected_wave: w.PlanarWave = None):
        """
        Record interference of two pulses
        """
        self._recent_energy = self.energy_distribution(forward_wave,
                                                       backward_wave=backward_wave,
                                                       reflected_wave=reflected_wave)
        self.deposited_energy += self.energy_response(self._recent_energy)

    def plot(self, ax, imaginary_axis=None, change_only=False, **kwargs):
        """
        Plot index of refraction as a function of the depth of the material

        Args:
            ax: matplotlib axis, created by plt.figure() or plt.subplots(), used
            to arrange plots.
            imaginary_axis: if provided, create new plot showing imaginary part
            of the index of refraction. This axis can be the same as ax.
            change_only: if true, plot only the change of the index of refraction
            **kwargs:  all other arguments than can be passed to matplotlib's
            plot function
        """
        index = self.index_of_refraction()
        if change_only:
            index -= self.n0
        ax.step(self.z, np.real(index), where='post', **kwargs)
        color = "k" if "color" in kwargs else ax.get_lines()[-1].get_color()
        ax.set_ylabel("real part", color=color)
        ax.tick_params(axis='y', labelcolor=color)
        if imaginary_axis is not None:
            if "color" not in kwargs:
                # accessing private property because otherwise get_next_color does not work
                color = ax._get_lines.get_next_color()
                kwargs["color"] = color
            else:
                color = "k"
            imaginary_axis.step(self.z, np.imag(index), where="post", **kwargs)
            imaginary_axis.set_ylabel("imaginary part", color=color)
            imaginary_axis.tick_params(axis='y', labelcolor=color)
        ax.set_xlabel("z [m]")
        ax.xaxis.set_major_formatter(c.FORMATTER)
        if change_only:
            ax.set_title("change in \n refractive index")
        else:
            ax.set_title("refractive index")

    def shade_plot(self, ax, alpha=0.1, **kwargs):
        pass

    def plot_recent_energy(self, ax, **kwargs):
        """Plot the energy of the recently recorded pulses as a function of
        the depth of the material.

        Args:
            ax: matplotlib axis, created by plt.figure() or plt.subplots(), used
            to arrange plots.
            **kwargs:  all other arguments than can be passed to matplotlib's
            plot function
        """
        ax.plot(self.z, self._recent_energy, **kwargs)
        ax.set_xlabel("z [m]")
        ax.xaxis.set_major_formatter(c.FORMATTER)
        ax.set_title("propagated energy")
        ax.set_ylabel("intensity")

    def reflect(self, wave: w.PlanarWave = None) -> w.PlanarWave:
        """Reflect a wave from the material"""
        if wave is not None:
            return wave * self.matrix[0, 1, :]
        else:
            return self.matrix[0, 1, :]

    def transmit(self, wave: w.PlanarWave = None) -> w.PlanarWave:
        """Transmit a wave through the material"""
        if wave is not None:
            return wave * self.matrix[0, 0, :]
        else:
            return self.matrix[0, 0, :]

    def reflectivity(self):
        return np.zeros_like(self.z)

    def set_beginning(self, z0):
        self.z += z0 - self.z[0]


class ConstantMaterial(Material):
    """A class describing a dielectric with constant index of refraction n0
    that cannot be modified"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.n0, (complex, float, int)):
            raise ValueError("Fixed dielectric must have constant index n0")

    def index_of_refraction(self):
        """Constant index of refraction."""
        return self.n0 * np.ones_like(self.z)

    def energy_response(self, energy):
        """Constant material does not react to light"""
        return np.zeros_like(energy)

    def energy_distribution(self, forward_wave, backward_wave=None, reflected_wave=None):
        """Semi-analytic interference calculation, where the amplitude at each
        point is calculated analytically using _propagate method,
        but then intensity is calculated numerically point-wise"""

        if backward_wave is None:
            forward_wave, backward_wave = p.swap_waves(p.change_basis(self.matrix), forward_wave, reflected_wave)
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

    def material_matrix(self, backward=False):
        return p.propagation_matrix(self.n0, self.length, backward=backward)


class EmptySpace(ConstantMaterial):
    """A class describing empty space with index of refraction 1,
    that cannot be modified. Used mostly for testing"""
    def __init__(self, **kwargs):
        super().__init__(n0=1.0, **kwargs)

    def energy_distribution(self, forward_wave: w.PlanarWave, backward_wave: w.PlanarWave = None, reflected_wave=None) \
            -> \
            np.ndarray:
        """Different implementation of interference based on correlation rather
        than matrix theory"""

        if backward_wave is None:
            forward_wave, backward_wave = p.swap_waves(p.change_basis(self.matrix), forward_wave, reflected_wave)
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
        """Index of refraction depending on the deposited energy.

        Index of refraction is linear with deposited energy up to max_energy,
        when it's constant and equal to max_dn."""
        index = self.n0 + self.max_dn * np.clip(self._deposited_energy / self.max_energy, a_max=1, a_min=None)
        return index

    def energy_response(self, energy: np.ndarray):
        """Energy deposited from incoming energy.

        The energy response is linear above min_energy threshold, and zero below
        this threshold."""
        return np.clip(energy - self.min_energy, a_min=0, a_max=None)

    def single_layer_matrix(self, idx_z):
        matrix = p.propagation_matrix(self._ns[idx_z - 1], self.z[idx_z] - self.z[idx_z - 1])
        # add boundary:
        # the last element of ns does not correspond to any physical space
        # thus the last boundary is between second to last and last index
        if idx_z < len(self.z) - 2:
            boundary = p.boundary_matrix(self._ns[idx_z - 1], self._ns[idx_z])
            matrix = np.tensordot(boundary, matrix, axes=(1, 0))
        return matrix

    def material_matrix(self, backward=False):
        self._ns = self.index_of_refraction()
        self._r = self.reflectivity()
        matrix = p.identity_matrix()
        for idx_z in range(1, len(self.z)):
            matrix = np.einsum('ijk,jlk->ilk', self.single_layer_matrix(idx_z), matrix)
        if backward:
            matrix = p.change_basis(matrix)
        return matrix

    def energy_distribution(self, forward_wave, backward_wave=None, reflected_wave=None):
        if reflected_wave is None:
            assert (backward_wave is not None)
            forward_wave, reflected_wave = p.swap_waves(self.matrix, forward_wave, backward_wave)

        # Calculate the outgoing waves
        left_wave = [forward_wave.s, reflected_wave.s]
        energy = np.zeros_like(self.z)
        energy[0] = np.sum(np.abs(np.sum(left_wave, axis=0))**2) * w.PlanarWave.dk()
        self._r = self.reflectivity()
        self._ns = self.index_of_refraction()  # TODO make sure the index is always correct, like the matrix
        matrix = p.identity_matrix()
        # Calculate the wave at each step, moving forward
        for idx_z in range(1, len(self.z)):
            matrix = np.einsum('ijk,jlk->ilk', self.single_layer_matrix(idx_z), matrix)
            current_wave = np.einsum('ijk,jk->ik', matrix, left_wave)
            energy[idx_z] = np.sum(np.abs(np.sum(current_wave, axis=0))**2) * w.PlanarWave.dk()
        self._tmp_matrix = p.change_basis(matrix)
        return energy

    def shade_plot(self, ax, alpha=0.1, **kwargs):
        ax.axvspan(self.z[0], self.z[-1], alpha=alpha, **kwargs)


class PhotoSensitiveMaterial(SimpleDielectric):
    def __init__(self,
                 min_energy=0.0,
                 max_energy=1 * c.MICRO,
                 max_dr=1e-3,
                 r: Union[complex, np.ndarray] = 0,
                 **kwargs):
        self.max_dr = max_dr
        super().__init__(**kwargs)
        self.min_energy = min_energy
        self.max_energy = max_energy
        self._r = np.ones_like(self.z) * r

    def single_layer_matrix(self, idx_z):
        propagation = p.propagation_matrix(self.n0, self.z[idx_z] - self.z[idx_z - 1])
        reflection = p.reflection_matrix(self._r[idx_z])
        matrix = np.tensordot(reflection, propagation, axes=(1, 0))
        return matrix

    def index_of_refraction(self):
        """Index of refraction"""
        return self.n0

    def reflectivity(self):
        index = self.max_dr * np.clip(self._deposited_energy / self.max_energy, a_max=1, a_min=None)
        # fill dummy index of refraction
        index[0] = index[1]
        return index

    def plot(self, ax, imaginary_axis=None, **kwargs):
        """
        Plot index of refraction as a function of the depth of the material

        Args:
            ax: matplotlib axis, created by plt.figure() or plt.subplots(), used
            to arrange plots.
            imaginary_axis: if provided, create new plot showing imaginary part
            of the index of refraction. This axis can be the same as ax.
            **kwargs:  all other arguments than can be passed to matplotlib's
            plot function
        """
        ref = self.reflectivity()
        ax.step(self.z, ref, where='post', **kwargs)
        color = "k" if "color" in kwargs else ax.get_lines()[-1].get_color()
        ax.set_ylabel("real part", color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_xlabel("z [m]")
        ax.xaxis.set_major_formatter(c.FORMATTER)
        ax.set_title("reflectivity")


class LayeredMaterial(SimpleDielectric):
    """A layered dielectric, that is a dielectric with simplified creation of layers"""
    def __init__(self, z, n_layers, layer_width, n0, n1, **kwargs):
        """

        Args:
            z: vector of depths, it is needed and has to have fine resolution
            in order to allow for fine material modification
            n_layers: number of layers of different index of refraction
            layer_width: with of the different layers
            n0: basic index of refraction
            n1: index of refraction of the layer
            **kwargs: any other arguments used by SimpleDielectric
        """
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
        self.left_boundaries = self.z[0] + (period - layer_width) / 2 + period * np.arange(0, n_layers)
        self.layer_width = layer_width

    def shade_plot(self, ax, alpha=0.1, **kwargs):
        for boundary in self.left_boundaries:
            ax.axvspan(boundary, boundary + self.layer_width, alpha=alpha, **kwargs)

    def set_beginning(self, z0):
        diff = z0 - self.z[0]
        self.z += diff
        self.left_boundaries += diff


class CompositeMaterial(SimpleDielectric):
    """A very simple implementation of a composite material.

    TODO this could be optmised

    Behaves the same way as layered material, but can be made of any stack of
    previously created materials."""
    def __init__(self, material_list, z0=0, **kwargs):
        """
        Create material from a list of composites.

        Args:
            material_list: list of composites. Note that materials might have
            internal state (e.g. recorded pattern) that is going to be flattened
            and internal material properties lost.
            **kwargs: any other arguments used by SimpleDielectric
        """
        self.materials = material_list
        boundary_z = 0
        z = np.array([])
        for idx, material in enumerate(self.materials):
            material.set_beginning(boundary_z)
            boundary_z = material.z[-1]
            # create those because they are required by the material
            z = np.concatenate([z, material.z[:-1]])
            if idx == len(self.materials) - 1:
                z = np.concatenate([z, [material.z[-1]]])
        super().__init__(z=z, n0=np.zeros_like(z), **kwargs)

    def material_matrix(self, backward=False):
        matrix = p.identity_matrix()
        for idx, material in enumerate(self.materials):
            matrix = np.einsum('ijk,jlk->ilk', p.change_basis(material.matrix), matrix)
            # add boundary:
            if idx < len(self.materials) - 1:
                boundary = p.boundary_matrix(material.index_of_refraction()[-2],
                                             self.materials[idx + 1].index_of_refraction()[0])
                matrix = np.tensordot(boundary, matrix, axes=(1, 0))
        if backward:
            matrix = p.change_basis(matrix)
        return matrix

    def plot(self, ax, imaginary_axis=None, **kwargs):
        for material in self.materials:
            material.plot(ax, imaginary_axis=imaginary_axis, **kwargs)

    def shade_plot(self, ax, alpha=0.1, **kwargs):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = cycle(prop_cycle.by_key()['color'])
        for material in self.materials:
            material.shade_plot(ax, alpha=alpha, color=next(colors), **kwargs)

    def plot_recent_energy(self, ax, **kwargs):
        for material in self.materials:
            material.plot_recent_energy(ax, **kwargs)

    def record(self,
               forward_wave: w.PlanarWave,
               backward_wave: w.PlanarWave = None,
               reflected_wave: w.PlanarWave = None):
        """
        Record interference of two pulses
        """
        if reflected_wave is None:
            assert (backward_wave is not None)
            forward_wave, reflected_wave = p.swap_waves(self.matrix, forward_wave, backward_wave)
        left_wave = [forward_wave.s, reflected_wave.s]
        for material in self.materials:
            left_wave = np.einsum('ijk,jk->ik', material.matrix, left_wave)
            material.record(forward_wave=forward_wave, reflected_wave=reflected_wave)
            forward_wave.s = left_wave[0]
            reflected_wave.s = left_wave[1]
        self.matrix = self.material_matrix(backward=True)
