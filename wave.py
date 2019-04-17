# -*- coding: utf-8 -*-
"""
Created on 17/04/19

@author: Michalina Pacholska
"""
from __future__ import annotations
import numpy as np

C = 1.0


class PlanarWave(object):
    """Basic class to describe planar light wave"""

    def __init__(self,
                 z: np.ndarray = np.empty(0),
                 k: np.ndarray = np.empty(0),
                 spectrum: np.ndarray = np.empty(0),
                 shape: np.ndarray = np.empty(0),
                 time_zero: float = 0):
        if spectrum.shape != k.shape:
            raise ValueError()  # TODO(michalina)
        if shape.shape != z.shape:
            raise ValueError()  # TODO(michalina)
        self.z = z
        self.k = k
        self.spectrum = spectrum
        self.shape = shape
        self.time_zero = time_zero

    def __mul__(self, alpha: complex) -> PlanarWave:
        spectrum = alpha * self.spectrum
        shape = alpha * self.shape
        return PlanarWave(self.z, self.k, spectrum, shape)

    __rmul__ = __mul__

    def propagate(self, timestamp: float):
        if self.spectrum.shape != (0,):
            self.spectrum = self.spectrum * np.exp(1j * self.k * C * (timestamp-self.time_zero))
        else:
            raise NotImplementedError()
        self.time_zero = timestamp

    def _spectrum_to_shape(self):
        raise NotImplementedError

    def _shape_to_spectrum(self):
        raise NotImplementedError

    def __add__(self, other: PlanarWave) -> PlanarWave:
        raise NotImplementedError

    def _interpolate_domain(self, domain, k=False): # TODO(michalina) some kind of enum?
        if k:
            old_domain = self.k
            old_functon = self.spectrum
        else:
            old_domain = self.z
            old_domain = self.shape
        raise NotImplementedError

    def get_power_spectrum(self):
        raise NotImplementedError

    def get_power(self):
        raise NotImplementedError

    def get_power(self):
        raise NotImplementedError

    def __get__(self, instance, owner):