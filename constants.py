# -*- coding: utf-8 -*-
"""
Created on 17.08.2019

@author: Michalina Pacholska

Constants used in Femto-Lippmann project.
"""
from matplotlib.ticker import EngFormatter
import numpy as np

C = 299792458  # speed of light
NANO = 1e-9
MICRO = 1e-6
VIOLET = 380 * NANO  # wavelength of violet light
GREEN = 500 * NANO  # wavelength of green light
RED = 740 * NANO  # wavelength of red light
OMEGA_STEPS = 10000  # default frequency/wavenumber discretization
SINGLE_PULSE_ENERGY = 50 * NANO  # nanoJules
DEFAULT_K = np.linspace(  # default wavenumber vector
    2 * np.pi / (1.0 * MICRO), 2 * np.pi / (300.0 * NANO), OMEGA_STEPS)
FORMATTER = EngFormatter(places=1, sep="")  # often used plot axis formatter
