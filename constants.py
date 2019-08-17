# -*- coding: utf-8 -*-
"""
Created on 17.08.2019

@author: Michalina Pacholska

Constants used in Femto-Lippmann project.
"""
from matplotlib.ticker import EngFormatter

C = 299792458
NANO = 1e-9
MICRO = 1e-6
VIOLET = 380 * NANO
GREEN = 500 * NANO
RED = 740 * NANO
OMEGA_STEPS = 100
SINGLE_PULSE_ENERGY = 50 * NANO  # nanoJules

FORMATTER = EngFormatter(places=1, sep="")
