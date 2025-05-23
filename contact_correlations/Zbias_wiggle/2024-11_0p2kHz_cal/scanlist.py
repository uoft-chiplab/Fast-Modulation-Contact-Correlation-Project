# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:56:33 2024

@author: coldatoms
"""

# paths
import os
import sys
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
if root not in sys.path:
	sys.path.insert(0, root)

from data_class import Data
from fit_functions import Sinc2
from scipy.optimize import curve_fit
from library import colors, dark_colors, light_colors, markers, FreqMHz

import numpy as np
import matplotlib.pyplot as plt
from cycler import Cycler
from tabulate import tabulate

def Bfield(t, A, phi, C):
	omega = 0.2 * (2*np.pi)
	return A*np.sin(omega*t + phi) + C

Bfield_popt = [0.128221, -1.52954, 202.14]
delay1 = np.arange(1000, 1420, 60)
delay2 = np.arange(9500, 9920, 60)
delays = list(delay1) + list(delay2)
delay3 = list(np.arange(5000, 5500, 60))
# convert delay from samples to ms
delays = np.array(delays + delay3)/100

Bs = Bfield(delays, *Bfield_popt)

# freq75 = np.array([FreqMHz(B, 9/2,-7/2, 9/2, -5/2) for B in Bs]
freq75 = FreqMHz(Bs,9/2,-5/2,9/2,-7/2)
det = 0.1 # 100 kHz
det75 = freq75+det

fig, ax = plt.subplots()
ax.plot(delays, det75)

fig, ax = plt.subplots()
ax.plot(delays, Bs)

pulsewidth = 0.2 # ms
fgendelay = delays - (pulsewidth/2)

scanlist = np.stack([Bs, det75, fgendelay], axis=1)
np.savetxt('HFT_0p2kHz_phaseshift_scanlist.txt', scanlist)
