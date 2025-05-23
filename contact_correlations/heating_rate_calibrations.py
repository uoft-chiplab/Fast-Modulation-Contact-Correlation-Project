# -*- coding: utf-8 -*-
"""
2024-03-27
@author: Chip

Calibrations for use in heating rate data analysis.
"""
import numpy as np
pi = np.pi

#####
##### Field wiggle calibrations
#####

freqs = [10, 10, 1, 2.5, 10, 5] # kHz
Vpps = [0.4, 0.9, 0.9, 0.9, 1.0, 1.8] # 50 Ohm term
fieldAmps = [0.021, 0.048, 0.01, 0.018, 0.054, 0.070] # fit amplitudes in Gauss
e_fieldAmps = [0.002, 0.004, 0.001, 0.002, 0.001, 0.003] # fill
e_fieldAmp = 2e-3 # estimated as 2 mG

Bamp_per_Vpp = {1:None,2.5:None,5:None,10:None} # dict to loop and fill below
for freq, val in Bamp_per_Vpp.items():
	# compute average scaled by Vpp if multiple calibrations at the same freq
	val_avg = np.mean([fieldAmps[i]/Vpps[i] for i in range(len(freqs)) if freqs[i] == freq])
	err_avg = np.mean([e_fieldAmps[i]/Vpps[i] for i in range(len(freqs)) if freqs[i] == freq])
	Bamp_per_Vpp[freq] = [val_avg, err_avg]
	
#####
##### Trap frequency calibrations
#####

### old
## ODTs = 0.2/4
# wx = 151.6
# wy = 429
# wz = 442

### 2024-03-06
## ODTs = 0.2/4
wx = 169.1 # Hz
wy = 453#429#*np.sqrt(2)
wz = 441#442#*np.sqrt(2)
mean_trapfreq = 2*pi*(wx*wy*wz)**(1/3)