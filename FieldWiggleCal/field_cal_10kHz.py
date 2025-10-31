"""This script compares several oscillating field calibrations with different pulse times."""

import sys, os
from pathlib import Path
analysis_path = str(os.path.join(Path(os.getcwd()).parent.parent, 'analysis'))
sys.path.append(analysis_path)

from data_class import Data
from fit_functions import Sinc2, Linear
from library import *
from tabulate import tabulate
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Select calibration to use
freq = 10 # kHz
period = 1/(freq*1000) # s

### Load the field wiggle calibration done previously 
field_cal_path = os.path.join(os.getcwd(), "field_cal_summary.csv")
field_cal_df = pd.read_csv(field_cal_path)
field_cal_df = field_cal_df[(field_cal_df['wiggle_freq'] == freq)]

params = ['B_amp', 'B_phase', 'B_offset']

units = ['(G)', '(rad)', '(G)']

linear, _, param_names = Linear([])

param_name = [param_names[0]]
linear_no_offset = partial(linear, b=0)

###
### Plotting
###

fig, ax = plt.subplots(len(params), 1, figsize=(6, 6), sharex=True)

# B amp fit
popt, pcov = curve_fit(linear_no_offset, field_cal_df['wiggle_amp'], 
					   field_cal_df['B_amp'], sigma=field_cal_df['e_B_amp'])
perr = np.sqrt(np.diag(pcov))
print("Linear fit of Vpp to Bamp gives:")
parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
								 headers=param_name)
print(parameter_table)
ax[0].plot(np.array([0, 1.8]), linear_no_offset(np.array([0, 1.8]), *popt), '--', color='tomato')

# B phase fit
popt, pcov = curve_fit(linear, field_cal_df['wiggle_amp'], 
					   field_cal_df['B_phase'], sigma=field_cal_df['e_B_phase'])
perr = np.sqrt(np.diag(pcov))
print("Linear fit of Vpp to Bphase gives:")
parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
								 headers=param_names)
print(parameter_table)
ax[1].plot(np.array([0, 1.8]), linear(np.array([0, 1.8]), *popt), '--', color='tomato')


for i, param in enumerate(params):
	ax[i].errorbar(field_cal_df['wiggle_amp'], field_cal_df[param], yerr=field_cal_df['e_'+param], fmt='o', color='cornflowerblue')
	ax[i].set(ylabel=param[2:] + " " + units[i])
	
ax[2].set(xlabel = 'Wiggle voltage (Vpp)')
fig.suptitle(f"Field Wiggle Calibrations at {freq} kHz")
fig.tight_layout()