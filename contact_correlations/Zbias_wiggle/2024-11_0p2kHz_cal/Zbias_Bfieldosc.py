# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:19:19 2024

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
from library import colors, dark_colors, light_colors, markers, B_from_FreqMHz

import numpy as np
import matplotlib.pyplot as plt
from cycler import Cycler
from tabulate import tabulate

styles = Cycler([{'color':dark_color, 'mec':dark_color, 'mfc':light_color,
					 'marker':marker} for dark_color, light_color, marker in \
						   zip(dark_colors, light_colors, markers)])

plt.rcParams.update({"figure.figsize": [5,3.5]})

plot = True

def sin(x, A, omega, phi, C):
	return A*np.sin(omega*x+phi) + C

def cos(x, A, omega, phi, C):
	return A*np.cos(omega*x+phi) + C

def fixed_sin(x, A, phi, C):
	return A*np.sin(2*np.pi*0.2*x+phi) + C

def fixed_cos(x, A, phi, C):
	return A*np.cos(2*np.pi*0.2*x+phi) + C

# create file names
filename = "2024-11-06_F_e_delay_samples="
delay1 = np.arange(1000, 1420, 60)
delay2 = np.arange(9500, 9920, 60)
delays = list(delay1) + list(delay2)

files = [filename+str(x)+'.dat' for x in delays]

filename = "2024-11-07_B_e_delay_samples="
delay3 = list(np.arange(5000, 5500, 60))
files = files + [filename+str(x)+'.dat' for x in delay3]

# convert delay from samples to ms
delays = np.array(delays + delay3)/100

# freq centre guesses for fits
guesses = 45.9 * np.ones(len(files))

# guesses[8] = 45.918

fit_func = Sinc2

# .dat file column names and guess params
names = ['freq', 'fraction95']
guess = [0.17, guesses, 0.01, 0.01]
	
if plot == True:
	# loop over files, fitting and taking peak frequencies
	freqs = []
	e_freqs = []
	for file, guess in zip(files, guesses):
		print("--------------------------")
		print("Fitting", file)
		guess = [0.5, guess, 0.1, 0]
		data = Data(file)
			
		data.fit(fit_func, names, guess=guess)
		freqs.append(data.popt[1])
		e_freqs.append(data.perr[1])
		print("freq = {:.4f}({:.0f}) MHz".format(data.popt[1], data.perr[1]*1e4)) 
		

# fit results
freqs = np.array(freqs)
e_freqs = np.array(e_freqs)
Bs = np.array([B_from_FreqMHz(freq) for freq in freqs])
e_Bs = e_freqs/freqs*Bs

# fit to sine function
fit_func = cos
fixed_func = fixed_cos

# fit all data
p0 = [0.12, 1.27, 0, 209.116]
popt, pcov = curve_fit(fit_func, delays, Bs, sigma=e_Bs, p0=p0)
perr= np.sqrt(np.diag(pcov))
p0 = [0.12, 1.8, 209.116]
bounds = ([0,0,0] , [np.inf, 2*np.pi, np.inf]) # A, phi, C
popt_fixed, pcov_fixed = curve_fit(fixed_func, delays, Bs, sigma=e_Bs, p0=p0, bounds=bounds)
perr_fixed = np.sqrt(np.diag(pcov_fixed))


# early times
delays_1 = delays[delays<15]
Bs_1 = Bs[delays<15]
e_Bs_1 = e_Bs[delays<15]
popt_1, pcov_1 = curve_fit(fit_func, delays_1, Bs_1, sigma=e_Bs_1)
perr_1 = np.sqrt(np.diag(pcov_1))

# mid times
delays_2 = delays[(delays>15) & (delays<80)] - 40
Bs_2 = Bs[(delays>15) & (delays<80)]
e_Bs_2 = e_Bs[(delays>15) & (delays<80)]
popt_2, pcov_2 = curve_fit(fit_func, delays_2, Bs_2, sigma=e_Bs_2)
perr_2 = np.sqrt(np.diag(pcov_2))

# later times
delays_3 = delays[delays>80] - 85
Bs_3 = Bs[delays>80]
e_Bs_3 = e_Bs[delays>80]
popt_3, pcov_3 = curve_fit(fit_func, delays_3, Bs_3, sigma=e_Bs_3)
perr_3 = np.sqrt(np.diag(pcov_3))

# plotting
sty_1 = list(styles)[0]
sty_2 = list(styles)[1]
sty_3 = list(styles)[2]
xs_1 = np.linspace(min(delays_1), max(delays_1), 1000)

# plot some more
plt.figure()
plt.xlabel(r'Time, $t$ [ms]')
plt.ylabel(r'Field, $B$ [G]')
plt.errorbar(delays_1, Bs_1, e_Bs_1, **sty_1)
plt.errorbar(delays_2, Bs_2, e_Bs_2, **sty_2)
plt.errorbar(delays_3, Bs_3, e_Bs_3, **sty_3)
plt.plot(xs_1, fit_func(xs_1, *popt_1), '-', color=colors[0], label='First period')
plt.plot(xs_1, fit_func(xs_1, *popt_2), '-', color=colors[1], label='mid period, shifted')
plt.plot(xs_1, fit_func(xs_1, *popt_3), '-', color=colors[2], label='last period, shifted')
plt.legend()

sty = list(styles)[2]
xs = np.linspace(0, max(delays), 1000)
fig, ax = plt.subplots()

ax.errorbar(delays, Bs, e_Bs, **sty)
ax.plot(xs, fit_func(xs, *popt), '--', color=colors[1])
ax.plot(xs, fixed_func(xs, *popt_fixed), '-', color=colors[2])

ax.set(xlim=[0,20], xlabel=r'Time, $t$ [ms]', ylabel=r'Field, $B$ [G]')
plt.tight_layout()

param_names = ['A', 'omega', 'phi', 'C']
parameter_table = tabulate([['Values', *popt_1], ['Errors', *perr_1]], 
 						 headers=param_names)
print('Fit to earlier set:')
print(parameter_table)
param_names = ['A', 'omega', 'phi', 'C']
parameter_table = tabulate([['Values', *popt_2], ['Errors', *perr_2]], 
 						 headers=param_names)
print('Fit to latter set:')
print(parameter_table)

param_names = ['A', 'omega', 'phi', 'C']
parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
 						 headers=param_names)
print('Fit to both sets:')
print(parameter_table)

param_names = ['A', 'phi', 'C']
parameter_table = tabulate([['Values', *popt_fixed], ['Errors', *perr_fixed]], 
 						 headers=param_names)
print('Fit to both sets:')
print(parameter_table)

