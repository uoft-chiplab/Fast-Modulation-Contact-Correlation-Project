# -*- coding: utf-8 -*-
"""
To summarize current phase shift measurements and compare to theory curves
Created on Thursday October 2 2025

@author: Chip Lab
"""

import os
# Fast-Modulation-Contact-Correlation-Project\contact_correlations
cc_folder = os.path.dirname(os.getcwd())
# Fast-Modulation-Contact-Correlation-Project\
root_folder = os.path.dirname(cc_folder)
# Fast-Modulation-Contact-Correlation-Project\analysis (contains metadata)
analysis_folder = os.path.join(root_folder, r"analysis")
# GitHub folder
github_folder = os.path.dirname(root_folder)
# analysis folder (that contains library.py)
library_folder = os.path.join(github_folder, r"analysis")
# Fast-Modulation-Contact-Correlation-Project\contact_correlations\phaseshift
ps_folder = os.path.join(root_folder, r"contact_correlations\phaseshift")

import sys
if library_folder not in sys.path:
	sys.path.append(library_folder)
from library import styles, colors, kB, h
from contact_correlations.UFG_analysis import BulkViscTrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from matplotlib import colors
import pickle
import pandas as pd
import ast

#2025 data comes from phase_shift.py
data = pd.read_csv(ps_folder + '\\phase_shift_2025_summary.csv')
if 'Unnamed: 0' in data.columns:
	data.rename(columns={'Unnamed: 0':'run'}, inplace=True)
metadata = pd.read_csv(analysis_folder + '\\metadata.csv')
# later plotting gets easier if I merge the summary df and the metadat df
data= data.merge(metadata, on='run')
data['T'] = data['ToTF'] * (data['EF']/h) # T in Hz
# pickle
pickle_file = cc_folder + '\\time_delay_BVTs_working.pkl'
load = False

# parameters for theory lines
ToTFs = np.array([metadata['ToTF'].min(), metadata['ToTF'].median(), metadata['ToTF'].max()])
# EFs = np.array([metadata['EF'].min(), metadata['EF'].median(), metadata['EF'].max()])/h # Hz
EFs = np.ones(len(ToTFs))*10000 # Hz
barnu = 377 # THIS IS JUST AN ESTIMATE; NOT VALID FOR LOOSE ODTs
num = 50

# set color via normalized ToTF
xs = np.linspace(data['ToTF'].min(), data['ToTF'].max(), 10)
norm = colors.Normalize(vmin = xs.min(), vmax = xs.max())
cmap = cm.get_cmap('RdYlBu').reversed()
def get_color(value, alpha=1):
	"""Map a numeric value to a color. Color map and normalization created outside of scope."""
	color = cmap(norm(value))
	if alpha != 1:
		color[-1]=alpha
	return color
def darken(rgba, factor=0.5):
	"""
	Darken an RGBA color by multiplying the RGB channels.
	factor < 1 = darker, factor = 1 = same color
	"""
	r, g, b, a = rgba
	return (r * factor, g * factor, b * factor, a)

def get_marker(value):
	"""Map a numeric value to a marker. Hardcoded to map mod freqs (kHz)."""
	# markers = ['s', 'o', '^', 'x', 'D', '*']  # circle, square, triangle, etc.
	marker_map = {
		6.0: 's',
		10.0: 'o',
		}
	return marker_map.get(value, 'x') # default to 'x' if not found

def contact_time_delay(phi, period):
	""" Computes the time delay of the contact response given the oscillation
		period and the phase shift.
		phi /  (omega) = phi/(2*pi) * period
	"""
	return phi/(2*np.pi) * period

# load pickle
if load == True:
	with open(pickle_file, 'rb') as f:
		BVTs = pickle.load(f)
	
	analysis = False
		
else: 
	# compute BVT and save to pickle
	BVTs = []
	analysis = True

BVT_T_list = []
BVT_tau_list = []
### generate BVT theory values for given ToTFs, EFs
for i in range(len(ToTFs)):
	ToTF = ToTFs[i]
	EF = EFs[i]
	
	if analysis == False:
		break
	
	T = ToTF*EF
	nus = T*np.logspace(-2, 1, num)
	# compute trap averaged quantities using Tilman's code
	BVT = BulkViscTrap(ToTF, EF, barnu, nus)
	BVT.tau_noz =  ((1.739) * BVT.T) * (2*np.pi)
	BVT.phiLR_noz = np.arctan(2*np.pi*BVT.nus * 1/ BVT.tau_noz)
	# compute time delays
	BVT.time_delay = contact_time_delay(BVT.phaseshiftsQcrit, 1/BVT.nus)
	BVT.time_delay_LR = contact_time_delay(BVT.phiLR, 1/BVT.nus)
	BVT.rel_amp = 1/(np.sqrt(1+(2*np.pi*BVT.nus*BVT.tau)**2))
	BVT_T_list.append(BVT.T)
	BVT_tau_list.append(BVT.tau *1e6) # [us]
	BVTs.append(BVT)

with open(pickle_file, 'wb') as f:
	pickle.dump(BVTs, f)

### PLOTTING ###
# figure for phase shift and related tau
fig, axs = plt.subplots(2,2, figsize=(8, 6))
axs = axs.flatten()

# phase shift 
ax = axs[0]
ax.set(xlabel = r"$h\nu/k_BT$",
	#    ylabel = rf'$\phi = \arctan(\omega \tau)$ [rad]',
	ylabel = rf'$\phi$ [rad]',
	   xscale='log')
# Loop over BVTs to plot theory curves
for j, BVT in enumerate(BVTs):
	x_theory = BVT.nus / BVT.T
	y_theory = BVT.phiLR
	EF_kHz = (BVT.T / BVT.ToTF) / 1e3
	linestyle = ':'
	marker =''
	label = f'ToTF={BVT.ToTF:.2f}, EF={EF_kHz:.0f} kHz'
	ax.plot(x_theory, y_theory, ls=linestyle, marker=marker, color=get_color(BVT.ToTF), label=label)

# Plot data
x_data = (data['Modulation Freq (kHz)']*1000)/data['T']
y_data_C = data['Phase Shift C-B (rad)']
y_data_C_err = data['Phase Shift C-B err (rad)']
colors_data = get_color(data['ToTF'])
colors_data = [tuple(sublist) for sublist in colors_data] 
markers_data = [get_marker(x) for x in (data['Modulation Freq (kHz)'])]

# in order to apply individual colors and markers, must loop 
for x, yC, eyC,color, marker in zip(x_data, y_data_C, y_data_C_err, colors_data, markers_data):	
	ax.errorbar(
		x,yC,yerr = eyC,color=color,marker=marker, mec=darken(color)
		)

# tau
ax=axs[1]
ax.set(xlabel = f"T [Hz]",
		ylabel = rf'$\tau = \tan(\phi)/\omega$ [us]'
		)
# theory curves
x_theory = BVT_T_list
y_theory = BVT_tau_list
linestyle = ':'
marker='.'
ax.plot(x_theory, y_theory, ls=linestyle, marker=marker, color='black', label=label)
		
# plot data
x_data = data['T']
y_tau = (np.tan(y_data_C) / (data['freq']*1e3) / 2/np.pi) * 1e6 # us
y_tau_err = 1/(np.cos(y_data_C))**2 * y_data_C_err # sec^2(x) * \delta x
# in order to apply individual colors and markers, must loop 
for x, y, ey,color, marker in zip(x_data, y_tau, y_tau_err, colors_data, markers_data):	
	ax.errorbar(
		x,y,yerr = ey,color=color,marker=marker, mec=darken(color)
		)
	

### time lag and time lag/tau plts 
ax=axs[2]
ax.set(xlabel = r"$h\nu/k_BT$",	
	ylabel = rf'$\tau_\mathrm{{lag}}=\phi/\omega$ [us]')
# Loop over BVTs to plot theory curves
for j, BVT in enumerate(BVTs):
	x_theory = BVT.nus/BVT.T # dimless
	y_theory = BVT.time_delay_LR * 1e6 # us
	linestyle = ':'
	marker =''
	label = f'ToTF={BVT.ToTF:.2f}, EF={EF_kHz:.0f} kHz'
	ax.plot(x_theory, y_theory, color=get_color(BVT.ToTF), marker=marker, ls=linestyle)
	if j ==0 or j==(len(BVTs)-1):
		ax.hlines(y=BVT.tau * 1e6, xmin=min(x_theory), xmax=max(x_theory), 
			ls='--',color=get_color(BVT.ToTF), label=rf'$\tau = {BVT.tau*1e6:.0f} us, T/T_F = {BVT.ToTF:.2f}$')
ax.legend()
# plot data
x_data = (data['freq']*1000)/data['T'] # dimless
y_taulag = contact_time_delay(y_data_C, 1/(data['freq'] * 1e3)) * 1e6 # us
y_taulag_err = y_data_C_err / y_data_C * y_taulag /(2*np.pi)# TODO: check
# in order to apply individual colors and markers, must loop 
for x, y, ey,color, marker in zip(x_data, y_taulag, y_taulag_err, colors_data, markers_data):	
	ax.errorbar(
		x,y,yerr = ey,color=color,marker=marker, mec=darken(color)
		)
	
ax=axs[3]
ax.set(xlabel = r"$h\nu/k_BT$",
		ylabel = rf'$\tau_\mathrm{{lag}}/\tau$')
# Loop over BVTs to plot theory curves
for j, BVT in enumerate(BVTs):
	x_theory = BVT.nus/BVT.T
	y_theory = BVT.time_delay_LR / BVT.tau
	linestyle = ':'
	marker =''
	label = f'ToTF={BVT.ToTF:.2f}, EF={EF_kHz:.0f} kHz'
	ax.plot(x_theory, y_theory, color=get_color(BVT.ToTF), marker=marker, ls=linestyle)

# plot data
y_scaledtau = y_taulag/y_tau
y_scaledtau_err = np.sqrt((y_taulag_err / y_taulag)**2 + (y_tau_err/y_tau)**2) * y_scaledtau
# in order to apply individual colors and markers, must loop 
for x, y, ey,color, marker in zip(x_data, y_scaledtau, y_scaledtau_err, colors_data, markers_data):	
	ax.errorbar(
		x,y,yerr = ey,color=color,marker=marker, mec=darken(color)
		)
fig.suptitle(r'Summary: analyzing $\phi$')
fig.tight_layout()

### AMPLITUDE ANALYSIS SUMMARY
fig, axs = plt.subplots(2,2, figsize=(8,6))
axs=axs.flatten()
axs[0].set(
	xlabel=r"$h\nu/k_BT$", 
	   ylabel=r"Amplitude of Sin Fit to $\alpha$",
)
axs[1].set(
	xlabel=r"$h\nu/k_BT$", 
	   ylabel=r"Amplitude of Sin Fit to $C/Nk_F$",
)
axs[2].set(
	xlabel=r"$h\nu/k_BT$", 
	   ylabel=r"Relative amplitude $\alpha_\mathrm{AC}/\alpha_\mathrm{DC}$",
)
axs[3].set(
	xlabel=r"$h\nu/k_BT$", 
	   ylabel=r"Relative amplitude $C_\mathrm{AC}/C_\mathrm{DC}$",
)
x_data = (data['freq']*1000)/data['T']
plot_params = ['Sin Fit of A', 'Sin Fit of C']
for i, plot_param in enumerate(plot_params):
	# ensures conversion of strings into actual lists of values
	data[plot_param] = data[plot_param].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
	data['Error of ' +plot_param] = data['Error of ' + plot_param].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
	# creates lists of only popt[0] and perr[0] (amplitude)
	y_amps = np.array(data[plot_param].apply(lambda x: x[0]).tolist())
	yerr = np.array(data['Error of ' + plot_param].apply(lambda x: x[0]).tolist())
	for x, y, ey,color, marker in zip(x_data, y_amps, yerr, colors_data, markers_data):	
		axs[i].errorbar(
		x,y,yerr= ey,color=color,marker=marker, mec=darken(color)
		)

for b, BVT in enumerate(BVTs):
	axs[2].plot(BVT.nus/BVT.T, BVT.rel_amp,':', color=get_color(BVT.ToTF),
		 label = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
		  )
	axs[3].plot(BVT.nus/BVT.T, BVT.rel_amp,':', color=get_color(BVT.ToTF),
		 label = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
		  )
for i, plot_param in enumerate(plot_params):
	DC_amps = 0.0844 # NOT CORRECT
	DC_amps_err = 0.0136
	AC_amps = np.array(data[plot_param].apply(lambda x: x[0]).tolist())
	AC_amps_err = np.array(data['Error of ' + plot_param].apply(lambda x: x[0]).tolist())

	rel_amps = AC_amps/DC_amps
	e_rel_amps = np.sqrt((AC_amps_err/AC_amps)**2 + (DC_amps_err/DC_amps)**2) * rel_amps
	for x, y, ey,color, marker in zip(x_data, rel_amps, e_rel_amps, colors_data, markers_data):	
		axs[i+2].errorbar(
		x,y,yerr= ey,color=color,marker=marker, mec=darken(color)
		)


# for (i, temp, EFs) in zip(data_2025.index, templist, EFslist):
# 	x = data_2025['Modulation Freq (kHz)'][i]/temp/EFs
# 	if data_2025['Unnamed: 0'][i] != '2025-10-01_L': # only have data to compare to 10 kHz for now -- 2025-10-15
# 		ax.errorbar(x, rel_amp, 
# 		yerr = e_rel_amp_abs,
# 		marker='o', ls='', color=colors_light[i], 
# 		label=f"{data_2025['Unnamed: 0'][i]}, A: {data_2025['Modulation Freq (kHz)'][i]}kHz"
# 		)
# 		# continue
# 		# Completely eyeballed guess right now
# 		# DC_amp = 0.05
# 		# e_DC_amp = 0.01
# 		# AC_amp = data_2025['Amplitude of Sin Fit of A'][i]
# 		# e_AC_amp = data_2025['Error of Amplitude of Sin Fit of A'][i]
# 		# rel_amp = AC_amp/DC_amp
# 		# e_rel_amp = np.sqrt((e_AC_amp/AC_amp)**2 + (e_DC_amp/DC_amp)**2)
# 		# e_rel_amp_abs = e_rel_amp * rel_amp
# 		# ax.errorbar(x, rel_amp, 
# 		# 		yerr = e_rel_amp_abs,
# 		# 		marker='o', ls='', color=colors_light[i], 
# 		# 		label=f"A: {data_2025['Modulation Freq (kHz)'][i]}kHz, cf. static measurements")
# 	else :
# 		DC_amp = 0.0844
# 		e_DC_amp = 0.0136
# 		AC_amp = data_2025['Amplitude of Sin Fit of A'][i]
# 		e_AC_amp = data_2025['Error of Amplitude of Sin Fit of A'][i]
# 		rel_amp = AC_amp/DC_amp
# 		e_rel_amp = np.sqrt((e_AC_amp/AC_amp)**2 + (e_DC_amp/DC_amp)**2)
# 		e_rel_amp_abs = e_rel_amp * rel_amp
# 		ax.errorbar(x, rel_amp, 
# 				yerr = e_rel_amp_abs,
# 				marker='o', ls='', color=colors_light[i], 
# 				label=f"A: {data_2025['Modulation Freq (kHz)'][i]}kHz, cf. static measurements"
# 				)
# ax.legend()
	
