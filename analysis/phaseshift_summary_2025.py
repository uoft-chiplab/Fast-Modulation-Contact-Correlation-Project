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
root_folder = os.getcwd()
# Fast-Modulation-Contact-Correlation-Project\analysis (contains metadata)
analysis_folder = os.path.join(root_folder, r"analysis")
# GitHub folder
github_folder = os.path.dirname(root_folder)
# analysis folder (that contains library.py)
library_folder = os.path.join(github_folder, r"analysis")
# Fast-Modulation-Contact-Correlation-Project\contact_correlations\phaseshift
ps_folder = os.path.join(root_folder, r"contact_correlations\phaseshift")

from preamble import *

# Fast-Modulation-Contact-Correlation-Project\FieldWiggleCal
field_cal_folder = os.path.join(root_analysis, r"FieldWiggleCal")

import sys
if library_folder not in sys.path:
	sys.path.append(library_folder)
if analysis_folder not in sys.path:
	sys.path.append(analysis_folder)
if cc_folder not in sys.path:
	sys.path.append(cc_folder)
from library import styles, colors, a97, mK, hbar
from constants import kB, h, hbar, pi
from unitary_fermi_gas import TrappedUnitaryGas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from matplotlib import colors
import pickle
import pandas as pd
import ast
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

#2025 data comes from phase_shift.py
data = pd.read_csv(root_project + os.sep+'phase_shift_2025_summary.csv')
if 'Unnamed: 0' in data.columns:
	data.rename(columns={'Unnamed: 0':'run'}, inplace=True)
metadata = pd.read_csv(root_project + os.sep + 'metadata.csv')
# later plotting gets easier if I merge the summary df and the metadat df
data= data.merge(metadata, on='run')
data['T'] = data['ToTF'] * (data['EF']/h) # T in Hz
# pickle
pickle_file = cc_folder + os.sep + 'time_delay_TUGs_working.pkl'
load = False

# parameters for theory lines
ToTFs = np.array([metadata['ToTF'].min(), metadata['ToTF'].median(), metadata['ToTF'].max()])
# EFs = np.array([metadata['EF'].min(), metadata['EF'].median(), metadata['EF'].max()])/h # Hz
EFs = np.ones(len(ToTFs))*10000 # Hz
barnu = 377 # THIS IS JUST AN ESTIMATE; NOT VALID FOR LOOSE ODTs
num = 50

# set color via normalized ToTF
xs = np.linspace(data['T'].min(), data['T'].max(), 10)
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
def get_marker(value, HFT_or_dimer):
	"""Map a numeric value to a marker. Hardcoded to map mod freqs (kHz) and also pulse type (HFT_or_dimer)."""
	# markers = ['s', 'o', '^', 'x', 'D', '*']  # circle, square, triangle, etc.
	if HFT_or_dimer == 'dimer':
		marker_map = {
			6.0: 's',
			10.0: 'o',
			}
	elif HFT_or_dimer == 'HFT':
		marker_map = {
			6.0: 'd',
			10.0: 'P',
			20.0 : 'v'
		}
	return marker_map.get(value, 'x') # default to 'x' if not found

def contact_time_delay(phi, period):
	""" Computes the time delay of the contact response given the oscillation
		period and the phase shift.
		phi /  (omega) = phi/(2*pi) * period
	"""
	return phi/(2*pi) * period

def Linear(x, m, b):
	return m*x + b

# load pickle
TUG_T_list = []
TUG_tau_list = []

if load == True:
	with open(pickle_file, 'rb') as f:
		TUGs = pickle.load(f)
	analysis = False

else: 
	# compute TUG and save to pickle
	TUGs = []
	analysis = True

	### generate TUG theory values for given ToTFs, EFs
	for i, ToTF in enumerate(ToTFs):
		EF = EFs[i]

		T = ToTF*EF
		nus = T*np.logspace(-2, 1, num)
		# compute trap averaged quantities using Tilman's code
		TUG = TrappedUnitaryGas(ToTF, EF, barnu)
		TUG.modulate_field(nus)
		TUG.tau_noz =  ((1.739) * TUG.T) * (2*pi)
		TUG.phiLR_noz = np.arctan(2*pi*TUG.nus * 1/ TUG.tau_noz)
		# compute time delays
		TUG.time_delay = contact_time_delay(TUG.phaseshiftsQcrit, 1/TUG.nus)
		TUG.time_delay_LR = contact_time_delay(TUG.phiLR, 1/TUG.nus)
		TUG.rel_amp = 1/(np.sqrt(1+(2*pi*TUG.nus*TUG.tau)**2))
		TUG_T_list.append(TUG.T)
		TUG_tau_list.append(TUG.tau *1e6) # [us]
		TUGs.append(TUG)

	with open(pickle_file, 'wb') as f:
		pickle.dump(TUGs, f)

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
# Loop over TUGs to plot theory curves
for j, TUG in enumerate(TUGs):
	x_theory = TUG.nus / TUG.T
	y_theory = TUG.phiLR
	EF_kHz = (TUG.T / TUG.ToTF) / 1e3
	label = f'ToTF={TUG.T:.2f}, EF={EF_kHz:.0f} kHz'
	ax.plot(x_theory, y_theory, ls=':', marker='', color=get_color(TUG.T), label=label)

# Plot data
x_data = (data['Modulation Freq (kHz)']*1000)/data['T']
y_data_C = data['Phase Shift C-B (rad)']
y_data_C_err = data['Phase Shift C-B err (rad)']
colors_data = get_color(data['T'])
colors_data = [tuple(sublist) for sublist in colors_data] 
markers_data = [get_marker(x, HFT_or_dimer) for x, HFT_or_dimer in zip(data['Modulation Freq (kHz)'], data['HFT_or_dimer'])]

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
x_theory = TUG_T_list
y_theory = TUG_tau_list
linestyle = ':'
marker='.'
ax.plot(x_theory, y_theory, ls=linestyle, marker=marker, color='black', label=label)
		
# plot data
x_data = data['T']
y_tau = (np.tan(y_data_C) / (data['freq']*1e3) / 2/np.pi) * 1e6 # us
y_tau_err = 1/(np.cos(y_data_C))**2 * y_data_C_err / (data['freq']*1e3) / 2/np.pi * 1e6 # sec^2(x) * \delta x
# in order to apply individual colors and markers, must loop 
for x, y, ey,color, marker in zip(x_data, y_tau, y_tau_err, colors_data, markers_data):	
	ax.errorbar(
		x,y,yerr = ey,color=color,marker=marker, mec=darken(color)
		)
	

### time lag and time lag/tau plts 
ax=axs[2]
ax.set(xlabel = r"$h\nu/k_BT$",	
	ylabel = rf'$\tau_\mathrm{{lag}}=\phi/\omega$ [us]')
# Loop over TUGs to plot theory curves
for j, TUG in enumerate(TUGs):
	x_theory = TUG.nus/TUG.T # dimless
	y_theory = TUG.time_delay_LR * 1e6 # us
	linestyle = ':'
	marker =''
	label = f'ToTF={TUG.ToTF:.2f}, EF={EF_kHz:.0f} kHz'
	ax.plot(x_theory, y_theory, color=get_color(TUG.T), marker=marker, ls=linestyle)
	if j ==0 or j==(len(TUGs)-1):
		ax.hlines(y=TUG.tau * 1e6, xmin=min(x_theory), xmax=max(x_theory), 
			ls='--',color=get_color(TUG.T), label=rf'$\tau = {TUG.tau*1e6:.0f} us, T/T_F = {TUG.ToTF:.2f}$')
ax.legend()
# plot data
x_data = (data['freq']*1000)/data['T'] # dimless
y_taulag = contact_time_delay(y_data_C, 1/(data['freq'] * 1e3)) * 1e6 # us
y_taulag_err = y_data_C_err / y_data_C * y_taulag # TODO: check
# in order to apply individual colors and markers, must loop 
for x, y, ey,color, marker in zip(x_data, y_taulag, y_taulag_err, colors_data, markers_data):	
	ax.errorbar(
		x,y,yerr = ey,color=color,marker=marker, mec=darken(color)
		)
	
ax=axs[3]
ax.set(xlabel = r"$h\nu/k_BT$",
		ylabel = rf'$\tau_\mathrm{{lag}}/\tau$')
# Loop over TUGs to plot theory curves
for j, TUG in enumerate(TUGs):
	x_theory = TUG.nus/TUG.T
	y_theory = TUG.time_delay_LR / TUG.tau
	linestyle = ':'
	marker =''
	label = f'ToTF={TUG.ToTF:.2f}, EF={EF_kHz:.0f} kHz'
	ax.plot(x_theory, y_theory, color=get_color(TUG.T), marker=marker, ls=linestyle)

# plot data
y_scaledtau = y_taulag/y_tau
y_scaledtau_err = (np.cos(y_data_C)/np.sin(y_data_C) - y_data_C*(np.cos(y_data_C))**2)*y_data_C_err #np.sqrt((y_taulag_err / y_taulag)**2 ) * y_scaledtau
# in order to apply individual colors and markers, must loop 
for i, (x, y, ey,color, marker) in enumerate(zip(x_data, y_scaledtau, y_scaledtau_err, colors_data, markers_data)):	
	try:
		ax.errorbar(
		x,y,yerr = ey,color=color,marker=marker, mec=darken(color)
		)
	except ValueError:
		print(f'yerr negative: {y_scaledtau_err[i]:0.3f} for {i}')
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
	   ylabel=r"$\tau = \sqrt{(C_{DC}/C_{AC} - 1)}/\omega$",
)
axs[2].set(
	xlabel=r"$h\nu/k_BT$", 
	   ylabel=r"Relative amplitude $\alpha_\mathrm{AC}/\alpha_\mathrm{DC}$",
	   ylim=[0,1]
)
axs[3].set(
	xlabel=r"$h\nu/k_BT$", 
	   ylabel=r"Relative amplitude $C_\mathrm{AC}/C_\mathrm{DC}$",
	   ylim=[0,1]
)
x_data = (data['freq']*1000)/data['T']
plot_params = ['Sin Fit of A', 'Sin Fit of C']
seen_markers = set()
for i, plot_param in enumerate(plot_params):
	# ensures conversion of strings into actual lists of values
	data[plot_param] = data[plot_param].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
	data['Error of ' +plot_param] = data['Error of ' + plot_param].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
	# creates lists of only popt[0] and perr[0] (amplitude)
	y_amps = np.array(data[plot_param].apply(lambda x: x[0]).tolist())
	yerr = np.array(data['Error of ' + plot_param].apply(lambda x: x[0]).tolist())
	if i != 1:
		for x, y, ey,color, marker, mod_freq, pulse_type in zip(x_data, y_amps, yerr, colors_data, markers_data, data['Modulation Freq (kHz)'], data['HFT_or_dimer']):	
			label = f"{pulse_type} {mod_freq}kHz" if marker not in seen_markers else None
			if label:
				seen_markers.add(marker)

			axs[i].errorbar(
			x,y,yerr= ey,color=color,marker=marker, mec=darken(color), 
			label=label
			)
axs[0].legend()
ax_tau = axs


for b, TUG in enumerate(TUGs):
	axs[2].plot(TUG.nus/TUG.T, TUG.rel_amp,':', color=get_color(TUG.T),
		 label = f'ToTF={TUG.ToTF}, EF={(TUG.T/TUG.ToTF)/10e2:.0f} kHz'
		  )
	axs[3].plot(TUG.nus/TUG.T, TUG.rel_amp,':', color=get_color(TUG.T),
		 label = f'ToTF={TUG.ToTF}, EF={(TUG.T/TUG.ToTF)/10e2:.0f} kHz'
		  )

# comparing AC to DC contact. This code is very ugly because of different data processing over time.
# get DC dimer amplitudes (currently only valid for ToTF=0.3)
####to do choose btw dimer or HFT and what temp 
DCdf = pd.read_csv(os.path.join(root_analysis, 'exploratory_and_misc/DC_contact.csv'))
# DCdf['HFT_or_dimer'] = 'dimer'
DCdf['alpha'] = DCdf['popts'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
DCdf.sort_values(by='Bfield')
Bs_d = np.linspace(DCdf['Bfield'].min(), DCdf['Bfield'].max(), 10)

###yes this is dumb but i think it works 
cold_subset = DCdf[DCdf['ToTF'] < 0.3]
hot_subset = DCdf[DCdf['ToTF'] > 0.3]

cold_subset.sort_values(by='Bfield')
alphas_cold = np.array(cold_subset['alpha'].apply(lambda x: x[0]).tolist())
field_to_contact_dimer_cold = interp1d(cold_subset['Bfield'], cold_subset['contact'], kind='linear', fill_value='extrapolate')
field_to_alpha_dimer_cold = interp1d(cold_subset['Bfield'], alphas_cold, kind='linear', fill_value='extrapolate')
Cs_d_cold = field_to_contact_dimer_cold(Bs_d)
As_d_cold = field_to_alpha_dimer_cold(Bs_d)

hot_subset.sort_values(by='Bfield')
alphas_hot = np.array(hot_subset['alpha'].apply(lambda x: x[0]).tolist())
field_to_contact_dimer_hot = interp1d(hot_subset['Bfield'], hot_subset['contact'], kind='linear', fill_value='extrapolate')
field_to_alpha_dimer_hot = interp1d(hot_subset['Bfield'], alphas_hot, kind='linear', fill_value='extrapolate')
Cs_d_hot = field_to_contact_dimer_hot(Bs_d)
As_d_hot = field_to_alpha_dimer_hot(Bs_d)



DCdf = DCdf[DCdf['ToTF'] < 0.3]
# DCdf_hot = DCdf[DCdf['ToTF'] > 0.3]
DCdf.sort_values(by='Bfield')
alphas = np.array(DCdf['alpha'].apply(lambda x: x[0]).tolist())
field_to_contact_dimer = interp1d(DCdf['Bfield'], DCdf['contact'], kind='linear', fill_value='extrapolate')
field_to_alpha_dimer = interp1d(DCdf['Bfield'], alphas, kind='linear', fill_value='extrapolate')
Bs_d = np.linspace(DCdf['Bfield'].min(), DCdf['Bfield'].max(), 10)
Cs_d = field_to_contact_dimer(Bs_d)
As_d = field_to_alpha_dimer(Bs_d)

# DC HFT contact susceptibility from early November runs (ToTF <~ 0.3)
sus_df = pd.read_csv(os.path.join(root_analysis, 'corrections//saturation_HFT.csv'))
sus_df['HFT_or_dimer']='HFT'
popt, pcov = curve_fit(Linear, sus_df['Bfield'], sus_df['C'], sigma=sus_df['e_C'])
popt_fudged, pcov_fudged = curve_fit(Linear, sus_df['Bfield'], sus_df['fudgedC'], sigma=sus_df['e_fudgedC'])
field_to_contact_HFT_unfudged = lambda B :Linear(B, *popt)
field_to_contact_HFT_cold = lambda B: Linear(B, *popt_fudged)
Bs_HFT = np.linspace(sus_df['Bfield'].min(), sus_df['Bfield'].max(), 10)
Cs_HFT = field_to_contact_HFT_cold(Bs_HFT)
Cs_HFT_unfudged = field_to_contact_HFT_unfudged(Bs_HFT)
# fig2, ax = plt.subplots()
# ax.plot(DCdf['Bfield'], DCdf['contact'], marker='o', ls='', color='orchid', label='DC dimer')
# ax.plot(Bs_d, Cs_d, ls='--', marker='', color='orchid', label='dimer fit')
# ax.errorbar(sus_df['Bfield'], sus_df['fudgedC'], sus_df['e_fudgedC'], marker='P', ls='', color='salmon', label='DC HFT')
# ax.plot(Bs_HFT, Cs_HFT, ls= '--', marker='', color='salmon', label='HFT fudged')
# ax.plot(Bs_HFT, Cs_HFT_unfudged, ls= '--', marker='', color='orange', label='HFT unfudged')
# ax.set(xlabel = 'Bfield [G]', ylabel = r'$\widetilde{C}$', title = r'DC C, $T/T_F \lesssim 0.3$')
# ax.legend()
# dirty implementation for hot HFT sus, TODO clean up, want to comebine all temps
sus_df_hot = pd.read_csv(os.path.join(root_analysis, 'corrections//saturation_HFT_hot.csv'))
sus_df_hot['HFT_or_dimer']='HFT'
popt, pcov = curve_fit(Linear, sus_df_hot['Bfield'], sus_df_hot['C'], sigma=sus_df_hot['e_C'])
popt_fudged_hot, pcov_fudged_hot = curve_fit(Linear, sus_df_hot['Bfield'], sus_df_hot['fudgedC'], sigma=sus_df_hot['e_fudgedC'])
field_to_contact_HFT_hot = lambda B: Linear(B, *popt_fudged_hot)



# need field cal amplitudes for each run
# load wiggle field calibration
field_cal_df_path = os.path.join(field_cal_folder, "field_cal_summary.csv")
field_cal_df = pd.read_csv(field_cal_df_path)
field_cal_df['field_cal_run'] = field_cal_df['run']
field_cal_df = field_cal_df[['field_cal_run','B_amp']]
data=pd.merge(data, field_cal_df, on='field_cal_run')

# from field_cal_run, find field amplitude for each run, then evaluate DC contact and alpha amplitude
# for 202.14 +/- Bamp. Calculate difference. Call it DC amp.

data['alpha_DC_amp'] = np.where(
    data['HFT_or_dimer'] == 'dimer',
    np.abs(field_to_alpha_dimer(202.14 - data['B_amp']) - field_to_alpha_dimer(202.14 + data['B_amp'])) / 2,
	0
    # np.abs(field_to_contact_HFT(202.14 - data['B_amp']) - field_to_contact_HFT(202.14 + data['B_amp'])) / 2
)



ToTFcutoff = 0.4
data_dimer = data[data['HFT_or_dimer'] == 'dimer']
data_dimer['contact_DC_amp'] = np.where(
data_dimer['ToTF'] < ToTFcutoff, 
    (field_to_contact_dimer_cold(202.14 - data_dimer['B_amp']) - 
     field_to_contact_dimer_cold(202.14 + data_dimer['B_amp'])) / 2,
    (field_to_contact_dimer_hot(202.14 - data_dimer['B_amp']) - 
     field_to_contact_dimer_hot(202.14 + data_dimer['B_amp'])) / 2
)
data_HFT = data[data['HFT_or_dimer'] == 'HFT']
data_HFT['contact_DC_amp'] = np.where(
data_HFT['ToTF'] < ToTFcutoff, 
    (field_to_contact_HFT_cold(202.14 - data_HFT['B_amp']) - 
     field_to_contact_HFT_cold(202.14 + data_HFT['B_amp'])) / 2,
    (field_to_contact_HFT_hot(202.14 - data_HFT['B_amp']) - 
     field_to_contact_HFT_hot(202.14 + data_HFT['B_amp'])) / 2
)

data = pd.concat([data_dimer,data_HFT])
for i, plot_param in enumerate(plot_params):
	if i==0:
		key = 'alpha'
	else:
		key = 'contact'

	DC_amps = data[f'{key}_DC_amp']
	DC_amps_err = data[f'{key}_DC_amp']/5 # TODO: PROPAGATE UNCERTAINTY CORRECTLY
	data[f'{key}_AC_amp'] = np.array(data[plot_param].apply(lambda x: x[0]).tolist())
	data[f'{key}_AC_amp_err'] = np.array(data['Error of ' + plot_param].apply(lambda x: x[0]).tolist())
	data[f'{key}_rel_amp'] = data[f'{key}_AC_amp']/data[f'{key}_DC_amp']
	data[f'{key}_rel_amp_err'] = np.sqrt((data[f'{key}_AC_amp_err']/data[f'{key}_AC_amp'])**2 +\
									(DC_amps_err/DC_amps)**2) * data[f'{key}_rel_amp']
	rel_amps = data[f'{key}_rel_amp']
	e_rel_amps = data[f'{key}_rel_amp_err']

	for x, y, ey,color, marker in zip(x_data, rel_amps, e_rel_amps, colors_data, markers_data):	
		axs[i+2].errorbar(
		x,y,yerr= ey,color=color,marker=marker, mec=darken(color)
		)


fig.suptitle(r'Summary: analyzing amplitude (uncertainties not prop. correctly)')
fig.tight_layout()


# show EF, T, ToTF of every data point
fig, ax=plt.subplots(3,1)
xd = data['T']
yd = data['EF']/h
yd_mean = yd.mean()
yd_std = yd.std()

for x, y,color, marker in zip(xd, yd, colors_data, markers_data):	
	ax[0].plot(x, y, color=color, marker=marker, mec=darken(color))
ax[0].hlines(y=yd_mean, xmin=xd.min(), xmax=xd.max(), ls='--', color='gray')
ax[0].fill_between(xd.sort_values(ascending=True), yd_mean-yd_std, yd_mean + yd_std, color='gray', alpha=0.2)
ax[0].set(xlabel=r'$T$', ylabel=r'$E_F$')

xd = data['ToTF']

for x, y,color, marker in zip(xd, yd, colors_data, markers_data):	
	ax[1].plot(x, y, color=color, marker=marker, mec=darken(color))
ax[1].hlines(y=yd_mean, xmin=xd.min(), xmax=xd.max(), ls='--', color='gray')
ax[1].fill_between(xd.sort_values(ascending=True), yd_mean-yd_std, yd_mean + yd_std, color='gray', alpha=0.2)
ax[1].set(xlabel=r'$T/TF$', ylabel=r'$E_F$')

xd = data['run']
for x, y,color, marker in zip(xd, yd, colors_data, markers_data):	
	ax[2].plot(x[5:], y, color=color, marker=marker, mec=darken(color))
ax[2].set(xlabel=r'run name', ylabel=r'$E_F$')
for label in ax[2].get_xticklabels():
	label.set_rotation(90)

fig.tight_layout()

x_data = data['T']	
y_data = np.sqrt((1/data['contact_rel_amp']) - 1)/(data['Modulation Freq (kHz)']*2*np.pi)	
for x, y,color, marker in zip(x_data, y_data, colors_data, markers_data):	
	ax_tau[1].plot(x, y, color=color, marker=marker, mec=darken(color))
	ax_tau[1].set(
		xlabel = 'T (Hz)'
	)


# some additional checks
fig, ax = plt.subplots()
x_data = data['alpha_AC_amp']
y_data = data['contact_AC_amp']
y_err = data['contact_AC_amp_err']
popt, pcov = curve_fit(lambda x,m: Linear(x, m, b=0), x_data, y_data, sigma=y_err)
# plot linear fit
xs = np.linspace(x_data.min(), x_data.max(), 20)
ax.plot(xs, Linear(xs, *popt, 0), ls='-', marker='', color='gray', label=f'Linear Fixed: y={popt[0]:.2f}x')

# plot data points
for x, y, yerr, color, marker in zip(x_data, y_data, y_err, colors_data, markers_data):
	ax.errorbar(x, y, yerr, color=color, marker=marker, mec=darken(color), ls='')

ax.legend()
ax.set(
	xlabel=r'Amplitude of $\alpha_\mathrm{AC}$',
	ylabel=r'Amplitude of $\widetilde{C}_\mathrm{AC}$'
)

# plot C vs T, and scale sus vs. T
fig, ax = plt.subplots(2, 2, figsize=(8,6))
ax = ax.flatten()
kF = np.sqrt(2*mK*data['EF'])/hbar
Cs = [Cfit[-1] for i, Cfit in enumerate(data['Sin Fit of C'])]
Cerrs = [Cfit[-1] for i, Cfit in enumerate(data['Error of Sin Fit of C'])]

B0 = 202.14 # assume avg B is 202.14
Bamp = data['B_amp']
dkFa0_inv = 1/(kF*a97(B0 - Bamp)) - 1/(kF*a97(B0 + Bamp)) # max B - min B 

dC_kFda0 = 2*data['contact_AC_amp']/dkFa0_inv # dC/d(kF a0)^-1 or change in measured contact per change in field/scattering length 
dCkFda0_err = 2*dC_kFda0*np.sqrt((data['contact_AC_amp_err']/data['contact_AC_amp'])**2 + (data['eEF']/data['EF'])**2)

for x, y, yerr, color, marker in zip(data['ToTF'], Cs, Cerrs, colors_data, markers_data):
	ax[0].errorbar(x, y, yerr, color=color, marker=marker, mec=darken(color), ls='')

popts, pcov = curve_fit(lambda x,m,b: Linear(x, m, b), data['ToTF'], Cs)
xss = np.linspace(0.2 , max(data['ToTF']), 100)
ax[0].plot(xss, Linear(xss, *popts), ls=linestyle, color='darkgray', marker='')

# ratio = popts[0] / 
# plot C from some recent DC measurements
# these are from HFT measurements
HFT_meas = pd.concat([sus_df[sus_df['Bfield']==202.14][['Bfield', 'ToTF','fudgedC','e_fudgedC']],
					sus_df_hot[sus_df_hot['Bfield'] == 202.14][['Bfield','ToTF','fudgedC','e_fudgedC']] ]
					)
ax[0].errorbar(HFT_meas['ToTF'], HFT_meas['fudgedC'], HFT_meas['e_fudgedC'], 
			   mec='black',marker='*', markersize=10,
			   label='HFT DC')

# Loop over TUGs to plot theory curves
ToTFs = np.linspace(0.2, 0.6, 5)
EF = 10000
ytugs=[TrappedUnitaryGas(x, EF, barnu) for x in ToTFs]
Ctrap_list = [y.Ctrap for y in ytugs]
scale_sus_list = [y.dCdkFa_inv for y in ytugs]

# simple linear fit to data points
popts, pcov = curve_fit(lambda x,m,b: Linear(x, m, b), data['ToTF'], Cs)
yss = Linear(data['ToTF'], *popts)
data['C_exp_fit'] = yss
# had to sort to make plotting work
ax[0].plot(data.sort_values(by='ToTF')['ToTF'], data.sort_values(by='ToTF')['C_exp_fit'],
		    ls='--', color='darkgray', marker='')
# interpolate theory over ToTF
Ctheory = interp1d(ToTFs, Ctrap_list, kind='linear', fill_value='extrapolate')
# interpolate correction ratio as function of ToTF
ratio = Ctheory(data['ToTF']) / yss
ratio_totf = interp1d(data['ToTF'], ratio, kind='linear', fill_value='extrapolate')
data['C_rescaled'] = ratio_totf(data['ToTF']) * Cs 
data['C_rescaled_err'] = ratio_totf(data['ToTF']) * Cerrs
data['chi_rescaled'] = ratio_totf(data['ToTF']) * dC_kFda0 
data['chi_rescaled_err'] = ratio_totf(data['ToTF']) * dCkFda0_err
label = r'$\langle \widetilde{C}_\mathrm{eq} \rangle_\mathrm{trap}$'
ax[0].plot(ToTFs, Ctrap_list, marker='', ls='-', color='black', label='Trap avg')
ax[0].errorbar(data['ToTF'], data['C_rescaled'] ,
			   yerr=data['C_rescaled_err'],
				marker='.', color='dimgrey', label='scaled')
ax[0].set(
	xlabel=r'$T/T_F$',
	ylabel=r'$\langle\widetilde{C}_\mathrm{eq}\rangle$'
)
ax[0].legend()

seen_markers = set()
# plot data
# something's buggy; data['Modulation ...'] seems to be out of order
for x, y, yerr, color, marker, mod_freq, pulse_type in zip(data['ToTF'], dC_kFda0, dCkFda0_err, colors_data, markers_data, data['Modulation Freq (kHz)'], data['HFT_or_dimer']):
	label = f"{pulse_type} {mod_freq}kHz" if marker not in seen_markers else None
	if label:
		seen_markers.add(marker)
	ax[1].errorbar(x, y, yerr, color=color, marker=marker, mec=darken(color), ls='', label=label)
linestyle = '--'
marker ='o'
label = r'$\langle S \rangle_\mathrm{trap}$'
# plot Tilman theory
ax[1].plot(ToTFs, scale_sus_list, marker='', ls=linestyle, color='dimgrey')
# plot rescaled data
ax[1].errorbar(data['ToTF'], data['chi_rescaled'],
		   yerr = data['chi_rescaled_err'],
			marker='.', color='dimgrey', label='scaled')
ax[1].legend()
ax[1].set(
	xlabel=r'$T/T_F$',
	ylabel=r'$\partial\widetilde{C}_\mathrm{AC}/\partial(k_F a_0)^{-1}$'
)

# plots of chi/S
Stheory = interp1d(ToTFs, scale_sus_list, kind='linear', fill_value='extrapolate')
data['chi'] = dC_kFda0
data['chi_err'] = dCkFda0_err
data['Stheory'] = Stheory(data['ToTF'])
data['chi/S'] = data['chi']/data['Stheory']
data['chi/S_err'] = data['chi_err']/data['Stheory'] # TODO estimate uncertainty in theoretical S
data['chi_rescaled/S'] = data['chi_rescaled'] / data['Stheory']
data['chi_rescaled/S_err'] = data['chi_rescaled_err']/data['Stheory']
data['nu/T'] = data['freq']*1000/data['T']
data['EF_Hz'] = data['EF']/h

# plot data
# something's buggy; data['Modulation ...'] seems to be out of order
for x, y, yerr, color, marker, mod_freq, pulse_type in zip(data['nu/T'], data['chi/S'], data['chi/S_err'], colors_data, markers_data, data['Modulation Freq (kHz)'], data['HFT_or_dimer']):
	ax[2].errorbar(x, y, yerr, color=color, marker=marker, mec=darken(color), ls='')
for x, y, yerr, color, marker, mod_freq, pulse_type in zip(data['nu/T'], data['chi_rescaled/S'], data['chi_rescaled/S_err'], colors_data, markers_data, data['Modulation Freq (kHz)'], data['HFT_or_dimer']):
	ax[3].errorbar(x, y, yerr, color=color, marker=marker, mec=darken(color), ls='')


# plot theory
num=20
for b, TUG in enumerate(ytugs):
	nus = TUG.T*np.logspace(-2, 1, num)
	TUG.modulate_field(nus)
	TUG.rel_amp = 1/(np.sqrt(1+(2*pi*TUG.nus*TUG.tau)**2))
	ax[2].plot(TUG.nus/TUG.T, TUG.rel_amp,':', color=get_color(TUG.T),
		 label = f'ToTF={TUG.ToTF}, EF={(TUG.T/TUG.ToTF)/10e2:.0f} kHz'
		  )
	ax[3].plot(TUG.nus/TUG.T, TUG.rel_amp,':', color=get_color(TUG.T),
		 label = f'ToTF={TUG.ToTF}, EF={(TUG.T/TUG.ToTF)/10e2:.0f} kHz'
		  )
ax[2].set(xlim=[0, data['nu/T'].max()+1],
		  xlabel=r'$h\nu/k_BT$',
		  ylabel=r'$\chi/S$')

ax[3].set(xlim=[0, data['nu/T'].max()+1],
		  xlabel=r'$h\nu/k_BT$',
		  ylabel=r'rescaled $\chi/S$')
ax[2].legend(fontsize=6)

fig.tight_layout()


### a large number of alternative ways to plot the amplitude data
binning_selection = ['EF_Hz', 'ToTF', 'T', 'freq']
num_bins = 8
fig, ax = plt.subplots(len(binning_selection),2,
					   sharex='col', 
					   sharey='col', 
					   figsize=(10,8))
ax[len(binning_selection)-1, 0].set(xlim=[0, data['nu/T'].max()+1],
		  xlabel=r'$h\nu/k_BT$',
		  ylabel=r'$\chi/S$')

ax[len(binning_selection)-1, 1].set(xlim=[0, data['nu/T'].max()+1],
		  xlabel=r'$h\nu/k_BT$',
		  ylabel=r'rescaled $\chi/S$')

# plot data with different types of filtering
for i, bin in enumerate(binning_selection):
	ax[i,0].set_title(f"{binning_selection[i]}, [{data[bin].min():.1f}, {data[bin].max():.1f}]")
	ax[i,1].set_title(f"{binning_selection[i]}, [{data[bin].min():.1f}, {data[bin].max():.1f}]")

	data['bin_number'] = pd.cut(data[bin], bins=num_bins, labels=False)
	data['bin'] = pd.cut(data[bin], bins=num_bins)
	# For plot labels, you might want formatted strings
	data['midpoint'] = data['bin'].apply(lambda x: (x.left+x.right)/2)
	
	# normalize some colors
	xs = np.linspace(data[bin].min(), data[bin].max(), 10)
	my_norm = colors.Normalize(vmin = xs.min(), vmax = xs.max())
	my_cmap = cm.get_cmap('RdYlBu').reversed()
	for bin_num in data['bin_number'].unique():
		bindata = data[data['bin_number']==bin_num]
		color = my_cmap(my_norm(bindata['midpoint'].values[0]))
		ax[i,0].errorbar(bindata['nu/T'], bindata['chi/S'], bindata['chi/S_err'], color=color)
		ax[i,1].errorbar(bindata['nu/T'], bindata['chi_rescaled/S'], bindata['chi_rescaled/S_err'], color=color)
	# plot theory
	num=20
	for b, TUG in enumerate(ytugs): # this relies on the fact it was calculated previously
		ax[i,0].plot(TUG.nus/TUG.T, TUG.rel_amp,':', color=get_color(TUG.T),
			# label = f'ToTF={TUG.ToTF}, EF={(TUG.T/TUG.ToTF)/10e2:.0f} kHz'
			)
		ax[i,1].plot(TUG.nus/TUG.T, TUG.rel_amp,':', color=get_color(TUG.T),
			# label = f'ToTF={TUG.ToTF}, EF={(TUG.T/TUG.ToTF)/10e2:.0f} kHz'
			)
		
	# ax[i,0].legend()
	# ax[i,1].legend()


fig.tight_layout()
