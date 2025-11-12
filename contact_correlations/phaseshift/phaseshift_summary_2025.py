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

# Fast-Modulation-Contact-Correlation-Project\FieldWiggleCal
field_cal_folder = os.path.join(root_folder, r"FieldWiggleCal")

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
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

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
		}
	return marker_map.get(value, 'x') # default to 'x' if not found

def contact_time_delay(phi, period):
	""" Computes the time delay of the contact response given the oscillation
		period and the phase shift.
		phi /  (omega) = phi/(2*pi) * period
	"""
	return phi/(2*np.pi) * period

def Linear(x, m, b):
	return m*x + b
def Linear_no_intercept(x, m):
	return m*x

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
	label = f'ToTF={BVT.T:.2f}, EF={EF_kHz:.0f} kHz'
	ax.plot(x_theory, y_theory, ls=linestyle, marker=marker, color=get_color(BVT.T), label=label)

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
x_theory = BVT_T_list
y_theory = BVT_tau_list
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
# Loop over BVTs to plot theory curves
for j, BVT in enumerate(BVTs):
	x_theory = BVT.nus/BVT.T # dimless
	y_theory = BVT.time_delay_LR * 1e6 # us
	linestyle = ':'
	marker =''
	label = f'ToTF={BVT.ToTF:.2f}, EF={EF_kHz:.0f} kHz'
	ax.plot(x_theory, y_theory, color=get_color(BVT.T), marker=marker, ls=linestyle)
	if j ==0 or j==(len(BVTs)-1):
		ax.hlines(y=BVT.tau * 1e6, xmin=min(x_theory), xmax=max(x_theory), 
			ls='--',color=get_color(BVT.T), label=rf'$\tau = {BVT.tau*1e6:.0f} us, T/T_F = {BVT.ToTF:.2f}$')
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
# Loop over BVTs to plot theory curves
for j, BVT in enumerate(BVTs):
	x_theory = BVT.nus/BVT.T
	y_theory = BVT.time_delay_LR / BVT.tau
	linestyle = ':'
	marker =''
	label = f'ToTF={BVT.ToTF:.2f}, EF={EF_kHz:.0f} kHz'
	ax.plot(x_theory, y_theory, color=get_color(BVT.T), marker=marker, ls=linestyle)

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


for b, BVT in enumerate(BVTs):
	axs[2].plot(BVT.nus/BVT.T, BVT.rel_amp,':', color=get_color(BVT.T),
		 label = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
		  )
	axs[3].plot(BVT.nus/BVT.T, BVT.rel_amp,':', color=get_color(BVT.T),
		 label = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
		  )

# comparing AC to DC contact. This code is very ugly because of different data processing over time.
# get DC dimer amplitudes (currently only valid for ToTF=0.3)
DCdf = pd.read_csv(os.path.join(root_folder, 'exploratory_and_misc/DC_contact.csv'))
DCdf['HFT_or_dimer'] = 'dimer'
DCdf['alpha'] = DCdf['popts'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
DCdf = DCdf[DCdf['ToTF'] < 0.3]
DCdf.sort_values(by='Bfield')
alphas = np.array(DCdf['alpha'].apply(lambda x: x[0]).tolist())
field_to_contact_dimer = interp1d(DCdf['Bfield'], DCdf['contact'], kind='linear', fill_value='extrapolate')
field_to_alpha_dimer = interp1d(DCdf['Bfield'], alphas, kind='linear', fill_value='extrapolate')
Bs_d = np.linspace(DCdf['Bfield'].min(), DCdf['Bfield'].max(), 10)
Cs_d = field_to_contact_dimer(Bs_d)
As_d = field_to_alpha_dimer(Bs_d)

# DC HFT contact susceptibility from early November runs (ToTF <~ 0.3)
sus_df = pd.read_csv(os.path.join(root_folder, 'corrections//saturation_HFT.csv'))
sus_df['HFT_or_dimer']='HFT'
popt, pcov = curve_fit(Linear, sus_df['Bfield'], sus_df['C'], sigma=sus_df['e_C'])
popt_fudged, pcov_fudged = curve_fit(Linear, sus_df['Bfield'], sus_df['fudgedC'], sigma=sus_df['e_fudgedC'])
field_to_contact_HFT_unfudged = lambda B :Linear(B, *popt)
field_to_contact_HFT = lambda B: Linear(B, *popt_fudged)
Bs_HFT = np.linspace(sus_df['Bfield'].min(), sus_df['Bfield'].max(), 10)
Cs_HFT = field_to_contact_HFT(Bs_HFT)
Cs_HFT_unfudged = field_to_contact_HFT_unfudged(Bs_HFT)

fig2, ax = plt.subplots()
ax.plot(DCdf['Bfield'], DCdf['contact'], marker='o', ls='', color='orchid', label='DC dimer')
ax.plot(Bs_d, Cs_d, ls='--', marker='', color='orchid', label='dimer fit')
ax.errorbar(sus_df['Bfield'], sus_df['fudgedC'], sus_df['e_fudgedC'], marker='P', ls='', color='salmon', label='DC HFT')
ax.plot(Bs_HFT, Cs_HFT, ls= '--', marker='', color='salmon', label='HFT fudged')
ax.plot(Bs_HFT, Cs_HFT_unfudged, ls= '--', marker='', color='orange', label='HFT unfudged')
ax.set(xlabel = 'Bfield [G]', ylabel = r'$\widetilde{C}$', title = r'DC C, $T/T_F \lesssim 0.3$')
ax.legend()

# need field cal amplitudes for each run
# load wiggle field calibration
field_cal_df_path = os.path.join(field_cal_folder, "field_cal_summary.csv")
field_cal_df = pd.read_csv(field_cal_df_path)
field_cal_df['field_cal_run'] = field_cal_df['run']
field_cal_df = field_cal_df[['field_cal_run','B_amp']]
data=pd.merge(data, field_cal_df, on='field_cal_run')

# from field_cal_run, find field amplitude for each run, then evaluate DC contact and alpha amplitude
# for 202.14 +/- Bamp. Calculate difference. Call it DC amp.
# data['HFT_or_dimer'] = "HFT" # test
data['contact_DC_amp'] = np.where(
    data['HFT_or_dimer'] == 'dimer',
    np.abs(field_to_contact_dimer(202.14 - data['B_amp']) - field_to_contact_dimer(202.14 + data['B_amp'])) / 2,
    np.abs(field_to_contact_HFT(202.14 - data['B_amp']) - field_to_contact_HFT(202.14 + data['B_amp'])) / 2
)
data['alpha_DC_amp'] = np.where(
    data['HFT_or_dimer'] == 'dimer',
    np.abs(field_to_alpha_dimer(202.14 - data['B_amp']) - field_to_alpha_dimer(202.14 + data['B_amp'])) / 2,
	0
    # np.abs(field_to_contact_HFT(202.14 - data['B_amp']) - field_to_contact_HFT(202.14 + data['B_amp'])) / 2
)

# data['contact_DC_amp'] = np.abs(field_to_contact_dimer(202.14-data['B_amp']) - field_to_contact_dimer(202.14+data['B_amp'])) /2 # divide by 2 to compare to sine fit amp
# data['alpha_DC_amp'] = np.abs(field_to_alpha_dimer(202.14-data['B_amp']) - field_to_alpha_dimer(202.14+data['B_amp'])) /2

for i, plot_param in enumerate(plot_params):
	if i==0:
		DC_amps = data['alpha_DC_amp']
		DC_amps_err = data['alpha_DC_amp']/5 # TODO: PROPAGATE UNCERTAINTY CORRECTLY
		data['alpha_AC_amp'] = np.array(data[plot_param].apply(lambda x: x[0]).tolist())
		data['alpha_AC_amp_err'] = np.array(data['Error of ' + plot_param].apply(lambda x: x[0]).tolist())
		data['alpha_rel_amp'] = data['alpha_AC_amp']/data['alpha_DC_amp']
		data['alpha_rel_amp_err'] = np.sqrt((data['alpha_AC_amp_err']/data['alpha_AC_amp'])**2 +\
									   (DC_amps_err/DC_amps)**2) * data['alpha_rel_amp']
		rel_amps = data['alpha_rel_amp']
		e_rel_amps = data['alpha_rel_amp_err']

	else:
		DC_amps = data['contact_DC_amp']
		DC_amps_err = data['contact_DC_amp']/5 # TODO: PROPAGATE UNCERTAINTY CORRECTLY
		data['contact_AC_amp'] = np.array(data[plot_param].apply(lambda x: x[0]).tolist())
		data['contact_AC_amp_err'] = np.array(data['Error of ' + plot_param].apply(lambda x: x[0]).tolist())
		data['contact_rel_amp'] = data['contact_AC_amp']/data['contact_DC_amp']
		data['contact_rel_amp_err'] = np.sqrt((data['contact_AC_amp_err']/data['contact_AC_amp'])**2 +\
									   (DC_amps_err/DC_amps)**2) * data['contact_rel_amp']
		rel_amps = data['contact_rel_amp']
		e_rel_amps = data['contact_rel_amp_err']
	

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
y_data = np.sqrt((1/data['contact_rel_amp']) - 1)/(data['Modulation Freq (kHz)'])	
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
popt, pcov = curve_fit(Linear_no_intercept, x_data, y_data, sigma=y_err)
# plot linear fit
xs = np.linspace(x_data.min(), x_data.max(), 20)
ax.plot(xs, Linear_no_intercept(xs, popt), ls='-', marker='', color='gray', label=f'Linear Fixed: y={popt[0]:.2f}x')

# plot data points
for x, y, yerr, color, marker in zip(x_data, y_data, y_err, colors_data, markers_data):
	ax.errorbar(x, y, yerr, color=color, marker=marker, mec=darken(color), ls='')

ax.legend()
ax.set(
	xlabel=r'Amplitude of $\alpha_\mathrm{AC}$',
	ylabel=r'Amplitude of $\widetilde{C}_\mathrm{AC}$'
)

