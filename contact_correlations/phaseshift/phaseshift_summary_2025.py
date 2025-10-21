# -*- coding: utf-8 -*-
"""
To summarize current phase shift measurements and compare to theory curves
Created on Thursday October 2 2025

@author: Chip Lab
"""

import os
#analysis_folder = 'E:\\\\Analysis Scripts\\analysis\\'
library_folder = '\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\analysis'
import sys
if library_folder not in sys.path:
	sys.path.append(library_folder)
from library import styles, colors, kB, h
from contact_correlations.UFG_analysis import BulkViscTrap
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

analysis_folder = '\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\Fast-Modulation-Contact-Correlation-Project'
summary_folder = '\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\Fast-Modulation-Contact-Correlation-Project\\contact_correlations\\phaseshift'
#2025 data comes from phase_shift.py

# summary of phase shift measurements
summary = pd.read_excel(summary_folder + '\\phaseshift_summary.xlsx')
summary = summary[summary['exclude']!=1]
metadata = pd.read_excel(analysis_folder + '\\contact_correlations\\phaseshift\\phaseshift_metadata.xlsx')
df = pd.merge(summary, metadata, how='inner', on='filename')
df['EF_kHz'] = (df['EF_i_kHz']+df['EF_f_kHz'])/2
df['e_EF_kHz'] = np.sqrt(df['EF_i_sem_kHz']**2 + df['EF_f_sem_kHz']**2)
df['ToTF'] =  (df['ToTF_i']+df['ToTF_f'])/2
df['e_ToTF'] = np.sqrt(df['ToTF_i_sem']**2 + df['ToTF_f_sem']**2)
df['kBT'] = df['EF_kHz']*df['ToTF'] # kHz
df['ScaledFreq'] = df['freq']/df['kBT']
df['time_lag'] = df['ps']/df['freq']/2/np.pi
df['e_time_lag'] = df['e_ps']/df['freq']/2/np.pi

data_2025 = pd.read_csv(summary_folder + '\\phase_shift_2025_summary.csv')

# pickle
pickle_file = analysis_folder + '\\contact_correlations\\time_delay_BVTs_working.pkl'
load = False

# parameters
ToTFs = np.array(df[df['ToTF'] < 0.35]['ToTF'])
EFs = np.array(df['EF_kHz']*1000)

ToTFs = [0.2, 0.25, 0.3,0.35, 0.4
		#  0.3,0.3,0.3,0.3, 0.3, 0.3
		]
EFs = [12*1000,12*1000,12*1000,12*1000,12*1000,
	#    12*1000,14*1000,16*1000,18*1000, 20*1000, 5*1000
	   ]
barnu = 377
num = 50

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
# analysis loop
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
	# BVT_tau_list.append((np.tan(BVT.phiLR)/nus/2/np.pi * 2*np.pi)[0])
	BVT_tau_list.append(BVT.tau *1e6) # [us]
	BVTs.append(BVT)

with open(pickle_file, 'wb') as f:
	pickle.dump(BVTs, f)
# plot phase shift
subfigs = 1
fig, axs = plt.subplots(1,2, figsize=(8*subfigs, 4))
axs = axs.flatten()
# phase shift
ax = axs


templist = [0.2954, 0.356, 
			# 0.2691
			]
EFslist  = [10.7,10.56, 
			# 12.62
			]

colors_light = ['powderblue', 'plum', 
				'lightpink'
				]
colors_dark = ['steelblue', 'orchid', 'palevioletred']

all_handles = []
all_labels = []

#making 2 subplots for phase shift plots 
for b in range(2):
	current_ax = ax[b]
	
	if b == 0:
		xlabel = r"$h\nu/k_BT$"	
		ylabel = rf'$\phi = \arctan(\omega \tau)$ [rad]'
	else:
		xlabel = f"T [nK]"
		ylabel = rf'$\tau = \tan(\phi)/\omega$ [us]'


	# Loop over BVTs to plot theory curves
	for j, BVT in enumerate(BVTs):
		if b == 0:
			x_theory = BVT.nus / BVT.T
			y_theory = BVT.phiLR

			EF_kHz = (BVT.T / BVT.ToTF) / 1e3
			
			linestyle = ':'
			marker =''
			label2 = f'ToTF={BVT.ToTF}, EF={EF_kHz:.0f} kHz'
		else:
			x_theory = BVT_T_list
			y_theory = BVT_tau_list

			linestyle = ':'
			marker='.'

			label2 = None
		

		handle, = current_ax.plot(x_theory, y_theory, linestyle = linestyle, 
				  marker = marker, color=colors[j], label=label2)

		if b == 0 and label2 is not None:
			all_handles.append(handle)
			all_labels.append(handle.get_label())
	# Plot data for current axis
	for i, (temp, ef_val) in enumerate(zip(templist, EFslist)):
		if b == 0:
			x_data = data_2025['Modulation Freq (kHz)'][i] / temp / ef_val
			y_data_C = data_2025['Phase Shift C-B (rad)'][i]
			y_data_A = data_2025['Phase Shift A-f0 (rad)'][i]

			y_data_C_err = data_2025['Phase Shift C-B err (rad)'][i]
			y_data_A_err = data_2025['Phase Shift A-f0 err (rad)'][i]
		else:
			x_data = temp*ef_val*1000
			y_data_C = data_2025['Phase Shift C-B (rad)'][i] 
			y_data_A = data_2025['Phase Shift A-f0 (rad)'][i] 
			y_data_C = np.tan(y_data_C) / data_2025['Modulation Freq (kHz)'][i]/2/np.pi *1e3
			y_data_A = np.tan(y_data_A) / data_2025['Modulation Freq (kHz)'][i]/2/np.pi *1e3

			y_data_C_err = np.sqrt((1/(np.cos(data_2025['Phase Shift C-B err (rad)'][i]))**2/2/np.pi/data_2025['Modulation Freq (kHz)'][i])/(np.tan(data_2025['Phase Shift C-B err (rad)'][i]*0)/2/np.pi/data_2025['Modulation Freq (kHz)'][i]))
			y_data_A_err = data_2025['Phase Shift A-f0 err (rad)'][i]*0

		err_C = current_ax.errorbar(
			x_data,
			y_data_C,
			yerr=y_data_C_err,
			color=colors_light[i],
			# label=f'C-B: {data_2025["Modulation Freq (kHz)"][i]}kHz',
			marker='.',
		)
		line_C = err_C[0]
		line_C.set_label(f'C-B: {data_2025["Modulation Freq (kHz)"][i]}kHz')

		err_A = current_ax.errorbar(
			x_data,
			y_data_A,
			yerr=y_data_A_err,
			color=colors_dark[i],
			# label=f'A-f0: {data_2025["Modulation Freq (kHz)"][i]}kHz, {data_2025["Unnamed: 0"][i]}',
			marker='.',
		)

		line_A = err_A[0]
		line_A.set_label(f'A-f0: {data_2025["Modulation Freq (kHz)"][i]}kHz, {data_2025["Unnamed: 0"][i]}')

# Only collect legend handles for b == 0
		if b == 0:
			all_handles.append(line_C)
			all_labels.append(line_C.get_label())

			all_handles.append(line_A)
			all_labels.append(line_A.get_label())
	# Set axis properties
	current_ax.set(
	# xlabel=r"Frequency, $\nu/T$", 
	xlabel=xlabel, 
	#    ylabel=r"Phase Shift, $\phi$ [rad]",
	 xscale='log',
ylabel = ylabel,
	#    ylim=[0,1]
	)
ax[0].legend(all_handles, all_labels, fontsize=8)
fig.tight_layout()

#time lag and time lag/tau plts 
subfigs = 1
fig, axs = plt.subplots(1,2, figsize=(8*subfigs, 3))
axs = axs.flatten()
ax = axs
for b in range(2):
	current_ax = ax[b]
	
	if b == 0:
		xlabel = r"$h\nu/k_BT$"	
		ylabel = rf'$\tau_\mathrm{{lag}}=\phi$/2/$\pi$/$\omega$ [ms]'
	else:
		xlabel = r"$h\nu/k_BT$"	
		ylabel = rf'$\tau_\mathrm{{lag}}/\tau$'

	# Loop over BVTs to plot theory curves
	for j, BVT in enumerate(BVTs):
		BVT.time_delay_LR = contact_time_delay(BVT.phiLR, 1/BVT.nus)
		BVT.time_delay_LR_noz = contact_time_delay(BVT.phiLR_noz, 1/BVT.nus)
		x_theory = BVT.nus/BVT.T
		if b == 0:
			y_theory = BVT.time_delay_LR * 1e3

		else:
			y_theory = BVT.time_delay_LR/BVT.tau

		EF_kHz = (BVT.T / BVT.ToTF) / 1e3
		label2 = f'ToTF={BVT.ToTF}, EF={EF_kHz:.0f} kHz'

		if j == 1:
			current_ax.plot(x_theory, y_theory, ':', color=colors[j], label=label2)
		if b == 0 and j == 1:
			current_ax.hlines(y=BVT.tau * 1000, xmin=min(x_theory), xmax=max(x_theory),ls='--', marker='',color='black', label=rf'$\tau = {BVT.tau*1000:.2f}, ToTF = {BVT.ToTF}')
			current_ax.legend()
	# Plot data for current axis
	for i, (temp, ef_val) in enumerate(zip(templist, EFslist)):
		x_data = data_2025['Modulation Freq (kHz)'][i]/temp/ef_val
		T = temp*ef_val*1000
		betaomegas = T*np.array([np.log(data_2025['Modulation Freq (kHz)'][i])])
		BVT_data = BulkViscTrap(temp, ef_val*1000, barnu, betaomegas)
		BVT_data.tau = BVT_data.tau * 1e3 #ms
		# print(BVT_data.tau)
		y_A = contact_time_delay(data_2025['Phase Shift A-f0 (rad)'][i], 1/(data_2025['Modulation Freq (kHz)'][i]))
		y_A_err = contact_time_delay(data_2025['Phase Shift A-f0 err (rad)'][i], 1/(data_2025['Modulation Freq (kHz)'][i]))
		y_C = contact_time_delay(data_2025['Phase Shift C-B (rad)'][i], 1/(data_2025['Modulation Freq (kHz)'][i]))
		y_C_err = contact_time_delay(data_2025['Phase Shift C-B err (rad)'][i], 1/(data_2025['Modulation Freq (kHz)'][i]))
		if b == 1:
			y_A = y_A/BVT_data.tau
			y_A_err = y_A_err/BVT_data.tau
			y_C = y_C/BVT_data.tau
			y_C_err = y_C_err/BVT_data.tau
		
		current_ax.errorbar(x_data, y_C, yerr=y_C_err, 
			color=colors_light[i], 
			label=f"C-B: {data_2025['Modulation Freq (kHz)'][i]}kHz, {data_2025['Unnamed: 0'][i]}"
			)
		current_ax.errorbar(x_data, y_A, yerr=y_A_err, 
			color=colors_dark[i], 
			label=f"A-f0: {data_2025['Modulation Freq (kHz)'][i]}kHz"
			)
	# Set axis properties
	current_ax.set(
	# xlabel=r"Frequency, $\nu/T$", 
	xlabel=xlabel, 
	xlim = [0.1, 10],
	 xscale='log',
ylabel = ylabel,
	#    ylim=[0,1]
	)
current_ax.legend(fontsize=8)
fig.tight_layout()


###time lag plot
# fig, axs = plt.subplots(1, subfigs, figsize=(5*subfigs, 4))
# for b, BVT in enumerate(BVTs):
# 	BVT.time_delay_LR = contact_time_delay(BVT.phiLR, 1/BVT.nus)
# 	BVT.time_delay_LR_noz = contact_time_delay(BVT.phiLR_noz, 1/BVT.nus)

# 	axs.plot(BVT.nus/BVT.T, BVT.time_delay_LR*1e3, ':', color=colors[b], 
# 		  label = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
# 		  )
# 	# axs.plot(BVT.nus/BVT.T, BVT.time_delay_LR_noz*1e3, ':', color=colors[b+1], 
# 	# 	  label = f'no z in tau'
# 	# 	  )
# 	for (i, temp, EFs) in zip(data_2025.index, templist, EFslist):
# 		x = data_2025['Modulation Freq (kHz)'][i]/temp/EFs
# 		y_A = contact_time_delay(data_2025['Phase Shift A-f0 (rad)'][i], 1/(data_2025['Modulation Freq (kHz)'][i]*1e3))*1e3
# 		y_A_err = contact_time_delay(data_2025['Phase Shift A-f0 err (rad)'][i], 1/(data_2025['Modulation Freq (kHz)'][i]*1e3))*1e3
# 		y_C = contact_time_delay(data_2025['Phase Shift C-B (rad)'][i], 1/(data_2025['Modulation Freq (kHz)'][i]*1e3))*1e3
# 		y_C_err = contact_time_delay(data_2025['Phase Shift C-B err (rad)'][i], 1/(data_2025['Modulation Freq (kHz)'][i]*1e3))*1e3

# 		axs.errorbar(x, y_C, yerr=y_C_err, 
# 			color=colors_light[i], 
# 			label=f"C-B: {data_2025['Modulation Freq (kHz)'][i]}kHz, {data_2025['Unnamed: 0'][i]}"
# 			)
# 		axs.errorbar(x, y_A, yerr=y_A_err, 
# 			color=colors_dark[i], 
# 			label=f"A-f0: {data_2025['Modulation Freq (kHz)'][i]}kHz"
# 			)

# axs.legend()
# axs.set(
# 	xlabel=r"$h\nu/k_BT$", 
# 	#    ylabel=r"Time Delay [ms]",
# 	ylabel = rf'$\phi$/2/$\pi$/$\omega$ [ms]', 

# )


###amplitude of sin fits plot 
fig, axs = plt.subplots(1, subfigs, figsize=(5*subfigs, 4))
for (i, temp, EFs) in zip(data_2025.index, templist, EFslist):
	x = data_2025['Modulation Freq (kHz)'][i]/temp/EFs
	axs.errorbar(x, data_2025['Amplitude of Sin Fit of A'][i], 
			  yerr = data_2025['Error of Amplitude of Sin Fit of A'][i],
			  marker='o', ls='', color=colors_light[i], 
			  label=f"A: {data_2025['Modulation Freq (kHz)'][i]}kHz, "
			  )
	axs.errorbar(x, data_2025['Amplitude of Sin Fit of C'][i], 
		  yerr = data_2025['Error of Amplitude of Sin Fit of C'][i],
		  marker='o', ls='', color=colors_dark[i], 
		  label=f"C: {data_2025['Modulation Freq (kHz)'][i]}kHz, {data_2025['Unnamed: 0'][i]}"
		  )

axs.set(
	xlabel=r"$h\nu/k_BT$", 
	   ylabel=r"Amplitude of Sin Fit",
)
axs.legend()

### Relative amplitude plot
fig, ax=plt.subplots(figsize=(5*subfigs, 4))
ax.set(
	xlabel=r"$h\nu/k_BT$", 
	   ylabel=r"Relative amplitude $\alpha_\mathrm{DC}/\alpha_\mathrm{AC}$",
)
ax.legend()

for b, BVT in enumerate(BVTs):
	ax.plot(BVT.nus/BVT.T, BVT.rel_amp,':', color=colors[b],
		 label = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'
		  )
for (i, temp, EFs) in zip(data_2025.index, templist, EFslist):
	x = data_2025['Modulation Freq (kHz)'][i]/temp/EFs
	if data_2025['Unnamed: 0'][i] != '2025-10-01_L': # only have data to compare to 10 kHz for now -- 2025-10-15
		ax.errorbar(x, rel_amp, 
		yerr = e_rel_amp_abs,
		marker='o', ls='', color=colors_light[i], 
		label=f"{data_2025['Unnamed: 0'][i]}, A: {data_2025['Modulation Freq (kHz)'][i]}kHz"
		)
		# continue
		# Completely eyeballed guess right now
		# DC_amp = 0.05
		# e_DC_amp = 0.01
		# AC_amp = data_2025['Amplitude of Sin Fit of A'][i]
		# e_AC_amp = data_2025['Error of Amplitude of Sin Fit of A'][i]
		# rel_amp = AC_amp/DC_amp
		# e_rel_amp = np.sqrt((e_AC_amp/AC_amp)**2 + (e_DC_amp/DC_amp)**2)
		# e_rel_amp_abs = e_rel_amp * rel_amp
		# ax.errorbar(x, rel_amp, 
		# 		yerr = e_rel_amp_abs,
		# 		marker='o', ls='', color=colors_light[i], 
		# 		label=f"A: {data_2025['Modulation Freq (kHz)'][i]}kHz, cf. static measurements")
	else :
		DC_amp = 0.0844
		e_DC_amp = 0.0136
		AC_amp = data_2025['Amplitude of Sin Fit of A'][i]
		e_AC_amp = data_2025['Error of Amplitude of Sin Fit of A'][i]
		rel_amp = AC_amp/DC_amp
		e_rel_amp = np.sqrt((e_AC_amp/AC_amp)**2 + (e_DC_amp/DC_amp)**2)
		e_rel_amp_abs = e_rel_amp * rel_amp
		ax.errorbar(x, rel_amp, 
				yerr = e_rel_amp_abs,
				marker='o', ls='', color=colors_light[i], 
				label=f"A: {data_2025['Modulation Freq (kHz)'][i]}kHz, cf. static measurements"
				)
ax.legend()
	
