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

ToTFs = [0.3,
		#  0.3,0.3,0.3,0.3, 0.3, 0.3
		]
EFs = [10*1000,
	#    12*1000,14*1000,16*1000,18*1000, 20*1000, 5*1000
	   ]
barnu = 377
num = 50

def contact_time_delay(phi, period):
	""" Computes the time delay of the contact response given the oscillation
		period and the phase shift.
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
	BVTs.append(BVT)

with open(pickle_file, 'wb') as f:
	pickle.dump(BVTs, f)
# plot phase shift and time delay of contact response
subfigs = 1
fig, axs = plt.subplots(1, subfigs, figsize=(5*subfigs, 4))
# axs = axs.flatten()
# phase shift
ax = axs
ax.set(
	# xlabel=r"Frequency, $\nu/T$", 
	xlabel=r"Frequency, $h\nu/k_BT$", 
	   ylabel=r"Phase Shift, $\phi$ [rad]", xscale='log',
	   ylim=[0,1])

templist = [0.3, 0.275]
EFslist  = [10,10]

for b, BVT in enumerate(BVTs):
		BVT.time_delay_LR = contact_time_delay(BVT.phiLR, 1/BVT.nus)	
		label = r"T={:.0f} kHz".format(BVT.T)
		label2 = f'ToTF={BVT.ToTF}, EF={(BVT.T/BVT.ToTF)/10e2:.0f} kHz'

		ax.plot(BVT.nus/BVT.T, BVT.phiLR, ':', color=colors[b], 
		  label = label2
		  )
		ax.plot(BVT.nus/BVT.T, BVT.phiLR_noz, ':', color=colors[b+1], label ='tau no z')
		plot_df = df[(df['EF_kHz'] == EFs[b]/1000)]
		# axs[1].errorbar(plot_df['ScaledFreq'], 
		# 		  plot_df['time_lag']*1e3,
		# 		  yerr=plot_df['e_time_lag']*1e3, color=colors[b])
	#####2024 data######
		# ax.errorbar(plot_df['ScaledFreq'], plot_df['ps'], yerr=plot_df['e_ps'], \
		# 		  color=colors[b])
	#####2025 data######
		for (i, temp, EFs) in zip(data_2025.index, templist, EFslist):
			ax.errorbar(data_2025['Modulation Freq (kHz)'][i]/temp/EFs,data_2025['Phase Shift C-B (rad)'][i], yerr= data_2025['Phase Shift C-B err (rad)'][i],
            color = 'powderblue', 
			label = f'C-B: {data_2025['Modulation Freq (kHz)'][i]}kHz'
			# label=f'C-B{data_2025['Unnamed: 0'][i]}'
			)
			ax.errorbar(data_2025['Modulation Freq (kHz)'][i]/temp/EFs,data_2025['Phase Shift A-f0 (rad)'][i], yerr= data_2025['Phase Shift A-f0 err (rad)'][i],
            color = 'steelblue', label=f'A-f0: {data_2025['Modulation Freq (kHz)'][i]}kHz')
# ax.errorbar(data_2025['Modulation Freq (kHz)']/0.275/10,data_2025['Phase Shift C-B (rad)'], yerr= data_2025['Phase Shift C-B err (rad)'],
#             color = 'powderblue', label='C-B')
# ax.errorbar(data_2025['Modulation Freq (kHz)']/0.275/10,data_2025['Phase Shift A-f0 (rad)'], yerr= data_2025['Phase Shift A-f0 err (rad)'],
#             color = 'steelblue', label='A-f0')
ax.set(ylim=[0, np.pi/2])
ax.legend()
