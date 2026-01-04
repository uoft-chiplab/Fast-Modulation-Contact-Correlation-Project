"""
Another summary analysis script.
Takes every result and compares residuals and correlations between model and various parameters.
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
import seaborn as sns


def contact_time_delay(phi, period):
	""" Computes the time delay of the contact response given the oscillation
		period and the phase shift.
		phi /  (omega) = phi/(2*pi) * period
	"""
	return phi/(2*pi) * period

def Linear(x, m, b):
	return m*x + b

def create_TUG(ToTF, EF, barnu, num=50):
	"""
	Create TrappedUnitaryGas objects for given ToTFs and EFs.
	Returns a TUG object with additionally computed properties.
	"""
	Ts = ToTF * EF
	nus = Ts * np.logspace(-2, 1, num)
	TUG = TrappedUnitaryGas(ToTF, EF, barnu)
	TUG.modulate_field(nus)
	TUG.tau_noz =  ((1.739) * TUG.T) * (2*pi)
	TUG.phiLR_noz = np.arctan(2*pi*TUG.nus * 1/ TUG.tau_noz)
	# compute time delays
	TUG.time_delay = contact_time_delay(TUG.phaseshiftsQcrit, 1/TUG.nus)
	TUG.time_delay_LR = contact_time_delay(TUG.phiLR, 1/TUG.nus)
	TUG.rel_amp = 1/(np.sqrt(1+(2*pi*TUG.nus*TUG.tau)**2))
	return TUG

#2025 data comes from phase_shift.py
data = pd.read_csv(root_project + os.sep+'phase_shift_2025_summary.csv')
if 'Unnamed: 0' in data.columns:
	data.rename(columns={'Unnamed: 0':'run'}, inplace=True)
metadata = pd.read_csv(root_project + os.sep + 'metadata.csv')
# later plotting gets easier if I merge the summary df and the metadat df
data= data.merge(metadata, on='run')
data['T'] = data['ToTF'] * (data['EF']/h) # T in Hz
data['EF_Hz'] = (data['EF']/h).round().astype(int) # EF in Hz and rounded
data['barnu'] = 300 # TODO FIX THIS ESTIMATE; HAVE TO LOOK AT OLD DATA AND REFERENCE
data.rename(columns={'Modulation Freq (kHz)':'mod_freq_kHz',
					 'Phase Shift C-B (rad)':'phaseshift',
					 'Phase Shift C-B err (rad)':'phaseshift_err'}, inplace=True)
data['tau_from_ps'] = data['phaseshift'] / (2 * pi * data['mod_freq_kHz'] * 1e3) * 1e6  # [us]
data['tau_from_ps_err'] = 1/(np.cos(data['phaseshift']))**2 * data['phaseshift_err']/ (data['mod_freq_kHz']*1e3) / 2/np.pi * 1e6 # sec^2(x) * \delta x
data['time_delay'] = contact_time_delay(data['phaseshift'], 1/(data['mod_freq_kHz']*1e3)) * 1e6  # [us]
data['time_delay_err'] = data['phaseshift_err'] / data['phaseshift'] * data['time_delay'] # TODO: check
data['betaomega'] = data['mod_freq_kHz'] * 1e3 / data['T']  # dimless
data = data[data['HFT_or_dimer']=='HFT'] # Hmm I get big differences depending on measurement
# pickle
pickle_file = cc_folder + os.sep + 'time_delay_TUGs_working.pkl'

### Load or create TUG objects
if os.path.exists(pickle_file):
	with open(pickle_file, 'rb') as f:
		TUGs = pickle.load(f)
	print(f"Loaded {len(TUGs)} TUGs from pickle file.")
else: 
	TUGs = []
	print("No pickle file found, creating TUGs from scratch. This may take a while...")

### Map of existing TUGs for comparison
existing_map = {
	(round(tug.ToTF,3), int(round(tug.EF)), tug.barnu): tug for tug in TUGs
}
TUGs = []
update_tugs = False

### Compare TUGs in pickle to those needed from data; create any that are missing
for row in data.itertuples():
	current_params = (round(row.ToTF,3), row.EF_Hz, row.barnu)
	if current_params in existing_map:
		# print("Using existing TUG for ToTF={:.3f}, EF={}, barnu={}".format(row.ToTF, row.EF_Hz, row.barnu))
		TUGs.append(existing_map[current_params])
	else:
		print(f"Creating TUG for ToTF={row.ToTF:.3f}, EF={row.EF_Hz}, barnu={row.barnu}")
		new_TUG = create_TUG(row.ToTF, row.EF_Hz, row.barnu, num=50)
		TUGs.append(new_TUG)
		update_tugs=True

### Save updated TUGs if new ones were created
if update_tugs:
	with open(pickle_file, 'wb') as f:
		pickle.dump(TUGs, f)
	print("Updated pickle file with new/modified objects.")
else:
	print("No new TUGs created; pickle file remains unchanged.")

### COMPUTE RESIDUALS AND CORRELATIONS FOR PHASE SHIFTS AND TIME DELAYS

# predicted values from TUGs
pred_ps = [tug.evaluate(row.betaomega, 'phiLR') for tug, row in zip(TUGs, data.itertuples())] # [rad]
pred_tau = [tug.tau * 1e6 for tug in TUGs] # [us]
pred_delay = [tug.evaluate(row.betaomega, 'time_delay_LR') * 1e6 for tug, row in zip(TUGs, data.itertuples())] # [us]
# pred_delay_over_tau = [np.array(pred_delay)/np.array(pred_tau)]
# dataframe residuals
data['res_ps'] = data['phaseshift'] - pred_ps
data['res_tau'] = data['tau_from_ps'] - pred_tau
data['res_delay'] = data['time_delay'] - pred_delay
# data['res_delay_over_tau'] = data['time_delay']/data['tau_from_ps'] - pred_delay_over_tau

# List of parameters to check for correlations
params_to_check = ['ToTF', 'EF_Hz', 'mod_freq_kHz', 'T', 'betaomega']
res_cols = ['res_ps', 'res_tau', 'res_delay'] 

# Set the visual style
sns.set_theme(style="whitegrid")

# Plot residuals and check for correlations
for res_col in res_cols:
	fig, axes = plt.subplots(1, len(params_to_check) + 1, figsize=(18, 5))
	fig.suptitle(f'Residual analysis for {res_col}', fontsize=16)
	sns.histplot(data[res_col], kde=True, ax=axes[0], color='gray')
	axes[0].axvline(0, color='red', linestyle='--')
	axes[0].set_title('Residual Distribution')
	axes[0].set_xlabel('Residual (Observed - Predicted)')

	# Scatter plots of residuals vs parameters
	for i, param in enumerate(params_to_check):
		ax = axes[i+1]
		sns.scatterplot(data=data, x=param, y=res_col, ax=ax, alpha=0.9)
		
		# Add a horizontal line at 0 for reference
		ax.axhline(0, color='red', linestyle='--')
		
		# Add a trend line (regression) to highlight hidden correlations
		sns.regplot(data=data, x=param, y=res_col, ax=ax, 
					scatter=False, color='blue', )
		
		ax.set_title(f'Residual vs {param}')

# Compute correlation coefficients and heat map
corr_matrix = data[res_cols + params_to_check].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Residuals and Parameters', fontsize=16)