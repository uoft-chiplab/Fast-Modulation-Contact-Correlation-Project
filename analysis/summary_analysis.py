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
import seaborn as sns

def get_marker(type):
	if type == 'dimer':
		return 'd'
	elif type == 'HFT':
		return 'o'
def get_color(type):
	if type == 'dimer':
		return sns.color_palette()[0]
	elif type=='HFT':
		return sns.color_palette()[1]
	
def filter_dataframe(df, filters):
	"""
	filters: list of tuples [('col', 'op', val), ...]
	Applies all filters with AND logic
	"""
	result = df
	
	for col, op, val in filters:
		if op == '==':
			result = result[result[col] == val]
		elif op == '>':
			result = result[result[col] > val]
		elif op == '<':
			result = result[result[col] < val]
		elif op == '>=':
			result = result[result[col] >=val]
		elif op == '<=':
			result = result[result[col] <= val]
		elif op == 'in':
			result = result[result[col].isin(val)]
		# ... add other operators
	
	return result

def plot_correlations_comparison(input_df, obs_cols, thr_cols, res_cols, compare_params, TUGs, 
								  hue_col='HFT_or_dimer', filters=None, show_comparison=False):
	"""
	Plot residuals and correlations. If show_comparison=True and filters are provided,
	shows original and filtered data side-by-side for comparison.
	"""
	if show_comparison and filters:
		# Plot both original and filtered side-by-side
		plot_side_by_side_comparison(input_df, obs_cols, thr_cols, res_cols, compare_params, 
									  TUGs, hue_col, filters)
	else:
		# Use original single-dataset plotting
		plot_correlations(input_df, obs_cols, thr_cols, res_cols, compare_params, 
						 TUGs, hue_col, filters)


def plot_side_by_side_comparison(input_df, obs_cols, thr_cols, res_cols, compare_params, 
								   TUGs, hue_col, filters):
	"""
	Create side-by-side plots comparing original vs filtered datasets.
	"""
	sns.set_theme(style="whitegrid")
	fixed_plot_num = 2
	
	# Prepare both datasets
	df_original = input_df
	df_filtered = filter_dataframe(input_df, filters)
	filter_str = ' '.join(str(x) for x in filters)
	
	print(f'Comparing original (n={len(df_original)}) vs filtered (n={len(df_filtered)}) on {filter_str}')
	
	datasets = [
		('Original', df_original),
		(f'Filtered:', df_filtered)
	]
	
	# Plot residuals and correlations for each res_col
	for res_idx, res_col in enumerate(res_cols):
		# Calculate grid dimensions
		total_plots = len(compare_params) + fixed_plot_num
		n_cols = total_plots
		n_rows = 2
		
		fig, axes = plt.subplots(n_rows, n_cols, 
								figsize=(n_cols*4,n_rows*4))
		
		for dataset_idx, (dataset_name, df) in enumerate(datasets):
			# Select columns for this dataset (left half or right half)
			dataset_axes = axes[dataset_idx, :].flatten()
			
			# PLOT 1: Measurements vs predicted
			obs_x = obs_cols[res_idx][0]
			obs_y = obs_cols[res_idx][1]
			yerr = df[obs_y + '_err'] if obs_y + '_err' in df.columns else None
			
			if hue_col:
				for hue in df[hue_col].unique():
					subset_df = df[df[hue_col] == hue]
					subset_yerr = subset_df[obs_y + '_err'] if yerr is not None else None
					dataset_axes[0].errorbar(subset_df[obs_x], subset_df[obs_y], 
											yerr=subset_yerr, marker=get_marker(hue), 
											color=get_color(hue),
											ls='None', alpha=0.7)
			else:
				dataset_axes[0].errorbar(df[obs_x], df[obs_y], yerr=yerr,
										color='grey', marker='o', ls='None', alpha=0.7)
			
			# Add theory curves
			thr_x = thr_cols[res_idx][0]
			thr_y = thr_cols[res_idx][1]
			thr_scale = thr_cols[res_idx][2]
			for tug in TUGs:
				marker = 'o' if thr_x == 'T' else ''
				dataset_axes[0].plot(getattr(tug, thr_x), getattr(tug, thr_y) * thr_scale, 
									ls='-', marker=marker, alpha=0.3, color='grey')
			dataset_axes[0].set(xlabel=obs_x, ylabel=obs_y)
			dataset_axes[0].set_title(f'{dataset_name}: Obs vs Theory')
			
			# PLOT 2: Histogram of residuals
			plot_params = {
				"data": df,
				"x": res_col,
				"kde": True,
				"ax": dataset_axes[1],
				"element": "step"
			}
			if hue_col:
				plot_params["hue"] = hue_col
			sns.histplot(**plot_params)
			dataset_axes[1].axvline(0, color='red', linestyle='--')
			dataset_axes[1].set_title(f'{dataset_name}: Residual Distribution')
			
			# # PLOT 3: Residuals over time
			# if hue_col:
			#     for hue in df[hue_col].unique():
			#         subset_df = df[df[hue_col] == hue]
			#         dataset_axes[2].scatter(subset_df['date'], subset_df[res_col],
			#                               alpha=0.9, marker=get_marker(hue),
			# 							  color=get_color(hue),)
			# else:
			#     dataset_axes[2].scatter(df['date'], df[res_col], alpha=0.9)
			# dataset_axes[2].axhline(0, color='red', linestyle='--')
			# dataset_axes[2].set(xlabel='Date', ylabel=res_col)
			# dataset_axes[2].set_title(f'{dataset_name}: Residuals vs Time')
			
			# PLOTS 4+: Scatter plots of residuals vs parameters
			for param_idx, param in enumerate(compare_params):
				ax = dataset_axes[param_idx + fixed_plot_num]
				plot_params = {
					"data": df,
					"x": param,
					"y": res_col,
					"ax": ax,
					"alpha": 0.9,
					"legend": False,
				}
				if hue_col:
					plot_params["hue"] = hue_col
				sns.scatterplot(**plot_params)
				ax.axhline(0, color='red', linestyle='--')
				
				# Add regression lines
				if hue_col and len(df[hue_col].unique())>1:
					for category in df[hue_col].unique():
						subset = df[df[hue_col] == category]
						sns.regplot(data=subset, x=param, y=res_col, ax=ax,
								   scatter=False, marker='',
								   line_kws={'lw': 0.5, 'linestyle': ':', 'alpha': 0.7})
				else:
					sns.regplot(data=df, x=param, y=res_col, ax=ax,
							   scatter=False, marker='',
							   line_kws={'lw': 0.5, 'linestyle': ':', 'alpha': 0.7,
})
				
				ax.set_title(f'{param}')
			
			# Hide unused subplots for this dataset
			for idx in range(total_plots, len(dataset_axes)):
				dataset_axes[idx].set_visible(False)
		
		# Overall figure formatting
		fig.suptitle(f'Comparison: {res_col} (Original vs Filtered),\nfilter is {filter_str}', fontsize=16)
		handles, labels = axes[0, 1].get_legend_handles_labels()
		if handles:
			fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
		fig.tight_layout(rect=[0, 0, 1, 0.96])
		plt.savefig(f'comparison_res={res_col}_filter={filter_str}.png', dpi=300)

	if hue_col:
		for hue in input_df[hue_col].unique():
			# Correlation heatmaps side-by-side
			fig_corr, axes_corr = plt.subplots(1, 2, figsize=(15, 10))

			for idx, (dataset_name, df) in enumerate(datasets):
				subset_df = df[df[hue_col]==hue]
				corr_matrix = subset_df[res_cols+compare_params].corr()
				subset_corr = corr_matrix.loc[res_cols, :]
				print(subset_corr.isna().sum().sum())  # Check for NaN values
				sns.heatmap(subset_corr, annot=True, fmt=".1f", cmap='coolwarm', center=0,
					annot_kws={"size": 6}, ax=axes_corr[idx],square=True)
				axes_corr[idx].set_title(f'{dataset_name}')
				fig_corr.suptitle(f'Corr matrix comparison for {hue}')
			plt.savefig(f'corrmat_res={res_col}_hue={hue}.png', dpi=300)
	else:
		# Correlation heatmaps side-by-side
		fig_corr, axes_corr = plt.subplots(1, 2, figsize=(12, 4))
		for idx, (dataset_name, df) in enumerate(datasets):	
			corr_matrix = df[res_cols + compare_params].corr()
			sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
					annot_kws={"size": 6}, ax=axes_corr[idx])
			axes_corr[idx].set_title(f'{dataset_name}')
			fig_corr.suptitle('Correlation Matrix Comparison (all data)', fontsize=16)
		plt.savefig(f'corrmat_res={res_col}.png', dpi=300)
	fig_corr.tight_layout(rect=[0, 0, 1, 0.96])

def plot_correlations(input_df, obs_cols, thr_cols, res_cols, compare_params,  TUGs, hue_col='HFT_or_dimer', filters=None):
	"""
	Plot residuals and correlations between residuals and various parameters.
	"""
	# Set the visual style
	sns.set_theme(style="whitegrid")
	fixed_plot_num = 3

	if filters:
		df = filter_dataframe(input_df,filters) #return and reassign to avoid reference issues
		filter_str = ' '.join(str(x) for x in filters)
		print(f'filtering df on {filter_str}')
	else:
		df = input_df
		filter_str='None'

	# Plot residuals and check for correlations
	for i, res_col in enumerate(res_cols):
		# Calculate grid dimensions
		total_plots = len(compare_params) + fixed_plot_num
		n_cols = (len(compare_params) + fixed_plot_num) // 3  # Half the number of columns
		n_rows = (total_plots + n_cols - 1) // n_cols  # Ceiling division

		fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, 4 * n_rows))
		axes = axes.flatten()  # Flatten to 1D array for easier indexing
 
		# first plot the measurements vs predicted
		
		obs_x = obs_cols[i][0]
		obs_y = obs_cols[i][1]
		if hue_col:
			for j, hue in enumerate(df[hue_col].unique()):
				subset_df = df[df[hue_col] == hue]
				axes[0].errorbar(subset_df[obs_x], subset_df[obs_y], 
						yerr = [subset_df[obs_y + '_err'] if obs_y + '_err' in subset_df.columns else 0], 
						marker=get_marker(hue), color=get_color(hue),ls='None', alpha=0.7)
		else:
			axes[0].errorbar(df[obs_x], df[obs_y], 
						yerr = [df[obs_y + '_err'] if obs_y + '_err' in df.columns else 0], 
						color='grey', marker='o', ls='None', alpha=0.7)
			
		thr_x = thr_cols[i][0]
		thr_y = thr_cols[i][1]
		thr_scale = thr_cols[i][2]
		for tug in TUGs:
			marker = 'o' if thr_x == 'T' else ''
			axes[0].plot(getattr(tug, thr_x), getattr(tug, thr_y) * thr_scale, ls='-', marker=marker,  alpha=0.3, color='grey')
		axes[0].set(xlabel = obs_x, ylabel=obs_y)

		# second plot the histogram of residuals
		plot_params = {
			"data": df,
			"x": res_col,
			"kde": True,
			"ax": axes[1],
			"element": "step"
		}
		if hue_col:
			plot_params["hue"] = hue_col
		sns.histplot(**plot_params)
		axes[1].axvline(0, color='red', linestyle='--')
		axes[1].set_title('Residual Distribution')

		# third plot residuals over time
		if hue_col:
			for j, hue in enumerate(df[hue_col].unique()):
				subset_df = df[df[hue_col]==hue]
				axes[2].scatter(subset_df['date'], subset_df[res_col],
					alpha=0.9, marker=get_marker(hue),
					color=get_color(hue))
		else:
			axes[2].scatter(df['date'], df[res_col], alpha=0.9)
		axes[2].axhline(0, color='red', linestyle='--')
		axes[2].set(xlabel = 'Date', ylabel=res_col)

		# Scatter plots of residuals vs parameters
		for i, param in enumerate(compare_params):
			ax = axes[i+fixed_plot_num]
			plot_params = {
				"data": df,
				"x": param,
				"y": res_col,
				"ax": ax,
				"alpha":0.9,
				"legend":False,
			}
			if hue_col:
				plot_params["hue"] = hue_col
			sns.scatterplot(**plot_params)
			# Add a horizontal line at 0 for reference
			ax.axhline(0, color='red', linestyle='--')
			
			# Add a trend line (regression) to highlight hidden correlations
			if hue_col:
				for category in df[hue_col].unique():
					subset = df[df[hue_col] == category]
					sns.regplot(
					data=subset, x=param, y=res_col, 
					ax=ax, label=category,
					# scatter_kws={'alpha': 0.8, 's': 20},
					scatter=False,
					marker='',
					line_kws={'lw': 0.5, 'linestyle': ':', 
				  'alpha': 0.7
				  }
			)
			else: # bad repeated code
				sns.regplot(
				data=df, x=param, y=res_col, 
				ax=ax,
				scatter=False,
				marker='',
				line_kws={'lw': 0.5, 'linestyle': ':', 
			  'alpha': 0.7
			  }
			)

			ax.set_title(f'Residual vs {param}')
			   
		# Hide any unused subplots
		for idx in range(total_plots, len(axes)):
			axes[idx].set_visible(False)
		fig.suptitle(f'Residual analysis for {res_col}, filter={filter_str}', fontsize=16)
		handles, labels = axes[2].get_legend_handles_labels()
		fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
		fig.tight_layout(rect=[0, 0, 1, 0.96], 
				#    h_pad=3, w_pad=3
				   )
	# Compute correlation coefficients and heat map
	corr_matrix = df[res_cols + compare_params].corr()
	# Slice the matrix to get only res_cols vs compare_params
	# subset_corr = corr_matrix.loc[res_cols, :]
	plt.figure(figsize=(6, 4))
	sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
			 annot_kws={"size":6})
	plt.title('Correlation Matrix of Residuals and Parameters', fontsize=16)

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
	# fugacity
	TUG.mu = TUG.betamu * TUG.T
	return TUG

# Load summary data, metadata and merge
data = pd.read_csv(root_project + os.sep+'phase_shift_2025_summary.csv')
if 'Unnamed: 0' in data.columns:
	data.rename(columns={'Unnamed: 0':'run'}, inplace=True)
metadata = pd.read_csv(root_project + os.sep + 'metadata.csv')
data= data.merge(metadata, on='run')
print(f"Loaded summary data and metadata with {len(data)} entries.")

# Load field amplitude calibration and merge
field_cal_df_path = os.path.join(field_cal_folder, "field_cal_summary.csv")
field_cal_df = pd.read_csv(field_cal_df_path)
field_cal_df['field_cal_run'] = field_cal_df['run']
field_cal_df = field_cal_df[['field_cal_run','B_amp']]
data=pd.merge(data, field_cal_df, on='field_cal_run')
print(f"Merged field amplitude calibration, total entries now {len(data)}.")

# Process extra derived results
data['T'] = data['ToTF'] * (data['EF']/h) # T in Hz
data['EF_Hz'] = (data['EF']/h).round().astype(int) # EF in Hz and rounded
data['date'] = pd.to_datetime(data['run'].str[:10], format='%Y-%m-%d')
data.rename(columns={'Modulation Freq (kHz)':'mod_freq_kHz',
					 'Phase Shift C-B (rad)':'phaseshift',
					 'Phase Shift C-B err (rad)':'phaseshift_err',
					 }, inplace=True)
# data['phaseshift'] = data['phaseshift'].apply(lambda x: x+2*pi if x < 0 else x)
data['tau_from_ps'] = data['phaseshift'] / (2 * pi * data['mod_freq_kHz'] * 1e3) * 1e6  # [us]
data['tau_from_ps_err'] = 1/(np.cos(data['phaseshift']))**2 * data['phaseshift_err']/ (data['mod_freq_kHz']*1e3) / 2/np.pi * 1e6 # sec^2(x) * \delta x
data['time_delay'] = contact_time_delay(data['phaseshift'], 1/(data['mod_freq_kHz']*1e3)) * 1e6  # [us]
data['time_delay_err'] = data['phaseshift_err'] / data['phaseshift'] * data['time_delay'] # TODO: check
data['betaomega'] = data['mod_freq_kHz'] * 1e3 / data['T']  # dimless

# creates lists of only popt[0] and perr[0] (amplitude)
C_param = 'Sin Fit of C'
# ensures conversion of strings into actual lists of values
data[C_param] = data[C_param].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
data['Error of ' +C_param] = data['Error of ' + C_param].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
data['C_amp'] = [p[0] for i, p in enumerate(data[C_param])]
data['C_amp_err'] = [p[0] for i, p in enumerate(data['Error of ' + C_param])]
data['Ceq'] = [p[-1] for i, p in enumerate(data[C_param])]
data['Ceq_err'] = [p[-1] for i, p in enumerate(data['Error of ' + C_param])]

B0 = 202.14 # this assumes the average B0 for all measurements
data['kF'] = np.sqrt(2*mK*data['EF'])/hbar  # [1/m]
data['dkFa0_inv'] = 1/(data['kF']*a97(B0 - data['B_amp'])) - 1/(data['kF']*a97(B0 + data['B_amp'])) # max B - min B 
data['dC_kFda0'] = 2*data['C_amp']/data['dkFa0_inv'] # dC/d(kF a0)^-1 
data['dC_kFda0_err'] = 2*data['dC_kFda0']*np.sqrt((data['C_amp_err']/data['C_amp_err'])**2 + (data['eEF']/data['EF'])**2)

# List of TrappedUnitaryGas objects pickle file
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
# dataframe residuals
data['res_ps'] = data['phaseshift'] - pred_ps
data['res_tau'] = data['tau_from_ps'] - pred_tau
data['res_delay'] = data['time_delay'] - pred_delay
# plot residuals and correlations
compare_params = ['ToTF', 'EF_Hz','T','mod_freq_kHz', 'B_amp',  'betaomega']
params_from_tug = ['barnu']
if params_from_tug:
	for param in params_from_tug:
		data[param] = [getattr(tug, param) for tug in TUGs]
		compare_params.append(param)
# res_cols = ['res_ps', 'res_tau', 'res_delay']
res_cols = ['res_ps'] 
obs_cols = [('betaomega', 'phaseshift'), ('T', 'tau_from_ps'), ('betaomega','time_delay')] # tuples of (x,y) observables
thr_cols = [('betaomegas', 'phiLR', 1), ('T', 'tau', 1e6), ('betaomegas','time_delay_LR', 1e6)] # (x, y, scale)
hue_col = 'HFT_or_dimer'
# hue_col=None
# filters = [('ToTF', '<=', 0.35)]
filters = [('betaomega', '>=', 2),
		   ('betaomega', '<=', 3)]
# filters = [('ToTF', '<=', 0.35),
# 		   ('betaomega', '>=', 2),
# 		   ('betaomega', '<=', 3)]
# filters=None
show_comparison=True
plot_correlations_comparison(data, 
				  obs_cols, 
				  thr_cols, 
				  res_cols, 
				  compare_params, 
				  TUGs, 
				  hue_col=hue_col, 
				  filters=filters,
				  show_comparison=show_comparison)


### COMPUTE RESIDUALS AND CORRELATIONS FOR AMPLITUDES AND CONTACTS
# predicted values from TUGs
pred_Ctrap =[tug.Ctrap for tug in TUGs] # dimensionless, <Ceq>
pred_S = [tug.dCdkFa_inv for tug in TUGs] # scale sus
pred_chioverS = [tug.evaluate(row.betaomega, 'rel_amp') for tug, row in zip(TUGs, data.itertuples())] # chi / S
# Using theory to compute S, data for chi
data['theoryS'] = pred_S
data['chioverS'] = data['dC_kFda0'] / data['theoryS']

# dataframe residuals
data['res_Ceq'] = data['Ceq'] - pred_Ctrap
data['res_dC_kFda0'] = data['dC_kFda0'] - pred_S
data['res_chioverS'] = data['chioverS'] - pred_chioverS

# plot residuals and correlations
# compare_params = ['ToTF', 'EF_Hz', 'N','T','mod_freq_kHz', 'B_amp','betaomega']
res_cols = ['res_chioverS'] 
obs_cols = [('betaomega', 'chioverS'),] # tuples of (x,y) observables
thr_cols = [('betaomegas', 'rel_amp', 1),] # (x, y, scale)
plot_correlations_comparison(data, 
				  obs_cols, 
				  thr_cols, 
				  res_cols, 
				  compare_params, 
				  TUGs, 
				  hue_col=hue_col, 
				  filters=filters,
				  show_comparison=show_comparison)





### Extra TUG plots
# fig, axs=plt.subplots(1, 4, figsize=(10,6))
# axs=axs.flatten()
# for tug in TUGs:
# 	axs[0].plot(tug.betaomegas, tug.rel_amp, marker='', ls='-', label=f'barnu={tug.barnu:.1f}')
# 	axs[1].plot(tug.betamu, tug.tau*1e6)
# 	axs[2].plot( 1/tug.T,tug.tau*1e6)
# 	axs[3].plot(tug.T,tug.betamu)

# axs[0].set(xlabel=r'$\nu/T$',
# 	   ylabel=r'$1/\sqrt{1+(2\pi\nu \tau)^2}$')
# axs[1].set(ylabel=r'$\tau$ [us]',
# 		xlabel='fugacity')
# axs[2].set(ylabel=r'$\tau$ [us]',
# 		xlabel='1/T')
# axs[3].set(ylabel=r'fugacity',
# 		   xlabel='T')

# fig.tight_layout()

