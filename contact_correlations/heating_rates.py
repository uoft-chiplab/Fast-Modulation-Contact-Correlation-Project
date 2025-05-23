# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Calculations below assume errors are not correlated. I'm not so sure how to get
around that. Perhaps we need to do some like Monte Carlo or bootstrapping thing?
Or maybe we need to stop computing quantities that depend on so many related things.

Please check all uncertainty analysis. ctrl-F for uncertainty_flag
"""
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)

from data_class import Data
from library import deBroglie_kHz, a97, mK, hbar, h, kB, pi, \
	plt_settings, colors, markers, set_marker_color, save_to_Excel
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

# field wiggle calibration fit
from contact_correlations.field_wiggle_calibration import Bamp_from_Vpp 

# Unitary Fermi Gas analysis code
from contact_correlations.UFG_analysis import BulkViscTrap, BulkViscUniform

### paths
#data_path = os.path.join(proj_path, 'data')
data_path = os.path.join(proj_path, 'heating/reanalyzed')
figfolder_path = os.path.join(proj_path, 'figures')

### flags
Analysis = False
Save = False # analysis results and figures
Summary_Plot = True
Load = True

Debug = True
hijack_EF = False # if EF determined in metadata
Auto_Exclude = True # exclude runs if criteria

### metadata
metadata_filename = 'heating_metadata.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)	

### analysis settings
slope_variables = ['TkHz', 'EFkHz', 'ToTF', 'N', 'EkHz', 'EkHz2', 'SNkB', 
				   'smomEkHz']
time_name = 'time'
files =  metadata.loc[metadata['exclude'] == 0]['filename'].values
# files = ["2024-04-10_D_UHfit"]

error_band = 0.14
band_alpha = 0.2

# plotting frequency ranges in kHz
nu_min = 1
nu_max = 140

sheet_name = 'Sheet1'

### save file path
savefilename = 'heating_rates_results.xlsx'
savefile = os.path.join(proj_path, savefilename)

UFG_pkl_filename = 'UFG.pkl'
UFG_pkl_file = os.path.join(proj_path, UFG_pkl_filename)

### plot settings
plt.rcParams.update(plt_settings) # from library.py

def calc_A(B0, T, Bamp):
	""" Returns A """ 
	A = deBroglie_kHz(T)/a97(B0-Bamp)
	return A

def compute_slope_midpoint(df, x_name, y_name):
	""" Computes the rise over the run and the midpoint from a dataframe
		given x and y column names. Assumes there are only two points. 
		If there are more than two points, use fit_slope_midpoint. """
	if len(df[x_name].unique()) > 2:
		raise ValueError("Too many values to compute slope.")
		
	if len(df[x_name].unique()) < 2:
		print(df)
		print(df[y_name])
		print(df[x_name])
		raise ValueError("""Too few values to compute slope. Something strange 
				   with the data.""")
	
	deltay = df[y_name][1] - df[y_name][0]
	deltax = df[x_name][1] - df[x_name][0]
	
	midpoint = (df[y_name][1] + df[y_name][0])/2
	e_midpoint = np.sqrt(df['em_'+y_name][1]**2 + df['em_'+y_name][0]**2)
	
	slope = deltay/deltax
	e_slope = slope * e_midpoint/deltay
		
	return slope, e_slope, midpoint, e_midpoint

def linear(x, a, b):
	return a*x + b

def fit_slope_midpoint(df, x_name, y_name):
	""" Fits slope and computes midpoint given x and y column names and
		dataframe. """
	if len(df[x_name].unique()) <= 2:
		raise ValueError("Too few values to fit slope.")
		
	popt, pcov = curve_fit(linear, df[x_name], df[y_name], sigma=df['em_'+y_name])
	perr = np.sqrt(np.diag(pcov))
	
	midpoint = linear((min(df[x_name])+max(df[x_name]))/2, *popt)
	# this is wrong, assumes no covariance
	e_midpoint = np.sqrt((perr[0]/popt[0])**2*(min(df[x_name])+ \
						max(df[x_name]))/2 + (perr[1]/popt[1])**2)
		
	return popt[0], perr[0], midpoint, e_midpoint

def heating_zeta_C(run, E_name, postfix=''):
	""" Computes the dimensionless heating rate, zeta and C from the run
		dict using the energy determined by E_name. This is likely either the 
		second moment determined energy, or the hydrodynamics determined E. 
		postfix is appended to the dataframe column names to separate
		different results."""
	# compute heating rate
	run['heating'+postfix] = run[E_name+'_rate']/run['EFkHz']**2/run['A']**2
	run['e_heating'+postfix] = run['heating'+postfix] * np.sqrt( 
		# 2x here is because it's EF**2 
		(2*run['e_EFkHz']/run['EFkHz'])**2 + \
			(run['e_'+E_name+'_rate']/run[E_name+'_rate'])**2) # uncertainty_flag
	# zeta
	run['zeta'+postfix] = (run['kF']*run['lambda'])**2/(9*pi*run['freq']**2) * \
		run[E_name+'_rate']/run['A']**2
	run['e_zeta'+postfix] = run['zeta'+postfix] * np.sqrt(
		# kF**2 is like EF, so
		(run['e_EFkHz']/run['EFkHz'])**2 + \
		(run['e_TkHz']/run['TkHz'])**2 + \
			(run['e_'+E_name+'_rate']/run[E_name+'_rate'])**2) # uncertainty_flag
	# C
	pi_factors = 2/9*36*pi*(2*pi)**(3/2)
	fandT_factors = (run['freq']/run['TkHz'])**(3/2)/ \
		(run['lambda']*run['kF'])/(2*pi*run['freq']**2)
	run['C'+postfix] = pi_factors*fandT_factors*run[E_name+'_rate']/run['A']**2
	# C propto Edot/kF, so.., but there is also a factor of temperature. Hmm
	run['e_C'+postfix] = run['C'+postfix]*np.sqrt((run['e_kF']/run['kF'])**2 \
						+(run['e_'+E_name+'_rate']/run[E_name+'_rate'])**2 \
							+(run['e_TkHz']/run['TkHz'])**2) # uncertainty_flag
		
	return run

##################
#### Analysis ####
##################

# loop number iterable
i = 0

# initialize results df
results = pd.DataFrame([])

### loop analysis over selected datasets
for filename in files:
	j = 0 # multiscan iterable
	
	if Analysis == False:
		break # exit loop if no analysis required, this just elims one 
			  # indentation for an if statement
			  
	print("----------------")
	print("Analyzing " + filename)
	
	# select metadata row
	metadf = metadata.loc[metadata.filename == filename].reset_index()
	if metadf.empty:
		print("Metadata Dataframe is empty! The metadata likely needs updating.")
		break
	
	# create data structure
	filename = filename + ".dat"
	dat = Data(filename, path=data_path)
	df = dat.data
	
	# convert T to kHz
	df['TkHz'] = df['T']*kB/h/1e3
	
	# compute Bamp
	df['Bamp'] = df.apply(lambda x: Bamp_from_Vpp(x['amplitude'], x['freq'])[0], axis=1)

	# groupby time, averaging and computing sem
	df = pd.concat([df.groupby([time_name, 'freq']).mean().reset_index(),
			 df.groupby([time_name, 'freq']).sem().reset_index().add_prefix('em_')], axis=1)
	
	# loop over freq in multiscan
	for freq in df['freq'].unique():
		# subdf for the loop frequency
		subdf = df.loc[df['freq'] == freq].reset_index()
		
		# initialize arrays to hold computed slopes
		slopes = []
		e_slopes = []
		midpoints = []
		e_midpoints = []
		
		# compute slopes and midpoints for requested values
		for name in slope_variables:
			if len(subdf[time_name].unique()) <= 2: # rise over run
				slope, e_slope, midpoint, e_midpoint = compute_slope_midpoint(subdf, 
														 time_name, name)
			else: # fit to linear function
				slope, e_slope, midpoint, e_midpoint = fit_slope_midpoint(subdf, 
															   time_name, name)
			# slopes are in kHz cause time is in ms
			slopes.append(slope)
			e_slopes.append(e_slope)
			midpoints.append(midpoint)
			e_midpoints.append(e_midpoint)
			
		# initialize run dict from metadata
		run = metadf.to_dict(orient='records')[0]
		run.pop('index', None) # remove index
		
		# fill run dict with slopes and midpoints
		run.update(dict(zip([name+'_rate' for name in slope_variables], 
						  slopes)))
		run.update(dict(zip(['e_'+name+'_rate' for name in slope_variables], 
						  e_slopes)))
		run.update(dict(zip(slope_variables, midpoints)))
		run.update(dict(zip(['e_'+name for name in slope_variables], 
						  e_midpoints)))
		
		# set EF from metadata rather than hydrodynamic code
		if hijack_EF == True:
			run['EFkHz'] = metadf['EF']
		
		if Auto_Exclude == True:
			if run['EkHz_rate'] < 0 and run['smomEkHz_rate'] < 0:
				run['exclude'] = 1
		
		# add from df to run dict
		run['time'] = max(subdf.time) - min(subdf.time)
		run['Bamp'] = subdf.Bamp[0]
		run['freq'] = subdf.freq[0]
		
		# compute A and lambda_db
		run['A'] = calc_A(run['B'], run['TkHz'], run['Bamp'])
		run['lambda'] = deBroglie_kHz(run['TkHz'])
		run['kF'] = np.sqrt(2*mK*h*1e3*run['EFkHz'])/hbar
		run['e_kF'] = run['kF']*(run['e_EFkHz']/run['EFkHz'])/2 # uncertainty_flag
		
		# compute heating rate, zeta and C for various energies, 
		# adding them to the dict
		run = heating_zeta_C(run, 'EkHz')
		run = heating_zeta_C(run, 'EkHz2', postfix='2')
		run = heating_zeta_C(run, 'smomEkHz', postfix='_smom')
		
		if len(df['freq'].unique()) > 1: # modify run col to make unique identifier
			run['run'] = run['run'] + str(j)
			j += 1
		
		rundf = pd.DataFrame(run, index=[i]) # specify index, or ignore it when concat
		results = pd.concat([results, rundf])
		i += 1
		
	##################
	#### Plotting ####
	##################
	
	plt.rcParams.update({"figure.figsize": [10,7],
					     "font.size": 12,
						 "lines.markeredgewidth": 2,
						 "errorbar.capsize": 0})
	fig, axs = plt.subplots(2,3)
	ax_list = axs.flatten()
	xlabel = r"Time (ms)"
	
	### list plot y axes, and labels
	quantities = ["EkHz", "N", #"TkHz", 
			   "ToTF", "smomEkHz", "SNkB"]
	ylabels = [r"Energy $E$ [kHz]",
			 r"Number of Atoms $N$",
# 			 r"Temperature $T$ [kHz]",
			 r"$T/T_F$",
			 r"2nd Mom. $E$ [kHz]",
			 r"Entropy $S/N$ [kB]"]
	
	plt.suptitle(filename)
	
	# set labels
	for ax, ylabel in zip(ax_list, ylabels):
		ax.set(ylabel=ylabel, xlabel=xlabel)
	
	### loop over freq, set colors and markers
	for k, freq in enumerate(df['freq'].unique()):
		marker = markers[k]
		set_marker_color(colors[k])
		
		# subdf for the loop frequency
		subdf = df.loc[df['freq'] == freq].reset_index()
		
		# plot label
		letter = metadf.run[0] + str(k) # could pull results.run, but...
		label = r"{}: freq={:.0f} kHz, Bamp={:.0f} mG".format(letter, 
												freq, 1000* subdf.Bamp[0])
		
		# plot each quantity
		for ax, y in zip(ax_list, quantities):
			ax.errorbar(subdf.time, subdf[y], yerr=subdf["em_"+y], 
					fmt=marker+'-', label=label)
	
	# add legend
	lax = ax_list[-1]
	han, lab = ax_list[0].get_legend_handles_labels() # get legend from first plot
	lax.legend(han, lab, borderaxespad=0)
	lax.axis("off")
	
	fig.tight_layout()
	
	### save figure
	if Save == True:
		runfolder = filename 
		figpath = os.path.join(figfolder_path, runfolder)
		os.makedirs(figpath, exist_ok=True)
	
		sumrulefig_name = "Heating_Rates_Plots.png"
		sumrulefig_path = os.path.join(figpath, sumrulefig_name)
		fig.savefig(sumrulefig_path)
		
	plt.show()
		
####################
### Save Results ###
####################
if Analysis == True and Save == True:
	save_to_Excel(savefile, results, sheet_name=sheet_name, mode='replace')
	
##########################
#### Summary Plotting ####
##########################
if Summary_Plot:
	plt.rcParams.update({"figure.figsize": [15,8],
					     "font.size": 12,
						 "lines.markeredgewidth": 2,
						 "errorbar.capsize": 0})
	
	# load data
	df = pd.read_excel(savefile, sheet_name=sheet_name)
	
	# remove excludes runs
	df = df.loc[df.exclude == 0]
	
	plot_set_conditions = [#df['ToTF'] < 0.4, 
						(df['ToTF'] >= 0.4) & (df['ToTF'] < 0.7),
						df['ToTF'] >= 0.7]
	plot_set_conditions
	
	fig, axs = plt.subplots(2,3)
	ax_list = axs.flatten()
	xlabel = r"Drive Frequency $\hbar\omega/E_F$"
	num = nu_max-nu_min+1
	nus = np.linspace(nu_min, nu_max, num)
	
	fig.suptitle("Hydrodynamic model (top) vs. 2nd moment (bottom)")
	
	### list plot y axes, and labels
	ylabels = [r"Heating Rate $h \langle\dot{E}\rangle/(E_F\,A)^2$",
			 r"Contact Correlation $\langle\tilde{\zeta}(\omega)\rangle $",
			 r"Residual ratio",
			 r"Heating Rate $h \langle\dot{E}\rangle/(E_F\,A)^2$",
			 r"Contact Correlation $\langle\tilde{\zeta}(\omega)\rangle $",
			 r"Residual ratio"]
	
	# set labels
	for i, ax, ylabel in zip(range(len(ax_list)), ax_list, ylabels):
		ax.set(ylabel=ylabel, xlabel=xlabel)
		if i == 1 or i == 4:
			ax.set(xscale='log', yscale='log')
			
	### LOAD THEORY 
	if Load == True:
		with open(UFG_pkl_file, 'rb') as f:  
			BVTs = pickle.load(f)
	else:
		BVTs = []
		
	# loop on condisions, grouping data and theory
	for condition, color, marker in zip(plot_set_conditions, colors, markers):
		set_marker_color(color)
		
		# subdf for the loop frequency
		subdf = df.loc[condition].reset_index()
		
		# mean quantities in sets
		ToTF = subdf.ToTF.mean()
		EF = subdf.EFkHz.mean()
		barnu = (subdf.wx.mean()*subdf.wy.mean()*subdf.wz.mean())**(1/3)
		
		if Load == False: # calculate trap theory
			BVT = BulkViscTrap(ToTF, EF, barnu, nus)
		else: # find the right theory
			for BVT in BVTs:
				if BVT.ToTF == ToTF:
					foundBVT = True
					break
			if not foundBVT: # if can't find, then throw error
				raise FileNotFoundError("No BVT in pickle matches ToTF {ToTF} \n Recalculate BVTs.")
		
		# label
		label = r"$T/T_F$={:.2f}, $E_F$={:.0f}kHz".format(ToTF, EF)
		
		# calculate residual ratios
		heating_theory = np.array([BVT.calc_Edot(f) for f in subdf.freq])
		res = subdf['heating']/heating_theory
		e_res = subdf['e_heating']/heating_theory
		res_smom = subdf['heating_smom']/heating_theory
		e_res_smom = subdf['e_heating_smom']/heating_theory
		
		# plot quantities
		x = subdf.freq/subdf.EFkHz
		ys = [subdf['heating'], subdf['zeta'], res,
				subdf['heating_smom'], subdf['zeta_smom'], res_smom]
		yerrs = [subdf['e_heating'], subdf['e_zeta'], e_res,
			   subdf['e_heating'], subdf['e_zeta'], e_res_smom]
		
		
		xt = BVT.nus/BVT.EF
		yts = [BVT.Edot, BVT.zeta, np.ones(len(xt)),
			 BVT.Edot, BVT.zeta, np.ones(len(xt))]
		
		for ax, y, yerr, yt in zip(ax_list, ys, yerrs, yts):
			ax.errorbar(x, y, yerr=yerr, fmt=marker, label=label)
			ax.plot(xt, yt, '-', color=color)
			ax.fill_between(xt, yt*(1-error_band), yt*(1+error_band), 
				   alpha=band_alpha, color=color)
			
		BVTs.append(BVT)
		
	ax_list[1].legend()
	fig.tight_layout()
	plt.show()

	########### SAVE THEORY ###########
	if Load == False:
		with open(UFG_pkl_file, 'wb') as f:  
			pickle.dump(BVTs, f)
	


