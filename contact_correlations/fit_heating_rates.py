# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Fits heating rates from unitary_expansion_analysis.m output .dat files.
Fit results are compiled and stored as a Dataframe in a pkl file for later 
analysis or plotting, and a xlsx for viewing.

Datasets must be placed in the heating data folder, and a metadata row
needs to be filled in the heating_metadata.xlsx sheet. 

Multiscans and double multiscans are handled by looking for ms_param and 
ms_param2 names in the metadata file. Those are stored in the output to be used
for data sorting after fitting.

New files can be analyzed and added to the pickled DataFrame and xlsx by
just analyzing the new files.

Relies on data_class.py, library.py

WARNING: The saved DataFrame will not be updated correctly with new analysis if 
there are any empty DataFrame entries in the new "update" DataFrame. I'm not 
sure how to get around this. Currently, this analysis file does not produce
any DataFrame rows that have empty cells, so this shouldn't be an issue.
"""
from data_class import Data
from library import deBroglie, deBroglie_kHz, a97, chi_sq, mK, hbar, h, kB, pi, \
	plt_settings, colors, markers, tintshade, tint_shade_color
from scipy.optimize import curve_fit
from itertools import chain
from matplotlib import cm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import pickle

from fit_functions import Linear

# field wiggle calibration fit
from contact_correlations.field_wiggle_calibration import Bamp_from_Vpp 

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)

# select data folder
data_folder = 'data'
# data_folder = 'data//reanalysed'
data_path = os.path.join(proj_path, data_folder)

### files and paths
pkl_filename = 'heating_rate_fit_results.pkl'
pkl_file = os.path.join(proj_path, pkl_filename)
xlsx_results_filename = 'heating_data_results.xlsx'
xlsx_results_file = os.path.join(proj_path, xlsx_results_filename)
metadata_filename = 'heating_metadata.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)

### load metadata dataframe
metadata = pd.read_excel(metadata_file)

### script options
plotting = True
save = True
select_plotting = False

# only fit data up to max time in ms
max_time = 120

# class attributes to delete after analysis, before storage
del_attr = ['data', 'avg_data', 'fit_func', 'popt', 
			'pcov', 'err', 'poptT', 'pcovT', 'errT', 
			'poptEF', 'pcovEF', 'errEF', 'poptToTF', 
			'pcovToTF', 'errToTF','poptN', 'pcovN', 
			'errN', 'label', 'res_label', 'exclude']

##############################################
#### add list of files to analyze here #######
##############################################

# this selects all files that are not excluded e.g. have exclude = 0
files =  metadata.loc[metadata['exclude'] == 0]['filename'].values

# or select files manually, e.g. files = [filename1, filename2, ...]
# files = ["2024-05-10_S_UHfit"]

# names for select_plotting
# plot_names = ["2024-03-19_L_f=20_Vpp=0.70",
# 			  "2024-03-19_L_f=40_Vpp=0.50",
# 				"2024-03-21_B_f=35_Vpp=0.45",
# 				"2024-03-21_D_f=55_Vpp=0.30"]
plot_names = []

# initialize list of runs to fill
runs = []

# convert metadata into dict of dicts
metadata = metadata.set_index('filename')
metadata = metadata.to_dict(orient='index')

# analysis options
settings = {'param':'time',
			'temp_param':'meanEtot_kHz',
			'xlabel':'Time (ms)',
			'files':files}

### Functions
def calc_A(B0, T, e_T, Bamp, e_Bamp):
	''' Returns A and e_A '''
	A = deBroglie_kHz(T)/a97(B0-Bamp)
	e_a97 = np.abs(a97(B0-Bamp) - a97(B0-Bamp+e_Bamp)) # assumes symmetric
	e_dB = np.abs(deBroglie(T) - deBroglie(T+e_T)) # same
	e_A = A*np.sqrt((e_a97/a97(B0-Bamp))**2 + (e_dB/deBroglie(T))**2)
	return A, e_A

# shitty hard-coded linear fit of C/kFN vs. B field [G]
def calc_Ca(B0, Bamp):
	m = -2.07499885
	b = 420.18726779
	Ca = np.abs(m*Bamp)
	return Ca

# read description!!!
def fix_attribute(run, attr, attr_names):
	""" .... okay so like, if you're trying to generalize over multiscans, 
		you need to get the freq or amplitude from the .dat file, and use that 
		for analysis. That's what this does. It checks if those are not in the 
		metadata, and if they are't (i.e. they're nan) then it fills that attr
		with an entry from the .dat file, trying a list of names. """
	if getattr(run, attr) != getattr(run, attr): # nan check
		for name in attr_names:
			try:
				setattr(run, attr, run.data[name].values[0])
				break
			except KeyError:
				if name != attr_names[-1]:
					continue

### fitting and parameter calculation functions
### both take class as input, modify class, then return the same class
def fit_analysis(run):
	""" takes Data class run, fits and computes some stuff"""
	# fit energy to fit function
	run.fit_func, guess, param_names = Linear([]) # need to pass empty set... LOL
	
	run.popt, run.pcov = curve_fit(run.fit_func, run.avg_data[run.param], 
						run.avg_data[settings['temp_param']], 
						sigma = run.avg_data['em_'+settings['temp_param']])
	run.err = np.sqrt(np.diag(run.pcov))
	run.dof = len(run.avg_data.index) - len(run.err) # num data - num params in the fit
	
	try:
		run.chi_sq = chi_sq(run.avg_data[settings['temp_param']], 
					run.fit_func(run.avg_data[run.param], *run.popt), 
					run.avg_data['em_'+settings['temp_param']], run.dof)
	except ZeroDivisionError:
		print("dof = 0, so no chi^2 to calculate.")
		run.chi_sq = 1

	# fit ToTF to get offset
	run.poptToTF, run.pcovToTF = curve_fit(run.fit_func, run.avg_data[run.param], 
						run.avg_data['ToTF'], 
						sigma = run.avg_data['em_'+'ToTF'])
	run.errToTF = np.sqrt(np.diag(run.pcovToTF))
	run.ToTF = (run.poptToTF[1]+5*run.poptToTF[0])
	run.e_ToTF = run.errToTF[1]
	
	# fit EF to get offset
	run.poptEF, run.pcovEF = curve_fit(run.fit_func, run.avg_data[run.param], 
						run.avg_data['EF'], 
						sigma = run.avg_data['em_'+'EF'])
	run.errEF = np.sqrt(np.diag(run.pcovEF))
	run.EF = (run.poptEF[1]+5*run.poptEF[0])/h/1e3
	run.e_EF = run.errEF[1]/h/1e3
	run.kF = np.sqrt(2*mK*run.poptEF[1])/hbar
	run.e_kF = (run.e_EF/run.EF)/2 * run.kF
	
	# fit T to get offset
	run.poptT, run.pcovT = curve_fit(run.fit_func, run.avg_data[run.param], 
						run.avg_data['T'], 
						sigma = run.avg_data['em_'+'T'])
	run.errT = np.sqrt(np.diag(run.pcovT))
	run.T = (run.poptT[1]+5*run.poptT[0])*kB/h/1e3
	run.e_T = run.errT[1]*kB/h/1e3
	run.lamda = np.sqrt(hbar/(mK*run.T*1e3))
	
	# fit N to get offset
	run.poptN, run.pcovN = curve_fit(run.fit_func, run.avg_data[run.param], 
						run.avg_data['N'], 
						sigma = run.avg_data['em_'+'N'])
	run.errN = np.sqrt(np.diag(run.pcovN))
	run.N = (run.poptN[1]+5*run.poptN[0])
	run.e_N = run.errN[1]
	run.loss_rate = run.poptN[0]/run.N
	run.e_loss_rate = run.loss_rate*np.sqrt((run.e_N/run.N)**2 + (run.poptN[0]/run.errN[0])**2)

	# calculate other run params and errors
	run.Bamp, run.e_Bamp = Bamp_from_Vpp(run.Vpp, run.freq)
	run.A, run.e_A = calc_A(run.B, run.T, run.e_T, run.Bamp, run.e_Bamp)
	run.Ca = calc_Ca(run.B, run.Bamp)
	
	run.Edot = run.popt[0] # kHz^2
	run.e_Edot = run.err[0]
	run.Ei = run.popt[1]
	run.e_Ei = run.err[1]
	
	# Edot/EF**2/A**2
	run.rate = run.Edot/run.EF**2/run.A**2
	run.e_rate= run.rate*np.sqrt((run.e_Edot/run.Edot)**2+ \
					 (2*run.e_EF/run.EF)**2+(2*run.e_A/run.A))
		

def analysis_for_dof_equals_two(run):
	"""For use when dof = 2, i.e. you can't fit."""
	# dumbest shit I've ever written
	df1 = run.avg_data.loc[run.avg_data[run.param] == run.avg_data[run.param].min()]
	df2 = run.avg_data.loc[run.avg_data[run.param] == run.avg_data[run.param].max()]
	
	run.ToTF = (df2.ToTF.iloc[0]+df1.ToTF.iloc[0])/2
	run.e_ToTF = df1.em_ToTF.iloc[0]
	run.EF = (df2.EF.iloc[0]+df1.EF.iloc[0])/2/h/1e3
	run.e_EF = df1.em_EF.iloc[0]/h/1e3
	run.kF = np.sqrt(2*mK*h*1e3*run.EF)/hbar
	run.e_kF = (run.e_EF/run.EF)/2 * run.kF
	run.T = (df2['T'].iloc[0]+df1['T'].iloc[0])/2*kB/h/1e3 # .T is the transpose operation... LOL
	run.e_T = df1.em_T.iloc[0]*kB/h/1e3
	run.Ei = df1[settings['temp_param']].iloc[0]
	run.e_Ei = df1['em_'+settings['temp_param']].iloc[0]
	Ef = df2[settings['temp_param']].iloc[0]
	e_Ef = df2['em_'+settings['temp_param']].iloc[0]
	run.N = (df2.N.iloc[0]+df1.N.iloc[0])/2
	run.e_N = df1.em_N.iloc[0]
	Ni = df2.N.iloc[0] # this needs to be df1?
	e_Ni = df2.em_N.iloc[0]
	Nf = df2.N.iloc[0]
	e_Nf = df2.em_N.iloc[0]
	
	run.dof = 0
	run.chi_sq = 1
	
	run.Bamp, run.e_Bamp = Bamp_from_Vpp(run.Vpp, run.freq)
	run.lamda = np.sqrt(hbar/(mK*run.T*1e3))
	run.A, run.e_A = calc_A(run.B, run.T, run.e_T, run.Bamp, run.e_Bamp)
	run.Ca = calc_Ca(run.B, run.Bamp)
	# rise over run...
	deltax = (df2[run.param].iloc[0]-df1[run.param].iloc[0]) 
	deltay = Ef-run.Ei
	
	# I think this is done right... plz check.
	run.loss_rate = (Nf-Ni)/deltax/run.N
	run.e_loss_rate = run.loss_rate * np.sqrt((e_Nf/Nf)**2 + (run.e_N/run.N)**2)
	
	run.Edot = deltay/deltax # kHz^2
	run.e_Edot = run.Edot*np.sqrt((e_Ef/Ef)**2 + (run.e_Ei/run.Ei)**2)
	
	# Edot/EF**2/A**2
	run.rate = run.Edot/run.EF**2/run.A**2
	run.e_rate = run.rate*np.sqrt((run.e_Edot/run.Edot)**2+ \
							 (2*run.e_EF/run.EF)**2 + (2*run.e_A/run.A)**2)
	
def zeta_and_C(run):
	run.zeta = (run.kF*run.lamda)**2/(9*pi*run.freq**2)*run.Edot/run.A**2
	# zeta propto EF * Edot, so
	run.e_zeta = run.zeta*np.sqrt((run.e_EF/run.EF)**2+(run.e_Edot/run.Edot)**2)
	
	pi_factors = 2/9*36*pi*(2*pi)**(3/2)
	fandT_factors = (run.freq/run.T)**(3/2)/(run.lamda*run.kF)/(2*pi*run.freq**2)
	run.C = pi_factors*fandT_factors*run.Edot/run.A**2
	# C propto Edot/kF, so
	run.e_C = run.C*np.sqrt((run.e_kF/run.kF)**2+(run.e_Edot/run.Edot)**2)
	
	
############## FITTING ##############

# loop over files, split into runs if they are multiscans
for file in files:
	dat = Data(file+".dat", path=data_path, metadata=metadata[file])
	print("Fitting data from {}".format(dat.filename))
	
	# default to no multicans
	multiscan = False
	multiscan2 = False
	run_param_sets = [0,0] # dummy, only one loop per file
	
	# check for multiscan
	if dat.ms_param == dat.ms_param: # nan check
		multiscan = True
		print("Found multiscan param: {}".format(dat.ms_param))
		ms_vals = dat.data[dat.ms_param].unique()
		
		run_param_sets = [[ms_val] for ms_val in ms_vals] # number of loops
		
		# check for double multiscan
		if dat.ms_param2 == dat.ms_param2: # nan check
			multiscan2 = True
			print("Found 2nd multiscan param: {}".format(dat.ms_param2))
			run_param_sets = []
			for ms_val in ms_vals:
				# find multiscan 2 values for each multiscan 1 value
				ms2_vals = dat.data.loc[dat.data[dat.ms_param] == ms_val][dat.ms_param2].unique()
				# stuff pairs of multiscan values in list to loop over
				for ms2_val in ms2_vals:
					run_param_sets.append([ms_val, ms2_val]) # number of loops
	
	# loop over multiscan param pairs, if they exist
	for run_params in run_param_sets:
		# reload the .dat file, required when using drop, as we do below
		run = Data(file+".dat", path=data_path, metadata=metadata[file])
			
		if multiscan: # then print multiscan value, and only keep relevant df portion
			print("{} = {}".format(dat.ms_param, run_params[0]))
			run.data = run.data.drop(run.data[run.data[run.ms_param]!=run_params[0]].index)
		else: # required to have saved DataFrame update correctly, can't leave blank.
			run.ms_param = 'nope'
		if multiscan2:
			print("{} = {}".format(dat.ms_param2, run_params[1]))
			run.data = run.data.drop(run.data[run.data[run.ms_param2]!=run_params[1]].index)
		else:
			run.ms_param2 = 'nada'
			
		# take out long time data if we want to
		run.data = run.data.drop(run.data[run.data[run.param]>max_time].index)
		
		# make sure there isn't a repeated point at the end of the file
		# FILL
		
		# calculate correct energy from Will's code output in kHz
		run.data[settings['temp_param']] = run.data['ENoEF'] * run.data['EF']/1000/h
		
		# average data
		run.group_by_mean(run.param)
		
		# find attributes needed for analysis that were not in metadata
		# because they change during scan, and weren't written in metadata file
		fix_attribute(run, 'freq', ['freq', 'wigglefreq', 'wiggle freq'])
		fix_attribute(run, 'Vpp', ['Vpp', 'amp', 'amplitude'])
		
		if run.date < pd.Timestamp(2024,3,1):
			run.Vpp = run.Vpp/2 # we had a funny setting on func. gen.
		
		# if it's just a run with two values... then dof = 0 so we have to 
		# do some calculations of rates without fitting
		if len(run.avg_data.index) == 2:
			analysis_for_dof_equals_two(run)
			
		else: # we just do linear fits
			fit_analysis(run)
			
		# calculate the rest
		zeta_and_C(run)
			
		try:
			run.beta = run.avg_data['beta'].values[0]
		except KeyError: # if old .dat file, then there is no beta, so define beta
# 			print("This must be an old .dat file, adding old beta value")
			run.beta = 1.8339e-05
	
		# make a legend label for this run, assumes typical sig figs
		run.label = "$B_{{amp}}=${:.0f}({:.0f}) mG, $f=${:.0f} kHz, $T/T_F=${:.2f}({:.0f}), $E_F=${:.1f}({:.0f}) kHz".format(
					   run.Bamp*1e3, run.e_Bamp*1e3, run.freq, run.ToTF, 
					      run.e_ToTF*1e2, run.EF, run.e_EF*1e1)
		run.label = "$B_{{amp}}=${:.0f}({:.0f}) mG, $f=${:.0f} kHz".format(
					   run.Bamp*1e3, run.Bamp*1e3*0.07, run.freq)
		run.res_label = r"$\chi^2=${:.2f}".format(run.chi_sq)
		
		run.date = run.date.strftime('%Y-%m-%d')
		
		# label for run in results file, will be unique
		# format e.g. "B_2024-03-31_f=10_Vpp=0.5"
		run.name = "{}_{}_f={:.0f}_Vpp={:.2f}".format(run.date, run.run,
												run.freq, run.Vpp)
		
		# put run class into list, for later plotting and saving
		if select_plotting == True:
			if run.name in plot_names:
				runs.append(run)
				
		else:
			runs.append(run)
		

############## PLOTTING ##############		
if plotting == True:
	### matplotlib options
	plt.rcParams.update(plt_settings)
	plt.rcParams.update({"figure.figsize": [12,6],
						  "legend.fontsize": 9,
						  "font.size": 14,
						   "errorbar.capsize": 0})
	fig, axs = plt.subplots(2,3)
	num = 500
	font = {'size'   : 12}
	plt.rc('font', **font)
	legend_font_size = 9 # makes legend smaller, so plot is visible
			
	###
	### Heating rate data, fit
	###
	ax = axs[0,0]
	xlabel = settings['xlabel']
	ylabel = r"Energy per particle $E_\mathrm{tot}/N$ (kHz)"
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	### loop over runs, plotting fits
	max_plots = len(colors)
	if len(runs) > max_plots: # check if we are ignoring some fits when plotting
		print("Only showing {} datasets. Add more colors, ".format(max_plots))
		print("or reduce the number of runs to fit.")
	for run, color, marker in zip(runs, colors, markers): 
		light_color = tint_shade_color(color, amount=1+tintshade)
		dark_color = tint_shade_color(color, amount=1-tintshade)
		
		plt.rcParams.update({
					 "lines.markeredgecolor": dark_color,
					 "lines.markerfacecolor": light_color,
					 "lines.color": dark_color})
		
		xx = np.array(run.avg_data[settings['param']])
		yy = np.array(run.avg_data[settings['temp_param']])
		yerr = np.array(run.avg_data['em_'+settings['temp_param']])
		
		ax.errorbar(xx, yy, yerr=yerr, fmt=marker, label=run.label)
		
		xx = np.array(run.avg_data[settings['param']])
		yy = np.array(run.avg_data[settings['temp_param']])
		if len(xx) == 2: # if no fit, then just connect the two points
			ax.plot(xx, yy, color=color, linestyle='--')
		else:
			xlist = np.linspace(xx.min(), xx.max(), num)
			ax.plot(xlist, run.fit_func(xlist, *run.popt), color=color, linestyle='--')

	###
	### Residuals
	###
	ax = axs[1,0]
	xlabel = settings['xlabel']
	ylabel = "Residual Energy (kHz)"
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	# zero line

# 	ax.plot(xlist, np.zeros(num), color='k', linestyle='--')
	
	# residuals
	for run, color, marker in zip(runs, colors, markers):
		light_color = tint_shade_color(color, amount=1+tintshade)
		dark_color = tint_shade_color(color, amount=1-tintshade)
		plt.rcParams.update({
					 "lines.markeredgecolor": dark_color,
					 "lines.markerfacecolor": light_color,
					 "lines.color": dark_color})
		xx = np.array(run.avg_data[settings['param']])
		yy = np.array(run.avg_data[settings['temp_param']])
		yerr = np.array(run.avg_data['em_'+settings['temp_param']])
		if len(xx) == 2: # no fit, no residual
			continue 
		residuals =  yy - run.fit_func(xx, *run.popt)
		ax.errorbar(xx, residuals, yerr=yerr, fmt=marker, label=run.res_label)
		
	###
	### ToTF
	###
	ax = axs[0,1]
	xlabel = settings['xlabel']
	ylabel = r"$T/T_F$"
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	# loop over runs, plotting fits
	for run, color, marker in zip(runs, colors, markers):
		light_color = tint_shade_color(color, amount=1+tintshade)
		dark_color = tint_shade_color(color, amount=1-tintshade)
		plt.rcParams.update({
					 "lines.markeredgecolor": dark_color,
					 "lines.markerfacecolor": light_color,
					 "lines.color": dark_color})
		
		xx = np.array(run.avg_data[settings['param']])
		yy = np.array(run.avg_data['ToTF'])
		yerr = np.array(run.avg_data['em_'+'ToTF'])
		
		ax.errorbar(xx, yy, yerr=yerr, fmt=marker, label=run.label)
		
		if len(xx) == 2:
			ax.plot(xx, yy, color=color, linestyle='--')
		else:
			xlist = np.linspace(xx.min(), xx.max(), num)
			ax.plot(xlist, run.fit_func(xlist, *run.poptToTF), color=color, linestyle='--')
		
	###
	### EF
	###
	ax = axs[1,1]
	xlabel = settings['xlabel']
	ylabel = r"$E_F$ (kHz)"
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	# loop over runs, plotting fits
	for run, color, marker in zip(runs, colors, markers):
		light_color = tint_shade_color(color, amount=1+tintshade)
		dark_color = tint_shade_color(color, amount=1-tintshade)
		plt.rcParams.update({
					 "lines.markeredgecolor": dark_color,
					 "lines.markerfacecolor": light_color,
					 "lines.color": dark_color})
		xx = np.array(run.avg_data[settings['param']])
		yy = np.array(run.avg_data['EF']/1e3/h)
		yerr = np.array(run.avg_data['em_'+'EF']/h/1e3)
		
		ax.errorbar(xx, yy, yerr=yerr, fmt=marker, label=run.label)
		
		if len(xx) == 2:
			ax.plot(xx, yy, color=color, linestyle='--')
		else:
			xlist = np.linspace(xx.min(), xx.max(), num)
			ax.plot(xlist, run.fit_func(xlist, *run.poptEF)/h/1e3, color=color, linestyle='--')
	
	###
	### N
	###
	ax = axs[0,2]
	xlabel = settings['xlabel']
	ylabel = r"$N$ (atom no.)"
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	# loop over runs, plotting fits
	for run, color, marker in zip(runs, colors, markers):
		light_color = tint_shade_color(color, amount=1+tintshade)
		dark_color = tint_shade_color(color, amount=1-tintshade)
		
		plt.rcParams.update({
					 "lines.markeredgecolor": dark_color,
					 "lines.markerfacecolor": light_color,
					 "lines.color": dark_color})
		xx = np.array(run.avg_data[settings['param']])
		yy = np.array(run.avg_data['N'])
		yerr = np.array(run.avg_data['em_'+'N'])
		
		ax.errorbar(xx, yy, yerr=yerr,  fmt=marker, label=run.label)
		
		if len(xx) == 2:
			ax.plot(xx, yy, color=color, linestyle='--')
		else:
			xlist = np.linspace(xx.min(), xx.max(), num)
			ax.plot(xlist, run.fit_func(xlist, *run.poptN), linestyle='--')
		
	###
	### Legend in own subplot
	###
	lax = axs[1,2]
	h, l = axs[0,0].get_legend_handles_labels() # get legend from first plot
	lax.legend(h, l, borderaxespad=0)
	lax.axis("off")
	
	fig.tight_layout()
	plt.show()
	#fig.savefig('figures/heating_fits.pdf')

############ SAVE RESULTS ############	
if save == True:
	save_results = pd.DataFrame() # df for storing results, to be pickled
	for run in runs:
		results = vars(run) # makes dict out of class attributes, this gives me goosebumps
		
		# delete attributes from dict results that you don't want to save
		for attr in del_attr:
			results.pop(attr, None) # None in case of key error
		
		# stuff result (dict) in dataframe OH YEAH BABY
		# pandas doesn't let you use append on dataframes anymore as of 2.0, now it's ugly
# 		save_results = save_results.append(results, ignore_index=True)
		save_results = pd.concat([save_results, pd.DataFrame([results])], ignore_index=True)
# 		print(save_results.head())
	# set index as name
	save_results = save_results.set_index('name')
	try: # open pkl file if it's there
		with open(pkl_file, 'rb') as f:
			loaded_results = pickle.load(f) # load all results in file
# 			loaded_results['Ca'] = save_results['Ca']
# 			print(loaded_results.columns)
			# update index, then rows of loaded df
			loaded_results = loaded_results.reindex(loaded_results.index.union(save_results.index))
# 			print(loaded_results.columns)
# 			print(loaded_results.head())
			loaded_results.update(save_results) # this only updates values in shared columns, does not merge new columns
# 			print(loaded_results.columns)
		with open(pkl_file, 'wb') as f:
			pickle.dump(loaded_results, f)  # write new pkl file
			loaded_results.to_excel(xlsx_results_file)
	except FileNotFoundError: # if file not there then make it
		with open(pkl_file, 'wb') as f:
			pickle.dump(save_results, f)
			save_results.to_excel(xlsx_results_file)
