# -*- coding: utf-8 -*-
"""
@author: Chip Lab
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def exponential(x, a, b, c):
	return a*(1-np.exp(x/b))+c

def linear(x, a, b):
	return a*x+b

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)

###
### Load measurement analysis results
###
data_path = os.path.join(proj_path, "FieldWiggleCal")
csv_filename = "field_cal_summary.csv"
df = pd.read_csv(os.path.join(data_path, csv_filename))
num = 500

###
### Fit
###

# Fix wiggle_amp at 0.9 Vpp, fit vs freq
sub_df = df[df['wiggle_amp']==0.9]

xx = sub_df['wiggle_freq']
yy = sub_df['B_amp']
yerr = sub_df['e_B_amp']
guess = [0.07,-10, 0]

popt_freq, pcov_freq = curve_fit(exponential, xx, yy, sigma=yerr, p0=guess)
err_freq = np.sqrt(np.diag(pcov_freq))

# Fix 10 kHz wiggle freq, fit vs amp
sub_df = df[df['wiggle_freq']==10.0]
xx = sub_df['wiggle_amp']
yy = sub_df['B_amp']
yerr = sub_df['e_B_amp']

popt_amp, pcov_amp = curve_fit(linear, xx, yy, sigma=yerr)
err_amp = np.sqrt(np.diag(pcov_amp))


###
### Define calibration functions for importing
###

def Bamp_from_Vpp(Vpp, freq):
	''' Returns Bamp and e_Bamp in an array '''
	Bamp = linear(Vpp, *popt_amp)/linear(0.9, *popt_amp) * exponential(freq, *popt_freq)
	# the below is wrong
	e_Bamp = Bamp*0.00 # 0.07 average error ratio for the measurements, 
						# i.e. average error bar size
	return Bamp, e_Bamp


###
### Plotting
###

if __name__ == '__main__':
	# plot B amp vs. wiggle freq
	fig, ax = plt.subplots(1,1)
	sub_df = df[df['wiggle_amp']==0.9]
	xx = sub_df['wiggle_freq']
	yy = sub_df['B_amp']
	yerr = sub_df['e_B_amp']
	
	ax.errorbar(sub_df['wiggle_freq'], sub_df['B_amp'], sub_df['e_B_amp'],fmt='bo')
	
	xlist = np.linspace(np.min(xx), np.max(xx), num)
	ax.plot(xlist, exponential(xlist, *popt_freq), 'b--')
	ax.plot(xlist, exponential(xlist, *guess), '--', color='orange')
	
	ax.set_xlabel('drive freq (kHz)')
	ax.set_ylabel('B field fit amplitude (G)')
	ax.set_title('Fixed 0.9 Vpp drive amplitude')
	
	# Fix wiggle_amp at 0.9 Vpp, plot B phase vs. wiggle freq
	fig, ax = plt.subplots(1,1)
	sub_df = df[df['wiggle_amp']==0.9]
	ax.errorbar(sub_df['wiggle_freq'], sub_df['B_phase'], sub_df['e_B_phase'],fmt='bo')
	ax.set_xlabel('drive freq (kHz)')
	ax.set_ylabel('B field fit phase (rad)')
	ax.set_title('Fixed 0.9 Vpp drive amplitude')
	
	# Fix wiggle_freq at 10 kHz, plot B amp vs. wiggle amp
	fig, ax = plt.subplots(1,1)
	sub_df = df[df['wiggle_freq']==10.0]
	xx = sub_df['wiggle_amp']
	yy = sub_df['B_amp']
	yerr = sub_df['e_B_amp']
	
	xlist = np.linspace(np.min(xx), np.max(xx), num)
	ax.errorbar(xx, yy, sub_df['e_B_amp'],fmt='bo')
	ax.plot(xlist, linear(xlist, *popt_amp), 'b--')
	ax.set_xlabel('drive amp (Vpp)')
	ax.set_ylabel('B field fit amplitude (G)')
	ax.set_title('Fixed 10 kHz drive freq')