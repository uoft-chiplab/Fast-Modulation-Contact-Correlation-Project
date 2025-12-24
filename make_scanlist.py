###file to make scan list for phase shift measurements 

import os
import sys
# Always base paths on this file's location, not the working directory
this_file = os.path.abspath(__file__)
root_project = os.path.dirname(this_file)  
root_analysis = os.path.dirname(root_project)
library_folder = os.path.join(root_analysis, "analysis")
if library_folder not in sys.path:
	sys.path.append(library_folder)	
root = os.path.dirname(root_analysis)
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import random
from scipy.interpolate import interp1d
from library import FreqMHz
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
###fitting sin to field calibration
from scipy.optimize import curve_fit
# settings for directories, standard packages...
def Linear(x, m, b):
	return m*x + b
def sinc2(x, A, x0, sigma, C):
	return np.abs(A)*(np.sinc((x-x0) / sigma)**2) + C

def generate_spectra_scanlist(trf, f0, VVA, n_peak, n_wings, n_bg, max_wings_freq_in_width=2, reps=1,randomize=True):
	"""Generates a scanlist for a sinc^2 spectra s.t. the peak is well sampled
	   and only a few points are sampled beyond the Fourier width.
	   trf - time of the pulse in us
	   f0 - frequency centre of the spectra in MHz detuning (i.e: 4)
	   n_peak - number of points to sample in peak
	   n_wings - num points outside Fourier width
	   n_bg - number of points to sample as bg points
	   max_bg_freq_in_width - max freq in bg list as multiple of width
	"""
	freq_width = 1/trf
	x0 = 47.2227
	# sample around peak with arcsin, s.t. you weigh more near the peak.
	peak_shift_dets = np.arcsin(np.linspace(0, 1, n_peak // 2 + 1, endpoint=False))/(np.pi/2) * freq_width/0.75
	# add the flipped shifts, ignoring the centre
	peak_shift_dets = np.concat([-peak_shift_dets[1:], peak_shift_dets])
	# actual dets obtained by adding to centre det
	peak_dets =  peak_shift_dets + f0
	peak_freqs = x0 - peak_dets
	# sample uniformly in the wings, alternating sides
	if n_wings > 0:
		max_wings = max_wings_freq_in_width * freq_width
		distance_from_centre = np.linspace(max_wings, freq_width, n_wings)
		distance_from_centre[::2] = -1 * distance_from_centre[::2]
		wing_dets = distance_from_centre + f0
		wing_freqs = x0-wing_dets
	else: 
		wing_dets = []
		wing_freqs=[]
	
	all_freqs = np.append(peak_freqs, wing_freqs)
	all_dets = np.append(peak_dets, wing_dets)
	signals = np.ones((len(all_freqs),3))
	signals[:,0] = all_freqs
	signals[:,1] = all_dets
	signals[:,2] = VVA
	
	# sample zero VVA points
	bgs = np.ones((n_bg,3))
	bgs[:,0] = bgs[:,0] * 43
	bgs[:,1] = bgs[:,1] * (x0-43)
	bgs[:,2] = bgs[:,2] * 0
	if randomize:
		scanlist = np.concat([signals, bgs])
		np.random.shuffle(scanlist)
	# TODO: logic to regularly interleave the bg point based on a chosen interval
	# for _ in range(reps - 1):
	# 	scanlist.extend(scanlist[:]) # Use a slice to create a copy of the list
	scanlist = np.vstack([scanlist]*reps)
	return scanlist


def generate_singleshot_scanlist(f0, VVA, n_peak, n_bg, wings=False, randomize=True):
	"""Generates a scanlist for a single shot dimer spectrum. 
		Can randomly distribute bg points or place them at intervals.
		f0 -- frequency centre of the spectra in MHz (positive) detuning 
		n_peak -- num points with signal
		n_bg -- num points at bg
	"""
	x0 = 47.2227 # b to c res at 202.14 G
	if wings:
		signals = np.ones((n_peak*3,3))
		signals[:n_peak,0] = signals[:n_peak,0] * (x0-f0) # freq i.e: 43.2
		signals[n_peak:2*n_peak,0] = signals[n_peak:2*n_peak,0] * (x0-f0-wings) 
		signals[2*n_peak:3*n_peak,0] = signals[2*n_peak:3*n_peak,0] * (x0-f0+wings) 
		signals[:n_peak,1] = signals[:n_peak,1] * f0 # detuning i.e 4
		signals[n_peak:2*n_peak,1] = signals[n_peak:2*n_peak,1] * (f0-wings) 
		signals[2*n_peak:3*n_peak,1] = signals[2*n_peak:3*n_peak,1] * (f0+wings) 
		signals[:,2] = signals[:,2] * VVA # VVA
		bgs = np.ones((n_bg, 3))
		bgs[:,0] = bgs[:,0] * 43
		bgs[:,1] = bgs[:,1] * (x0-43)
		bgs[:,2] = bgs[:,2] * 0

	else:
		
		signals = np.ones((n_peak, 3))
		signals[:,0] = signals[:,0] * (x0-f0) # freq i.e: 43.2
		signals[:,1] = signals[:,1] * f0 # detuning ~4
		signals[:,2] = signals[:,2] * VVA # VVA

		bgs = np.ones((n_bg, 3))
		bgs[:,0] = bgs[:,0] * 43
		bgs[:,1] = bgs[:,1] * (x0-43)
		bgs[:,2] = bgs[:,2] * 0

	if randomize:
		scanlist = np.concat([signals, bgs])
		np.random.shuffle(scanlist)
	# TODO: logic to regularly interleave the bg point based on a chosen interval

	return scanlist

def field_to_odt(Bval):
	fields = np.array([202.14      , 202.15265925, 202.1651148 , 202.17716625,
       202.18861967, 202.19929079, 202.2090079 , 202.21761465,
       202.22497254, 202.2309632 , 202.23549022, 202.23848078,
       202.23988673, 202.23968548, 202.23788024, 202.23450008,
       202.22959938, 202.22325699, 202.21557496, 202.2066769 ,
       202.19670599, 202.18582265, 202.17420201, 202.16203105,
       202.1495056 ])
	odts = np.array([1.0003318773524044, 0.9852500707824112,0.9710777926925142,
	0.9576756013558121,0.9452119910350403,0.9338321559859135,0.9236591925419738,0.9147956404675881,
	0.9073251732267774,0.9013142807641008,0.8968138214330148,0.8938603496021237,0.8924771512327716,
	0.892674941315125,0.8944521950886813,0.8977951005050037,0.9026771337019462,
	0.9090582737476196,0.9168838889564076,0.9260833459222179,0.9365684150129047,
	0.9482315729031997,0.9609443335381477,0.9745557724521371,0.9888914430220057])
	df = pd.DataFrame({'odts': odts, 'Field': fields})

	df_sorted = df.sort_values('odts')
	
	interp_func = interp1d(df_sorted['Field'], df_sorted['odts'],  
                          kind='linear', fill_value='extrapolate')
	
	return interp_func(Bval)

export =True
# randomizes order freqs in scan list for each time
randomize = True
# single shot scan list vs. multiple detunings
singleshot = False
# HFT single shot list
HFT = True
#if we want to oscillate the ODT values to try to keep the density const
odt_wiggle = False
# reference voltages for the scale
odt1_ref = 0.2 
odt2_ref = 4.0
detuning = 0.150 # MHz
# times
pulsetime = 0.020
# this is the time delay time stamp, the actual time at which pulse starts is defind later
# t = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18])
t = np.array([ 0.21, 0.22, 0.24,0.26, 0.27, 0.29, 
 					])
np.random.shuffle(t)
f = 10 #kHz
amp = 0.9 # Vpp
vva= 8
reps = 15

###dimer is formed in the middle of the pulsetime
t_pulse = t - pulsetime/2

# frequencies
wait=0.02

fname = "phase_shift_scanlist.xlsx"
#saving in E:\Analysis Scripts\Fast-Modulation-Contact-Correlation-Project
x0 = 47.2227

###getting the field wiggle calibration done previously 
field_cal_df = pd.read_csv(os.path.join(os.getcwd(), 'FieldWiggleCal//field_cal_summary.csv'))
if amp in field_cal_df['wiggle_amp'].unique():
	field_cal_df = field_cal_df[(field_cal_df['wiggle_freq'] == f) & (field_cal_df['wiggle_amp'] == amp)]
	field_params = field_cal_df[['B_amp', 'B_phase', 'B_offset']].values[-1]
	print(f'Using field cal from {field_cal_df['run'].values[-1]} with {field_cal_df['wiggle_freq'].values[-1]}kHz freq and {field_cal_df['wiggle_amp'].values[-1]}Vpp amp')
else:
	print("Requested field cal doesn't exist, probably because that specific amplitude wasn't calibrated.\nAttempting to fudge it.")
	field_cal_df = field_cal_df[(field_cal_df['wiggle_freq'] == f)]
	popt_b, pcov_b = curve_fit(Linear, field_cal_df['wiggle_amp'], field_cal_df['B_amp'], sigma=field_cal_df['e_B_amp'])
	fudged_bamp = Linear(amp, *popt_b)
	fudged_bphase, fudged_boffset = field_cal_df['B_phase'].mean(), 202.14
	field_params = fudged_bamp, fudged_bphase, fudged_boffset
	print(f'Fudged field params are: {fudged_bamp:.3f} G, {fudged_bphase:.2f} rad, {fudged_boffset:.3f} G')

# also get the experimental phase shift results for f0
results_df = pd.read_csv(os.path.join(os.getcwd(),'analysis//phase_shift_2025_summary.csv'))
# thisis because I annoyingly saved the result as a string of a list and need to split it
# results_df['Sin Fit of f0'] = results_df['Sin Fit of f0'].apply(lambda x: x[1:-1].split(' '))
# lst = results_df[(results_df['Modulation Freq (kHz)']==f) & (results_df['Modulation Amp (Vpp)']==amp)]['Sin Fit of f0'].values[0]
# f0_fit = [float(i) for i in lst]

###plotting field calibration 
t2 = np.linspace(min(t), max(t), 100)

def fit_fixedSinkHz(t, y, run_freq, eA, p0=None):
	"""
	need docstring, and maybe a better way to code this
	t in ms, freq in kHz
	"""
	def FixedSinkHz(t, A, p, C):
		omega = run_freq * 2 * np.pi # kHz
		return A*np.sin(omega * t - p) + C

	if p0 == None:
		A = (max(y)-min(y))/2
		C = (max(y)+min(y))/2
		p = np.pi
		p0 = [A,p,C]
	
	if np.any(eA != 0):
		popts, pcov = curve_fit(FixedSinkHz, t, y, p0, bounds=([0, 0, 0], [np.inf, 2*np.pi, np.inf]), sigma=eA)
		perrs = np.sqrt(np.diag(pcov))
	
	else: 
		popts, pcov = curve_fit(FixedSinkHz, t, y, p0, bounds=([0, 0, 0], [np.inf, 2*np.pi, np.inf]))
		perrs = np.sqrt(np.diag(pcov))

	# rescales phase by pi for label
		# plabel = fit_label(popts, perrs, ["A", "p", "C"])#, units=["", f"$\pi$", ""])

	return popts, perrs, FixedSinkHz

popts, pcov, sin = fit_fixedSinkHz(t, np.sin((f*2*np.pi)*t-field_params[1]), f, 0, p0=[0.07,6,202.1])

# plt.plot(t, np.sin((f*2*np.pi)*t-field_params[1]), marker="o", ls="", color="hotpink", 
# 		 label = 'Time delay timestamps')
fig, ax = plt.subplots()
ax.plot(t2, np.sin((f*2*np.pi)*t2-field_params[1]), marker="", ls="-", color="hotpink", label='Field cal')
# plt.plot(t_pulse, np.sin((f*2*np.pi)*t_pulse-field_params[1]), marker="o", ls="", color="cornflowerblue", 
# 		 label = 'Time of Beginning of Pulse')
expectedphaseshift = np.pi/4
expected_timedelay = t2 + expectedphaseshift / (2*np.pi*f)
ax.plot(t2, np.sin((f*2*np.pi)*t2-field_params[1]-expectedphaseshift), marker="", ls="-", color="crimson", 
		 label = f'Expected phase shift comp. field cal= {expectedphaseshift:.2f}')
ax.plot(t_pulse, np.sin((f*2*np.pi)*t_pulse-field_params[1]-expectedphaseshift), marker="o", ls="", color="cornflowerblue", 
		 label = 'Time of Beginning of Pulse')
ax.plot(t, np.sin((f*2*np.pi)*t-field_params[1]-expectedphaseshift), marker="o", ls="", color="crimson", 
 		 label = 'Time delay timestamps')

ax.set(
	xlabel = "time (ms)", 
	ylabel = "B (au)" )
ax.legend()

ax2 = ax.twinx()
#generate a list of f0s based on B for each time we choose 
#using t because I want the f0 associated with the time that the phase shift is measured at
Bs = sin(t, *field_params)

if odt_wiggle:
	ax2.plot(t, field_to_odt(Bs), marker = '+', ls='')
	ax2.set(
		ylabel = 'ODT'
	)

use_static_cal =False
if use_static_cal:
	# linear interpolation from static dimer measurements around unitarity
	# hard coded because I haven't saved the static measurement results anywhere yet
	xs = [202.04, 202.24]
	ys = [4.020, 3.96]
else :
	# linear interpolation from AC measurement
	xs = [Bs.min(), Bs.max()]
	# ys = [f0_fit[2] + f0_fit[0], f0_fit[2]-f0_fit[0]] # [const + amp, const-amp]

# #dimer center pts freqs based on 2025-10-01_L run 
xs = [4.008814, 4.008802, 4.002268, 3.99069,3.990477, 3.973251, 3.973131, 3.960835,   
	  3.95955]
ys = [ 202.04432777360583, 202.04432777360583, 202.08310682397507, 
	  202.14250352617975, 202.14250352617975, 202.20524934772862, 
		  202.20524934772862, 202.24195852851076,  202.24195852851076]
xs.reverse() # higher Eb = lower field

B_to_f0 = interp1d(xs, ys, fill_value='extrapolate')
predicted_f0s_list = B_to_f0(Bs)

if HFT == True:
	scanlist_df = pd.DataFrame({'field':Bs})
	# scanlist_df['freq_res'] = FreqMHz(scanlist_df['field'], 9/2, -5/2, 9/2, -7/2)
	scanlist_df['freq'] = FreqMHz(scanlist_df['field'], 9/2, -5/2, 9/2, -7/2) + detuning
	scanlist_df['VVA'] = vva
	chargetimes = 1 - t_pulse - pulsetime - wait
	scanlist_df['chargetime'] = chargetimes
	scanlist_df['wiggletime'] = t_pulse
	scanlist_df = pd.concat([scanlist_df] * reps, ignore_index=True)
	if odt_wiggle:
		scanlist_df['ODTscale'] = field_to_odt(scanlist_df['field'])
		scanlist_df['ODT1'] = odt1_ref * scanlist_df['ODTscale']
		scanlist_df['ODT2'] = odt2_ref * scanlist_df['ODTscale']

	bg_df = scanlist_df.copy()
	bg_df['VVA']=0
	bg_df_repeated=pd.concat([bg_df]*reps, ignore_index=True)

	final_df = pd.concat([scanlist_df, bg_df])
	if randomize:
		final_df = final_df.sample(frac=1).reset_index(drop=True) # randomize 100% of rows

	if export:
		final_df.to_excel(fname, index=False)
else:
	#generate scanlist for each f0
	scanlist = []   

	for f0 in predicted_f0s_list:
		if singleshot == True:
			scanlist.append(generate_singleshot_scanlist(f0, vva, 15, 7, wings = None, randomize=True))
		elif HFT == True:
			continue

		else:
			scanlist.append(generate_spectra_scanlist(pulsetime*1000, f0, vva, 7, 0, 2, 1, reps,randomize=True))


	scanlist = np.array(scanlist)
	SHOW_SCANLIST= True
	if SHOW_SCANLIST:
		fig, ax = plt.subplots()
		for i, fs in enumerate(scanlist):
			freqs = fs[:,0]
			filter_bg = np.where(freqs!=43)
			filtered_freqs = freqs[filter_bg]
			ax.plot(filtered_freqs, np.ones(len(filtered_freqs))*(i+1), marker="o", ls="", label=f't={t[i]:.3f} ms, f0={predicted_f0s_list[i]:.3f} MHz')
		ax.set(xlabel="Freq (MHz)", ylabel="Scan #")
		# ax.legend()
	####exporting scan list to excel
	# all_scans = []
	# f0s_repeated = f0slist * n_dimer_repeat

	chargetimes = 1 - t_pulse - pulsetime - wait
	full_time_arr = [x for x in t_pulse for _ in range(np.shape(scanlist)[1])]
	full_chargetime_arr = [x for x in chargetimes for _ in range(np.shape(scanlist)[1])]
	full_time_arrs_2D = np.column_stack((full_chargetime_arr, full_time_arr))
	scanlist_2D = scanlist.reshape(-1, scanlist.shape[2])
	final_array = np.concatenate((scanlist_2D, full_time_arrs_2D), axis=1)
	final_df = pd.DataFrame(final_array)
	if export:
		final_df.to_excel(fname, index=False)
