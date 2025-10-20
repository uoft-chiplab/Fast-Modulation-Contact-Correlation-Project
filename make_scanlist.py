import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys 
import random
import os
from scipy.interpolate import interp1d
# settings for directories, standard packages...

def sinc2(x, A, x0, sigma, C):
	return np.abs(A)*(np.sinc((x-x0) / sigma)**2) + C

def generate_spectra_scanlist(trf, f0, VVA, n_peak, n_wings, n_bg, max_wings_freq_in_width=2, reps=1,randomize=True):
	"""Generates a scanlist for a sinc^2 spectra s.t. the peak is well sampled
	   and only a few points are sampled beyond the Fourier width.
	   trf - time of the pulse in us
	   f0 - frequency centre of the spectra in MHz detuning (i.e: 4)
	   n_peak - number of points to sample in peak
	   n_bg - number of points to sample outside Fourier width
	   max_bg_freq_in_width - max freq in bg list as multiple of width
	"""
	freq_width = 1/trf
	x0 = 47.2227
	# sample around peak with arcsin, s.t. you weigh more near the peak.
	peak_shift_dets = np.arcsin(np.linspace(0, 1, n_peak // 2 + 1, endpoint=False))/(np.pi/2) * freq_width/1
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
	signals = np.ones((n_peak+n_wings+1,3))
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

export = True
# randomizes order freqs in scan list for each time
randomize = False
# single shot scan list vs. multiple detunings
singleshot=True

# times
pulsetime=0.020
t = np.array([0.22,0.25,0.28, 0.31,0.34,0.37, 0.40, 0.43, 0.46, 0.49, 0.52, 0.55, 0.58
					]) #np.linspace(min(x), max(), 7)
f = 10 #kHz
amp = 1.8 # Vpp



###expected phase shift based on frequency of mod (and temp etc in theory)
###expected time accounting for 1/2 pulse length since the dimer is formed in the middle of the pulsetime
t_pulse = t - pulsetime/2
# frequencies
wait=0.02
vva=4.5
n_dimer_repeat = 4
fname = "phase_shift_scanlist.xlsx"
#saving in E:\Analysis Scripts\Fast-Modulation-Contact-Correlation-Project
x0 = 47.2227

###getting the field wiggle calibration done previously 
field_cal_df = pd.read_csv(os.path.join(os.getcwd(), 'FieldWiggleCal//field_cal_summary.csv'))
field_cal_df = field_cal_df[(field_cal_df['wiggle_freq'] == f) & (field_cal_df['wiggle_amp'] == amp)]
# also get the experimental phase shift results for f0
results_df = pd.read_csv(os.path.join(os.getcwd(), 'contact_correlations//phaseshift//phase_shift_2025_summary.csv'))
# thisis because I annoyingly saved the result as a string of a list and need to split it
results_df['Sin Fit of f0'] = results_df['Sin Fit of f0'].apply(lambda x: x[1:-1].split(' '))
lst = results_df[(results_df['Modulation Freq (kHz)']==f) & (results_df['Modulation Amp (Vpp)']==amp)]['Sin Fit of f0'].values[0]
f0_fit = [float(i) for i in lst]

###plotting field calibration 
t2 = np.linspace(min(t), max(t), 100)
###fitting sin to field calibration
from scipy.optimize import curve_fit
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
field_params = field_cal_df[['B_amp', 'B_phase', 'B_offset']].values[0]
# field_params[1] += np.pi # add pi phase shift to comapre to freq

popts, pcov, sin = fit_fixedSinkHz(t, np.sin((f*2*np.pi)*t-field_params[1]), f, 0, p0=[0.07,6,202.1])

plt.plot(t, np.sin((f*2*np.pi)*t-field_params[1]), marker="o", ls="", color="hotpink", 
		 label = 'Expected Phase Shift Time')

plt.plot(t2, np.sin((f*2*np.pi)*t2-field_params[1]), marker="", ls="-", color="hotpink")
# plt.plot(t2, sin(t2, *popts), marker="", ls="--", color="cornflowerblue")
plt.plot(t_pulse, np.sin((f*2*np.pi)*t_pulse-field_params[1]), marker="o", ls="", color="cornflowerblue", 
		 label = 'Time of Beginning of Pulse')
plt.xlabel("time (ms)")
plt.ylabel("B (au)")
# plt.legend()


#generate a list of f0s based on B for each time we choose 
#using t because I want the f0 associated with the time that the phase shift is measured at
Bs = sin(t, *field_params)
use_static_cal =False
if use_static_cal:
	# linear interpolation from static dimer measurements around unitarity
	# hard coded because I haven't saved the static measurement results anywhere yet
	xs = [202.04, 202.24]
	ys = [4.020, 3.96]
else :
	# linear interpolation from AC measurement
	xs = [Bs.min(), Bs.max()]
	ys = [f0_fit[2] + f0_fit[0], f0_fit[2]-f0_fit[0]] # [const + amp, const-amp]

# #dimer center pts freqs based on 2025-10-01_L run 
# xs = [4.008814, 4.008802, 4.002268, 3.99069,3.990477, 3.973251, 3.973131, 3.960835,   
# 	  3.95955]
# ys = [ 202.04432777360583, 202.04432777360583, 202.08310682397507, 
# 	  202.14250352617975, 202.14250352617975, 202.20524934772862, 
# 		  202.20524934772862, 202.24195852851076,  202.24195852851076]
# xs.reverse() # higher Eb = lower field
B_to_f0 = interp1d(xs, ys, fill_value='extrapolate')
predicted_f0s_list = B_to_f0(Bs)

#generate scanlist for each f0
scanlist = []   

for f0 in predicted_f0s_list:
	if singleshot == True:
		scanlist.append(generate_singleshot_scanlist(f0, 5, 15, 5, wings = None, randomize=True))
	else:
		scanlist.append(generate_spectra_scanlist(pulsetime*1000, f0, 5, 6, 0, 3, 1, reps=2,randomize=True))
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

# blame KX for this
# ####exporting scan list to excel
# all_scans = []

# all_ds = []
# all_f0s = []
# all_vvas = []
# all_ts = []
# all_chargetimes = []

# if export:
# 	 #t_pulse here since the time the pulse starts is different than the expected ps shift time
# 	for idx, (f0s, time_val) in enumerate(zip(scanlist, t_pulse)):
# 		# ts = time_val #np.repeat(t, len(f0s) * n_dimer_repeat)  # shape: (len(f0s) * len(t),)
# 		# chargetimes = 1 - ts - pulsetime - wait
# 		# assert len(f0s_repeated) == len(t) * n_dimer_repeat
# 		# print(f0s)
# 		for _ in range(n_dimer_repeat):
# 			ds = x0 - f0s
# 			# print(f0s)

# 		# Make sure time arrays match length
# 			all_ts.append(np.full(len(ds), time_val, dtype=float))
# 			all_chargetimes.append(np.full(len(ds), 1 - time_val - pulsetime - wait, dtype=float))
# 			if randomize:
#     # Create a random permutation of indices
# 				indices = np.random.permutation(len(ds))
#     # Shuffle both ds and f0s in the same way
# 				ds_shuffled = ds[indices]
# 				# print(ds_shuffled)	
# 				f0s_shuffled = f0s[indices]

# 		# # Append shuffled data
# 				all_ds.append(ds_shuffled)
# 				all_f0s.append(f0s_shuffled)
# 				all_vvas.append(np.ones_like(ds_shuffled) * vva)
# 			else:
# 				all_ds.append(ds)
# 				all_f0s.append(f0s)
# 				all_vvas.append(np.ones_like(ds) * vva)	

# 		# Add one background point after each 3 repeats
# 		ds_bg = np.array([43])
# 		f0s_bg = np.array([x0 - 43])
# 		vvas_bg = np.array([0])
# 		times_bg = np.array([time_val])
# 		chargetimes_bg = np.array([1 - time_val - pulsetime - wait])

# 		all_ds.append(ds_bg)
# 		all_f0s.append(f0s_bg)
# 		all_vvas.append(vvas_bg)
# 		all_ts.append(times_bg)
# 		all_chargetimes.append(chargetimes_bg)


# 	all_ds = np.concatenate(all_ds)
# 	all_f0s = np.concatenate(all_f0s)
# 	all_vvas = np.concatenate(all_vvas)
# 	all_ts = np.concatenate(all_ts)
# 	all_chargetimes = np.concatenate(all_chargetimes)

# 	scan_df = pd.DataFrame({
# 		"freq": all_ds,
# 		"det": all_f0s,
# 		"vva": all_vvas,
# 		"chargetime": all_chargetimes,
# 		"time": all_ts
# 	})
# 	all_scans.append(scan_df)
	
# 	if randomize:
# 		fname = fname.replace(".xlsx", "_randomized.xlsx")
# 		final_df = pd.concat(all_scans, ignore_index=True)
# 		final_df.to_excel(fname, index=False)
# 	else:
# 		final_df = pd.concat(all_scans, ignore_index=True)
# 		final_df.to_excel(fname, index=False)