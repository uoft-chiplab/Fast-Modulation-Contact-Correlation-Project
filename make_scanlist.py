import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys 
import random

def sinc2(x, A, x0, sigma, C):
	return np.abs(A)*(np.sinc((x-x0) / sigma)**2) + C

def generate_spectra_scanlist(trf, f0, n_peak, n_bg, max_bg_freq_in_width=2):
	"""Generates a scanlist for a sinc^2 spectra s.t. the peak is well sampled
	   and only a few points are sampled beyond the Fourier width.
	   trf - time of the pulse in us
	   f0 - frequency centre of the spectra in MHz
	   n_peak - number of points to sample in peak
	   n_bg - number of points to sample outside Fourier width
	   max_bg_freq_in_width - max freq in bg list as multiple of width
	"""
	freq_width = 1/trf

	# sample around peak with arcsin, s.t. you weigh more near the peak.
	peak_shift_freqs = np.arcsin(np.linspace(0, 1, n_peak // 2 + 1, endpoint=False))/(np.pi/2) * freq_width

	# add the flipped shifts, ignoring the centre
	peak_shift_freqs = np.concat([-peak_shift_freqs[1:], peak_shift_freqs])

	# actual freqs obtained by adding to centre freq
	peak_freqs =  peak_shift_freqs + f0

	# sample uniformly in the bg, alternating sides
	max_bg = max_bg_freq_in_width * freq_width
	distance_from_centre = np.linspace(max_bg, freq_width, n_bg)
	distance_from_centre[::2] = -1 * distance_from_centre[::2]
	bg_freqs = distance_from_centre + f0

	return np.sort(np.concat([peak_freqs, bg_freqs]))

export = True
# randomizes order freqs in scan list for each time
randomize = False
#add in randomize once then repeat insetad of randomizing each time

# times
pulsetime=0.020
t = np.array([0.3, 0.35, 0.39, 0.43, 0.47, 0.51, 0.55, 0.59, 0.63, 0.67, 0.71
					]) #np.linspace(min(x), max(), 7)
f = 6 #kHz
amp = 1.8 # Vpp
###expected phase shift based on frequency of mod (and temp etc in theory)
# shift = 0.7/2/np.pi * 1/f*1000 
# t = t + np.round(shift, decimals=-1)/1000
###expected time accounting for 1/2 pulse length since the dimer is formed in the middle of the pulsetime
t_pulse = t - pulsetime/2
# frequencies
wait=0.02
vva=4.5
n_dimer_repeat = 4
fname = "phase_shift_scanlist.xlsx"
#saving in E:\Analysis Scripts\Fast-Modulation-Contact-Correlation-Project
x0 = 47.2227


module_folder = 'E:\\Analysis Scripts\\analysis'
if module_folder not in sys.path:
	sys.path.insert(0, 'E:\\Analysis Scripts\\analysis')
	
###getting the field wiggle calibration done previously 
field_cal_path = r"E:\Analysis Scripts\Fast-Modulation-Contact-Correlation-Project\FieldWiggleCal\field_cal_summary.csv"
field_cal_df = pd.read_csv(field_cal_path)
field_cal_df = field_cal_df[(field_cal_df['wiggle_freq'] == f) & (field_cal_df['wiggle_amp'] == amp)]

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
		 p0 = [A, p, C]

	if np.any(eA != 0):

		popts, pcov = curve_fit(FixedSinkHz, t, y, p0, bounds=([0, 0, 0], [np.inf, 2*np.pi, np.inf]), sigma=eA)
		perrs = np.sqrt(np.diag(pcov))



		# plabel = fit_label(popts, perrs, ["A", "p", "C"])#, units=["", f"$\pi$", ""])

	else: 
		popts, pcov = curve_fit(FixedSinkHz, t, y, p0, bounds=([0, 0, 0], [np.inf, 2*np.pi, np.inf]))
		perrs = np.sqrt(np.diag(pcov))

	# rescales phase by pi for label
		# plabel = fit_label(popts, perrs, ["A", "p", "C"])#, units=["", f"$\pi$", ""])

	return popts, perrs, FixedSinkHz
field_params = field_cal_df[['B_amp', 'B_phase', 'B_offset']].values[0]
field_params[1] += np.pi # add pi phase shift to comapre to freq

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

def B_to_f0(x):
	"""
	given a frequency corresponding to the dimer center position get a field out
	lists taken from the breit rabi notebook since I didn't want to code it all out lmao
	then I realized these frequencies are for on resonance for 7->5 at 202-202.5
	and I figured I could just subtract the frequency expected for the dimer at 202.14 to get the 
	detuning 
	"""
	#dimer center pts freqs based on 2025-10-01_L run 
	xs = [4.008814, 4.008802, 4.002268, 3.99069,3.990477, 3.973251, 3.973131, 3.960835,   
		  3.95955]
	ys = [ 202.04432777360583, 202.04432777360583, 202.08310682397507, 
		  202.14250352617975, 202.14250352617975, 202.20524934772862, 
			  202.20524934772862, 202.24195852851076,  202.24195852851076]

	xs.reverse() # higher Eb = lower field

	return np.interp(x,ys,xs)

#generate a list of f0s based on B for each time we choose 
#using t because I want the f0 associated with the time that the phase shift is measured at
Bs = sin(t, *field_params)
predicted_f0s_list = B_to_f0(Bs)

#generate scanlist for each f0
f0slist = []   
for f0s in predicted_f0s_list:
	f0slist.append(generate_spectra_scanlist(pulsetime*1000, f0s, 7, 0, max_bg_freq_in_width=2))

SHOW_SCANLIST= True
if SHOW_SCANLIST:
	fig, ax = plt.subplots()
	for i, fs in enumerate(f0slist):
		ax.plot(fs, np.ones(len(fs))*(i+1), marker="o", ls="", label=f't={t[i]:.3f} ms, f0={predicted_f0s_list[i]:.3f} MHz')
	ax.set(xlabel="-Detuning (MHz)", ylabel="Scan #")
	ax.legend()
####exporting scan list to excel
all_scans = []
f0s_repeated = f0slist * n_dimer_repeat

####exporting scan list to excel
all_scans = []

all_ds = []
all_f0s = []
all_vvas = []
all_ts = []
all_chargetimes = []
if export:
	 #t_pulse here since the time the pulse starts is different than the expected ps shift time
	for idx, (f0s, time_val) in enumerate(zip(f0slist, t_pulse)):
		# ts = time_val #np.repeat(t, len(f0s) * n_dimer_repeat)  # shape: (len(f0s) * len(t),)
		# chargetimes = 1 - ts - pulsetime - wait
		# assert len(f0s_repeated) == len(t) * n_dimer_repeat
		# print(f0s)
		for _ in range(n_dimer_repeat):
			ds = x0 - f0s
			# print(f0s)

		# Make sure time arrays match length
			all_ts.append(np.full(len(ds), time_val, dtype=float))
			all_chargetimes.append(np.full(len(ds), 1 - time_val - pulsetime - wait, dtype=float))
			if randomize:
    # Create a random permutation of indices
				indices = np.random.permutation(len(ds))
    # Shuffle both ds and f0s in the same way
				ds_shuffled = ds[indices]
				# print(ds_shuffled)	
				f0s_shuffled = f0s[indices]

		# # Append shuffled data
				all_ds.append(ds_shuffled)
				all_f0s.append(f0s_shuffled)
				all_vvas.append(np.ones_like(ds_shuffled) * vva)
			else:
				all_ds.append(ds)
				all_f0s.append(f0s)
				all_vvas.append(np.ones_like(ds) * vva)	

		# Add one background point after each 3 repeats
		ds_bg = np.array([43])
		f0s_bg = np.array([x0 - 43])
		vvas_bg = np.array([0])
		times_bg = np.array([time_val])
		chargetimes_bg = np.array([1 - time_val - pulsetime - wait])

		all_ds.append(ds_bg)
		all_f0s.append(f0s_bg)
		all_vvas.append(vvas_bg)
		all_ts.append(times_bg)
		all_chargetimes.append(chargetimes_bg)


	all_ds = np.concatenate(all_ds)
	all_f0s = np.concatenate(all_f0s)
	all_vvas = np.concatenate(all_vvas)
	all_ts = np.concatenate(all_ts)
	all_chargetimes = np.concatenate(all_chargetimes)

	scan_df = pd.DataFrame({
		"freq": all_ds,
		"det": all_f0s,
		"vva": all_vvas,
		"chargetime": all_chargetimes,
		"time": all_ts
	})
	all_scans.append(scan_df)
	
	if randomize:
		fname = fname.replace(".xlsx", "_randomized.xlsx")
		final_df = pd.concat(all_scans, ignore_index=True)
		final_df.to_excel(fname, index=False)
	else:
		final_df = pd.concat(all_scans, ignore_index=True)
		final_df.to_excel(fname, index=False)