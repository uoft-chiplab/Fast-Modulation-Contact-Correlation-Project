import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys 

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
# times
t = np.array([0.20, 0.22, 
              0.24, 0.26, 0.32, 0.34, 0.36, 0.38
                    ]) #np.linspace(min(x), max(), 7)
f = 10 #kHz
###expected phase shift based on frequency of mod (and temp etc in theory)
shift = 0.7/2/np.pi * 1/f*1000 
t = t + np.round(shift, decimals=-1)/1000
# frequencies
# x = np.array([43.34, 43.32, 43.3,43.285, 43.27, 43.26, 43.25, 43.24, 43.23, 43.22, 43.21, 43.20, 43.185, 43.17, 43.14, 43.12, 43.1])
pulsetime=0.02
wait=0.02
vva=4.5
n_dimer_repeat = 2
fname = "phase_shift_scanlist.xlsx"
#saving in E:\Analysis Scripts\Fast-Modulation-Contact-Correlation-Project
x0 = 47.2227

# randomizes order of x and t
randomize = True

module_folder = 'E:\\Analysis Scripts\\analysis'
if module_folder not in sys.path:
	sys.path.insert(0, 'E:\\Analysis Scripts\\analysis')
	
###getting the field wiggle calibration done previously 
field_cal_path = r"E:\Analysis Scripts\Fast-Modulation-Contact-Correlation-Project\FieldWiggleCal\field_cal_summary.csv"
field_cal_df = pd.read_csv(field_cal_path)
field_cal_df = field_cal_df[field_cal_df['run'] == '2024-04-05_G']

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

popts, pcov, sin = fit_fixedSinkHz(t, np.sin((f*2*np.pi)*t-field_cal_df['B_phase'].values), f, 0, p0=[0.07,6,202.1])

plt.plot(t, np.sin((f*2*np.pi)*t-field_cal_df['B_phase'].values), marker="o", ls="", color="hotpink")
plt.plot(t2, np.sin((f*2*np.pi)*t2-field_cal_df['B_phase'].values), marker="", ls="-", color="hotpink")
plt.plot(t2, sin(t2, *popts), marker="", ls="--", color="cornflowerblue")
plt.xlabel("time (ms)")
plt.ylabel("B (au)")

def B_to_f0(x):
    """
    given a frequency corresponding to the dimer center position get a field out
    lists taken from the breit rabi notebook since I didn't want to code it all out lmao
    then I realized these frequencies are for on resonance for 7->5 at 202-202.5
    and I figured I could just subtract the frequency expected for the dimer at 202.14 to get the 
    detuning 
    """
    xs = [47.1989, 47.2159, 47.2329, 47.2498, 47.2668, 47.2838]
    xs = np.array(xs) - 43.2227
    ys = [202., 202.1, 202.2, 202.3, 202.4, 202.5]
    # ys.reverse() # higher Eb = lower field

    return np.interp(x,ys,xs)

#generate a list of f0s based on B for each time we choose 
predicted_f0s_list = []
for times in t:
      #find f0s 
      B_to_f0(sin(times, *field_params))
      predicted_f0s_list.append(B_to_f0(sin(times, *field_params)))
#generate scanlist for each f0
f0slist = []   
for f0s in predicted_f0s_list:
    f0slist.append(generate_spectra_scanlist(20, f0s, 7, 3, max_bg_freq_in_width=2))

# ###exporting scan list to excel
all_scans = []
f0s_repeated = f0slist * n_dimer_repeat
if export:
     
	for idx, (f0s, time_val) in enumerate(zip(f0s_repeated, t)):
		ts = time_val #np.repeat(t, len(f0s) * n_dimer_repeat)  # shape: (len(f0s) * len(t),)
		chargetimes = 1 - ts - pulsetime - wait
		# f0s_repeated = f0slist * n_dimer_repeat #np.tile(f0s, len(t) * n_dimer_repeat)
		assert len(f0s_repeated) == len(t) * n_dimer_repeat
		# print(f0s)
            
		ds = x0 - f0s
    ###adding in points for 0 VVA bg shots
		x = np.ones_like(np.linspace(0,4,4))*43
		ds = np.append(ds, x)
		f0s = np.append(f0s, x0 -x)
		vvas = np.ones_like(ds) * vva
		vvas[ds <= 43.01] = 0

		scan_df = pd.DataFrame({
			"freq": f0s,
			"det": ds,
			"vva": vvas,
			"chargetime": chargetimes,
			"time": ts
		})
		# print(scan_df)
		all_scans.append(scan_df)

	final_df = pd.concat(all_scans, ignore_index=True)
	final_df.to_excel(fname, index=False)

      
###plotting 2 of the sinc2 
# plt.figure()

# p = [0.14272243, 3.98, 0.05228547, 0.01898835]
# d = x0-x
# d2 = np.linspace(min(d), max(d), 100)
# plt.plot(d, sinc2(d, *p), marker="o", ls="", color="hotpink", label=f"x0={p[1]}")
# plt.plot(d2, sinc2(d2, *p), marker="", ls="-", color="hotpink")

# p = [0.14272243, 4.00, 0.05228547, 0.01898835]
# plt.plot(d, sinc2(d, *p), marker="o", ls="", color="cornflowerblue", label=f"x0={p[1]}")
# plt.plot(d2, sinc2(d2, *p), marker="", ls="-", color="cornflowerblue")
# plt.xlabel("detuning (MHz)")
# plt.ylabel("transfer")
# plt.legend()


###exporting scan list to excel
# if export:
# 	if randomize:
# 		# shuffles in place
# 		np.random.shuffle(x)
# 		np.random.shuffle(t)

# 	# add background point
# 	x = np.append(x, 43)

# 	# resize time to match freq arr size to keep time constant for each dimer scan
# 	ts = np.repeat(t, len(x)*n_dimer_repeat) # makes array of [t1, t1, ... tn, tn...]
# 	chargetimes = 1 - ts - pulsetime - wait

# 	# repeat dimer scan list for all times
# 	xs = np.resize(x, len(x)*n_dimer_repeat*len(t)) # makes array [x1, x2, xn... x1, x2, xn]
# 	# repeat 3 times

# 	ds = x0-xs
# 	# update vva of background point
# 	vvas = np.ones(len(xs))*vva
# 	vvas[xs <= 43.001] = 0

# 	# concatenate time and frequency scan lists
# 	scan_list = np.stack([xs, ds, vvas, chargetimes, ts], axis=1)
# 	scan_list = pd.DataFrame(scan_list, columns=["freq", "det", "vva", "chargetime", "time"])
# 	scan_list.to_excel(fname, index=False)

