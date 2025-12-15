"""
This script analyzes the growth of the contact following an rf quench into unitarity.
"""


# settings for directories, standard packages...
from preamble import *
from library import colors, kB

using_midpoints = False
####plotting timing diagram 
# Data representing a sequence of bits (0s and 1s)
bits = [0, 0, 1, 0, 0, 1 , 0]
# Repeat each bit twice to create the square wave data
data = np.repeat(bits, 2)
# Create corresponding time points (e.g., 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, ...)
t = 0.5 * np.arange(len(data))
# Plot the signal using plt.step() with where='post'
plt.step(t, data, 'r', ls='-', marker='', linewidth=2, where='post')
plt.ylim([-0.5, 1.5]) # Set y-axis limits to clearly show 0 and 1
plt.xlim([0, len(bits)]) # Set x-axis limits to match the number of bits
plt.title("Digital Timing Diagram of Pulses")
# plt.xticks([], [])
plt.yticks([], [])
plt.grid(True, axis='x', linestyle='--', alpha=0.6) # Add vertical grid lines

if using_midpoints:
	plt.vlines(2.5, -0.5, 1.5, ls='-.', color='grey')
	plt.vlines(5.5, -0.5, 1.5, ls='-.', color='grey', label='time after quench')
else:
	plt.vlines(3, -0.5, 1.5, ls='-.', color='grey')
	plt.vlines(5, -0.5, 1.5, ls='-.', color='grey', label='time after quench')

plt.text(2, 1.2, 'Quench')
plt.text(5, 1.2, 'Probe')

plt.legend()
###5 us quench, 10 us dimer 

runs = pd.DataFrame(
	{"2025-12-10_J":{"EF":14500, "T":369, "quench time":5, "probe time":10, "ratio":5050},
	 "2025-12-11_E":{"EF":14250, "T":340, "quench time":10, "probe time":10, "ratio":5050},
	 "2025-12-11_G":{"EF":14250, "T":340, "quench time":5, "probe time":20, "ratio":5050},
	 "2025-12-11_H": {"EF":14250, "T":340, "quench time":8, "probe time":10, "ratio":5050},
	#  "2025-12-12_E": {"EF":14420, "T":377, "quench time":5, "probe time":10},
	 "2025-12-15_H": {"EF":13940, "T":386, "quench time":5, "probe time":10, "ratio":8515},
	 "2025-12-15_J": {"EF":13940, "T":386, "quench time":5, "probe time":10, "ratio":7030}

	 }) # EF in Hz, T in nK

defaults = {'freq':43.24} # default values for freq (MHz), trf (s)
xlim = [0,200]

fig, ax = plt.subplots(figsize=(4,3))
ax.set(xlabel = 'Time after quench (us)',
	ylabel = r'$\widetilde{\Gamma}_d / \widetilde{\Gamma}_{d,\mathrm{max}}$')

fig2, ax2= plt.subplots(1,2 , figsize=(7,3))
ax2[0].set(xlabel='Time after quench (us)',
	   ylabel=r'$C_d$',
	#    xlim = xlim
)
ax2[1].set(xlabel='Time after quench (us)',
	   ylabel=r'$C_d/C_{max}$',
	#    xlim = xlim
	   )
fig3, ax3 = plt.subplots(2,2 , figsize=(10,5))
ax3 = ax3.flatten()
ax3[0].set(
	xlabel = 'Time after quench (us)',
	ylabel = 'c5bg'
)
ax3[1].set(
	xlabel = 'Time after quench (us)',
	ylabel = 'c5'
)
ax3[2].set(
	xlabel = 'Time after quench (us)',
	# ylabel = 'c9bg/c5bg'
	ylabel='alpha'
)
ax3[3].set(
	xlabel = 'VVA',
	ylabel = 'c5'
)
for i, run in enumerate(runs):
	# find data files
	y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
	runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
	datfiles = glob(f"{runpath}*.dat")
	runname = datfiles[0].split(os.sep)[-2].lower() # get run folder name, should be same for all files

	filename = datfiles[0].split(os.sep)[-1]
	print("Filename is " + filename)
	data = Data(filename, path=datfiles[0])

	EF, T, quench_t, probe_t, ratio = runs[run]
	data.data['trf'] = probe_t/1e6
	ToTF = kB*(T*1e-9)/(h*EF)
	data.data['EF'] = EF
	for key in defaults.keys():
		if key not in data.data.keys():
			data.data[key] = defaults[key]

	# alldata = data.analysis(bgVVA = 0,nobg=True, pulse_type="square").data
	data.analysis(bgVVA = 0, pulse_type="square")
	data.group_by_mean('hold time (us)')
	df = data.avg_data
	print(data.data['trf'])

	if using_midpoints:
		midpoints = quench_t/2 + probe_t/2 
		df['hold time (us)'] = df['hold time (us)'] + midpoints
		print('Adding midpoints to the time')
	
	df['norm_sig'] = df['scaledtransfer_dimer'] / df['scaledtransfer_dimer'].max()
	df['em_norm_sig'] = df['em_scaledtransfer_dimer'] / df['scaledtransfer_dimer'].max()
	df['norm_C'] = df['contact_dimer'] / df['contact_dimer'].max()
	df['em_norm_C'] = df['em_contact_dimer'] / df['contact_dimer'].max()
	# I'm flip flopping on this. Let's turn this into some kind of fit. The speed is the the comparizon b/w the zero time offset and the maximum ~1. 
	C_at_0 = df[df['hold time (us)'] == min(df['hold time (us)'])]['contact_dimer']
	C_at_30 = df[df['hold time (us)'] == 30]['contact_dimer']
	# speed = float(C_at_0.iloc[0])/float(C_at_30.iloc[0])
	x = df['hold time (us)']
	popt, pcov = curve_fit(one_minus_e,x, df['norm_sig'])
	xs = np.linspace(min(x), max(x), 100)
	popt_C, pcov_C = curve_fit(one_minus_e, x, df['norm_C'])

	ax.errorbar(x,#+ (quench_t - np.min(runs.loc["quench time"])), # shift x axis so t=0 is at the end of the pulse, using 5us quench as a reference
		  df['norm_sig'], 
		  df['em_norm_sig'],
		  fmt='o', label=f'{quench_t} us quench, {probe_t} us probe, {T} nK',
		  color=colors[i])
	ax.plot(xs, one_minus_e(xs, *popt), marker='', ls='-',
		 color = colors[i])
	
	ax2[0].errorbar(x,
			  df['contact_dimer'],
			  df['em_contact_dimer'],
			  fmt='o', label=f'{quench_t} us quench, {probe_t} us probe, ToTF = {ToTF:.2f}',
			  color=colors[i])
	ax2[1].errorbar(x,
			  df['norm_C'],
			  df['em_norm_C'],
			  fmt='o',
			    label=f'{quench_t} us quench, {probe_t} us probe, ToTF = {ToTF:.2f}, Ratio = {ratio}',
			  color=colors[i])
	ax2[1].plot(xs, one_minus_e(xs, *popt_C), marker='', ls='-',
		 color = colors[i])
	print(run, popt, popt_C)
	ax3[0].plot(x,
		  df['c5bg'],
		  color=colors[i],
		  label=f'{quench_t} us quench, {probe_t} us probe, {T} nK')
	
	ax3[1].plot(x,
		  df['c5'],
		  color=colors[i],
		  label=f'{quench_t} us quench, {probe_t} us probe, {T} nK')
	# ax3[2].plot(
	# 	alldata['hold time (us)'],
	# 	alldata['fraction95bg'],
	# 			  color=colors[i],
	# 	  label=f'{quench_t} us quench, {probe_t} us probe, {T} nK'
	# )
	ax3[2].plot(
		df['hold time (us)'],
		df['alpha_dimer'],
				  color=colors[i],
		  label=f'{quench_t} us quench, {probe_t} us probe, {T} nK'
	)
	ax3[3].plot(
		alldata['VVA'],
		alldata['c5'],
				  color=colors[i],
		  label=f'{quench_t} us quench, {probe_t} us probe, {T} nK'
	)

	fig2.tight_layout()
	fig3.tight_layout()
	
ax.legend(loc=0, bbox_to_anchor=(1,-0.25))
ax2[1].legend(loc=0, bbox_to_anchor=(0.5,-0.25))
ax3[3].legend(loc=0, bbox_to_anchor=(0.5,-0.25))
