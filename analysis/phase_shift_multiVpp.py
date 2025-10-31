"""
Calculates relative contact amplitdue between 2 points (~peak and trough) for multiple Vpp Wiggle Amplitudes 
"""

# settings for directories, standard packages...
from preamble import *
from get_metadata import metadata

run = "2025-10-26_O"
meta_df = pd.read_csv('metadata.csv')
meta_df = meta_df[meta_df['run']==run]
if meta_df.empty:
    print(f'no meta data for {run}, running now')
    m = metadata([run], [[]], True, "metadata.csv", "run from phase_shift.py")
    m.output()
    meta_df = pd.read_csv('metadata.csv')
    # print(meta_df)
    meta_df = meta_df[meta_df['run']==run]

# put this into metadata?
time_column_name = "Wiggle Time" # I think we can set this to be automatic 
								# by looping through column names and finding which contains "time"
wiggle_amp = meta_df['Vpp'].values[0]  # Vpp
dropped_list = np.fromstring(meta_df['drop'].values[0].strip('[]'), sep= ' ') # list of time stamps to drop
pulse_time = meta_df['pulse_time'].values[0] # 
wiggle_freq = meta_df['freq'].values[0] # kHz
VVA = meta_df['VVA'].values[0] # assumes just the max VVA for everything
### TODO: incorporate into metadata
EF = meta_df['EF'].values[0]/h # Hz
# this fudges the Rabi calibrated at 47 MHz for the attenuation at 43, but a calibration directly at 43 would be better
RabiperVpp47 = 13.05 / 0.500 # kHz/Vpp on scope 2025-10-21
e_RabiperVpp47 = 0.22
phaseO_OmegaR = lambda VVA, freq: 2*pi*RabiperVpp47 * Vpp_from_VVAfreq(VVA, freq)
xlabel = 'Times [ms]'

show_dimer_plot = True
amp_cutoff = 0.01 # ignore runs with peak transfer below 0.01
fix_width = True # whether or not dimer spectra sinc2 fits have a fixed width
track_bg = True
rerun = False

def line(x, m, b):
	return m*x+b

def find_transfer(df, popts_c5bg=np.array([])):
	"""
	given df output from matlab containing atom counts, returns new df containing detuning, 
	5 and 9 counts, and transfer/loss
	TODO account for tracking bg over whole scan or not
	"""
	run_data = df[["cyc","detuning", "VVA", "c5", "c9"]]

	# assumes popts_c5bg is a np array
	if popts_c5bg.any():
		run_data['c5bg'] = line(run_data['cyc'], *popts_c5bg)
		run_data["c5transfer"] = (1-run_data["c5"]/run_data["c5bg"])/2
		data = run_data[run_data["VVA"]!=0].copy()
	else:
		bg = df[df["VVA"] == 0]

		if (len(bg.c5) > 1):
			c5bg, c9bg = np.mean(bg[["c5", "c9"]], axis=0)
			c5bg_err = np.std(bg.c5)/len(bg.c5)
		else:
			c5bg, c9bg = bg[['c5', 'c9']].values[0]
			c5bg_err = 0
		# c5bg = avg_bg_c5
		# c9bg = avg_bg_c9
		# c5bg_err = np.std(bg_df.c5)/len(bg_df.c5)

		print(c5bg, c9bg)

		data = run_data[df["VVA"] != 0].copy()
		data.loc[data.index, "c5transfer"] = (1-data["c5"]/c5bg)/2 # factor of 2 assumes atom-molecule loss
		data.loc[data.index, "c5bg"] = c5bg
		data.loc[data.index, "ec5bg"] = c5bg_err
	return data

def avg_transfer(df):

	bg_err_rel = (df.ec5bg/df.c5bg).values[0] # same for all points

	# uncertainty in mean
	count = df.groupby("detuning")["c5"].count()
	std = df.groupby("detuning")["c5"].std()
	mean = df.groupby("detuning")["c5"].mean()

	c5_err_rel = ((std/count)/mean).values
	transfer_mean = df.groupby("detuning")["c5transfer"].mean()

	transfer_df_avg = pd.DataFrame(
						{
						'c5transfer' : transfer_mean,
						'ec5transfer' : transfer_mean.values*np.sqrt(c5_err_rel**2 + bg_err_rel**2)}
						)

	return transfer_df_avg

def fit_sinc2(xy, width=False):
	"""
	fits xy, input as single array, to sinc2 function. returns fitted params, errors, and label

	if a width is provided, fits sinc with fixed width and 0 background
	"""
	f, p0, paramnames = Sinc2(xy)

	if width:
		sinc2 = lambda x, A, x0: f(x, A, x0, width,0)
		p0 = p0[:2]
		paramnames = paramnames[:2]
		
	else:
		#  sinc2 = f
		sinc2 = lambda x, A, x0, width: f(x, A, x0, width,0)
		p0 = p0[:3]
		paramnames = paramnames[:3]

	popts, pcov = curve_fit(sinc2, xy[:,0], xy[:,1], p0)
	perrs = np.sqrt(np.diag(pcov))
	plabel = fit_label(popts, perrs, paramnames)

	return popts, perrs, plabel, sinc2

def fit_fixedSinkHz(t, y, run_freq, eA, p0=None, param_labels=["A", "p", "C"]):
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

	else: 
		popts, pcov = curve_fit(FixedSinkHz, t, y, p0, bounds=([0, 0, 0], [np.inf, 2*np.pi, np.inf]))
		perrs = np.sqrt(np.diag(pcov))

	plabel = fit_label(popts, perrs, param_labels)

	return popts, perrs, plabel, FixedSinkHz

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

def Contact_from_amplitude(A, eA, EF, OmegaR, trf):
	"""
	Convert fitted amplitude to \alpha (raw transfer) into GammaTilde -> Id -> C
	A -- fitted amplitude to \alpha
	eA -- error on A
	EF -- Fermi energy in Hz
	OmegaR -- Rabi frequency in kHz
	trf -- rf pulse time in us
	returns dimer spectral weight (Id), error in Id, contact/NkF (C, technically Ctilde), error in C
	TODO: proper error propgation from A, error elsewhere
	"""
	# scale fitted amplitude to gammatilde
	gammatilde = h*EF / (hbar * pi * (OmegaR*1e3)**2 * (trf/1e6)) * A 
	### TODO: uncertainty analysis that considers uncertainty in EF and OmegaR
	e_gammatilde = eA/A * gammatilde
	# calculate Id (dimer spectral weight) from gammatilde and then C
	Id_conv = 2 # this makes the spectral weight a fraction out of 1; right side of Fig. 3
	Id = gammatilde / (trf/1e6) / (EF) * Id_conv
	e_Id = e_gammatilde/gammatilde * Id

	# use CC theory line / our empirical fit to convert Id -> C
	kF = np.sqrt(2*mK*EF*h)/hbar
	a13kF = kF * a13(202.14)
	spin_me = 32/42 # spin matrix element
	ell_d_CCC = spin_me * 42 * np.pi

	C =   Id * np.pi / (ell_d_CCC *kF* a0) 
	e_C = e_Id/Id * C

	return Id, e_Id, C, e_C

def bg_over_scan(datfiles, plot=False):
	# mscan = pd.read_csv(glob(f"{runpath}*.mscan")[0], skiprows=2)
	df0VVA = pd.DataFrame()
	for fpath in datfiles:
		run_df = pd.read_csv(fpath)
		df0VVA = pd.concat([df0VVA, run_df[run_df["VVA"] == 0]])
	# sort by cycle to match mscan list
	df0VVA.sort_values("cyc", inplace=True)
	# fit trend in c5 vs time to line (cyc # as proxy for time)
	popts, pcov = curve_fit(line, df0VVA.cyc, df0VVA.c5, [3000, -1])
	perrs = np.sqrt(np.diag(pcov))
	### plot if you want 
	if plot:
		fig, ax = plt.subplots(figsize=(3,2))
		ax.plot(df0VVA['cyc'], df0VVA['c5'], marker='.')
		ax.plot(df0VVA['cyc'], line(df0VVA['cyc'], *popts), ls='-', marker='')

		ax.set(
			ylabel = 'c5 bg shots',
			xlabel='cyc'
		)
	
	return popts, perrs

# find data files
y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
datfiles = glob(f"{runpath}*=*.dat")
runname = datfiles[0].split("\\")[-2].lower() # get run folder name, should be same for all files
# calculate bg over time; needs to load all the datfiles first
if track_bg:
	popts_c5bg, perrs_c5bg = bg_over_scan(datfiles, plot=True)
	
# set up dataframe to hold dimer fits
columns = ["A", "x0", "eA", "ex0", 'C', 'eC', 'Wiggle Amp', 'Wiggle Time'] if fix_width else ["A", "x0", "width", "eA", "ex0", "ewidth", 'C', 'eC', 'Wiggle Amp']
df = pd.DataFrame( columns =columns,  dtype=float)
df.index.name = xlabel

valid_results = []

for fpath in datfiles:
	print(fpath.split("\\")[-1])
	run_df = pd.read_csv(fpath)
	if run_df[time_column_name][0] in dropped_list:
		print(f'dropping time at {run_df[time_column_name][0]}')
		continue
	# if run_df['cyc'].mean() >200 : 
	# 	continue
	# adjust time to be at centre of pulse
	index = run_df[time_column_name][0] + (pulse_time/1000)/2 # ms, should be same for all cycles
	width = 1/pulse_time if fix_width else None
	run_df['OmegaR'] = phaseO_OmegaR(VVA, run_df['freq'])
	# if not single_shot:
	if 'Wiggle Amp' in run_df.columns:
		for w in run_df['Wiggle Amp'].unique():
			# print(run_df)
			subset = run_df[run_df['Wiggle Amp'] == w]  # <-- select data for this amplitude
			data = find_transfer(subset)
			popts, perrs, plabel, sinc2 = fit_sinc2(data[["detuning", "c5transfer"]].values, 
        										width=width)
			title = f'Wiggle Time: {index:0.2f} ms, Wiggle Amp: {w} Vpp'

	# calculate GammaTilde
			data['GammaTilde'] = GammaTilde(data['c5transfer'], 
								 h*EF, 
								 run_df['OmegaR'].values[0]*1e3,
								 pulse_time/1e6)
			detuning, c5transfer, c5gammatilde = data["detuning"], data["c5transfer"], data["GammaTilde"]

	# Save for later plotting
			valid_results.append({
                "time":index, 
                "detuning": detuning,
                "c5transfer": c5transfer,
                "sinc2": sinc2,
                "popts": popts,
                "perrs":perrs,
                "plabel": plabel,
                "title": title,
                "residuals": c5transfer - sinc2(detuning, *popts),
                "predicted": sinc2(detuning, *popts),
				'Wiggle Amp': w
                    })

			if popts[0] > amp_cutoff: 
		# C/NkF from amplitude
				Id, e_Id, contact, e_contact = \
			Contact_from_amplitude(popts[0], perrs[0], EF, run_df['OmegaR'].mean(), pulse_time)
				row_index = f"{index:.3f}_{w:.2f}"  # combine time and amplitude
				df.loc[row_index] = np.concatenate([popts, perrs, [contact, e_contact], [w], [index]]) # lists should be concatenated in order
			else:
				dropped_list.append(index)

###plot all the dimer fits on one multi grid plot
if show_dimer_plot and valid_results:
	num_plots = len(valid_results)
	cols = 3
	rows = int(np.ceil(num_plots / cols))

	fig = plt.figure(figsize=(4 * cols, 3 * rows), constrained_layout=True)
	outer = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.5)

	for i, result in enumerate(valid_results):
		row = i // cols
		col = i % cols

		# inner grid for each panel (main + residual)
		inner = gridspec.GridSpecFromSubplotSpec(
			2, 1, subplot_spec=outer[row, col],
			height_ratios=[3, 1], hspace=0.05  
		)

		main_ax = fig.add_subplot(inner[0])
		resid_ax = fig.add_subplot(inner[1], sharex=main_ax)

		# Main plot
		xs = np.linspace(max(result["detuning"]), min(result["detuning"]), 100)
		main_ax.plot(xs, result["sinc2"](xs, *result["popts"]), ls="-", marker="", label=result["plabel"], color='deeppink')
		main_ax.plot(result["detuning"], result["c5transfer"], marker='o', ls='None', color='orchid')
		main_ax.set_title(result["title"], fontsize=10)
		main_ax.legend(fontsize=6)
		main_ax.set_ylabel("transfer")
		main_ax.tick_params(labelbottom=False)  # Hide x labels for main plot

		# Residuals
		resid_ax.plot(result["detuning"], result["residuals"], color='mediumvioletred', marker='.', linestyle='None')
		resid_ax.axhline(0, color="lightgrey", linestyle="--", marker="")
		resid_ax.set_xlabel("detuning (MHz)")
		resid_ax.set_ylabel("resid.")
		resid_ax.tick_params(labelsize=8)

	fig.suptitle(run, y=0.93)
	

# drop NaN rows from dfs
df = df.dropna()

ampslist = sorted([f"{x:.2f}" for x in run_df['Wiggle Amp'].unique()], reverse=True)
rel_amp_A_dict = {}
rel_amp_x0_dict = {}
B_wiggle_to_field_dict = {}

for i in ampslist:
    rel_amp_A = df[df.index == f'0.225_{i}']['A'].iloc[0] - df[df.index == f'0.275_{i}']['A'].iloc[0]
    rel_amp_x0 = df[df.index == f'0.225_{i}']['x0'].iloc[0] - df[df.index == f'0.275_{i}']['x0'].iloc[0]
    
    rel_amp_A_dict[i] = rel_amp_A
    rel_amp_x0_dict[i] = rel_amp_x0
	
    B_wiggle_to_field_dict[i] = B_from_FreqMHz(rel_amp_A_dict[i])
	
heating_dict = {
    "1.80": 1.167,
    '1.35': 1.138,
    '0.90': 1,
    '0.45': 1
}
dict_values = np.array([rel_amp_A_dict[amp] for amp in ampslist])
rel_amp_A_dict = {key: heating_dict[key] * rel_amp_A_dict[key] for i, key in enumerate(ampslist)}

###plotting
colors = plt.cm.magma(np.linspace(0, 1, len(ampslist)))
# peak transfer
fig, (ax0,ax1,ax2) = plt.subplots(3,1, figsize = (6,6), constrained_layout=True)
for (i, amp) in enumerate(ampslist):
    subset = df[df['Wiggle Amp'] == np.array(amp, dtype=float)].copy()
    subset['A'] = subset['A'] * heating_dict[amp]
    error = np.sqrt((subset.eA.iloc[0]**2 + subset.eA.iloc[1]**2))
    ax0.errorbar(subset['Wiggle Time'], subset.A, 
				 yerr=subset.eA, color=colors[i],
				 label = f'Wiggle Amp: {amp}Vpp Rel Amp:{rel_amp_A_dict[amp]:0.3f}({error*1000:0.0f})'
				 )
ax0.set(ylabel = "peak transfer",
		xlabel = 'Time [ms]'
		)
ax0.legend(loc='upper center')


for (i, amp) in enumerate(ampslist):
    subset = df[df['Wiggle Amp'] == np.array(amp, dtype=float)].copy()
    error = np.sqrt((subset.eA.iloc[0]**2 + subset.eA.iloc[1]**2))
    ax1.errorbar(
		# subset['Wiggle Time'], subset.x0,
		amp, 
				 rel_amp_A_dict[amp], 
				 yerr=error, color=colors[i],
				# label = f'Wiggle Amp: {amp}Vpp Rel Amp:{rel_amp_x0_dict[amp]:0.3f}({error*1000:.0f})'
				 )
ax1.set(ylabel='Relative Amp', 
		xlabel = 'Wiggle Amp [Vpp]'
		)
# ax1.legend(loc='upper center')


for (i, amp) in enumerate(ampslist):
    subset = df[df['Wiggle Amp'] == np.array(amp, dtype=float)].copy()
    error = np.sqrt((subset.eA.iloc[0]**2 + subset.eA.iloc[1]**2))
    # error = np.sqrt((subset.ex0[0]**2 + subset.ex0[1]**2))
    ax2.errorbar(
		    B_wiggle_to_field_dict[amp], 
				 rel_amp_A_dict[amp],  
				 yerr=error, 
				 color=colors[i],
				#  label = f'Wiggle Amp: {amp}Vpp Rel Amp:{rel_amp_A_dict[amp]:0.3f}({subset.eA[0]:.3f})'
				 )
ax2.set(
	ylabel = 'Relative Amp',
	xlabel = 'B Field [G]'
)

