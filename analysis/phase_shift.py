"""
Calculates phase shift between amplitude/frequency/contact/field for modulated field from dimer spectra, or the contact given static field.

Plots dimer spectra for each wiggle time (ac) or field value (dc) if show_dimer_plot==True.

Assumes fixed sinc^2 width for each dimer fit, and 0 background transfer.

Set plot_dc==True to plot dc field points along with ac fits, given dc_cal_path

Add times to dropped_list to exclude. times in dropped_list must match wiggle times in df

Remember to change field_cal run name to match wiggle run for correct B plot!

TO DO:
- Put dropped_list in metadata somehow
"""

# settings for directories, standard packages...
from preamble import *
from get_metadata import metadata

# runs = ["2025-09-24_E", "2025-10-01_L","2025-10-17_E","2025-10-17_M","2025-10-18_O","2025-10-20_M",
# 		"2025-10-21_H", "2025-10-23_R","2025-10-23_S"]
# have to put run into metadata first; use get_metadata.py to fill
run = "2025-10-20_M"
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
# Load CCC data for f0 to B conversion
CCC = pd.read_csv("theory/ac_s_Eb_vs_B_220-225G.dat", header=None, names=['B', 'f0'], delimiter='\s')

# load wiggle field calibration
field_cal_run = meta_df['field_cal_run'].values[0]
field_cal_df_path = os.path.join(field_cal_folder, "field_cal_summary.csv")
field_cal_df = pd.read_csv(field_cal_df_path)
field_cal = field_cal_df[field_cal_df['run'] == field_cal_run]
if len(field_cal) > 1:
	field_cal =field_cal[field_cal['pulse_length']==40]
print(f'Field cal run being used is {field_cal_run}')

show_dimer_plot = True
Export = True
amp_cutoff = 0.01 # ignore runs with peak transfer below 0.01
plot_dc = False # whether or not to plot DC field points from DC_cal_csv
avg_dimer_spec = False # whether or not to average detunings before fitting dimer spectrum
fix_width = True # whether or not dimer spectra sinc2 fits have a fixed width
plot_bg = True # whether or not to plot background points and fit
single_shot = True
track_bg = True
rerun = False

def line(x, m, b):
	return m*x+b

def f0_to_B_CCC(x):
	"""
	Converts f0 to B using CCC data file.
	"""
	Eb = -x  # Have to invert sign
	return np.interp(Eb, np.array(CCC['f0']), np.array(CCC['B']))

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
columns = ["A", "x0", "eA", "ex0", 'C', 'eC'] if fix_width else ["A", "x0", "width", "eA", "ex0", "ewidth", 'C', 'eC']
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
	title = f'Wiggle Time: {index:0.2f} ms'
	width = 1/pulse_time if fix_width else None
	run_df['OmegaR'] = phaseO_OmegaR(VVA, run_df['freq'])
	if not single_shot:
		data = find_transfer(run_df)
		popts, perrs, plabel, sinc2 = fit_sinc2(data[["detuning", "c5transfer"]].values, 
													width=width)
	else : 
		data = find_transfer(run_df, popts_c5bg)
		# to keep structure consistent, fake the popts and perrs
		popts = [data['c5transfer'].mean(), data['detuning'].mean()]
		epsilon=1e-9
		perrs = [data['c5transfer'].sem(), data['detuning'].sem()+epsilon] #epsilon was needed to avoid divide by zero
		f, p0, paramnames = Sinc2(data[["detuning", "c5transfer"]].values)
		sinc2 = lambda x, A, x0: f(x, A, x0, width,0)
		p0 = p0[:2]
		paramnames = paramnames[:2]
		plabel = fit_label(popts, perrs, paramnames)

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
		"predicted": sinc2(detuning, *popts)
	})

	if popts[0] > amp_cutoff: 
		# C/NkF from amplitude
		Id, e_Id, contact, e_contact = \
			Contact_from_amplitude(popts[0], perrs[0], EF, run_df['OmegaR'].mean(), pulse_time)
		df.loc[index] = np.concatenate([popts, perrs, [contact, e_contact]]) # lists should be concatenated in order
	else:
		 dropped_list.append(index)
		
# field_df.dropna(how='all').to_csv(runpath + f'DC_field_cal.csv',  index=None, sep=',')      

###plot all the dimer fits on one multi grid plot
if show_dimer_plot and valid_results:
	num_plots = len(valid_results)
	cols = 3
	rows = int(np.ceil(num_plots / cols))

	fig = plt.figure(figsize=(4 * cols, 3 * rows))
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
		if single_shot:
			xs = np.linspace(result["detuning"].values[0]-2*width, result["detuning"].values[0]+2*width, 100)
		else:
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
	plt.tight_layout()
	plt.show()

### plot the disitrubtion of fitted widths if not fixed
if valid_results and not fix_width:
	fig, ax = plt.subplots(figsize=(3, 2))
	ax.set(ylabel=r'Fit width $\sigma$ [kHz]',
		   xlabel = 'Wiggle Time [ms]')
	for result in valid_results:
		ax.errorbar(result['time'], result['popts'][2]*1000, result['perrs'][2]*1000, color='crimson', marker='o', ls='')

	ax.hlines(y=(1/pulse_time)*1000, xmin=valid_results[0]['time'], xmax=valid_results[-1]['time'],color='lightcoral', ls='--')
	fig.suptitle(run)

# drop NaN rows from dfs
df = df.dropna()

# Calculate B field from f0
df['B'] = f0_to_B_CCC(df['x0'])
# Estimate error in B from error in f0
df['eB'] = np.abs(f0_to_B_CCC(df['x0'] + df['ex0']) - f0_to_B_CCC(df['x0'] - df['ex0']))/2

###plotting

# df = df[df.index != 0.3] #manually dropping 0.3 ms pt for now


fig = plt.figure(figsize=(8, 10))
# outer grid for each variable
outer = gridspec.GridSpec(3, 1, hspace=0.2)

# inner grid for each panel (main + residual)
gs = [gridspec.GridSpecFromSubplotSpec(
			2, 1, subplot_spec=out_gs,
			height_ratios=[3, 1], hspace=0.05  
			) for i, out_gs in enumerate(outer)]

# gs = gridspec.GridSpec(6, 1, height_ratios=[3,1, 3,1, 3,1], hspace=0.06)

# peak transfer
ax0 = fig.add_subplot(gs[0][0])
ax0.errorbar(df.index, df.A, yerr=df.eA)
ax0.set_ylabel("peak transfer")
ax0.tick_params(labelbottom=False)

# f0
ax1 = fig.add_subplot(gs[1][0])
ax1.errorbar(df.index, df.x0, yerr=df.ex0 )
ax1.set_ylabel("f0 [MHz]")
ax1.tick_params(labelbottom=False)

# contact
ax2 = fig.add_subplot(gs[2][0])
ax2.errorbar(df.index, df.C,
				yerr=df.eC )
ax2.set(ylabel = r'$C/Nk_F$')
ax2.tick_params(labelbottom=False)

# plot fits to sine
###Sine fits for the wiggle data pts 
# sine should be same for both data
poptsA, perrsA, plabelA, sine = fit_fixedSinkHz(df.index, df.A, wiggle_freq, df.eA)
poptsf0, perrsf0, plabelf0, __ = fit_fixedSinkHz(df.index, df.x0, wiggle_freq, df.ex0)
poptsC, perrsC, plabelC, sine_C = fit_fixedSinkHz(df.index, df.C, wiggle_freq, df.eC)

###plotting sin fit to 1/A and f0 and C 
ts = np.linspace(min(df.index), max(df.index), 100)
ax0.plot(ts, sine(ts, *poptsA), ls="-", marker="", 
			label=plabelA
			)
fit1 = ax1.plot(ts, sine(ts, *poptsf0), ls="-", marker="", label=plabelf0)
ax2.plot(ts, sine_C(ts, *poptsC), ls="-", marker="", label=plabelC)

###plotting the Field wiggle in B (G)
B_phase = field_cal['B_phase'].values[0] 
eB_phase = field_cal['e_B_phase'].values[0]
field_params = field_cal[['B_amp', 'B_phase', 'B_offset']].values[0]
Bs = sine(ts, *field_params)

B_label = fit_label(field_cal[['B_amp', 'B_phase', 'B_offset']].values[0], 
					field_cal[['e_B_amp', 'e_B_phase', 'e_B_offset']].values[0],
					["A", "p", "C"])
ax1_B = ax1.twinx()
B1 = ax1_B.plot(ts, Bs, color="cornflowerblue", ls='--', marker="")

ax1_B.set_ylabel("B(cal) [G]")

ax2_B = ax2.twinx()
ax2_B.plot(ts, Bs, color="cornflowerblue", ls='--', marker="")
ax2_B.set_ylabel("B(cal) [G]")

# flip y axis instead of phase shifting
ax1_B.invert_yaxis()
ax2_B.invert_yaxis()

fig.legend(B1, [B_label], loc='upper center', bbox_to_anchor=(1, 0.85), title="field fit")

###plotting residuals
resid0 = fig.add_subplot(gs[0][1], sharex=ax0)
resid1 = fig.add_subplot(gs[1][1], sharex=ax1)
resid2 = fig.add_subplot(gs[2][1], sharex=ax2)

resid0.axhline(0, ls="--", color="lightgrey", marker="")
resid1.axhline(0, ls="--", color="lightgrey", marker="")
resid2.axhline(0, ls="--", color="lightgrey", marker="")

resid0.errorbar(df.index, df.A-sine(df.index, *poptsA), yerr=df.eA, color='mediumvioletred')
resid1.errorbar(df.index, df.x0-sine(df.index, *poptsf0), yerr=df.ex0, color='mediumvioletred')
resid2.errorbar(df.index, df.C-sine(df.index, *poptsC), df.eC, color='mediumvioletred')

resid0.set_xlabel("time (ms)")
resid1.set_xlabel("time (ms)")
resid2.set_xlabel("time (ms)")


ax0.legend(loc=0)
ax1.legend(loc=0)
ax2.legend(loc=0)

# reorder axes so legend is not covered by B line
ax1.set_zorder(2)
ax1.patch.set_visible(False)
ax2.set_zorder(2)
ax2.patch.set_visible(False)

###Finding the phase shift by subtracting various fits
# add pi offset to B field to compare phase shift
phases = np.array([poptsA[1] - poptsf0[1],
		poptsf0[1] - (B_phase + np.pi),
		poptsC[1] - (B_phase+np.pi)
])
ephases = np.sqrt([
	(perrsA[1]**2 + perrsf0[1]**2),
	(perrsf0[1]**2 + eB_phase**2),
	(perrsC[1]**2 + eB_phase**2)
	])

# convert angles from [-2pi, 2pi] to between [-pi, pi]
for i, p in enumerate(phases):
	# prop error?
	phases[i] = (p + np.pi) % (2 * np.pi) - np.pi 

try:
	fits = fit_label(phases, ephases, [r"$\phi$ for $C$ - $E_\mathrm{d}$", 
									r"$\phi$ for $E_\mathrm{d}$ - $B$ cal.", 
									r"$\phi$ for $C$ - $B$ cal."], 
						units = ["rad", r"rad", r"rad"], digits=2)
except:
	# errors not available
	fits = (f"phase shift A-f0 = {phases[0]:.2f}" 
			f"\nphase shift f0-B = {phases[1]:.3f}" 
			f"\nphase shift C-B = {phases[2]:.2f}")
	
fig.suptitle(f"{run}\n{pulse_time}us Pulse, {wiggle_freq}kHz Modulation, {VVA} VVA\n{fits}" +\
			f'\nDropped Wiggle Times: {[float(x) for x in dropped_list]}', y=1.02)

###
### Condensed plot with just f0->B and 
### 

fig = plt.figure(figsize=(8, 4))
#  grid for each panel (main + residual)
gs = gridspec.GridSpec(
			2, 1, height_ratios=[3, 1], hspace=0.05)


ax = fig.add_subplot(gs[0])

# B field on secondary axis
ax_B = ax.twinx()
ax_B.errorbar(df.index, df.B, yerr=df.eB, marker='s', 
			color='cornflowerblue', markerfacecolor='white')
ax_B.set_ylabel("B [G]", color='cornflowerblue')
ax_B.tick_params(labelbottom=False)

# contact
ax.errorbar(df.index, df.C, yerr=df.eC, marker='o', 
			color='mediumvioletred')
ax.set_ylabel(r'$C/Nk_F$', color='mediumvioletred')
ax.tick_params(labelbottom=False)


# plot fits to sine
###Sine fits for the wiggle data pts 
# sine should be same for both data
poptsB, perrsB, plabelB, __ = fit_fixedSinkHz(df.index, df.B, wiggle_freq, df.eB, param_labels=[r"$B_\mathrm{amp}$", r"$\phi$", r"$B_0$"])
poptsC, perrsC, plabelC, sine_C = fit_fixedSinkHz(df.index, df.C, wiggle_freq, df.eC, param_labels=[r"$C_\mathrm{amp}$", r"$\phi$", r"$C_\mathrm{eq}$"])

###plotting sin fit C 
ts = np.linspace(min(df.index), max(df.index), 100)
ax.plot(ts, sine_C(ts, *poptsC), ls="-", color='mediumvioletred',
		marker="", label=plabelC)
ax_B.plot(ts, sine(ts, *poptsB), ls="-", color='cornflowerblue',	
			marker="", label=plabelB)

###plotting the Field wiggle in B (G)
B_phase = field_cal['B_phase'].values[0] 
eB_phase = field_cal['e_B_phase'].values[0]
field_params = field_cal[['B_amp', 'B_phase', 'B_offset']].values[0]
Bs = sine(ts, *field_params)

B_cal_label = fit_label(field_cal[['B_amp', 'B_phase', 'B_offset']].values[0], 
					field_cal[['e_B_amp', 'e_B_phase', 'e_B_offset']].values[0],
					[r"$B_\mathrm{amp}$", r"$\phi$", r"$B_0$"])

B_cal = ax_B.plot(ts, Bs, color="cornflowerblue", ls='--', marker="")

# flip y axis instead of phase shifting
ax_B.invert_yaxis()

fig.legend(B_cal, [B_cal_label], loc='upper center', bbox_to_anchor=(1, 1.1), title="field cal")

###plotting residuals
resid = fig.add_subplot(gs[1], sharex=ax)

resid.axhline(0, ls="--", color="lightgrey", marker="")

resid.errorbar(df.index, df.C-sine(df.index, *poptsC), df.eC, color='mediumvioletred')
resid.errorbar(df.index, df.B-sine(df.index, *poptsB), yerr=df.eB, 
			marker='s', color='cornflowerblue', markerfacecolor='white')

resid.set_xlabel("Time (ms)")

ax.legend(loc=0)
ax_B.legend(loc=0)

# reorder axes so legend is not covered by B line
ax.set_frame_on(False)
ax.set_zorder(2)
# ax.patch.set_visible(False)

###Finding the phase shift by subtracting various fits
# add pi offset to B field to compare phase shift
phases = np.array([poptsA[1] - poptsB[1] + np.pi,
		poptsB[1] - (B_phase),
		poptsC[1] - (B_phase) + np.pi
])
ephases = np.sqrt([
	(perrsA[1]**2 + perrsB[1]**2),
	(perrsB[1]**2 + eB_phase**2),
	(perrsC[1]**2 + eB_phase**2)
	])

# convert angles from [-2pi, 2pi] to between [-pi, pi]
for i, p in enumerate(phases):
	# prop error?
	phases[i] = (p + np.pi) % (2 * np.pi) - np.pi 

fits = fit_label(phases, ephases, [r"$\phi$ for $C$ - $E_\mathrm{d}$", 
									r"$\phi$ for $E_\mathrm{d}$ - $B$ cal.", 
									r"$\phi$ for $C$ - $B$ cal."], 
						units = ["rad", r"rad", r"rad"], digits=2)
	
fig.suptitle(f"{run}\n{pulse_time}us Pulse, {wiggle_freq}kHz Modulation, {VVA} VVA\n{fits}" +\
			f'\nDropped Wiggle Times: {[float(x) for x in dropped_list]}', y=1.3
			)

# fig.tight_layout()
	
if Export == True and fix_width == True: # this complains when fix_width is false,  because there are mismatched num of params now
	csv_path = os.path.join(analysis_folder, f'phase_shift_2025_summary.csv')
	write_header = not os.path.exists(csv_path)
	run_id = run

	if os.path.exists(csv_path):
		existing_df = pd.read_csv(csv_path, index_col=0)
		already_logged = run_id in existing_df.index
	else:
		already_logged = False
	
	if not already_logged:
		csv_df = pd.DataFrame({
			'Pulse Time (us)': pulse_time,
			'Modulation Freq (kHz)': wiggle_freq,
			'Modulation Amp (Vpp)': wiggle_amp,
			'Phase Shift A-f0 (rad)': phases[0],
			'Phase Shift A-f0 err (rad)': ephases[0],
			'Phase Shift f0-B (rad)': phases[1],
			'Phase Shift f0-B err (rad)': ephases[1],
			'Phase Shift C-B (rad)': phases[2],
			'Phase Shift C-B err (rad)': ephases[2],
			'Dropped Wiggle Times (ms)': str([float(x) for x in dropped_list]),
			'Sin Fit of A': [poptsA],
			'Error of Sin Fit of A': [perrsA],   
			'Sin Fit of C': [poptsC],
			'Error of Sin Fit of C': [perrsC],
			'Sin Fit of f0':[poptsf0],
			'Error of Sin Fit of f0':[perrsf0],
			'TRACK_BG':track_bg,
			'FIX_WIDTH':fix_width,
			'SINGLE_SHOT':single_shot
		}, index=[run_id])  

		csv_df.to_csv(csv_path, mode='a', header=write_header, index=True, sep=',')
		print(f"✅ Appended run '{run_id}' to {csv_path}")
	else:
		if rerun: 
			print(f'rerunning ')
		else: 	
			print(f"⚠️ Run '{run_id}' already logged. Skipping append.")
