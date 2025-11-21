"""
Calculates phase shift between amplitude/frequency/contact/field for modulated field from dimer spectra, or the contact given static field.

Plots dimer spectra for each wiggle time (ac) or field value (dc) if show_dimer_plot==True.

Assumes fixed sinc^2 width for each dimer fit, and 0 background transfer.

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
run = "2025-11-20_F"
#CONTROLS
SHOW_INTERMEDIATE_PLOTS = True
Export = True
amp_cutoff = 0.01 # ignore runs with peak transfer below 0.01
avg_dimer_spec = False # whether or not to average detunings before fitting dimer spectrum
fix_width = True # whether or not dimer spectra sinc2 fits have a fixed width
plot_bg = True # whether or not to plot background points and fit
single_shot = True
track_bg = False
rerun = False
hijack_freq = True  # Sets HFT detuning to 150kHz no matter what data.freq says.

#CORRECTIONS
CORR_PULSECONV = True
CORR_SAT = True
CORR_CFUDGE = False
sat_scale_df = pd.read_csv(os.path.join(root_analysis, "corrections//saturation_HFT.csv"))

# this fudges the Rabi calibrated at 47 MHz for the attenuation at 43, 
# but a calibration directly at 43 would be better
RabiperVpp47 = 13.05 / 0.500 # kHz/Vpp on scope 2025-10-21
e_RabiperVpp47 = 0.22
phaseO_OmegaR = lambda VVA, freq: 2*pi*RabiperVpp47 * Vpp_from_VVAfreq(VVA, freq)

# get metadata info
meta_df = pd.read_csv('metadata.csv')
meta_df = meta_df[meta_df['run']==run]
if meta_df.empty:
    print(f'no meta data for {run}, running now')
    m = metadata([run], [[]], True, "metadata.csv", "run from phase_shift.py")
    m.output()
    meta_df = pd.read_csv('metadata.csv')
    # print(meta_df)
    meta_df = meta_df[meta_df['run']==run]

# read HFT vs dimer settings
is_HFT = meta_df["is_HFT"].values[0] #else "dimer"
pulse_type =  meta_df["pulse_type"]
pulse_area_corr = np.sqrt(0.305) if (pulse_type.values[0] == "blackman") else 1

# put this into metadata?
xlabel = 'Times [ms]'

wiggle_amp = meta_df['Vpp'].values[0]  # Vpp
dropped_list = np.fromstring(meta_df['drop'].values[0].strip('[]'), sep= ' ') # list of time stamps to drop
pulse_time = meta_df['pulse_time'].values[0] # 
wiggle_freq = meta_df['freq'].values[0] # kHz
VVA = meta_df['VVA'].values[0] # assumes just the max VVA for everything
ToTF = meta_df['ToTF'].values[0] 
EF = meta_df['EF'].values[0]/h # Hz
# Load CCC data for f0 to B conversion
CCC = pd.read_csv("theory/ac_s_Eb_vs_B_220-225G.dat", header=None, names=['B', 'f0'], delimiter=r'\s+')

# load wiggle field calibration
field_cal_run = meta_df['field_cal_run'].values[0]
field_cal_df_path = os.path.join(field_cal_folder, "field_cal_summary.csv")
field_cal_df = pd.read_csv(field_cal_df_path)
field_cal = field_cal_df[field_cal_df['run'] == field_cal_run]
if len(field_cal) > 1:
	field_cal = field_cal[field_cal['pulse_length'] == 40]
print(f'Field cal run being used is {field_cal_run}')

def line(x, m, b):
	return m*x+b

def sine(x, omega, A, p, c):
	return A*np.sin(omega*x-p) + c
	
# I want to move this somewhere else. Maybe put it in preamble?
def saturation_scale(x, x0):
	""" x is OmegaR^2 and x0 is fit 1/e Omega_R^2 """
	return x/x0*1/(1-np.exp(-x/x0))

def f0_to_B_CCC(x):
	"""
	Converts f0 to B using CCC data file.
	"""
	Eb = -x  # Have to invert sign
	return np.interp(Eb, np.array(CCC['f0']), np.array(CCC['B']))

def find_transfer(data, popts_c5bg=np.array([])):
	"""
	given data df output from matlab containing atom counts, returns new df containing detuning, 
	5 and 9 counts, and transfer/loss
	TODO account for tracking bg over whole scan or not
	"""
	run_data = data.data[["cyc","detuning", "VVA", "c5", "c9"]].copy()

	# assumes popts_c5bg is a np array
	if popts_c5bg.any():
		run_data['c5bg'] = line(run_data['cyc'], *popts_c5bg)
		run_data["c5transfer"] = (1-run_data["c5"]/run_data["c5bg"])/2
		data = run_data[run_data["VVA"]!=0].copy()
	else:
		bg = data.data[data.data["VVA"] == 0]

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

		data = run_data[data.data["VVA"] != 0].copy()
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

def fit_fixedSinkHz(t, y, run_freq, eA, p0=None, param_labels=None):
	"""
	need docstring, and maybe a better way to code this
	t in ms, freq in kHz
	"""
	f = lambda t, A, p, c: sine(t, run_freq*2*np.pi, A, p, c)

	if p0 == None:
		A = (max(y)-min(y))/2
		C = (max(y)+min(y))/2
		p = np.pi
		p0 = [A, p, C]
	
	if np.any(eA != 0):
		# fit to fixed frequency
		popts, pcov = curve_fit(f, t, y, p0, bounds=([0, 0, 0], [np.inf, 2*np.pi, np.inf]), sigma=eA)
		perrs = np.sqrt(np.diag(pcov))

	else: 
		popts, pcov = curve_fit(f, t, y, p0, bounds=([0, 0, 0], [np.inf, 2*np.pi, np.inf]))
		perrs = np.sqrt(np.diag(pcov))

	if param_labels == None:
		param_labels=["A", "p", "C"]
	plabel = fit_label(popts, perrs, param_labels)

	return popts, perrs, plabel, f

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
runname = datfiles[0].split(os.sep)[-2].lower() # get run folder name, should be same for all files
# calculate bg over time; needs to load all the datfiles first
if track_bg:
	popts_c5bg, perrs_c5bg = bg_over_scan(datfiles, plot=True)
else: 
	popts_c5bg = np.array([])
	
# set up dataframe to hold dimer fits
columns = ["A", "x0", "eA", "ex0", 'C', 'eC', 'B', 'eB'] if fix_width else ["A", "x0", "width", "eA", "ex0", "ewidth", 'C', 'eC', 'B', 'eB']
df = pd.DataFrame( columns =columns,  dtype=float)
df.index.name = "middle_pulse_time"
dropped_list = []
valid_results = [] # only for dimer meas.

for i, fpath in enumerate(datfiles):
	filename = fpath.split(os.sep)[-1]
	print(filename)

	data = Data(filename, path=fpath)
	if i==0:
		time_column_name = data.find_column('wiggle') # find which col has the wiggle time, only run once
	
	
	if data.data[time_column_name][0] in dropped_list:
		print(f'dropping time at {data.data[time_column_name][0]}')
		continue

	if is_HFT:
		# adjust time to be at centre of pulse
		middle_pulse_time = data.data[time_column_name][0] + (pulse_time/1000)/2 + 0.001 # ms, should be same for all cycles
		title = f'Mid Pulse Time: {middle_pulse_time:0.2f} ms'
		width = 1/pulse_time if fix_width else None
		data.data['EF'] = EF # Hz
		data.data['trf'] = pulse_time / 1e6 # s

		if hijack_freq == True:
			data.data['freq'] = 47.2227 + 0.150

		data.analysis(pulse_type=pulse_type.values[0])
		if CORR_PULSECONV:
			corr_pulseconv = 1/1.12 # TODO: EXTRACT FROM ACTUAL SIMULATION DATA
			data.data[['alpha_HFT', 'scaledtransfer_HFT', 'contact_HFT']] *= corr_pulseconv
			data.data['CORR_PULSECONV'] = True # flag for ease of checking
			print(f'Applied pulse convolution correction of {corr_pulseconv:.2f}')

		if CORR_SAT:
			# We saw saturation was field-specific. This is a rough estimate to apply to all Bfields. 
			# TODO: Redo with better data.
			mean_sat_scale = sat_scale_df['OmegaR2_sat'].mean()
			e_mean_sat_scale = np.sqrt((sat_scale_df['e_OmegaR2_sat']**2).sum())/3 #unused
			# TODO: distribution is not normal; use MonteCarlo methods (see HFT_dimer_bg_analysis for example)
			data.data['corr_sat'] = saturation_scale(data.data['OmegaR2'], mean_sat_scale)
			data.data.loc[:, ['alpha_HFT', 'scaledtransfer_HFT', 'contact_HFT']] = \
				(data.data[['alpha_HFT', 'scaledtransfer_HFT', 'contact_HFT']].multiply(\
					data.data['corr_sat'], axis=0)) # the explicitness here avoids in-place broadcasting errors
			data.data['CORR_SAT'] = True # flag for checking
			print(f'Applied saturation correction of approximately = {data.data.corr_sat.mean():.2f}')
			
		if CORR_CFUDGE:
			# this fudge currently uses the B-field as the input parameter, but this is not general. It should really be a fudge for TTF.
			# It just so happens that the calibration set and our measurements are at similar TTFs.
			# TODO make general by fudging for TTF
			cfudge_df = pd.read_csv(os.path.join(root_analysis, "corrections//saturation_HFT.csv"))
			field_params = field_cal[['B_amp', 'B_phase', 'B_offset']].values[0]
			thisB = sine(middle_pulse_time, wiggle_freq*2*np.pi, *field_params) # omega in kHz
			corr_interp = interp1d(cfudge_df['Bfield'], cfudge_df['C_corrfactor'], fill_value="extrapolate")
			corr_cfudge = corr_interp(thisB)
			data.data['corr_cfudge'] = corr_cfudge
			data.data.loc[:, ['alpha_HFT', 'scaledtransfer_HFT', 'contact_HFT']] = \
				(data.data[['alpha_HFT', 'scaledtransfer_HFT', 'contact_HFT']].multiply(\
					data.data['corr_cfudge'], axis=0)) # the explicitness here avoids in-place broadcasting errors
		
			data.data['CORR_CFUDGE'] = True
			print(f'Applied C(TTF) fudge correction of {corr_cfudge:.2f} at B={thisB:.2f}G')

		if i == 6: test = data
		if filename == '2025-11-20_C_e_Wiggle Time=0.09.dat':
			data.data = data.data.iloc[:-1]
		if filename == '2025-11-20_C_e_Wiggle Time=0.13.dat':
			data.data = data.data.iloc[:-1]
		if filename == '2025-11-20_C_e_Wiggle Time=0.17.dat':
			data.data = data.data.iloc[:-1]
		data.group_by_mean('freq')
		data_avg = data.avg_data
		# df_list.append(data.avg_data)
		# if data_avg.empty: continue
		df.loc[middle_pulse_time, ["A", "x0", "eA", 'C', 'eC']] = data_avg[["alpha_HFT", "freq", "em_alpha_HFT", 'contact_HFT', 'em_contact_HFT']].values
		df.loc[middle_pulse_time,  "ex0"] = 0
		df.loc[middle_pulse_time, 'B'] = data_avg["field"].values
		df.loc[middle_pulse_time, 'eB'] = 0 # TODO
		df.loc[middle_pulse_time, ['c9_var', 'c5_var']] = [np.var(data.data['c9']), np.var(data.data['c5'])]
		df.loc[middle_pulse_time, ['c9bg_var', 'c5bg_var']] = data_avg[[ "c9bg_var", 'c5bg_var']].values
		df.loc[middle_pulse_time, 'number shots'] = len(data.data['c9'])


		
	else:
		# adjust time to be at centre of pulse
		middle_pulse_time = data.data[time_column_name][0] + (pulse_time/1000)/2  # ms, should be same for all cycles
		title = f'Mid Pulse Time: {middle_pulse_time:0.2f} ms'
		width = 1/pulse_time if fix_width else None
		data.data['EF']=EF # Hz
		data.data['trf']=pulse_time / 1e6 # s
		data.data['OmegaR'] = phaseO_OmegaR(VVA, data.data['freq'])

		if not single_shot:
			dimerdata = find_transfer(data)
			popts, perrs, plabel, sinc2 = fit_sinc2(dimerdata[["detuning", "c5transfer"]].values, 
														width=width)
		else : 
			dimerdata = find_transfer(data, popts_c5bg)
			# to keep structure consistent, fake the popts and perrs
			popts = [dimerdata['c5transfer'].mean(), dimerdata['detuning'].mean()]
			epsilon=1e-9
			perrs = [dimerdata['c5transfer'].sem(),dimerdata['detuning'].sem()+epsilon] #epsilon was needed to avoid divide by zero
			f, p0, paramnames = Sinc2(dimerdata[["detuning", "c5transfer"]].values)
			sinc2 = lambda x, A, x0: f(x, A, x0, width,0)
			plabel = fit_label(popts, perrs, paramnames[:2])

		# calculate GammaTilde
		dimerdata['GammaTilde'] = GammaTilde(dimerdata['c5transfer'], 
									h*EF, 
									data.data['OmegaR'].values[0]*1e3,
									pulse_time/1e6)
		detuning, c5transfer, c5gammatilde = dimerdata["detuning"], dimerdata["c5transfer"], dimerdata["GammaTilde"]

		# Save for later plotting
		valid_results.append({
			"time":middle_pulse_time, 
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
				Contact_from_amplitude(popts[0], perrs[0], EF, data.data['OmegaR'].mean(), pulse_time)

			df.loc[middle_pulse_time, ["A", "x0", "eA", "ex0"]] = np.concatenate([popts, perrs]) # lists should be concatenated in order
			df.loc[middle_pulse_time, ['C', 'eC']] = [contact, e_contact]

			# Estimate error in B from error in f0
			x0 = popts[1]
			ex0 = perrs[1]
			df.loc[middle_pulse_time, ['B', 'eB']] = [f0_to_B_CCC(x0), 
												np.abs(f0_to_B_CCC(x0 + ex0) - f0_to_B_CCC(x0 - ex0))/2]
		
		else:
			dropped_list.append(middle_pulse_time)

cmap = plt.colormaps.get_cmap('viridis')
colours = cmap(np.linspace(0, 1, len(df.index)))
if is_HFT:

	fig, ax = plt.subplots(1, 2, figsize=(8,3.5))
	for i, j in enumerate(df.index):
		row = df.loc[j]
		ax[0].plot(row['number shots'],row['c9bg_var']/row['c9_var'],
			 color=colours[i],
			 label = f'{np.round(j, 3)}'
			 )
		ax[1].plot(row['number shots'],row['c5bg_var']/row['c5_var'], color=colours[i])
	ax[0].legend(loc='upper left')
	
	ax[0].set(
		ylabel = 'var(c9bg)/var(c9)',
		xlabel = 'number shots'
	)
	ax[1].set(
		ylabel = 'var(c5bg)/var(c5)',
		xlabel = 'number shots'
	)
	fig.tight_layout()

###plot all the dimer fits on one multi grid plot
if (not is_HFT) and SHOW_INTERMEDIATE_PLOTS and valid_results:
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

###plotting
fig = plt.figure(figsize=(8, 10))
# outer grid for each variable
outer = gridspec.GridSpec(3, 1, hspace=0.2)

# inner grid for each panel (main + residual)
gs = [gridspec.GridSpecFromSubplotSpec(
			2, 1, subplot_spec=out_gs,
			height_ratios=[3, 1], hspace=0.05  
			) for i, out_gs in enumerate(outer)]

axs = [fig.add_subplot(gs[i][0]) for i in range(3)]
axrs = [fig.add_subplot(gs[i][1], sharex = ax) for i, ax in enumerate(axs)]

[ax.tick_params(labelbottom=False) for ax in axs]
ax0, ax1, ax2 = axs # for easier calling

def plot_and_fit_sine(df, attr, freq, ax, ax_resid, ylabel, param_labels=None, 
					  legend_loc=0, **kwargs):
	"""
	given attr in df, plots residuals fit to f with params popts.
	"""
	# get data
	xs = df.index
	ys = df[attr]
	eys = df[f"e{attr}"]

	# plot data
	ax.errorbar(xs, ys, yerr = eys, **kwargs)
	color = kwargs.get("color")
	if color != None: ax.set_ylabel(ylabel, color=color) 
	else: ax.set_ylabel(ylabel)

	# fit
	popts, perrs, plabel, sine_fit = fit_fixedSinkHz(xs, ys, freq, eys, param_labels=param_labels)

	# plot fit
	ts = np.linspace(min(xs), max(xs), 100)
	ax.plot(ts, sine_fit(ts, *popts), ls="-", marker="", 
				label=plabel, color=kwargs.get("color")
				)
	
	# plot residuals 
	ys_resid = ys - sine_fit(xs, *popts)

	if color == None:  ax_resid.errorbar(xs, ys_resid, yerr=eys, color="mediumvioletred", **kwargs)
	else: ax_resid.errorbar(xs, ys_resid, yerr=eys, **kwargs)
	ax_resid.axhline(0, ls="--", color="lightgrey", marker="")

	# formatting
	ax.tick_params(labelbottom = False)
	ax_resid.set_xlabel("time (ms)")
	ax.legend(loc=legend_loc)
	ax.ticklabel_format(useOffset=False, style='plain')

	return popts, perrs, plabel, sine_fit

def make_plot_title(fig, run, pulse_time, wiggle_freq, dropped_list, 
					popts_list, perrs_list, B_phase, eB_phase, plabels, y=1.02):

	###Finding the phase shift by subtracting various fits
	# add pi offset to B field to compare phase shift
	phases = np.array([popts_list[0]-popts_list[1]  + np.pi,
			popts_list[1] - (B_phase),
			popts_list[2] - (B_phase)  + np.pi
	])
	ephases = np.sqrt([
		(perrs_list[0]**2 + perrs_list[1]**2),
		(perrs_list[1]**2 + eB_phase**2),
		(perrs_list[2]**2 + eB_phase**2)
		])

	# convert angles from [-2pi, 2pi] to between [-pi, pi]
	for i, p in enumerate(phases):
		# prop error?
		phases[i] = (p + np.pi) % (2 * np.pi) - np.pi 

	try:
		fits = fit_label(phases, ephases,plabels, 
							units = ["rad", r"rad", r"rad"], digits=2)
	except:
		# errors not available
		# generalize this later
		fits = (f"phase shift A-f0 = {phases[0]:.2f}" 
				f"\nphase shift f0-B = {phases[1]:.3f}" 
				f"\nphase shift C-B = {phases[2]:.2f}")
		
	fig.suptitle(f"{run}\n{pulse_time}us Pulse, {wiggle_freq}kHz Modulation, {VVA} VVA\n{fits}" +\
				f'\nDropped Wiggle Times: {[float(x) for x in dropped_list]}', y=y)

	return phases, ephases

poptsA, perrsA, plabelA, sine_A = plot_and_fit_sine(df, "A", wiggle_freq, ax0, 
													axrs[0], ylabel="peak transfer")

poptsf0, perrsf0, plabelf0, __ = plot_and_fit_sine(df, "x0", wiggle_freq, ax1, 
												   axrs[1], ylabel="f0 [MHz]")

poptsC, perrsC, plabelC, sine_C = plot_and_fit_sine(df, "C", wiggle_freq, ax2, 
													axrs[2], ylabel=r'$C/Nk_F$')

ts = np.linspace(min(df.index), max(df.index), 100)

###plotting the Field wiggle in B (G)
B_phase, eB_phase = field_cal[['B_phase', 'e_B_phase']].values[0]
field_params = field_cal[['B_amp', 'B_phase', 'B_offset']].values[0]
Bs = sine_A(ts, *field_params)

B_label = fit_label(field_cal[['B_amp', 'B_phase', 'B_offset']].values[0], 
					field_cal[['e_B_amp', 'e_B_phase', 'e_B_offset']].values[0],
					[r"$B_\mathrm{amp}$", r"$\phi$", r"$B_0$"])

B_axs = [ax.twinx() for ax in [ax1, ax2]]
[ax.invert_yaxis() for ax in B_axs] # flip y axis instead of phase shifting
B_plots = [ax.plot(ts, Bs, color="cornflowerblue", ls='--', marker="") for ax in B_axs]
[ax.set_ylabel("B(cal) [G]") for ax in B_axs]

fig.legend(B_plots[0], [B_label], loc='upper center', bbox_to_anchor=(1, 0.85), title="field fit")

# reorder axes so legend is not covered by B line
ax1.set_zorder(2)
ax1.patch.set_visible(False)
ax2.set_zorder(2)
ax2.patch.set_visible(False)

phases, ephases = make_plot_title(fig, run, pulse_time, wiggle_freq, 
				dropped_list, [poptsA[1], poptsf0[1], poptsC[1]], [perrsA[1], perrsf0[1], perrsC[1]], 
				B_phase, eB_phase,
				 [r"$\phi$ for $C$ - $E_\mathrm{d}$", 
										r"$\phi$ for $E_\mathrm{d}$ - $B$ cal.", 
										r"$\phi$ for $C$ - $B$ cal."])

###
### Condensed plot with just f0->B and 
### 

fig = plt.figure(figsize=(8, 4))
#  grid for each panel (main + residual)
gs = gridspec.GridSpec(
			2, 1, height_ratios=[3, 1], hspace=0.05)

ax = fig.add_subplot(gs[0])
resid = fig.add_subplot(gs[1], sharex=ax)

# B field on secondary axis
ax_B = ax.twinx()
ax_B.invert_yaxis() # flip y axis instead of phase shifting

poptsC, perrsC, plabelC, sine_C = plot_and_fit_sine(df, "C", wiggle_freq, ax, resid, ylabel=r"$C/Nk_F$", 
				  param_labels=[r"$C_\mathrm{amp}$", r"$\phi$", r"$C_\mathrm{eq}$"], marker='o', 
				  color='mediumvioletred')
poptsB, perrsB, plabelB, __ = plot_and_fit_sine(df, "B", wiggle_freq, ax_B, resid, legend_loc=1, ylabel="B [G]", 
				  param_labels=[r"$B_\mathrm{amp}$", r"$\phi$", r"$B_0$"], marker='s', 
 				  color='cornflowerblue', markerfacecolor='white')

fig.legend(B_plots[0], [B_label], loc='upper center', bbox_to_anchor=(1, 1.1), title="field cal")

phases, ephases = make_plot_title(fig, run, pulse_time, wiggle_freq, 
				dropped_list, [poptsA[1], poptsB[1], poptsC[1]], [perrsA[1], perrsB[1], perrsC[1]], 
				B_phase, eB_phase,
				 [r"$\phi$ for $C$ - $E_\mathrm{d}$", 
					r"$\phi$ for $E_\mathrm{d}$ - $B$ cal.", 
					r"$\phi$ for $C$ - $B$ cal."], y=1.3)

fig.tight_layout()

if Export == True and fix_width == True: # this complains when fix_width is false,  because there are mismatched num of params now
	root_analysis2 = os.path.join(root_analysis, 'analysis')
	csv_path = os.path.join(root_analysis2, f'phase_shift_2025_summary.csv')
	write_header = not os.path.exists(csv_path)
	run_id = run

	if os.path.exists(csv_path):
		existing_df = pd.read_csv(csv_path, index_col=0)
		already_logged = run_id in existing_df.index
	else:
		already_logged = False
	HFT_or_dimer = 'HFT' if is_HFT else 'dimer'
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
			'SINGLE_SHOT':single_shot,
			'HFT_or_dimer':HFT_or_dimer
		}, index=[run_id])  

		csv_df.to_csv(csv_path, mode='a', header=write_header, index=True, sep=',')
		print(f"✅ Appended run '{run_id}' to {csv_path}")
	else:
		if rerun: 
			print(f'rerunning ')
		else: 	
			print(f"⚠️ Run '{run_id}' already logged. Skipping append.")
