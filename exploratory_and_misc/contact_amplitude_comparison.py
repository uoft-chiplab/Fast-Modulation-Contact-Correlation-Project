'''
This is an exploratory (temp) script that compares the contact amplitude from static B-field runs
with the amplitude from oscillating B-field runs.
Data is currently focused on recent (Sep/Oct 2025) data for a 10 kHz 1.8 Vpp drive.
'''
import os
import sys
root_project = os.path.dirname(os.getcwd())

# Fast-Modulation-Contact-Correlation-Project\analysis
module_folder = os.path.join(root_project, "analysis")
if module_folder not in sys.path:
	sys.path.append(module_folder)
# settings for directories, standard packages...
from preamble import *
from scipy.interpolate import interp1d
from scipy.interpolate import make_smoothing_spline
### FLAGS
TALK = 1

### NUMBERS
# runs
runs = [
	'2025-10-03_N',
		'2025-10-03_P',
		'2025-10-15_L',
		'2025-10-15_M',
		'2025-10-15_J',
		'2025-10-15_H'
		] # date and letter, here 202.24 and 202.04 G
# kept constant for both runs
pulse_time = 20
VVA = 4.5 
# From Q_UShots; just estimates
EF = 10.5 # kHz
ToTF = 0.27
Num = 8520

### FUNCTION DEFINITIONS
def find_transfer(df):
	"""
	given df output from matlab containing atom counts, returns new df containing detuning, 
	5 and 9 counts, and transfer/loss.
	
	"""
	try: 
		run_data = df[["detuning", "VVA", "c5", "c9"]]
	except:
		df['detuning'] = - df['freq'] + 47.2227
		run_data = df[["detuning", "VVA", "c5", "c9"]]

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

	TODO: keep background fixed at 0
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

CCC_df = pd.read_csv('ac_s_Eb_vs_B_220-225G.dat',
	delimiter='\s', header=None)
CCC_df.columns = ['B', 'E_MHz']
Eb_to_B_CCC = interp1d(CCC_df['E_MHz'], CCC_df['B'])

Eb_exp_df = pd.read_excel('Eb_results.xlsx')
# Eb_exp_df = Eb_exp_df[Eb_exp_df['sat'] != 1]
Eb_exp_df = Eb_exp_df.sort_values(by=['Eb'])
# Eb_to_B_exp = interp1d(Eb_exp_df['Eb'], Eb_exp_df['B'])
Eb_to_B_exp = make_smoothing_spline(Eb_exp_df['Eb'], Eb_exp_df['B'], lam = None)

Ebs = np.linspace(-4.09, -3.9, 100)
B_interps = [Eb_to_B_CCC(Ebs), Eb_to_B_exp(Ebs)]
fig, ax = plt.subplots()
ax.plot(Ebs, B_interps[0], ls='-', marker='', color='chartreuse')
ax.plot(CCC_df['E_MHz'], CCC_df['B'], marker='o', color='chartreuse', ls='', label='CCC')
ax.plot(Ebs, B_interps[1], ls='-', marker='',color='crimson')
ax.plot(Eb_exp_df['Eb'], Eb_exp_df['B'], label='Exp')
ax.set(xlabel='Eb [MHz]', ylabel='B [G]', xlim=[-4.1, -3.8], ylim=[201, 203])
ax.legend()


valid_results = []
# analysis loop
for run in runs:
	# find data files
	y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
	runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
	datfiles = glob(f"{runpath}*.dat")
	runname = datfiles[0].split("\\")[-2].lower() # get run folder name, should be same for all files
	print(f'Run name = {runname}')

	try:
		Bfield = float(runname[-6:].replace('p','.'))
	except:
		Bfield = float(runname[:runname.find("g")].split("_")[-1].replace('p','.'))
	print(f'B={Bfield}')
	run_df = pd.read_csv(datfiles[0])
	VVA = run_df['VVA'].max() # maximum VVA; assumes VVA is the same except for the 0 point
	data = find_transfer(run_df)
	data['detuning']=-data['detuning']
	width = 1/pulse_time 
	popts, perrs, plabel, sinc2 = fit_sinc2(data[["detuning", "c5transfer"]].values, 
												width=width)
	B_from_f0 = [Eb_to_B_CCC(popts[1]), Eb_to_B_exp(popts[1])]

	if run == '2025-10-15_L':
		data = data[data['c5transfer'] != max(data['c5transfer'])]
	if TALK:
		fig, ax = plt.subplots()
		xs = np.linspace(data['detuning'].min(), data['detuning'].max(), 100)
		ax.plot(data['detuning'], data['c5transfer'])
		ax.plot(xs, sinc2(xs, *popts), '-', color='plum')
		ax.set(xlabel='detuning [MHz]', ylabel=r'$\alpha_\mathrm{transfer}$',
		  title=f'{run}, amp={popts[0]:.3f}({round(perrs[0]*1000):d}), x0={popts[1]:.3f}({round(perrs[1]*1000):d})')

	# Save for later plotting
	valid_results.append({
		"c5transfer": data['c5transfer'],
		"sinc2": sinc2,
		"popts": popts,
		"perrs":perrs,
		"plabel": plabel,
		"run": run,
		"Bfield":Bfield,
		"B_from_f0":B_from_f0
	})

# check if B the scan was supposed to be taken at matches B from f0
fig, ax = plt.subplots()
CCC_or_Exp = 0 # 0 chooses CCC, 1 chooses Exp
ylabel_suf = '(CCC)' if CCC_or_Exp == 0 else '(Exp)'
ax.set(
	# xlim=[202, 202.5], 
	   xlabel='Bfield from field cal ', 
	   ylabel='Bfield from peak f0' +ylabel_suf)
ax.locator_params(axis='x', nbins=5)

Bfield = [result['Bfield'] for result in valid_results]
B_from_f0 = [result['B_from_f0'] for result in valid_results]
popts_results = [result['popts'][0] for result in valid_results]
perrs_results = [result['perrs'][0] for result in valid_results]

label = '2025-10-03'
label2 = '2025-10-15 cold' 
label3 = '2025-10-15 hot'


B_from_f0 = np.array(B_from_f0)
ax.plot(Bfield[:2], B_from_f0[:2,CCC_or_Exp], marker='o', color='dodgerblue', label=label)
ax.plot(Bfield[2:4], B_from_f0[2:4,CCC_or_Exp], marker='o', color='slategrey', label=label2)
ax.plot(Bfield[4:6], B_from_f0[4:6,CCC_or_Exp], marker='o', color='navy', label=label3)
ax.legend() 

fig, ax = plt.subplots()
ax.locator_params(axis='x', nbins=5)

ax.errorbar(Bfield[:2], popts_results[:2], yerr=perrs_results[:2], marker='o', color='dodgerblue', label=label)
ax.errorbar(Bfield[2:4], popts_results[2:4], yerr=perrs_results[2:4], marker='o', color='slategrey', label=label2)
ax.errorbar(Bfield[4:6], popts_results[4:6], yerr=perrs_results[4:6], marker='o', color='navy', label=label3)
ax.legend()

DC_alpha_amplitude = (valid_results[2]['popts'][0] - valid_results[3]['popts'][0])/2 # factor of 2 is because comparison is to AC amplitude which is half the peak-to-peak
e_DC_alpha_amplitude = np.sqrt(valid_results[1]['perrs'][0]**2 + valid_results[0]['perrs'][0]**2)
ax.set(
	# xlim=[202, 202.3], 
	ylabel=r'$\alpha_\mathrm{transfer}$', xlabel='B [G]',
	   title=rf'$\alpha_\mathrm{{amp}} = {DC_alpha_amplitude:.3f}({round(e_DC_alpha_amplitude*1000):d})$,comparing 2025-10-15 cold')

phase_shift_summary = pd.read_csv(os.path.join(analysis_folder, 'phase_shift_2025_summary.csv'), index_col=0)
phase_shift_summary = phase_shift_summary[phase_shift_summary.index =='2025-10-01_L']
AC_alpha_amplitude = phase_shift_summary['Amplitude of Sin Fit of A'].values[0]
e_AC_alpha_amplitude = phase_shift_summary['Error of Amplitude of Sin Fit of A'].values[0]

relative_amp = AC_alpha_amplitude/DC_alpha_amplitude
e_relative_amp = np.sqrt((e_AC_alpha_amplitude/AC_alpha_amplitude)**2 + (e_DC_alpha_amplitude/DC_alpha_amplitude)**2)
e_relative_amp_abs = e_relative_amp * relative_amp
print(f'Relative AC/DC transfer amplitude is {relative_amp:.2f}({int(e_relative_amp_abs*100):d})')