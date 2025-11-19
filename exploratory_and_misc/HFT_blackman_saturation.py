"""
2025 Nov 4
Chip Lab

First pass at fitting saturation curves to HFT-short-blackman data around unitarity. 
Also calculates DC contact susceptibility.
"""
import os
import sys

# paths
root_project = os.path.dirname(os.getcwd())
# Fast-Modulation-Contact-Correlation-Project\analysis
module_folder = os.path.join(root_project, "analysis")
if module_folder not in sys.path:
	sys.path.append(module_folder)
# settings for directories, standard packages...
from preamble import *
from library import paper_settings, styles, colors
from contact_correlations.contact_tabulated import ContactInterpolation
from contact_correlations.UFG_analysis import BulkViscTrap, calc_contact
from scipy.optimize import curve_fit

# Controls
EXPORT =True

### fit functions
def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

def Linear(x,m,b):
	return m*x + b

def quotient_propagation(f, A, B, sA, sB, sAB):
	return f* (sA**2/A**2 + sB**2/B**2 - 2*sAB/A/B)**(1/2)

def linear_extrapolation(popt, pcov):
	"""Calculates the slope of the linear extrapolation to the saturation
	   curve given the popt and pcov of the best fit saturation. Returns the 
	   slope and its error (with correct uncertainty propagation)."""
	slope = popt[0]/popt[1]
	perr = np.sqrt(np.diag(pcov))
	e_slope = quotient_propagation(slope, popt[0], popt[1], 
								perr[0], perr[1], pcov[0,1])
	return slope, e_slope

def contact_from_slope(EF, trf, detuning, slope):
	return 2**(3/2) * np.pi * (2 * np.pi * EF)/trf * (detuning/EF)**(3/2) * slope

runname = "2025-11-04_F"
runname = "2025-11-05_Q"
trf = 20e-6  # pulse time in seconds
# est
EF = 9459  #Hz  # 2025-11-05_Q
# EF = 10090  # 2025-11-04_F
ToTF = 0.2819

Params_11_05_Q = pd.DataFrame({
	'Field':[202.04, 202.14, 202.24],
	'ToTF':[0.303, 0.267, 0.253],
	'EF':[9770, 9876, 9901],
	'Num':[11280, 11280, 11280] # dummy values
})

barnu = 300 # guesstimated from my literal memory of recent trap freqs at 0.2/4

detuning = 150e3  # Hz
Num = 11280  # 2025-11-05_Q
# Num = 13936  # 2025-11-04_F
# find data files
y, m, d, l = runname[0:4], runname[5:7], runname[8:10], runname[-1]
runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
datfiles = glob(f"{runpath}*=*.dat")

# figure for data and sat curves
fig, ax = plt.subplots()
ax.set(xlabel='OmegaR2 [1/s^2]', ylabel=r'$\alpha_\mathrm{HFT}$', ylim=[-0.01, 0.14])

# figure for holding the fit parameters
fig2, axs = plt.subplots(1, 3, figsize=(8,4))
axs=axs.flatten()
axs[0].set(xlabel='Field [G]', ylabel=r'Amp $\alpha_\mathrm{max}$')
axs[1].set(xlabel='Field [G]', ylabel=r'1/e Rabi$^2$ $\Omega_{R,0}^2$ [Hz$^2$]')
axs[2].set(xlabel='Field [G]', ylabel=r'Slope $\alpha_\mathrm{max}/\Omega_{R,0}^2$ [Hz$^{-2}$]')

results_df = pd.DataFrame([])

for (fpath, style, color) in zip(datfiles, styles, colors):
	filename = fpath.split('\\')[-1]
	# initialize run
	run = Data(filename)
	
	# fix column headers
	run.data.rename(columns={'Freq':'freq'}, inplace=True)
	
	# fill in some extra data or else code will complain
	Bfield = run.data['Field'].values[0]

	# these runs are relevant as the most recent DC sus measurements.
	try:
		EF = Params_11_05_Q[Params_11_05_Q['Field'] == Bfield]['EF'].values[0]
		run.data['EF'] = EF
		ToTF = Params_11_05_Q[Params_11_05_Q['Field'] == Bfield]['ToTF'].values[0]	
		run.data['ToTF'] = ToTF
	except:
		run.data['EF'] = EF
		run.data['ToTF'] = ToTF
	run.data['trf'] = trf
	run.data['detuning'] = detuning

	# complete analysis of data
	run.analysis(bgVVA=0, pulse_type="blackman")
	
	# group and average
	run.group_by_mean('OmegaR2')
	df = run.avg_data
	
	# fit data to saturation model
	x = df['OmegaR2']
	y = df['alpha_HFT']
	yerr = df['em_alpha_HFT']
	p0 = [0.1, 0.5e11]
	popt, pcov = curve_fit(Saturation, x, y, sigma=yerr, p0=p0)
	perr = np.sqrt(np.diag(pcov))

	# Compute extrapolated linear slope and error
	slope, e_slope = linear_extrapolation(popt, pcov)

	# Compuate contact from slope
	C = contact_from_slope(EF, trf, detuning, slope)
	e_C = C * e_slope/slope  # Ignoring error in EF for now. TODO add e_EF

	# plot data and curves
	xs = np.linspace(x.min(), x.max(), 30)
	ax.plot(xs, Saturation(xs, *popt), ls='-', marker='', color=color)
	ax.plot(xs, Linear(xs, popt[0]/popt[1], 0), ls='--', marker='', color=color)
	ax.errorbar(x, y,yerr, label=Bfield, **style)
		# omegar2_cutoff = 0.05e11
	ax.plot(x, y, ls='', marker='d', markersize=2,color=color)

	axs[0].errorbar(Bfield, popt[0], perr[0],**style)
	axs[1].errorbar(Bfield, popt[1], perr[1], **style)
	axs[2].errorbar(Bfield, slope, e_slope, **style)

	results_dict = {'Bfield': Bfield, 'EF': EF, 'ToTF':ToTF, 'trf': trf, 'detuning': run.data['freq'],
				 	'slope': slope, 'e_slope': e_slope, 'OmegaR2_sat': popt[1], 'e_OmegaR2_sat': perr[1],
					'alpha_max': popt[0], 'e_alpha_max': perr[0], 'C': C, 'e_C': e_C}
	new_row_df = pd.DataFrame([results_dict])
	results_df = pd.concat([results_df, new_row_df])

ax.legend()
fig.suptitle(f'Sat. for run={runname}, ToTF={ToTF}')
fig.tight_layout()
fig2.suptitle('Fit params')
fig2.tight_layout()

popt, pcov = curve_fit(Linear, results_df['Bfield'], results_df['C'], sigma=results_df['e_C'])
perr = np.sqrt(np.diag(pcov))

fig, axes = plt.subplots(1,2,figsize=(7, 4), sharey = True)
ax = axes[0]
ax1 = axes[1]
ax.set(xlabel='B field [G]', ylabel=r'Contact, $\tilde C$')
ax1.set(
	xlabel = 'ToTF'
)


###Contacts from Tilman 
# dummy plot to visualize curve
ef = results_df[results_df['Bfield'] == 202.14]['EF'].values[0]
ttfs = np.linspace(0.2, 0.8, 20)
figC, axC = plt.subplots(figsize=(3,2))
Cs = [calc_contact(ttf, ef, barnu)[0] for ttf in ttfs]
axC.plot(ttfs, Cs, marker='', ls='--', color='salmon')
axC.vlines(results_df['ToTF'], ymin=np.min(np.array(Cs)), ymax=np.max(np.array(Cs)), ls=':', lw=1)
axC.set(xlabel=r'$T/T_F$', ylabel='Contact, ' + r'$\widetilde{C}$', title=fr'$\bar{{EF}}$ = {ef} Hz')
figC.tight_layout()

# calculate harmonic C from ToTF. However, have to use mean EF to allow dataset comparison, so for validity we
# want datasets ideally with the same EF.
meanEF = results_df['EF'].mean()
harmonic_C_calc_list = [calc_contact(ttf, meanEF, barnu)[0] for ttf in results_df['ToTF']]
results_df['C_harmonic'] = harmonic_C_calc_list

popt_calc_C_harmonic, pcovt_calc_C_harmonic = curve_fit(Linear, results_df['Bfield'], results_df['C_harmonic'], 
					#    sigma=results_df['e_C']
					   )
perr_calc_C_harmonic = np.sqrt(np.diag(pcovt_calc_C_harmonic))

# contact density (valid for uniform gas)
ContactInterpolation_list = []
for t in results_df['ToTF']:
    contactinterp = ContactInterpolation(t)
    ContactInterpolation_list.append(contactinterp)
results_df['C_uniform'] = ContactInterpolation_list

popt_calc_C, pcovt_calc_C = curve_fit(Linear, results_df['Bfield'], ContactInterpolation_list, 
					#    sigma=results_df['e_C']
					   )
perr_calc_C = np.sqrt(np.diag(pcovt_calc_C))

# fudge C based on ToTF change between points, referenced to unitary measurement
# assumes C(TTF) scaling doesn't change much with B field around unitarity.
# fudge C = \gamma * Experiment C where \gamma = Theory C(TTF(202.14)) / Theory C(TTF(B))
refC = results_df[results_df['Bfield']==202.14]['C_harmonic']
results_df['C_corrfactor'] =  refC / results_df['C_harmonic']
results_df['fudgedC'] = results_df['C_corrfactor'] * results_df['C']
results_df['e_fudgedC'] = results_df['C_corrfactor'] * results_df['e_C'] 

# fit to new fudged values
popt_fudge, pcov_fudge = curve_fit(Linear, results_df['Bfield'], results_df['fudgedC'], sigma = results_df['e_fudgedC'])
perr_fudge = np.sqrt(np.diag(pcov_fudge))

###plotting measured C and fudged C
Bs = np.linspace(results_df['Bfield'].min(), results_df['Bfield'].max(), 100)
ax.plot(Bs, Linear(Bs, *popt), '--', color='tab:blue')
ax.errorbar(results_df['Bfield'], results_df['C'], results_df['e_C'], **styles[0], label = r'Measured')
ax.plot(Bs, Linear(Bs, *popt_fudge), '--', color='sandybrown')
ax.errorbar(results_df['Bfield'], results_df['fudgedC'], results_df['e_fudgedC'], color='sandybrown', markeredgecolor = 'peru', label = 'Scaled')


###plotting C from Tilman's code 
# ax.plot(B_list, ContactInterpolation_list, color='violet', markeredgecolor='purple', label = r'Tilman Predicted Unitary')
# ax.plot(Bxs,Linear(Bxs,*popt_calc_C), marker='', ls = '--', color='hotpink')

###plotting calculated C with the harmonic approx applied
ax.plot(results_df['Bfield'], results_df['C_harmonic'], color='salmon', markeredgecolor='maroon', label = r'Harmonic')
ax.plot(Bs,Linear(Bs,*popt_calc_C_harmonic), marker='', ls = '--', color='tomato')

# ###plotting everything against ToTF
ax1.plot(results_df['ToTF'], results_df['C_uniform'], color='violet', markeredgecolor='purple', label = r'Tilman Predicted Unitary')
ax1.plot(results_df['ToTF'], results_df['C_harmonic'], color='salmon', markeredgecolor='maroon', label = r'Harmonic')
ax1.plot(results_df['ToTF'], results_df['C'],**styles[0], label = r'Measured')
ax1.plot(results_df['ToTF'], results_df['fudgedC'], color='sandybrown', markeredgecolor = 'peru', label = 'Scaled')

ax1.legend(loc = 'lower right')

fig.suptitle("Linear fit to DC contact vs Bfield")
fig.tight_layout()
plt.show()

# plot of C fudge factor vs. ToTF and B
fig, axs = plt.subplots(2)
axs[0].plot(results_df['ToTF'], results_df['C_corrfactor'], color='mediumorchid', mec='darkorchid', ls='-')
axs[0].set(xlabel=r'$T/T_F$', ylabel='Contact fudge factor')
axs[1].plot(results_df['Bfield'], results_df['C_corrfactor'], color='mediumorchid', mec='darkorchid', ls='-')
axs[1].set(xlabel=r'$B$', ylabel='Contact fudge factor')
fig.tight_layout()
plt.show()



print(r"The contact slope is $d\tilde C/dB = $" + f"{popt[0]:.2f}({1e2*perr[0]:.0f}) [1/G]")
print(r"After fudging, the contact slope is $d\tilde C/dB = $" + f"{popt_fudge[0]:.2f}({1e2*perr_fudge[0]:.0f}) [1/G]")

if EXPORT:

	csv_path = os.path.join(os.path.join(root_analysis, 'corrections'), f'saturation_HFT.csv')
	write_header = not os.path.exists(csv_path)

	try:
		results_df.to_csv(csv_path, mode='a', header=write_header, index=False, sep=',')
		print(f"✅ Created csv at {csv_path}")
	except:
		print(f"Something went wrong when trying to write {csv_path}")
