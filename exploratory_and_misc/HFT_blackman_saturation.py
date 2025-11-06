"""
2025 Nov 4
Chip Lab

First pass at fitting saturation curves to HFT-short-blackman data around unitarity.
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
from scipy.optimize import curve_fit
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
	run.data['EF'] = EF
	run.data['trf'] = trf
	run.data['detuning'] = detuning

	Bfield = run.data['Field'].values[0]
	# complete analysis of data
	run.analysis(bgVVA=0, pulse_type="blackman")
	
	# group and average
	# cutoff = 0.005
	# df = run.data[run.data['OmegaR2'] > 1]
	# alpha_cut = 0.005
	# omega_cut = 0.05e11
	# there were very low outliers in alpha when omegaR was large; filter
	# run.data = run.data[~((run.data['alpha_HFT'] < alpha_cut) & (run.data['OmegaR2'] > omega_cut))]	# omegar2_cutoff = 0.05e11
	# run.data = run.data[run.data['OmegaR2'] > omegar2_cutoff]
	# run.data = run.data[((run.data['OmegaR2'] > 0.2e11))]
	run.group_by_mean('OmegaR2')
	df = run.avg_data
	# sigmawindow = df['em_alpha_HFT']*3
	
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

	results_dict = {'Bfield': Bfield, 'EF': EF, 'trf': trf, 'detuning': run.data['freq'],
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

fig, ax = plt.subplots(figsize=(4, 4))
ax.set(xlabel='B field [G]', ylabel=r'Contact, $\tilde C$')

Bs = np.linspace(results_df['Bfield'].min(), results_df['Bfield'].max(), 100)
ax.plot(Bs, Linear(Bs, *popt), '--', color='tab:blue')
ax.errorbar(results_df['Bfield'], results_df['C'], results_df['e_C'], **styles[0])

fig.suptitle("Linear fit to DC contact vs Bfield")
fig.tight_layout()
plt.show()

print(r"The contact slope is $d\tilde C/dB = $" + f"{popt[0]:.2f}({1e2*perr[0]:.0f}) [1/G]")