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

runname = "2025-11-05_G"
trf = 20e-6 # pulse time in seconds
# est
EF = 10000 # Hz
ToTF = 0.3
Num = 13936
# find data files
y, m, d, l = runname[0:4], runname[5:7], runname[8:10], runname[-1]
runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
datfiles = glob(f"{runpath}*=*.dat")

# figure for data and sat curves
fig, ax = plt.subplots()
ax.set(xlabel='OmegaR2 [Hz^2]', ylabel=r'$\alpha_\mathrm{HFT}$', ylim=[-0.01, 0.14])

# figure for holding the fit parameters
fig2, axs = plt.subplots(1,2)
axs=axs.flatten()
axs[0].set(xlabel='Field [G]', ylabel='Amp')
axs[1].set(xlabel='Field [G]', ylabel=r'$\Omega_{R,0}^2$')

for (fpath, style, color) in zip(datfiles, styles, colors):
	filename = fpath.split('\\')[-1]
	# initialize run
	run = Data(filename)
	
	# fix column headers
	run.data.rename(columns={'Freq':'freq'}, inplace=True)
	
	# fill in some extra data or else code will complain
	run.data['EF']=EF
	run.data['trf'] = trf

	Bfield = run.data['Field'].values[0]
	# complete analysis of data
	run.analysis(bgVVA=0, pulse_type="blackman")
	
	# group and average
	# cutoff = 0.005
	# df = run.data[run.data['OmegaR2'] > 1]
	alpha_cut = 0.005
	omega_cut = 0.05e11
	# there were very low outliers in alpha when omegaR was large; filter
	run.data = run.data[~((run.data['alpha_HFT'] < alpha_cut) & (run.data['OmegaR2'] > omega_cut))]	# omegar2_cutoff = 0.05e11
	# run.data = run.data[run.data['OmegaR2'] > omegar2_cutoff]
	# run.data = run.data[((run.data['OmegaR2'] > 0.2e11))]
	run.group_by_mean('OmegaR2')
	df= run.avg_data
	# sigmawindow = df['em_alpha_HFT']*3
	
	# fit data to saturation model
	x = df['OmegaR2']
	y = df['alpha_HFT']
	yerr = df['em_alpha_HFT']
	p0 = [0.1, 0.5e11]
	popt, pcov = curve_fit(Saturation, x, y, sigma=yerr, p0=p0)
	perr = np.sqrt(np.diag(pcov))

	# plot data and curves
	xs = np.linspace(x.min(), x.max(), 30)
	ax.plot(xs, Saturation(xs, *popt), ls='-', marker='', color=color)
	ax.plot(xs, Linear(xs, popt[0]/popt[1], 0), ls='--', marker='', color=color)
	ax.errorbar(df['OmegaR2'], df['alpha_HFT'], df['em_alpha_HFT'], label=Bfield, **style)
		# omegar2_cutoff = 0.05e11
	ax.plot(run.data['OmegaR2'], run.data['alpha_HFT'], ls='', marker='d', markersize=2,color=color)
	
	axs[0].errorbar(Bfield, popt[0], perr[0],**style)
	axs[1].errorbar(Bfield, popt[1], perr[1], **style)

ax.legend()
fig.suptitle(f'Sat. for run={runname}, ToTF={ToTF}')
fig.tight_layout()
fig2.suptitle('Fit params')
fig2.tight_layout()