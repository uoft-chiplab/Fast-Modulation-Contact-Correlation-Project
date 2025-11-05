
import os
import sys
root_project = os.path.dirname(os.getcwd())

# Fast-Modulation-Contact-Correlation-Project\analysis
module_folder = os.path.join(root_project, "analysis")
if module_folder not in sys.path:
	sys.path.append(module_folder)
# settings for directories, standard packages...
from preamble import *
from matplotlib import colors as colorsmpl
from scipy.optimize import curve_fit
from library import colors
runs = ['2025-11-05_J', '2025-11-05_K', '2025-11-05_O', '2025-11-05_P']
plot_params = ['N','T', 'EFkHz', 'ToTF', 'peak_OD'] 
xparam = 'ramptime'
xlabel='ramptime'

def Linear(x, m, b):
	return m*x+b

fig2, ax2 = plt.subplots()
ax2.set(xlabel = 'ramptime', ylabel='ToTF')

for j,run in enumerate(runs):
	# find data files
	y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
	runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
	datfiles = glob(f"{runpath}*.dat")
	runname = datfiles[0].split("\\")[-2].lower() # get run folder name, should be same for all files
	print(f'Run name = {runname}')
	run_df = pd.read_csv(datfiles[0])
	
	fig, axs= plt.subplots(3,2)
	axs = axs.flatten()
	for i, param in enumerate(plot_params):
		axs[i].plot(run_df[xparam], run_df[param], ls='', marker='o')
		axs[i].set(xlabel=xlabel, ylabel=param)
		
		axs[i].hlines(run_df[param].mean(), xmin = run_df[xparam].min(), xmax=run_df[xparam].max(), ls='--', label=f'{run_df[param].mean():.3f}')
		popt, pcov = curve_fit(Linear, run_df[xparam], run_df[param], p0=[0.05, 0.3])
		xs = np.linspace(run_df[xparam].min(), run_df[xparam].max(), 20)
		axs[i].plot(xs, Linear(xs, *popt), ls='-', marker='')
		axs[i].legend()

		if param == 'ToTF' and (run[-1] == 'O' or run[-1] == 'P'):
			lab = 202.04 if run[-1] == 'O' else 202.24
			ax2.hlines(run_df[param].mean(), xmin = run_df[xparam].min(), xmax=run_df[xparam].max(), ls='--', label=f'{run_df[param].mean():.3f}')
			ax2.plot(xs, Linear(xs, *popt), ls='-', marker='')
			ax2.plot(run_df[xparam], run_df[param], ls='', marker='o', color= colors[j], label=lab)

	fig.tight_layout()
	fig.suptitle(run)
	ax2.legend()
	fig2.tight_layout()
	fig2.suptitle('Total time = 1 ms')
