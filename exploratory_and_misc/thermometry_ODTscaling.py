
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

def exponential(x, A, x0, C):
	return A*np.exp(-x/x0) + C

# runs = ['2025-10-23_G','2025-10-23_H','2025-10-23_I','2025-10-23_J', '2025-10-23_K']
runs = ['2025-11-21_H']
# ODTscaling = [0.5, 0.8, 1, 1.5, 2.25]
plot_params = ['N','T', 'EFkHz', 'ToTF', 'TF','peak_OD'] 
normalODT1 = 0.2 #V
xparam='odtscale'
xlabel=xparam
fig, axs= plt.subplots(3,2, sharex=True)
axs = axs.flatten()
for i, param in enumerate(plot_params):
	axs[i].set(ylabel=param)
	if i > len(plot_params)-3:
		axs[i].set(xlabel=xlabel)
for run in runs:
	# find data files
	y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
	runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
	datfiles = glob(f"{runpath}*.dat")
	runname = datfiles[0].split("\\")[-2].lower() # get run folder name, should be same for all files
	print(f'Run name = {runname}')
	run = Data(run, path = datfiles[0])
	run.group_by_mean('odtscale')
	run_df = run.avg_data
	for i, param in enumerate(plot_params):
		axs[i].errorbar(run_df[xparam], run_df[param], 
				   run_df['em_' + param], 
				  ls='', marker='o', color='hotpink')
		

fig.tight_layout()
