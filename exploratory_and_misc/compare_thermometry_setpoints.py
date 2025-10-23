
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
runs = ['2025-10-23_G','2025-10-23_H','2025-10-23_I','2025-10-23_J', '2025-10-23_K']
ODTscaling = [0.5, 0.8, 1, 1.5, 2.25]
plot_params = ['N','T', 'EFkHz', 'ToTF', 'TF','peak_OD'] 
normalODT1 = 0.2 #V
xparam='ODTscale'
xlabel='ODTscale'
fig, axs= plt.subplots(3,2, sharex=True)
axs = axs.flatten()
for i, param in enumerate(plot_params):
	axs[i].set(ylabel=param)
	if i > len(plot_params)-3:
		axs[i].set(xlabel=xlabel)
for run, ODTscale in zip(runs, ODTscaling):
	# find data files
	y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
	runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
	datfiles = glob(f"{runpath}*.dat")
	runname = datfiles[0].split("\\")[-2].lower() # get run folder name, should be same for all files
	print(f'Run name = {runname}')
	run_df = pd.read_csv(datfiles[0])
	run_df['ODTscale'] = ODTscale

	for i, param in enumerate(plot_params):
		axs[i].errorbar(run_df['ODTscale'].mean(), run_df[param].mean(), run_df[param].std(), ls='', marker='o', color='hotpink')
		
# fig.suptitle('comp. final ODT setpoints after evap')
fig.tight_layout()
