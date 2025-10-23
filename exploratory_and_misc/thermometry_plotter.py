
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
runs = ['2025-10-22_H', '2025-10-22_B', '2025-10-22_C', '2025-10-22_D']
plot_params = ['N','T', 'EFkHz', 'ToTF'] #'peak_OD'
normalODT1 = 0.2 #V


for run in runs:
	# find data files
	y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
	runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
	datfiles = glob(f"{runpath}*.dat")
	runname = datfiles[0].split("\\")[-2].lower() # get run folder name, should be same for all files
	print(f'Run name = {runname}')
	run_df = pd.read_csv(datfiles[0])
	if 'odt1' in run_df.columns:
		run_df['odt_scale'] = run_df['odt1']/normalODT1
		xparam = 'odt_scale'
		xlabel= 'ODT scaling'
	else:
		xparam = 'cyc'
		xlabel='cyc'

	fig, axs= plt.subplots(3,2)
	axs = axs.flatten()
	for i, param in enumerate(plot_params):
		axs[i].plot(run_df[xparam], run_df[param], ls='', marker='o')
		axs[i].set(xlabel=xlabel, ylabel=param)
	fig.tight_layout()
	fig.suptitle(run)