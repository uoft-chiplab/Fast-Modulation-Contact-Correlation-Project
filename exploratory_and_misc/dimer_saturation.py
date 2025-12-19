"""
2025 Dec 19
Chip Lab

Calibrating saturation curves for dimer data of <10 us pulse times.

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

# Controls
EXPORT =False

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

runs = pd.DataFrame({
	"2025-12-10_J":{"EF":11300, "T":183, "ToTF":0.338, "probe time":5},
	 }) # EF in Hz, T in nK

defaults = {'freq':43.24} 

fig, ax = plt.subplots()
ax.set(xlabel=r"$\Omega_R^2$",
	   ylabel=r"$\alpha_d$")


for i, run in enumerate(runs):
	# find data files
	y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
	runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
	datfiles = glob(f"{runpath}*.dat")
	runname = datfiles[0].split(os.sep)[-2].lower() # get run folder name, should be same for all files

	filename = datfiles[0].split(os.sep)[-1]
	print("Filename is " + filename)
	data = Data(filename, path=datfiles[0])

	EF, T, ToTF, probe_t = runs[run]

	data.data['trf'] = probe_t/1e6
	data.data['EF'] = EF
	for key in defaults.keys():
		if key not in data.data.keys():
			data.data[key] = defaults[key]


	# complete analysis of data
	data.analysis(bgVVA=0, pulse_type="square", rfsource="micrO")

	# group and average
	data.group_by_mean('OmegaR2')
	df = data.avg_data
	
	# fit data to saturation model
	x = df['OmegaR2']
	y = df['alpha_dimer']
	yerr = df['em_alpha_dimer']
	p0 = [0.1, 0.5e11]
	popt, pcov = curve_fit(Saturation, x, y, sigma=yerr, p0=p0)
	perr = np.sqrt(np.diag(pcov))

	# Compute extrapolated linear slope and error
	slope, e_slope = linear_extrapolation(popt, pcov)

	# plot data and curves
	xs = np.linspace(x.min(), x.max(), 30)
	ax.plot(xs, Saturation(xs, *popt), ls='-', marker='', color=color)
	ax.plot(xs, Linear(xs, popt[0]/popt[1], 0), ls='--', marker='', color=color)
	ax.plot(x, y, ls='', marker='d', markersize=2,color=color, label=f"trf={probe_t}us")

	

	# results_dict = {'Bfield': Bfield, 'EF': EF, 'ToTF':ToTF, 'trf': trf, 'detuning': run.data['freq'],
	# 			 	'slope': slope, 'e_slope': e_slope, 'OmegaR2_sat': popt[1], 'e_OmegaR2_sat': perr[1],
	# 				'alpha_max': popt[0], 'e_alpha_max': perr[0], 'C': C, 'e_C': e_C}
	# new_row_df = pd.DataFrame([results_dict])
	# results_df = pd.concat([results_df, new_row_df])

ax.legend()
fig.suptitle(f'Sat. for run={runname}, ToTF={ToTF}')
fig.tight_layout()


# if EXPORT:

# 	csv_path = os.path.join(os.path.join(root_analysis, 'corrections'), f'saturation_HFT.csv')
# 	write_header = not os.path.exists(csv_path)

# 	try:
# 		results_df.to_csv(csv_path, mode='a', header=write_header, index=False, sep=',')
# 		print(f"✅ Created csv at {csv_path}")
# 	except:
# 		print(f"Something went wrong when trying to write {csv_path}")
