
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
import matplotlib.lines as mlines
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
# some controls
SMOOTH =False
SAVEPLOT = True
def exponential(x, A, x0, C):
	return A*np.exp(-x/x0) + C

file = '2025-11-21_I'
plot_params = ['N','T', 'EFkHz', 'ToTF', 'TF','peak_OD'] 
xparam='odtscale'

y, m, d, l = file[0:4], file[5:7], file[8:10], file[-1]
runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
datfiles = glob(f"{runpath}*.dat")
runname = datfiles[0].split(os.sep)[-2].lower() # get run folder name, should be same for all files
print(f'Run name = {runname}')
run = Data(file, path = datfiles[0])
run_df = run.data

# initialize arrays
x = np.array(run_df['odtscale'])
y = np.array(run_df['evapscale'])
totf = np.array(run_df['ToTF'])
t = np.array(run_df['T']) *1e6 # uK

# meshgrid
xs = np.linspace(x.min(), x.max(), 200)
ys = np.linspace(y.min(), y.max(), 200)
X, Y = np.meshgrid(xs, ys)

# interpolated grid for Z values
ToTFgrid = griddata(
		(x, y),
		totf,
		(X, Y),
		method='linear' # 'cubic' or 'linear'
	)
Tgrid = griddata(
		(x, y),
		t,
		(X, Y),
		method='linear' # 'cubic' or 'linear'
	)
# plot heat map of scales vs. observable
# Interpolate ToTF across grid (b/c ToTF is not real 2D data)
params = [totf, t]
labels = ['ToTF', 'T [uK]']
paramgrids = [ToTFgrid, Tgrid]
for param, paramgrid, label in zip(params, paramgrids, labels):

	if SMOOTH:
		paramgrid = gaussian_filter(paramgrid, sigma=2)

	fig, ax = plt.subplots()
	h = ax.pcolormesh(X, Y, paramgrid, cmap='viridis')
	ax.scatter(x, y, c=param, edgecolor='k') 
	plt.colorbar(h, label=label)
	ax.set(xlabel = 'ODT scale',
		ylabel= 'Evap scale',
		title='Scale ref: 0.2/4.0 V')
	
	# plot contours of constant ToTF specifically
	targets = np.arange(0.2, 0.6, 0.05)
	if label == 'ToTF':
		CS = plt.contour(X, Y, ToTFgrid, levels=targets, linewidths=2)
	elif label == 'T [uK]':
		CS = plt.contour(X, Y, ToTFgrid, levels=targets, linewidths=2, colors='white')
	plt.clabel(CS, inline=True, fmt="%.2f", fontsize=10, colors='white')
	# Fake line for legend
	line = mlines.Line2D([],[],color='white', ls='-', marker='', linewidth=1.5, label='ToTF')
	plt.legend(handles=[line], loc='upper right', framealpha=0.3)  
	fig.tight_layout()
	if SAVEPLOT:
		thisdir = os.path.dirname(os.path.abspath(__file__))
		savepath = os.path.join(thisdir, 'ODT_and_evap_temp_heatmap.png')
		plt.savefig(savepath, dpi=300)
	plt.show()


# 4D-like visualization with T as color map
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(x, y, t, 
               c=totf,          
               cmap='viridis',
               s=40)

cbar = plt.colorbar(p, ax=ax)
cbar.set_label("ToTF")

ax.set(xlabel = 'ODT scale',
	   ylabel= 'Evap scale',
	   zlabel='T [uK]',
	   title='Scale ref: 0.2/4.0 V')
fig.tight_layout()
plt.show()