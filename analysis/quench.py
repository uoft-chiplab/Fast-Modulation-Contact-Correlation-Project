"""
This script analyzes the growth of the contact following an rf quench into unitarity.
"""


# settings for directories, standard packages...
from preamble import *

runs = ["2025-12-10_I", "2025-12-10_J"]
defaults = {'freq':43.24, 'EF':18000, 'trf':1e-5, 'T':300} # default values for freq (MHz), EF (Hz), trf (s)

fig, ax = plt.subplots(figsize=(4,3))
ax.set(xlabel = 'Time after quench (us)',
       ylabel = r'$\widetilde{\Gamma}_d / \widetilde{\Gamma}_{d,\mathrm{max}}$')

for run in runs:
    # find data files
    y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
    runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
    datfiles = glob(f"{runpath}*.dat")
    runname = datfiles[0].split(os.sep)[-2].lower() # get run folder name, should be same for all files

    filename = datfiles[0].split(os.sep)[-1]
    print("Filename is " + filename)
    data = Data(filename, path=datfiles[0])

    for key in defaults.keys():
        if key not in data.data.keys():
            data.data[key] = defaults[key]

    data.analysis(bgVVA = 0, pulse_type="square")
    data.group_by_mean('hold time (us)')
    df = data.avg_data
    df['norm_sig'] = df['scaledtransfer_dimer'] / df['scaledtransfer_dimer'].max()
    df['em_norm_sig'] = df['em_scaledtransfer_dimer'] / df['scaledtransfer_dimer'].max()
    ax.errorbar(df['hold time (us)'], 
                df['norm_sig'], 
                df['em_norm_sig'],
                fmt='o', label=f'{filename[-7]} {df['EF'].values[0]/1000:.1f} kHz')

ax.legend()
