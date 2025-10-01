import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import sys 

module_folder = 'E:\\Analysis Scripts\\analysis'
if module_folder not in sys.path:
	sys.path.insert(0, 'E:\\Analysis Scripts\\analysis')
from data_class import Data
from library import fit_label, a0, pi
from fit_functions import Sinc2
from glob import glob
from scipy.optimize import curve_fit

def fit_sinc2(xy, width=False):
    """
    fits xy, input as single array, to sinc2 function. returns fitted params, errors, and label
    if a width is provided, fits sinc with fixed width and 0 background
    TODO: keep background fixed at 0
    """
    f, p0, paramnames = Sinc2(xy)

    if width:
         sinc2 = lambda x, A, x0: f(x, A, x0, width, 0)
         p0 = p0[:2]
         paramnames = paramnames[:2]
    else:
         sinc2 = f

    popts, pcov = curve_fit(sinc2, xy[:,0], xy[:,1], p0)
    perrs = np.sqrt(np.diag(pcov))
    plabel = fit_label(popts, perrs, paramnames)

    return popts, perrs, plabel, sinc2

def find_transfer(df, ff=1):
    """
    given df output from matlab containing atom counts, returns new df containing detuning, 
    5 and 9 counts, and transfer/loss
    """
    run_data = df[["detuning", "VVA", "c5", "c9"]]
    run_data["c5"] = run_data["c5"]*ff
    if run_data['VVA'].isin([0]).any():
        bg = df[df["VVA"] == 0]
        c5bg, c9bg = np.mean(bg[["c5", "c9"]], axis=0)
    else:
        c5bg = run_data[run_data['detuning'] == max(run_data['detuning'])]['c5'].mean()

    data = run_data[df["VVA"] != 0].copy()
    data.loc[data.index, "c5transfer"] = (1-data["c5"]/c5bg)/2 # factor of 2 assumes atom-molecule loss
    data.loc[data.index, "c5bg"] = c5bg
    return data



root = "E:/Data"
runs = ["2025-09-26_B", "2025-09-26_C"] # need date and letter
fix_sinc_width=True
pulse_time = 20 #us
SHOW_PLOT=True
if SHOW_PLOT:
     fig, ax=plt.subplots(1,1)

for run in runs:
    # find data files
    y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
    runpath = glob(f"{root}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
    datfiles = glob(f"{runpath}*.dat")
    runname = datfiles[0].split("\\")[-2].lower() # get run folder name, should be same for all files
    run_df = Data(datfiles[0].split("\\")[-1])

    run_df.data['VVA'] = 9
    run_df.data['detuning'] = 47.2227 - run_df.data['freq']
    data = find_transfer(run_df.data, ff=1.3)
    # data = data.iloc[0:14]
    if fix_sinc_width:
        popts, perrs, plabel, sinc2 = fit_sinc2(data[["detuning", "c5transfer"]].values, 
                                                width=fix_sinc_width/pulse_time)
    detuning, c5transfer = detuning, c5transfer = data["detuning"], data["c5transfer"]

    if SHOW_PLOT:
        xs = np.linspace(max(detuning), min(detuning), 100)
        ax.plot(xs, sinc2(xs, *popts), ls="-", marker="",  label=plabel)
        ax.plot(detuning, c5transfer)
        ax.legend()
        ax.set(
            xlabel="detuning (MHz)",
            ylabel="transfer"
        )
       # ax.set_title(datfiles.split("\\")[-1])