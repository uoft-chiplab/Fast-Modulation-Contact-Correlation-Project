"""
calculates phase shift between amplitude/frequency/contact/field for modulated field
from dimer spectra, or the contact given static field.

plots dimer spectra for each wiggle time (ac) or field value (dc) if show_dimer_plot==True.

assumes fixed sinc^2 width for each dimer fit, and 0 background transfer.

set plot_dc==True to plot dc field points along with ac fits, given dc_cal_path

remember to change field_cal run name to match wiggle run for correct B plot!
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from glob import glob
import sys 
import os

# pretty plots
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
plt.rcParams['axes.prop_cycle'] = cycler(color=['hotpink', 'plum'])

# automatically read parent dir, assuming pwd is the directory containing this file

# contact correlation proj folder -- Fast-Modulation-Contact-Correlation-Project
root_project = os.path.dirname(os.getcwd())
# analysis folder
root_analysis = os.path.dirname(root_project)
# network machine
root = os.path.dirname(root_analysis)

module_folder = os.path.join(root_analysis, "analysis")
if module_folder not in sys.path:
	sys.path.append(module_folder)
from data_class import Data
from library import fit_label, a0, pi
from fit_functions import Sinc2

root_data = os.path.join(root, "Data")
run = "2025-09-24_E" # need date and letter
field_cal_path = os.path.join(root_project, r"FieldWiggleCal\field_cal_summary.csv")
field_cal_df = pd.read_csv(field_cal_path)
field_cal = field_cal_df[field_cal_df['run'] == '2025-09-19_D'] 

DC_cal_path = os.path.join(root_data, r"2025\09 September2025\31September2025\D_DCfield_dimer20us_5VVA_arggghhhh\DC_field_cal.csv")
DC_cal_csv = pd.read_csv(DC_cal_path)
DC_VVA = 5 # update this to pull automatically from csv or path

# 2025-09-19_D is for 6 kHz 1.8 Vpp
# 2024-04-05_G is for 10 kHz 1.8 Vpp

show_dimer_plot = True
amp_cutoff = 0.01 # ignore runs with peak transfer below 0.01
plot_dc = False # whether or not to plot DC field points from DC_cal_csv

def f0_to_B(x):
    """
    given a frequency corresponding to the dimer center position get a field out
    lists taken from the breit rabi notebook since I didn't want to code it all out lmao
    then I realized these frequencies are for on resonance for 7->5 at 202-202.5
    and I figured I could just subtract the frequency expected for the dimer at 202.14 to get the 
    detuning 
    """
    xs = [47.1989, 47.2159, 47.2329, 47.2498, 47.2668, 47.2838]
    xs = np.array(xs) - 43.2227
    ys = [202., 202.1, 202.2, 202.3, 202.4, 202.5]
    ys.reverse() # higher Eb = lower field

    return np.interp(x,xs,ys)

def find_transfer(df):
    """
    given df output from matlab containing atom counts, returns new df containing detuning, 
    5 and 9 counts, and transfer/loss
    """
    run_data = df[["detuning", "VVA", "c5", "c9"]]

    bg = df[df["VVA"] == 0]
    c5bg, c9bg = np.mean(bg[["c5", "c9"]], axis=0)

    data = run_data[df["VVA"] != 0].copy()
    data.loc[data.index, "c5transfer"] = (1-data["c5"]/c5bg)/2 # factor of 2 assumes atom-molecule loss
    data.loc[data.index, "c5bg"] = c5bg
    return data

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

def fit_fixedSinkHz(t, y, run_freq, eA, p0=None):
    """
    need docstring, and maybe a better way to code this
    t in ms, freq in kHz
    """
    def FixedSinkHz(t, A, p, C):
        omega = run_freq * 2 * np.pi # kHz
        return A*np.sin(omega * t - p) + C

    if p0 == None:
         A = (max(y)-min(y))/2
         C = (max(y)+min(y))/2
         p = np.pi
         p0 = [A, p, C]

    if np.any(eA != 0):

        popts, pcov = curve_fit(FixedSinkHz, t, y, p0, bounds=([0, 0, 0], [np.inf, 2*np.pi, np.inf]), sigma=eA)
        perrs = np.sqrt(np.diag(pcov))

    else: 
        popts, pcov = curve_fit(FixedSinkHz, t, y, p0, bounds=([0, 0, 0], [np.inf, 2*np.pi, np.inf]))
        perrs = np.sqrt(np.diag(pcov))
    return popts, perrs, plabel, FixedSinkHz

###amplitude to contact 
def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

def Contact_from_amplitude(A, eA, VVA):
     """
     from I_d_CCC =  kF / a13kF / pi * ell_d_CCC * a0 * C, rearranged to solve for C

     todo: make actually related to contact w/ rabi freq calibration

     """

    #  N = data["c5"].max()
    #  kF =(3*np.pi**2*N)**(1/3)
    #  a13kF = kF * a13(202.14)
    #  spin_me = 32/42 # spin matrix element
    #  ell_d_CCC = spin_me * 42 * np.pi

    #  C = A / kF * a13kF * np.pi / ell_d_CCC / a0 
    #  eC = eA / kF * a13kF * np.pi / ell_d_CCC / a0 

    #  return C, eC

     return A/(VVA/10)**2, eA/(VVA/10)**2

# find data files
y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
datfiles = glob(f"{runpath}*.dat")
runname = datfiles[0].split("\\")[-2].lower() # get run folder name, should be same for all files

# get run attributes
is_dc = runname.find("khz") < 0 # str.find returns -1 if not in string
if is_dc:
    wiggle_freq = 0
    xlabel = 'Field [G]'
    print(f'Not a modulation run, calibrating contact at DC')
else:
    wiggle_freq = float(runname[:runname.find("khz")].split("_")[-1])
    xlabel = 'Times [ms]'

i = runname.find("us")
pulse_time = float(runname[i-2:i]) # ms

# fit dimer
# index is field for DC run and wiggle time for modulated runs!
df = pd.DataFrame( columns = ["A", "x0", "eA", "ex0", 'C', 'eC'],  dtype=float)
df.index.name = xlabel

valid_results = []
dropped_list = []

for fpath in datfiles:
    run_df = Data(fpath.split("\\")[-1])
    data = find_transfer(run_df.data)
    # take max to ignore 0 and repeats, assume VVA is same for all data sets
    VVA = max(run_df.data["VVA"])

    if is_dc:
        index = run_df.data['Field'][0]
        title = f'Field: {index} G'
    else:
        index = run_df.data["Wiggle Time"][0] + (pulse_time/1000)/2 # ms, should be same for all cycles
        title = f'Wiggle Time: {index:0.2f} ms'

    popts, perrs, plabel, sinc2 = fit_sinc2(data[["detuning", "c5transfer"]].values, 
                                                width=1/pulse_time)
    detuning, c5transfer = data["detuning"], data["c5transfer"]

    # Save for later plotting
    valid_results.append({
        "detuning": detuning,
        "c5transfer": c5transfer,
        "sinc2": sinc2,
        "popts": popts,
        "plabel": plabel,
        "title": title,
        "residuals": c5transfer - sinc2(detuning, *popts),
        "predicted": sinc2(detuning, *popts)
    })

    if popts[0] > amp_cutoff: 
        # C from amplitude
        contact = Contact_from_amplitude(popts[0], perrs[0], VVA)
        df.loc[index] = np.concatenate([popts, perrs, contact]) # lists should be concatenated in order
    else:
         dropped_list.append(index)
        
# field_df.dropna(how='all').to_csv(runpath + f'DC_field_cal.csv',  index=None, sep=',')      

###plot all the dimer fits on one multi grid plot
if show_dimer_plot and valid_results:
    num_plots = len(valid_results)
    cols = 3
    rows = int(np.ceil(num_plots / cols))

    # fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True, sharey=True)
    # axes = axes.flatten()

    # for ax, result in zip(axes, valid_results):
    #     xs = np.linspace(max(result["detuning"]), min(result["detuning"]), 100)
    #     ax.plot(xs, result["sinc2"](xs, *result["popts"]), ls="-", marker="", label=result["plabel"])
    #     ax.plot(result["detuning"], result["c5transfer"], marker='o', ls='None')
    #     ax.set_title(result["title"], fontsize=10)
    #     ax.legend(fontsize=6)
    #     ax.set(xlabel="detuning (MHz)", ylabel="transfer")
        
    #     # plot residuals as inset
    #     inset_ax = inset_axes(ax,
    #                             width="50%", height="30%",
    #                             loc='lower right',
    #                             bbox_to_anchor=(0.01, 0.09, 1,1),
    #                             bbox_transform=ax.transAxes,
    #                             #   borderpad=1,
    #                             )
    #     inset_y = result['residuals']
    #     inset_x = result['predicted']
    #     inset_ax.plot(inset_x, inset_y, color='mediumvioletred', marker='.',
    #                         #   label='Residuals'
    #                         )
    #     inset_ax.set(
    #             xlabel = 'predicted',
    #             ylabel = 'residuals'
    #             )
        
    #     inset_ax.tick_params(labelsize=10)

    #     inset_ax.axhline(0, ls="--", color="lightgrey")

    # # Hide any unused subplots
    # for ax in axes[num_plots:]:
    #     ax.set_visible(False)

    fig = plt.figure(figsize=(4 * cols, 3 * rows))
    outer = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.5)

    for i, result in enumerate(valid_results):
        row = i // cols
        col = i % cols

        # inner grid for each panel (main + residual)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col],
            height_ratios=[3, 1], hspace=0.05  
        )

        main_ax = fig.add_subplot(inner[0])
        resid_ax = fig.add_subplot(inner[1], sharex=main_ax)

        # Main plot
        xs = np.linspace(max(result["detuning"]), min(result["detuning"]), 100)
        main_ax.plot(xs, result["sinc2"](xs, *result["popts"]), ls="-", marker="", label=result["plabel"], color='deeppink')
        main_ax.plot(result["detuning"], result["c5transfer"], marker='o', ls='None', color='orchid')
        main_ax.set_title(result["title"], fontsize=10)
        main_ax.legend(fontsize=6)
        main_ax.set_ylabel("transfer")
        main_ax.tick_params(labelbottom=False)  # Hide x labels for main plot

        # Residuals
        resid_ax.plot(result["detuning"], result["residuals"], color='mediumvioletred', marker='.', linestyle='None')
        resid_ax.axhline(0, color="lightgrey", linestyle="--")
        resid_ax.set_xlabel("detuning (MHz)")
        resid_ax.set_ylabel("resid.")
        resid_ax.tick_params(labelsize=8)

    fig.suptitle(run, y=0.93)
    plt.tight_layout()
    plt.show()

# drop NaN rows from dfs
df = df.dropna()

###plotting

if is_dc:
    # plot without residuals
    fig, axs = plt.subplots(3,1, sharex=True, figsize=(8,6))

    axs[0].errorbar(df.index, df.A, yerr=df.eA)
    axs[0].set_ylabel("transfer")
    axs[1].errorbar(df.index, df.x0, 
                    yerr=df.ex0
                    )
    axs[1].set_ylabel("f0 [MHz]")
    axs[2].errorbar(df.index, df.C,
                    yerr=df.eC
                    )
    axs[2].set(
        ylabel = r'transf/VVA$^2$',
        xlabel = xlabel
    )
    axs[0].set_title(f"{run}: {pulse_time}us Pulse, {VVA} VVA")

else:

    fig = plt.figure(figsize=(8, 10))
    # outer grid for each variable
    outer = gridspec.GridSpec(3, 1, hspace=0.2)

    # inner grid for each panel (main + residual)
    gs = [gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=out_gs,
                height_ratios=[3, 1], hspace=0.05  
                ) for i, out_gs in enumerate(outer)]
    
    # gs = gridspec.GridSpec(6, 1, height_ratios=[3,1, 3,1, 3,1], hspace=0.06)

    # peak transfer
    ax0 = fig.add_subplot(gs[0][0])
    ax0.errorbar(df.index, df.A, yerr=df.eA)
    ax0.set_ylabel("peak transfer")
    ax0.tick_params(labelbottom=False)

    # f0
    ax1 = fig.add_subplot(gs[1][0])
    ax1.errorbar(df.index, df.x0, yerr=df.ex0 )
    ax1.set_ylabel("f0 [MHz]")
    ax1.tick_params(labelbottom=False)

    # contact
    ax2 = fig.add_subplot(gs[2][0])
    ax2.errorbar(df.index, df.C,
                    yerr=df.eC )
    ax2.set(ylabel = r'transf/VVA$^2$')
    ax2.tick_params(labelbottom=False)

    # plot fits to sine
    ###Sine fits for the wiggle data pts 
    # sine should be same for both data
    poptsA, perrsA, plabelA, sine = fit_fixedSinkHz(df.index, df.A, wiggle_freq, df.eA)
    poptsf0, perrsf0, plabelf0, __ = fit_fixedSinkHz(df.index, df.x0, wiggle_freq, df.ex0)

    poptsC, perrsC, plabelC, sine_C = fit_fixedSinkHz(df.index, 
                                                    df.C, 
                                                    wiggle_freq, df.eC)

    ###plotting sin fit to 1/A and f0 and C 
    ts = np.linspace(min(df.index), max(df.index), 100)
    ax0.plot(ts, sine(ts, *poptsA), ls="-", marker="", 
                label=plabelA
                )
    fit1 = ax1.plot(ts, sine(ts, *poptsf0), ls="-", marker="", label=plabelf0)
    ax2.plot(ts, sine_C(ts, *poptsC), ls="-", marker="", label=plabelC)

    ###plotting the Field wiggle in B (G)
    B_phase = field_cal['B_phase'].values[0]
    eB_phase = field_cal['e_B_phase'].values[0]
    field_params = field_cal[['B_amp', 'B_phase', 'B_offset']].values[0]
    Bs = sine(ts, *field_params)

    B_label = fit_label(field_cal[['B_amp', 'B_phase', 'B_offset']].values[0], 
                        field_cal[['e_B_amp', 'e_B_phase', 'e_B_offset']].values[0],
                        ["A", "p", "C"])
    ax1_B = ax1.twinx()
    B1 = ax1_B.plot(ts, Bs, color="cornflowerblue", ls='--', marker="")

    ax1_B.set_ylabel("B(cal) [G]")

    ax2_B = ax2.twinx()
    ax2_B.plot(ts, Bs, color="cornflowerblue", ls='--', marker="")
    ax2_B.set_ylabel("B(cal) [G]")

    # flip y axis instead of phase shifting
    ax1_B.invert_yaxis()
    ax2_B.invert_yaxis()

    fig.legend(B1, [B_label], loc='upper center', bbox_to_anchor=(1, 0.85), title="field fit")

    ###plotting residuals
    resid0 = fig.add_subplot(gs[0][1], sharex=ax0)
    resid1 = fig.add_subplot(gs[1][1], sharex=ax1)
    resid2 = fig.add_subplot(gs[2][1], sharex=ax2)

    resid0.axhline(0, ls="--", color="lightgrey")
    resid1.axhline(0, ls="--", color="lightgrey")
    resid2.axhline(0, ls="--", color="lightgrey")

    resid0.errorbar(df.index, df.A-sine(df.index, *poptsA), yerr=df.eA, color='mediumvioletred')
    resid1.errorbar(df.index, df.x0-sine(df.index, *poptsf0), yerr=df.ex0, color='mediumvioletred')
    resid2.errorbar(df.index, df.C-sine(df.index, *poptsC), df.eC, color='mediumvioletred')

    resid0.set_xlabel("time (ms)")
    resid1.set_xlabel("time (ms)")
    resid2.set_xlabel("time (ms)")

    if plot_dc:
        # plot DC cal'd field/amp values
        B_DC = DC_cal_csv["Field"]
        imax, imin, imean = np.argmin(np.abs(Bs-B_DC[0])), np.argmin(np.abs(Bs-B_DC[1])), np.argmin(np.abs(Bs-B_DC[2]))
        t_DC = ts[[imax, imin, imean]] - 1/(2*wiggle_freq) # shift field phase by pi 
        # plot DC f0
        ax1.errorbar(t_DC, DC_cal_csv["x0"].values, DC_cal_csv["ex0"].values, color='navy', label='DC Field')
        # plot DC amplitude
        C_DC = Contact_from_amplitude(DC_cal_csv["A"].values, DC_cal_csv["eA"].values, DC_VVA)
        ax2.errorbar(t_DC, C_DC[0], C_DC[1], color='navy', label='DC Field')

        # ###finding max, min, middle points of f0 fit
        # maxy = np.argmax(sine(ts, *poptsf0))
        # maxx = ts[maxy]
        # miny = np.argmin(sine(ts, *poptsf0))
        # minx = ts[miny]

        # #add this to DC cal csv 
        # DC_cal_csv['time'] = [maxx, (maxx+minx)/2, minx]
        # #putting on plot
        # axs[1].plot(DC_cal_csv['time']+0.083,DC_cal_csv['x0'], color='navy', label='DC Field')
        # # axs[0].plot(DC_cal_csv['time']+0.083,DC_cal_csv['A'], color='navy', label='DC Field')
        # #the time is being shifted overy by 1/2 a period, unsure why? 

    ax0.legend(loc=0)
    ax1.legend(loc=0)
    ax2.legend(loc=0)

    # reorder axes so legend is not covered by B line
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    ax2.set_zorder(2)
    ax2.patch.set_visible(False)

    ###Finding the phase shift by subtracting various fits
    # add pi offset to B field to compare phase shift
    phases = [poptsA[1] - poptsf0[1],
            poptsf0[1] - (B_phase + np.pi),
            poptsC[1] - (B_phase + np.pi)
    ]
    ephases = [
        np.sqrt(perrsA[1]**2 + perrsf0[1]**2),
        np.sqrt(perrsf0[1]**2 + eB_phase**2),
        np.sqrt(perrsC[1]**2 + eB_phase**2)
        ]

    # if phase is negative, mod by 2pi
    # for i, p in enumerate(phases):
    #     if p<0:
    #         phases[i] %= 2*np.pi

    # convert angles from [-2pi, 2pi] to between [-pi, pi]
    [(phases[i] + np.pi) % (2 * np.pi) - np.pi for i in range(len(phases))]

    try:
        fits = fit_label(phases, ephases, ["phase shift A-f0", "phase shift f0-B", "phase shift C-B"], 
                         units = ["", r"+$\pi$", r"+$\pi$"])
    except:
        # errors not available
        fits = (f"phase shift A-f0 = {phases[0]:.2f}" 
                f"\nphase shift f0-B = {phases[1]:.3f}" 
                f"\nphase shift C-B = {phases[2]:.2f}")
        
    fig.suptitle(f"{run}\n{pulse_time}us Pulse, {wiggle_freq}kHz Modulation, {VVA} VVA\n{fits}" +\
                f'\nDropped Wiggle Times: {[float(x) for x in dropped_list]}', y=1.02)
    
###need to pi shift C to compare phase shift 