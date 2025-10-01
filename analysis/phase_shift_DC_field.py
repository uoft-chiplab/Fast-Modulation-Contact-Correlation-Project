import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from glob import glob
import sys 
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=['hotpink', 'plum'])

module_folder = 'E:\\Analysis Scripts\\analysis'
if module_folder not in sys.path:
	sys.path.insert(0, 'E:\\Analysis Scripts\\analysis')
from data_class import Data
from library import fit_label, a0, pi
from fit_functions import Sinc2

root = "E:/Data"
run = "2025-09-31_D" # need date and letter
field_cal_path = r"E:\Analysis Scripts\Fast-Modulation-Contact-Correlation-Project\FieldWiggleCal\field_cal_summary.csv"
field_cal_df = pd.read_csv(field_cal_path)
field_cal = field_cal_df[field_cal_df['run'] == '2024-04-05_G'] 
# 2025-09-19_D is for 6 kHz 1.8 Vpp
# 2024-04-05_G is for 10 kHz 1.8 Vpp

show_dimer_plot = True
fix_sinc_width = True

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

        plabel = fit_label(popts, perrs, ["A", "p", "C"])#, units=["", f"$\pi$", ""])

    else: 
        popts, pcov = curve_fit(FixedSinkHz, t, y, p0, bounds=([0, 0, 0], [np.inf, 2*np.pi, np.inf]))
        perrs = np.sqrt(np.diag(pcov))

    # rescales phase by pi for label
        plabel = fit_label(popts, perrs, ["A", "p", "C"])#, units=["", f"$\pi$", ""])

    return popts, perrs, plabel, FixedSinkHz

###amplitude to contact 
def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

def Contact_from_amplitude(A, eA):
     N = data["c5"].max()
     kF =(3*np.pi**2*N)**(1/3)
     a13kF = kF * a13(202.14)
     spin_me = 32/42 # spin matrix element
     ell_d_CCC = spin_me * 42 * np.pi

     C = A / kF * a13kF * np.pi / ell_d_CCC / a0 
     eC = eA / kF * a13kF * np.pi / ell_d_CCC / a0 
     return C, eC

#I_d_CCC =  kF / a13kF / pi * ell_d_CCC * a0 * C

# find data files
y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
runpath = glob(f"{root}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
datfiles = glob(f"{runpath}*.dat")
runname = datfiles[0].split("\\")[-2].lower() # get run folder name, should be same for all files
times = np.unique(np.loadtxt(glob(f"{runpath}commands/incremental.txt*")[0], 
                   delimiter=" ", usecols=[5])) # read time col from incremental file

# get run attributes
run_attr = runname.split("_")
wiggle_freq = float(runname[:runname.find("khz")].split("_")[-1])
i = runname.find("us")
pulse_time = float(runname[i-2:i]) # ms

# fit dimer
wiggle_df = pd.DataFrame(index = times, 
                         columns = ["A", "x0", "eA", "ex0"] if fix_sinc_width else ["A", "x0", "sigma", "b", "eA", "ex0", "esigma", "eb"], 
                         dtype=float)
###amplitude to contact 
contact_from_amplitdue_df = pd.DataFrame(index = times, 
                            columns = ['C', 'eC'], 
                            dtype=float)

valid_results = []
dropped_list = []
for fpath in datfiles:
    run_df = Data(fpath.split("\\")[-1])
    wiggle_time = run_df.data["Wiggle Time"][0] + (pulse_time/1000)/2 # ms, should be same for all cycles

    data = find_transfer(run_df.data)
    # data = data.iloc[0:14]
    if fix_sinc_width:
        popts, perrs, plabel, sinc2 = fit_sinc2(data[["detuning", "c5transfer"]].values, 
                                                width=fix_sinc_width/pulse_time)
    detuning, c5transfer = detuning, c5transfer = data["detuning"], data["c5transfer"]

        # Save for later plotting
    valid_results.append({
        "detuning": detuning,
        "c5transfer": c5transfer,
        "sinc2": sinc2,
        "popts": popts,
        "plabel": plabel,
        "title": f'Wiggle Time: {wiggle_time:0.2f} ms' #fpath.split("\\")[-1]
    })

    # if show_dimer_plot:
    #     fig, ax = plt.subplots(1,1)
    #     xs = np.linspace(max(detuning), min(detuning), 100)
    #     ax.plot(xs, sinc2(xs, *popts), ls="-", marker="",  label=plabel)
    #     ax.plot(detuning, c5transfer)
    #     ax.legend()
    #     ax.set(
    #         xlabel="detuning (MHz)",
    #         ylabel="transfer"
    #     )
    #     ax.set_title(fpath.split("\\")[-1])

    if popts[0] > 0.01: 

        wiggle_df.loc[wiggle_time] = np.concatenate([popts, perrs]) # lists should be concatenated in order
    # want time, f0, A -- fit f0, A to time
    # C from amplitude
        contact_from_amplitdue_df.loc[wiggle_time] = Contact_from_amplitude(popts[0], perrs[0])
    if popts[0] < 0.01:
        dropped_list.append(wiggle_time)

###plot all the dimer fits on one multi grid plot
if show_dimer_plot and valid_results:
    num_plots = len(valid_results)
    cols = 3
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, result in zip(axes, valid_results):
        xs = np.linspace(max(result["detuning"]), min(result["detuning"]), 100)
        ax.plot(xs, result["sinc2"](xs, *result["popts"]), ls="-", marker="", label=result["plabel"])
        ax.plot(result["detuning"], result["c5transfer"], marker='o', ls='None')
        ax.set_title(result["title"], fontsize=10)
        ax.legend(fontsize=6)
        ax.set(xlabel="detuning (MHz)", ylabel="transfer")

    # Hide any unused subplots
    for ax in axes[num_plots:]:
        ax.set_visible(False)

    fig.suptitle(str(fpath.split("\\")[-1])[0:12])
    plt.tight_layout()
    plt.show()

# drop NaN rows from dfs
wiggle_df = wiggle_df.dropna()
contact_from_amplitdue_df = contact_from_amplitdue_df.dropna()

###plotting
fig, axs = plt.subplots(3,1, sharex=True, figsize=(8,6))
axs[0].errorbar(wiggle_df.index, wiggle_df.A, yerr=wiggle_df.eA)
axs[0].set_ylabel("amplitude")
axs[1].errorbar(wiggle_df.index, wiggle_df.x0, 
                yerr=wiggle_df.ex0
                )
axs[1].set_ylabel("f0 [MHz]")
axs[2].errorbar(contact_from_amplitdue_df.index, contact_from_amplitdue_df.C,
                yerr=contact_from_amplitdue_df.eC
                )
axs[2].set(
     ylabel = 'C (scaled A)',
     xlabel = 'time [ms]'
)
###Sine fits for the wiggle data pts 
# sine should be same for both data
poptsA, perrsA, plabelA, sine = fit_fixedSinkHz(wiggle_df.index, wiggle_df.A, wiggle_freq, wiggle_df.eA)
poptsf0, perrsf0, plabelf0, __ = fit_fixedSinkHz(wiggle_df.index, wiggle_df.x0, wiggle_freq, wiggle_df.ex0)

poptsC, perrsC, plabelC, sine_C = fit_fixedSinkHz(contact_from_amplitdue_df.index, 
                                                contact_from_amplitdue_df.C, 
                                                wiggle_freq, contact_from_amplitdue_df.eC)

###plotting sin fit to 1/A and f0 and C 
ts = np.linspace(min(wiggle_df.index), max(wiggle_df.index), 100)
axs[0].plot(ts, sine(ts, *poptsA), ls="-", marker="", label=plabelA)
fit1 = axs[1].plot(ts, sine(ts, *poptsf0), ls="-", marker="", label=plabelf0)
axs[2].plot(ts, sine_C(ts, *poptsC), ls="-", marker="", label=plabelC)

###plotting the Field wiggle in B (G)
B_phase = field_cal['B_phase'].values[0]
eB_phase = field_cal['e_B_phase'].values[0]
field_params = field_cal[['B_amp', 'B_phase', 'B_offset']].values[0]
field_params[1] += np.pi # add pi phase shift to comapre to freq
Bs = sine(ts, *field_params)

B_label = fit_label(field_cal[['B_amp', 'B_phase', 'B_offset']].values[0], 
                    field_cal[['e_B_amp', 'e_B_phase', 'e_B_offset']].values[0],
                    ["A", "p", "C"], units=["", r"+$\pi$", ""])
axs1_B = axs[1].twinx()
B1 = axs1_B.plot(ts, Bs, color="cornflowerblue", ls='--', marker="")
axs1_B.set_ylabel("B(cal) [G]")

axs2_B = axs[2].twinx()
axs2_B.plot(ts, Bs, color="cornflowerblue", ls='--', marker="")
axs2_B.set_ylabel("B(cal) [G]")

fig.legend(B1, [B_label], loc='upper center', bbox_to_anchor=(1, 0.8), title="field fit")
axs[0].legend(loc=0)
axs[1].legend(loc=0)
axs[2].legend(loc=0)

# reorder axes so legend is not covered by B line
axs[1].set_zorder(2)
axs[1].patch.set_visible(False)
axs[2].set_zorder(2)
axs[2].patch.set_visible(False)

###Finding the phase shift by subtracting various fits

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
for i, p in enumerate(phases):
     if p<0:
          phases[i] %= 2*np.pi

title = fit_label(phases, ephases, ["phase shift A-f0", "phase shift f0-B", "phase shift C-B"])
axs[0].set_title(f"{run} {pulse_time}us Pulse {wiggle_freq}kHz Modulation\n{title}"
                 f'\nDropped Wiggle Times: {[float(x) for x in dropped_list]}')

###need to pi shift C to compare phase shift 