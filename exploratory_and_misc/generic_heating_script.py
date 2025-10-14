import os
import sys
root_project = os.path.dirname(os.getcwd())

# Fast-Modulation-Contact-Correlation-Project\analysis
module_folder = os.path.join(root_project, "analysis")
if module_folder not in sys.path:
	sys.path.append(module_folder)
# settings for directories, standard packages...
from preamble import *

from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from UFG_analysis import BulkViscTrap #??
# blherggg = BulkViscTrap(ToTF, EF, barnu, nus)
# blherggg.EdotDrude


line = lambda x, m, b: m*x + b

plot = True
trap_freqs = np.array([146.7, 385.7, 434])*np.sqrt(1.5) # in Hz
barnu = np.mean(trap_freqs)
# run name
runs = [f'2025-10-06_{i}' for i in ['F', 'G', 'H']]
df = pd.DataFrame({'run': runs, 'Edot': np.nan, 'e_Edot': np.nan, 
                   'ToTF': np.nan, 'EF': np.nan,
                   'EdotDrude':np.nan})

for run in runs:
    # should make these into 1 figure grid

    print(f"Processing run {run}...")
    # find data files
    y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
    runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
    datfiles = glob(f"{runpath}*.dat")

    run_df = pd.read_csv(datfiles[0])
    zoom_df = run_df[run_df['time'] < 1] 
    time_zoom, T_zoom, ToTF_zoom = zoom_df['time'], zoom_df['T']*1e9, zoom_df['ToTF']

    # get run params 
    runname = runpath.split("\\")[-2].lower().split("_")[1]
    i = runname.find("khz")
    wiggle_freq = float(runname[:i])
    i = runname.find("vpp")
    Vpp = float(runname[i-3:i].replace("p", "."))

    ## linear fit to 1 ms
    popt_T, pcov_T = curve_fit(line, time_zoom, T_zoom)
    popt_ToTF, pcov_ToTF = curve_fit(line, time_zoom, ToTF_zoom)
    perr_T = np.sqrt(np.diag(pcov_T))
    perr_ToTF = np.sqrt(np.diag(pcov_ToTF))
    
    ToTF, EF = run_df[run_df['time'] <=0.1][["ToTF", "EF"]].mean()
    blherggg = BulkViscTrap(ToTF, EF, barnu, np.array([wiggle_freq]))
    EdotDrude = blherggg.EdotDrude
    #add to df
    df.loc[df['run'] == run, 
           ['Edot', 'e_Edot', 'ToTF', 'EF', 'EdotDrude']] = \
            [popt_ToTF[0], perr_ToTF[0], ToTF, EF, *EdotDrude]

    if plot:
        time, T, ToTF = run_df['time'], run_df['T']*1e9, run_df['ToTF'] # convert T to nK

        ## plot data
        fig, (ax0, ax1) = plt.subplots(2, sharex=True, figsize=(8, 6))
        ax0.plot(time, T, marker='o', ls='', color='magenta')
        ax1.plot(time, ToTF, marker='o', ls='', color='magenta')

        ax1.set(xlabel='Time (ms)', ylabel=r'T/T$_\text{F}$')
        ax0.set(ylabel='T (nK)')

        # plot averaged points
        run_df_avg = run_df.groupby('time', as_index=False)
        run_df_avg = run_df_avg.mean()[run_df_avg.count()['cyc'] >=2] # only plot points with more than 3 counts
        ax0.plot(run_df_avg['time'], run_df_avg['T']*1e9, marker='o', ls='', color='darkmagenta', label='averaged')
        ax1.plot(run_df_avg['time'], run_df_avg['ToTF'], marker='o', ls='', color='darkmagenta', label='averaged')

        ## add inset for 1 ms zoom
        inset0 = inset_axes(ax0, width="30%", height="40%", loc='lower right', borderpad=2.)
        inset0.plot(time_zoom, T_zoom, color="navy")

        # Add inset to ax2
        inset1 = inset_axes(ax1, width="30%", height="40%", loc='lower right', borderpad=2.)
        inset1.plot(time_zoom, ToTF_zoom, color="navy")
        # inset1.plot(line(time_zoom, *popt_T), color="navy", ls='--')

        # get y limit, since fit will change it
        ylim0 = ax0.get_ylim()
        ylim1 = ax1.get_ylim()

        times_fine = np.linspace(min(time), max(time), 2) # just two points for linear fit
        
        ax0.plot(times_fine, line(times_fine, *popt_T), ls='--', marker="", color='navy', 
                    label=fit_label(popt_T, perr_T, ['m', 'b'], units=[' nK/ms', ' nK']))
        ax1.plot(times_fine, line(times_fine, *popt_ToTF), ls='--', marker="", color='navy',
                    label=fit_label(popt_ToTF, perr_ToTF, ['m', 'b'], units=[r' ms$^{-1}$','']))

        ax0.set_ylim(ylim0[0], ylim0[1])
        ax1.set_ylim(ylim1[0], ylim1[1])
        
        # legend
        ax0.legend(loc=2, title="linear fit (t < 1 ms)", framealpha=0.7, title_fontsize='small')
        ax1.legend(loc=2, title="linear fit (t < 1 ms)", framealpha=0.7, title_fontsize='small')


        fig.suptitle( f'{run} heating: {wiggle_freq:.0f} kHz, {Vpp} Vpp', y=0.95)


# compare the heating rate result to Tilman’s heating rate code
# hmmmmmm
# run.Bamp, run.e_Bamp = Bamp_from_Vpp(run.Vpp, run.freq)
# run.A, run.e_A = calc_A(run.B, run.T, run.e_T, run.Bamp, run.e_Bamp)
# run.rate = run.Edot/run.EF**2/run.A**2
# run.e_rate= run.rate*np.sqrt((run.e_Edot/run.Edot)**2+ \
#                     (2*run.e_EF/run.EF)**2+(2*run.e_A/run.A))