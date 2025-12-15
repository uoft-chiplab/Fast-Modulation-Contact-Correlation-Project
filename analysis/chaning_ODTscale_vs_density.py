#%%
from preamble import *
from unitary_fermi_gas import TrappedUnitaryGas as TUG 

bar_nu_scale1 = (157*370*431)**(1/3) #geometric trap freq based on most recent cals

def bar_nu_var(scale):
    return bar_nu_scale1*np.sqrt(scale)

ToTF = 0.3
EF = 12000 #Hz 

scale_list = [0.5,1,1.5]
EF_list = [np.sqrt(x) * EF for x in scale_list] #EF propto N * barn_nu 
dict_list = []

for (scale, EF) in zip(scale_list, EF_list):
    TUGs = TUG(ToTF, EF, bar_nu_var(scale))

    dict_list.append({
        'scale': scale,
        'EF': TUGs.EF, 
        'T': TUGs.T, 
        'ToTF': ToTF,
        'density': TUGs.density,
        'thermal_debrog': TUGs.lambda_T,
        'trap_den_x_thermal_debrog**3': TUGs.psd_trap/TUGs.Ns,
        'Ns': TUGs.Ns,
        'kF': TUGs.kF,
        'n_peak': TUGs.kF**3*3*np.pi**2,
        'Contact': TUGs.calc_contact()
    })

#creating a dataframe from the dictionary created from the loop above
df = pd.DataFrame(dict_list)
#%%
fig, ax = plt.subplots(1,1)

ax.plot(scale_list, df['density'])
ax.set(
    xlabel = 'ODT Scale', 
    ylabel = rf'$\langle n\rangle$'
)
ax2 = ax.twinx()
# ax2.plot(scale_list, df['EF'])
ax2.set(
    ylim = [df['EF'].min(), df['EF'].max()],
    ylabel = 'EF'
)

fig.tight_layout()
# %%
###Grabbing data for 202.24 (2025-11-05_P) and 202.04 (2025-11-05_O) and plotting

data_202p24 = Data('2025-11-05_P').data
data_202p04 = Data('2025-11-05_O').data

avg_by = 'ramptime'
avgdata_202p24 = Data('2025-11-05_P', average_by = avg_by).avg_data
avgdata_202p04 = Data('2025-11-05_O', average_by = avg_by).avg_data

def Linear(x, m, b):
    return m*x + b

xs = np.linspace(0, 1, 100)
avgx = data_202p24['ramptime'].mean()

to_plot_list = ['N', 'EFkHz', 'ToTF', 'T']
decimals_list = [0, 3, 3, 'e']

avgs_df = pd.DataFrame(columns = to_plot_list, dtype=float)

fig, ax = plt.subplots(int(len(to_plot_list)/2), int(len(to_plot_list)/2),
                       figsize = (10,8))

for i, (key, decimals) in enumerate(zip((to_plot_list),decimals_list)):
    popt, pcov = curve_fit(Linear, data_202p24['ramptime'], data_202p24[key])

    row = i // 2
    col = i % 2
    ax[row, col].plot(avgdata_202p24['ramptime'], avgdata_202p24[key], 'o', 
                    #   label = '202.24 Data'
                      )
    ax[row, col].plot(xs, Linear(xs, *popt), '-', 
                    #   label = f'Linear Fit'
                      )
    
    if decimals == 'e':
        label_text = f'Average {key}, {Linear(avgx, *popt):.2e}'
    else:
        label_text = f'Average {key}, {Linear(avgx, *popt):0.{decimals}f}'
    ax[row, col].hlines(Linear(avgx, *popt), xmin = xs.min(), xmax = xs.max(), colors = 'C1', linestyles = '--', label = label_text)

    ax[row, col].set(
        xlabel = 'Ramp Time (s)',
        ylabel = key
    )
    ax[row, col].legend()
    fig.suptitle('202.24 Data')
    fig.tight_layout()

    avgs_df.loc['202.24', key] = Linear(avgx, *popt)
    # print(f'{key}: {Linear(avgx, *popt)}')


fig, ax = plt.subplots(int(len(to_plot_list)/2), int(len(to_plot_list)/2),figsize = (10,8))

for i, (key, decimals) in enumerate(zip((to_plot_list),decimals_list)):
    popt, pcov = curve_fit(Linear, data_202p04['ramptime'], data_202p04[key])

    row = i // 2
    col = i % 2
    ax[row, col].plot(avgdata_202p04['ramptime'], avgdata_202p04[key], 'o', 
                    #   label = '202.24 Data'
                      )
    ax[row, col].plot(xs, Linear(xs, *popt), '-', 
                    #   label = f'Linear Fit'
                      )
    if decimals == 'e':
        label_text = f'Average {key}, {Linear(avgx, *popt):.2e}'
    else:
        label_text = f'Average {key}, {Linear(avgx, *popt):0.{decimals}f}'
    ax[row, col].hlines(Linear(avgx, *popt), xmin = xs.min(), xmax = xs.max(), colors = 'C1', linestyles = '--', label = label_text)

    ax[row, col].set(
        xlabel = 'Ramp Time (s)',
        ylabel = key,
        
    )
    ax[row, col].legend()
    fig.suptitle('202.04 Data')
    fig.tight_layout()

    avgs_df.loc['202.04', key] = Linear(avgx, *popt)

avgs_df.loc['202.24', 'ODT Scale'] = 1
avgs_df.loc['202.04', 'ODT Scale'] = 1
# %%
dict_list_changing_fields_varied = []
for scale in scale_list:
    print(scale)
    TUG_202p24_varied_scale = TUG(avgs_df.loc['202.24', 'ToTF'], 
                 avgs_df.loc['202.24', 'EFkHz']*1e3*np.sqrt(scale),
                 bar_nu_var(scale))
    
    dict_list_changing_fields_varied.append({
        'Field': '202.24',  
        'EF': TUG_202p24_varied_scale.EF,
        'T': TUG_202p24_varied_scale.T,
        'ToTF': avgs_df.loc['202.24', 'ToTF'],
        'density': TUG_202p24_varied_scale.density,
        'thermal_debrog': TUG_202p24_varied_scale.lambda_T,
        'trap_den_x_thermal_debrog**3': TUG_202p24_varied_scale.psd_trap/TUG_202p24_varied_scale.Ns,
        'Ns': TUG_202p24_varied_scale.Ns,
        'kF': TUG_202p24_varied_scale.kF,
        'n_peak': TUG_202p24_varied_scale.kF**3*3*np.pi**2 ,
        'ODT Scale': scale,
        'Contact': TUG_202p24_varied_scale.calc_contact()
    }) 

TUG_202p24 = TUG(avgs_df.loc['202.24', 'ToTF'], 
                 avgs_df.loc['202.24', 'EFkHz']*1e3,
                 bar_nu_var(avgs_df.loc['202.24', 'ODT Scale']))
TUG_202p04 = TUG(avgs_df.loc['202.04', 'ToTF'], 
                 avgs_df.loc['202.04', 'EFkHz']*1e3,
                 bar_nu_var(avgs_df.loc['202.04', 'ODT Scale']))

dict_list_changing_fields = []

dict_list_changing_fields.append({
    'Field': '202.24',  
    'EF': TUG_202p24.EF,
    'T': TUG_202p24.T,
    'ToTF': avgs_df.loc['202.24', 'ToTF'],
    'density': TUG_202p24.density,
    'thermal_debrog': TUG_202p24.lambda_T,
    'trap_den_x_thermal_debrog**3': TUG_202p24.psd_trap/TUG_202p24.Ns,
    'Ns': TUG_202p24.Ns,
    'kF': TUG_202p24.kF,
    'n_peak': TUG_202p24.kF**3*3*np.pi**2 ,
    'ODT Scale': avgs_df.loc['202.24', 'ODT Scale'],
    'Contact': TUG_202p24.calc_contact()
}) 

dict_list_changing_fields.append({
    'Field': '202.04',
    'EF': TUG_202p04.EF,
    'T': TUG_202p04.T,
    'ToTF': avgs_df.loc['202.04', 'ToTF'],
    'density': TUG_202p04.density,
    'thermal_debrog': TUG_202p04.lambda_T,
    'trap_den_x_thermal_debrog**3': TUG_202p04.psd_trap/TUG_202p04.Ns,
    'Ns': TUG_202p04.Ns,
    'kF': TUG_202p04.kF,
    'n_peak': TUG_202p04.kF**3*3*np.pi**2 ,
    'ODT Scale': avgs_df.loc['202.04', 'ODT Scale'],
    'Contact': TUG_202p04.calc_contact()
})

df_changing_fields = pd.DataFrame(dict_list_changing_fields)
# %%
fig, ax = plt.subplots()

ax.plot(df_changing_fields['Field'], df_changing_fields['density'], 'o-', ls='')
ax.set(
    xlabel = 'Magnetic Field (G)',
    ylabel = r'$\langle n \rangle$'
)

ax2 = ax.twinx()
ax2.plot(df_changing_fields['Field'], df_changing_fields['EF'], 'o-', ls='')
ax2.set(
    ylabel = 'EF (Hz)'
)
#%%
###theory based on ToTF and EF from 202.24/202.04 data 
###i want the number to be the same as in the data so i can compare densities
###since the # from the theory above is ~11k and in the data its ~6k
TUG_202p14_theory = TUG(0.3, 9800, bar_nu_var(1))

dict_list = []

dict_list.append(
{
     'scale': 1,
        'EF': TUG_202p14_theory.EF, 
        'T': TUG_202p14_theory.T, 
        'ToTF': 0.3,
        'density': TUG_202p14_theory.density,
        'thermal_debrog': TUG_202p14_theory.lambda_T,
        'trap_den_x_thermal_debrog**3': TUG_202p14_theory.psd_trap/TUG_202p14_theory.Ns,
        'Ns': TUG_202p14_theory.Ns,
        'kF': TUG_202p14_theory.kF,
        'n_peak': TUG_202p14_theory.kF**3*3*np.pi**2,
        'Contact': TUG_202p14_theory.calc_contact()
}
)

df_theory = pd.DataFrame(dict_list)
#%%
fig, ax = plt.subplots(2,2, figsize=(11,8))

ax = ax.flatten()

N_value = df_changing_fields[df_changing_fields['Field'] == '202.24']['Ns'].values[0]
N_value_202p04 = df_changing_fields[df_changing_fields['Field'] == '202.04']['Ns'].values[0]

ax[0].plot(df_theory['scale'], df_theory['density'], 'o', color='cornflowerblue',
        label=f'Theory 202.14, ToTF=0.3, EF=9800Hz, N={df_theory.loc[0, "Ns"]:.0f}'
        )
ax[0].plot(df_changing_fields[df_changing_fields['Field'] == '202.24']['ODT Scale'], 
        df_changing_fields[df_changing_fields['Field'] == '202.24']['density'], 
        label=f'202.24, ToTF={avgs_df.loc["202.24", "ToTF"]:.2f}, EF={avgs_df.loc["202.24", "EFkHz"]*1e3:.0f}Hz, N={N_value:.1f}'
        )
ax[0].plot(df_changing_fields[df_changing_fields['Field'] == '202.04']['ODT Scale'], 
        df_changing_fields[df_changing_fields['Field'] == '202.04']['density'], 
        label=f'202.04, ToTF={avgs_df.loc["202.04", "ToTF"]:.2f}, EF={avgs_df.loc["202.04", "EFkHz"]*1e3:.0f}Hz, N={N_value_202p04:.1f}'
        )
ax[0].set(
    xlabel = 'ODT Scale', 
    ylabel = r'$\langle n \rangle$'
)

ax[1].plot(df_theory['scale'], df_theory['Contact'], color='cornflowerblue', label=f'202.14, ToTF={ToTF}'
        )
ax[1].plot(df_changing_fields[df_changing_fields['Field'] == '202.24']['ODT Scale'], 
        df_changing_fields[df_changing_fields['Field'] == '202.24']['Contact'], 
        label=f'202.24, ToTF={avgs_df.loc["202.24", "ToTF"]:.2f}, EF={avgs_df.loc["202.24", "EFkHz"]*1e3:.0f}Hz'
        )
ax[1].plot(df_changing_fields[df_changing_fields['Field'] == '202.04']['ODT Scale'], 
        df_changing_fields[df_changing_fields['Field'] == '202.04']['Contact'], 
        label=f'202.04, ToTF={avgs_df.loc["202.04", "ToTF"]:.2f}, EF={avgs_df.loc["202.04", "EFkHz"]*1e3:.0f}Hz'
        )
ax[1].set(
    xlabel = 'ODT Scale', 
    ylabel = r'$\langle C \rangle$'
)
ax[0].legend()

ax[2].plot(scale_list, df['density'], color='cornflowerblue', label=f'202.14, ToTF={ToTF}'
        )

ax[2].set(
    xlabel = 'ODT Scale',
    ylabel = r'$\langle n \rangle$'
)
ax2 = ax[2].twinx()
ax2.set(
    ylim = [df['EF'].min(), df['EF'].max()],
    ylabel = 'EF'
)

fig.tight_layout()
# %%
fig, ax = plt.subplots()

ax.plot(scale_list, df['density'], color='cornflowerblue', label=f'202.14, ToTF={ToTF}'
        )
ax.plot(df_changing_fields[df_changing_fields['Field'] == '202.24']['ODT Scale'], 
        df_changing_fields[df_changing_fields['Field'] == '202.24']['density'], 
        label=f'202.24, ToTF={avgs_df.loc["202.24", "ToTF"]:.2f}, EF={avgs_df.loc["202.24", "EFkHz"]*1e3:.0f}Hz'
        )
ax.plot(df_changing_fields[df_changing_fields['Field'] == '202.04']['ODT Scale'], 
        df_changing_fields[df_changing_fields['Field'] == '202.04']['density'], 
        label=f'202.04, ToTF={avgs_df.loc["202.04", "ToTF"]:.2f}, EF={avgs_df.loc["202.04", "EFkHz"]*1e3:.0f}Hz'
        )
ax.set(
    xlabel = 'ODT Scale', 
    ylabel = r'$\langle n \rangle$'
)
# ax2 = ax.twinx()
# ax2.plot(df['scale'], df['EF'], color='cornflowerblue')
# ax2.set(
#     ylabel = 'EF (Hz)'
# )
ax.legend()
# %%


