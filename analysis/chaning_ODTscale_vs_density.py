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
        'n_peak': TUGs.kF**3*3*np.pi**2
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
