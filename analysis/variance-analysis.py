#%%
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

var_csv = pd.read_csv('variance.csv')

cmap = plt.colormaps.get_cmap('viridis')
colours = cmap(np.linspace(0, 2.5, (len(var_csv['Wiggle Time']))))
df = var_csv
# %%
# fig, ax = plt.subplots(1,2,figsize=(8,3.5))

# for i, (wiggle_time, group) in enumerate(df.groupby('Wiggle Time')):
#     wiggle_time_str = f'{np.round(wiggle_time, 3)}'
    
#     for _, row in group.iterrows():
#         x = row['Number Shots']
#         ax[0].plot(x, row['Var c9bg']/row['Var c9'],
#                    color=colours[i],
#                    label=wiggle_time_str if _ == group.index[0] else None)
#         ax[1].plot(x, row['Var c5bg']/row['Var c5'], color=colours[i])
	
# ax[0].legend(loc='upper left')
# %%
fig, ax = plt.subplots(1,2,figsize=(8,3.5))

# Get the wiggle time values for the colormap
wiggle_times = df['Wiggle Time'].values

# Create a colormap and normalization
import matplotlib.cm as cm
from matplotlib.colors import Normalize

norm = Normalize(vmin=wiggle_times.min(), vmax=wiggle_times.max())
cmap = cm.viridis 

for i, j in enumerate(df['Wiggle Time']):
    row = df.loc[i]
    x = row['Number Shots']
    color = cmap(norm(j))
    
    ax[0].plot(x, row['Var c9bg']/row['Var c9'], color=color)
    ax[1].plot(x, row['Var c5bg']/row['Var c5'], color=color)

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='Wiggle Time [ms]')

x = df['Number Shots']

ax[0].hlines(1, min(x), max(x), color='grey', linestyle = '--')

ax[1].hlines(1, min(x), max(x), color='grey', linestyle = '--')

ax[0].set(
    xlabel = 'Number of Shots',
    ylabel = 'Var(c9bg)/Var(c9)'
)
ax[1].set(
    xlabel = 'Number of Shots',
    ylabel = 'Var(c5bg)/Var(c5)'
)

fig.suptitle(f'Variance of bg/signal vs # shots for all HFT runs')
# %%
