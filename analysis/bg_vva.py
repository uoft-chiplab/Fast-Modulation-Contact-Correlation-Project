"""
looks at change in background VVA overtime
"""

from preamble import *
import matplotlib.dates as mdates

def line(x, m, b):
    return m*x+b

run = "2025-10-01_L" # need date and letter

y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}*/")[0] # note backslash included at end
datfiles = glob(f"{runpath}*Time*.dat")
mscan = pd.read_csv(glob(f"{runpath}*.mscan")[0], skiprows=2)
runname = datfiles[0].split("\\")[-2].lower() # get run folder name, should be same for all files

df0VVA = pd.DataFrame()
for fpath in datfiles:
    run_df = pd.read_csv(fpath)
    df0VVA = pd.concat([df0VVA, run_df[run_df["VVA"] == 0]])
# sort by cycle to match mscan list
df0VVA.sort_values("cyc", inplace=True)

# set cycle to be equal to index
mscan = mscan[mscan.cycle != -1]
mscan.reset_index(inplace = True)
# get times of each 0VVA run
mscan = mscan.iloc[df0VVA.cyc]

# add time column to main df and convert to datetime obj (use dt.time to remove date info)
df0VVA["time"]  = mscan.time.values
df0VVA["time"] = pd.to_datetime(df0VVA["time"].astype(str))

c5_mean = np.mean(df0VVA.c5)

# fit trend in c5 vs time to line (cyc # as proxy for time)
popts, pcov = curve_fit(line, df0VVA.cyc, df0VVA.c5, [3000, -1])
perrs = np.sqrt(np.diag(pcov))

### plot c5 vs time
# fig, ax = plt.subplots(3,1, sharex=False, figsize=(6,8), height_ratios = [3,1.5,3])
fig = plt.figure(figsize=(6,6))

outer = gridspec.GridSpec(2, 1, hspace=0.3)

# inner grid for each panel (main + residual)
gs01 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0],
        height_ratios=[3,1], hspace=0.05)
gs2 = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[1])

ax0 = fig.add_subplot(gs01[0])
ax1 = fig.add_subplot(gs01[1])
ax2 = fig.add_subplot(gs2[0])

ax0.plot(df0VVA.time, df0VVA.c5, color="plum")
ax0.set(ylabel="c5")
ax0.tick_params(labelbottom=False)

# plot mean
ax0.axhline(c5_mean, ls="--", color="lightgrey", label="avg", marker="")

# plot fit
ax0.plot(df0VVA.time.values[[0,-1]], line(df0VVA.cyc.values[[0,-1]],*popts), ls="-", marker="", 
           color="mediumvioletred", label= fit_label(popts, perrs, ["linear fit: m", "b"], sep=", "))
ax0.legend(framealpha=0.6)

# plot residuals
ax1.plot(df0VVA.time, df0VVA.c5 - line(df0VVA.cyc,*popts), color="plum")
ax1.axhline(0, ls="--", color="lightgrey", label="avg", marker="")

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
ax1.tick_params(axis='x', labelrotation=45)
ax1.set(xlabel="cycle time", ylabel="residuals")

#### plot c5 vs delay time
# adjust c5 count by cycle time from linear fit
df0VVA["c5_adjusted"] = df0VVA.c5 - (line(df0VVA.cyc,*popts) - c5_mean)

ax2.plot(df0VVA["Wiggle Time"], df0VVA.c5_adjusted, color="plum")
ax2.set(xlabel="delay time (ms)", ylabel=r"c5")

# plot averages
c5_time_avg = df0VVA.groupby("Wiggle Time")["c5_adjusted"].agg(["mean", "std", "count"])
c5_time_avg_uncorrected = df0VVA.groupby("Wiggle Time")["c5"].agg(["mean", "std", "count"])

ax2.errorbar(c5_time_avg.index, c5_time_avg_uncorrected["mean"], 
               c5_time_avg_uncorrected["std"], color="indigo", label="uncorrected")
ax2.errorbar(c5_time_avg.index, c5_time_avg["mean"], c5_time_avg["std"], color="mediumvioletred",
               label = "(linear fit  - mean) subtracted") #/c5_time_avg["count"]?

ax2.axhline(c5_mean, ls="--", color="lightgrey", marker="")
ax2.legend(loc=0, framealpha=0.6)

plt.tight_layout()
fig.suptitle(f"{run}, 0 VVA points", y=0.92)