import numpy as np 
import pandas as pd 

# set correct path dependencies
from preamble import *

class metadata:

    field_cal_df_path = os.path.join(field_cal_folder, "field_cal_summary.csv")
    field_cal_df = pd.read_csv(field_cal_df_path)

    def __init__(self, runs, drop_list, overwrite, fpath, notes):
        """
        runs: list of runs to include
        drop_list: list of list of cycles to drop for each run
        overwrite (bool): whether or not to overwrite precious rows in df
        fpath (str): name of output path
        """
        self.overwrite = overwrite
        self.fpath = fpath

        self.df = pd.DataFrame({"run": runs, "drop":drop_list, "notes":notes})

        for i, r in enumerate(runs):
            self.get_run_attributes(r, i)
            self.get_ushots(r, i)
            self.get_field_cal(i)

        return

    def get_run_path(self, run):
        """
        returns all dat files for a given run, each corresponding to a different wiggle time
        """
        y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
        runpath = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{l}_*")[0]
        runname = os.path.basename(runpath)
        datfiles = glob(f"{runpath}/*=*.dat")
        
        return runname, datfiles

    def get_run_attributes(self, run, j):
        """
        updates VVA, wiggle_freq, pulse_time, Vpp
        """
        r_name, r_paths = self.get_run_path(run)
        r_name = r_name.lower()
        # just get first path since VVA's are all the same
        r_df = pd.read_csv(r_paths[0])

        VVA = np.max(r_df.VVA)
        
        # get other metadata from run name
        wiggle_freq = float(r_name[:r_name.find("khz")].split("_")[-1])

        i = r_name.find("us")
        try:
            pulse_time = float(r_name[i-2:i]) # in us
        except:
            pulse_time = 20 # default to 20 us if not found

        i = r_name.find("vpp")
        try:
            Vpp = float(r_name[i-3:i]) # in V
        except:
            Vpp = 1.8 # default to 1.8 V if not found

        self.df.loc[j, ["freq", "pulse_time", "Vpp", "VVA"]] = [wiggle_freq, pulse_time, Vpp, VVA]

    def get_field_cal(self, i):
        """
        dc field file
        """
        wiggle_freq, wiggle_amp = self.df.loc[i, ["freq", "Vpp"]]

        field_cal = self.field_cal_df[(self.field_cal_df['wiggle_freq']==wiggle_freq) & \
                        (self.field_cal_df['wiggle_amp']==wiggle_amp)
                            ]
        
        if len(field_cal) > 1:
            field_cal = field_cal[field_cal['pulse_length']==40]
        
        field_cal_run = field_cal['run'].values[0]

        self.df.loc[i, "field_cal_run"] = field_cal_run

    def get_ushots(self, run, j):
        # interpolate between values before and after 
        y, m, d, l = run[0:4], run[5:7], run[8:10], run[-1]
        runpaths = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/*UShots*") # case insensitive on windows

        if len(runpaths) > 1:
            # get ushot run closest to run

            # get list of all letters
            letters = [os.path.basename(p)[0] for p in runpaths]
            # convert list to number of differences from run
            run_diffs = [abs(ord(letter.upper()) - ord(l.upper())) for letter in letters]
            i = np.argmin(run_diffs)
            u_run = os.path.basename(runpaths[i])
        
        else:
            u_run = os.path.basename(runpaths[0])

        u_path = glob(f"{root_data}/{y}/{m}*{y}/{d}*{y}/{u_run}/*.dat")[0]
        u_df = pd.read_csv(u_path)

        self.df.loc[j, ["ToTF", "EF", "N"]] = np.mean(u_df[["ToTF", "EF", "N"]], axis=0)
        self.df.loc[j, ["eToTF", "eEF", "eN"]] = np.std(u_df[["ToTF", "EF", "N"]], axis=0)/len(u_df)

        self.df.loc[j, ["wx", "wy", "wz"]] = np.mean(u_df[["wx", "wy", "wz"]], axis=0)

    def output(self):
        if self.overwrite: # I'm not 100% sure this works all the time
            if os.path.exists(self.fpath):
                # make able to overwrite individual dfs?
                old_df = pd.read_csv(self.fpath)
                for i, r in enumerate(self.df.run):
                    if r in old_df.run.values:
                        i = old_df.index[old_df['run'] == r].tolist()[0]
                        old_df.loc[i] = self.df.loc[i]
                        
                old_df.to_csv(self.fpath, index=False)

            else:
                self.df.to_csv(self.fpath, index=False)
        else:
            self.df.to_csv(self.fpath, mode='a', index=False, header=False)




# VVA to rabi frequency
# trap depth in UShots
# time column name?

if __name__ == "__main__":
    # attributes
    runs = ["2025-09-24_E", 
            "2025-10-01_L",
            "2025-10-17_E",
            "2025-10-21_H", 
            "2025-10-23_R",
            "2025-10-23_S"]
    #"2025-09-24_E" is 6kHz 20us pulse 1.8Vpp
    #2025-10-01_L is 10kHz 20us pulse 1.8Vpp

    # change format later?
    drop_list = [
          [],
          [0.29], 
          [],
          [],
          [],
          []
    ]

    notes = [
        "good", 
        "good",
        "?",
        "decent",
        "bad",
        "?"
    ]
    # 2025-09-24_E is 6kHz 20us pulse 1.8Vpp
# 2025-10-01_L is 10kHz 20us pulse 1.8Vpp
# 2025-10-17_E is ???
# 2025-10-17_M is 10kHz 20us pulse 1.8Vpp single shot measurements
# 2025-10-18_O is 10kHz 20us pulse 1.8Vpp single shot measurements take 2
# 2025-10-20_M is 10kHz 10us pulse 1.8Vpp for small scan list around spectra peaks
# 2025-10-21-H is 10kHz 10us pulse 1.8Vpp for small scan list around spectra peaks
# 2025-10-23-R is 10kHz 10us pulse 1.8Vpp for small scan list around spectra peaks

    overwrite = True 
    fpath = "metadata.csv"
    
    a = metadata(runs, drop_list, overwrite, fpath, notes)
    a.output()