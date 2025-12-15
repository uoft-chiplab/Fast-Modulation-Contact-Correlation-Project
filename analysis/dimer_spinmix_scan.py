
# settings for directories, standard packages...
from preamble import *
from library import colors, kB
import os

runs = {"data": "2025-12-15_L"}#, "bg":"2025-12-15_N"}
	

filename = r'e:\Data\2025\12 December2025\15December2025\L_dimer_spinmix_scan\2025-12-15_L_e.dat'
bg_filename = r'e:\Data\2025\12 December2025\15December2025\N_spinmix_bg_scan\2025-12-15_N_e.dat'
# '2025-12-15_L_e.dat'
# bg_filename = '2025-12-15_N_e.dat'

data = Data(filename, path=datfiles[0])
databg = Data(bg_filename, path=datfiles[0])




    # data.data['trf'] = 1e-6 # s
    # data.data['freq'] = 43.240e6 # Hz
    # data.data['EF'] = 13.94e3 # Hz

    # # alldata = data.analysis(bgVVA = 0,nobg=True, pulse_type="square").data
    # data.analysis(bgVVA = 0, pulse_type="square")
    # data.group_by_mean('VVA')
    # df = data.avg_data

    # df['norm_sig'] = df['scaledtransfer_dimer'] / df['scaledtransfer_dimer'].max()
    # df['em_norm_sig'] = df['em_scaledtransfer_dimer'] / df['scaledtransfer_dimer'].max()
    # df['norm_C'] = df['contact_dimer'] / df['contact_dimer'].max()
    # df['em_norm_C'] = df['em_contact_dimer'] / df['contact_dimer'].max()
