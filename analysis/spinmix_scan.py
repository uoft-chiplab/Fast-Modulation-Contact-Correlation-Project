
# settings for directories, standard packages...
from preamble import *
from library import colors, kB
import os

### dimer

dimer_filename = r'e:\Data\2025\12 December2025\15December2025\L_dimer_spinmix_scan\2025-12-15_L_e.dat'
HFT_filename = r'E:\Data\2025\12 December2025\15December2025\O_HFT_spinmix_scan\2025-12-15_O_e.dat'
bg_filename = r'e:\Data\2025\12 December2025\15December2025\N_spinmix_bg_scan\2025-12-15_N_e.dat'

dimerdata = Data(dimer_filename.split("\\")[-1], path=dimer_filename)
HFTdata = Data(HFT_filename.split("\\")[-1], path=HFT_filename)
databg = Data(bg_filename.split("\\")[-1], path=bg_filename)

# rename VVA columns on first page to spin_mix_VVA
dimerdata.data['spin_mix_VVA'] = dimerdata.data['VVA'].values
HFTdata.data['spin_mix_VVA'] = HFTdata.data['VVA'].values
databg.data['spin_mix_VVA'] = databg.data['VVA'].values

# add new VVA col for dimer pulse VVA
dimerdata.data['VVA'] = 9
databg.data['VVA'] = 0 # no dimer pulse here

# stitch two datasets together
dimerdata.data = pd.concat([dimerdata.data, databg.data])
dimerdata.data.reset_index(inplace=True)

dimerdata.data[['trf', 'freq', 'EF']] = 10e-6, 43.240, 13.94e3 # s, MHz, Hz
HFTdata.data[['trf', 'freq', 'EF', 'VVA']] = 20e-6, 47.373, 13.94e3, 9

c9i_HFT = 28308/2 # from HFT data for norminal VVA
c9i_dimer = 11847

df_spinmix = pd.DataFrame({'VVA':np.unique(dimerdata.data['spin_mix_VVA'])})
    
for VVA in np.unique(dimerdata.data['spin_mix_VVA']):

    # dimer
    datVVA = Data(dimer_filename.split("\\")[-1], path=dimer_filename)
    datVVA.data = dimerdata.data[dimerdata.data['spin_mix_VVA'] == VVA].copy(deep=True)
    datVVA.data['EF'] *= (np.mean(datVVA.data['c9'])/c9i_dimer)**(1/3)

    datVVA.analysis(bgVVA = 0, pulse_type="square")
    datVVA.group_by_mean('spin_mix_VVA')
    dimer_df = datVVA.avg_data

    df_spinmix.loc[df_spinmix['VVA'] == VVA, ['c5_dimer', 'c9_dimer', 'fraction95', 'contact_dimer', 'em_contact_dimer', 'scaledtransfer_dimer', 'em_scaledtransfer_dimer']] = \
    dimer_df[['c5', 'c9', 'fraction95', 'contact_dimer', 'em_contact_dimer', 'scaledtransfer_dimer', 'em_scaledtransfer_dimer']].iloc[0].values

    # HFT
    datVVA = Data(HFT_filename.split("\\")[-1], path=HFT_filename)
    datVVA.data = data.data[data.data['spin_mix_VVA'] == VVA].copy(deep=True)
    datVVA.data['EF'] *= (np.mean(datVVA.data['c9'])/c9i_HFT)**(1/3)

    datVVA.analysis(nobg=True, pulse_type="blackman")
    datVVA.group_by_mean('spin_mix_VVA')
    HFT_df = datVVA.avg_data

    df_spinmix.loc[df_spinmix['VVA'] == VVA, ['c5_HFT', 'c9_HFT', 'contact_HFT', 'em_contact_HFT', 'scaledtransfer_HFT', 'em_scaledtransfer_HFT']] = \
    HFT_df[['c5', 'c9', 'contact_HFT', 'em_contact_HFT', 'scaledtransfer_HFT', 'em_scaledtransfer_HFT']].iloc[0].values

    
plt.errorbar(df_spinmix['fraction95'], df_spinmix['contact_dimer'], df_spinmix['em_contact_dimer'],
             label="dimer")

plt.errorbar(df_spinmix['fraction95'], df_spinmix['contact_HFT'], df_spinmix['em_contact_HFT'],
             label="HFT")

plt.xlabel("fraction95")
plt.ylabel("single shot measured C")
plt.legend(loc=0)