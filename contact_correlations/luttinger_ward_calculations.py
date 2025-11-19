import os
import pandas as pd
import numpy as np

# Load tabulated data from
# Thermodynamics of unitary Fermi gas
# Haussmann, Rantner, Cerrito, Zwerger 2007; and Enss, Haussmann, Zwerger 2011
# Density:  n=k_F^3/(3\pi^2)
# columns:  T/T_F, mu/E_F, u/(E_F*n), s/(k_B*n), p/(E_F*n), C/k_F^4
data_folder = 'tilman_data'

C_file = 'luttward-thermodyn.txt'
df_C = pd.read_csv(os.path.join('..', data_folder, C_file), skiprows=4, sep=' ')
x_label = 'T/T_F'
C_label = 'C/k_F^4'


def contact_density(ToTF):
    """Functions that interpolates contact density \mathcal{C}/(k_F^4) using
       tabulated data from Haussmann, Rantner, Cerrito, Zwerger 2007; 
       and Enss, Haussmann, Zwerger 2011."""
    return np.interp(ToTF, df_C[x_label], df_C[C_label])


S_file = 'sumrule-bulk-ufg.txt'
df_S = pd.read_csv(os.path.join('..', data_folder, S_file), skiprows=3, sep=' ')
S_label = 'S(T)(k_Fa)^2/(n*EF)'


def scale_susceptibility(ToTF):
    """Functions that computes scale susceptibility (k_F a)^2 S/(E_F n) using
       tabulated data from Tilman 2019."""
    return np.interp(ToTF, df_S[x_label], df_S[S_label])

