import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from glob import glob
import sys 
import os
import ast
# pretty plots
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
plt.rcParams['axes.prop_cycle'] = cycler(color=['hotpink', 'plum'])

# automatically read parent dir, assuming pwd is the directory containing this file

# get paths
# Fast-Modulation-Contact-Correlation-Project\
# root_project = os.path.dirname(os.getcwd())
# # Carmen_Santiago\Analysis Scripts
# root_analysis = os.path.dirname(root_project)
# # Carmen_Santiago\
# root = os.path.dirname(root_analysis)
# # Fast-Modulation-Contact-Correlation-Project\contact_correlations\phaseshift
# analysis_folder = os.path.join(root_project, r"contact_correlations\phaseshift")
# # Carmen_Santiago\\Data
# root_data = os.path.join(root, "Data")

# Always base paths on this file's location, not the working directory
this_file = os.path.abspath(__file__)
root_project = os.path.dirname(this_file)  # directory where get_metadata.py lives
root_analysis = os.path.dirname(root_project)
root = os.path.dirname(root_analysis)

analysis_folder = os.path.join(root_analysis, r"contact_correlations\phaseshift")
root_data = os.path.join(os.path.dirname(root), "Data")

# Fast-Modulation-Contact-Correlation-Project\FieldWiggleCal
field_cal_folder = os.path.join(root_analysis, r"FieldWiggleCal")

# Fast-Modulation-Contact-Correlation-Project\analysis
module_folder = os.path.join(root, "analysis")

if module_folder not in sys.path:
	sys.path.append(module_folder)	
from data_class import Data
from library import fit_label, a0, pi, hbar, GammaTilde, h, mK, B_from_FreqMHz
from fit_functions import Sinc2
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
