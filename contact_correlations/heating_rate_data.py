# -*- coding: utf-8 -*-
"""
2024-02-26
@author: Chip


Okay this file is dumb. Issue with multiscans.
"""
import numpy as np
pi = np.pi

#####
##### Field wiggle calibrations
#####

freqs = [10, 10, 1, 2.5, 10, 5] # kHz
Vpps = [0.4, 0.9, 0.9, 0.9, 1.0, 1.8] # 50 Ohm term
fieldAmps = [0.021, 0.048, 0.01, 0.018, 0.054, 0.070] # fit amplitudes in Gauss
e_fieldAmps = [0.002, 0.004, 0.001, 0.002, 0.001, 0.003] # fill
e_fieldAmp = 2e-3 # estimated as 2 mG

Bamp_per_Vpp = {1:None,2.5:None,5:None,10:None} # dict to loop and fill below
for freq, val in Bamp_per_Vpp.items():
	# compute average scaled by Vpp if multiple calibrations at the same freq
	val_avg = np.mean([fieldAmps[i]/Vpps[i] for i in range(len(freqs)) if freqs[i] == freq])
	err_avg = np.mean([e_fieldAmps[i]/Vpps[i] for i in range(len(freqs)) if freqs[i] == freq])
	Bamp_per_Vpp[freq] = [val_avg, err_avg]
	
#####
##### Trap frequency calibrations
#####

### old
## ODTs = 0.2/4
# wx = 151.6
# wy = 429
# wz = 442

### 2024-03-06
## ODTs = 0.2/4
wx = 169.1 # Hz
wy = 453#429#*np.sqrt(2)
wz = 441#442#*np.sqrt(2)
mean_trapfreq = 2*pi*(wx*wy*wz)**(1/3)

#####
##### hydrodynamic TOF at 202.14G
#####

### 202.1 Mar 26 varying amp at different frequencies
C_0326_f5_A0p7 = {'name':'C_0326_f5_A0p7','param':'time',
				  'multiscan_param':'amplitude',
				  'filename':'2024-03-26_C_UHfit','freq':5e3,'Vpp':0.7,
				   'B':202.14,'trapfreq':mean_trapfreq}

C_0326_f5_A0p9 = {'name':'C_0326_f5_A0p9','param':'time',
				  'multiscan_param':'amplitude',
				  'filename':'2024-03-26_C_UHfit','freq':5e3,'Vpp':0.9,
				   'B':202.14,'trapfreq':mean_trapfreq}

C_0326_f5_A1p1 = {'name':'C_0326_f5_A1p1','param':'time',
				  'multiscan_param':'amplitude',
				  'filename':'2024-03-26_C_UHfit','freq':5e3,'Vpp':1.1,
				   'B':202.14,'trapfreq':mean_trapfreq}

C_0326_f5_A1p3 = {'name':'C_0326_f5_A1p3','param':'time',
				  'multiscan_param':'amplitude',
				  'filename':'2024-03-26_C_UHfit','freq':5e3,'Vpp':1.3,
				   'B':202.14,'trapfreq':mean_trapfreq}

C_0326_f5_A1p5 = {'name':'C_0326_f5_A1p5','param':'time',
				  'multiscan_param':'amplitude',
				  'filename':'2024-03-26_C_UHfit','freq':5e3,'Vpp':1.5,
				   'B':202.14,'trapfreq':mean_trapfreq}

runs_0326_C_f5 = [C_0326_f5_A0p7,C_0326_f5_A0p9,C_0326_f5_A1p1,
			   C_0326_f5_A1p3,C_0326_f5_A1p5]

C_0326_f120_A0p25 = {'name':'C_0326_f120_A0p25','param':'time',
				  'multiscan_param':'amplitude',
				  'filename':'2024-03-26_C_UHfit','freq':120e3,'Vpp':0.25,
				   'B':202.14,'trapfreq':mean_trapfreq}

C_0326_f120_A0p45 = {'name':'C_0326_f120_A0p45','param':'time',
				  'multiscan_param':'amplitude',
				  'filename':'2024-03-26_C_UHfit','freq':120e3,'Vpp':0.45,
				   'B':202.14,'trapfreq':mean_trapfreq}

C_0326_f120_A0p65 = {'name':'C_0326_f120_A1p65','param':'time',
				  'multiscan_param':'amplitude',
				  'filename':'2024-03-26_C_UHfit','freq':120e3,'Vpp':0.65,
				   'B':202.14,'trapfreq':mean_trapfreq}

C_0326_f120_A0p85 = {'name':'C_0326_f120_A1p85','param':'time',
				  'multiscan_param':'amplitude',
				  'filename':'2024-03-26_C_UHfit','freq':120e3,'Vpp':0.85,
				   'B':202.14,'trapfreq':mean_trapfreq}

C_0326_f120_A1p05 = {'name':'C_0326_f120_A1p05','param':'time',
				  'multiscan_param':'amplitude',
				  'filename':'2024-03-26_C_UHfit','freq':120e3,'Vpp':1.05,
				   'B':202.14,'trapfreq':mean_trapfreq}

runs_0326_C_f120 = [C_0326_f120_A0p25,C_0326_f120_A0p45,C_0326_f120_A0p65,
			   C_0326_f120_A0p85,C_0326_f120_A1p05]

# runs_0326_B = [B_0326_f40_A0p3]

### 202.1 Mar 26 varying amp at different frequencies
B_0326_f40_A0p3 = {'name':'B_0326_f40_A0p3', 'filename':'2024-03-26_B_UHfit','freq':40e3,
			 'Vpp':0.3,'B':202.14,'trapfreq':mean_trapfreq}

B_0326_f40_A0p5 = {'name':'B_0326_f40_A0p5', 'filename':'2024-03-26_B_UHfit','freq':40e3,
			 'Vpp':0.5,'B':202.14,'trapfreq':mean_trapfreq}

B_0326_f40_A0p7 = {'name':'B_0326_f40_A0p7', 'filename':'2024-03-26_B_UHfit','freq':40e3,
			 'Vpp':0.7,'B':202.14,'trapfreq':mean_trapfreq}

B_0326_f40_A1p1 = {'name':'B_0326_f40_A1p1', 'filename':'2024-03-26_B_UHfit','freq':40e3,
			 'Vpp':1.1,'B':202.14,'trapfreq':mean_trapfreq}

B_0326_f40_A1p3 = {'name':'B_0326_f40_A1p3', 'filename':'2024-03-26_B_UHfit','freq':40e3,
			 'Vpp':1.1,'B':202.14,'trapfreq':mean_trapfreq}


runs_0326_B = [B_0326_f40_A0p3, B_0326_f40_A0p5, B_0326_f40_A0p7,
			    B_0326_f40_A1p1, B_0326_f40_A1p3]


### 202.1 Mar 21 varying freq using Will's code 
B_0321_f15 = {'name':'B_0321_f15', 'filename':'2024-03-21_B_UHfit','freq':15e3,
			 'Vpp':0.75,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

B_0321_f35 = {'name':'B_0321_f35', 'filename':'2024-03-21_B_UHfit','freq':35e3,
			 'Vpp':0.45,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

B_0321_f55 = {'name':'B_0321_f55', 'filename':'2024-03-21_B_UHfit','freq':55e3,
			 'Vpp':0.4,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

B_0321_f75 = {'name':'B_0321_f75', 'filename':'2024-03-21_B_UHfit','freq':75e3,
			 'Vpp':0.25,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

B_0321_f95 = {'name':'B_0321_f95', 'filename':'2024-03-21_B_UHfit','freq':95e3,
			 'Vpp':0.25,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

B_0321_f115 = {'name':'B_0321_f115', 'filename':'2024-03-21_B_UHfit','freq':115e3,
			 'Vpp':0.25,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

D_0321_f35 = {'name':'D_0321_f35', 'filename':'2024-03-21_D_UHfit','freq':35e3,
			 'Vpp':0.35,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

D_0321_f55 = {'name':'D_0321_f55', 'filename':'2024-03-21_D_UHfit','freq':55e3,
			 'Vpp':0.3,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

runs_0321 = [B_0321_f15, B_0321_f35, B_0321_f55,
			 B_0321_f75, B_0321_f95, B_0321_f115, 
			 D_0321_f35, D_0321_f55]


### 202.1 Mar 20 varying freq using Will's code 
D_0320_f20 = {'name':'D_0320_f20', 'filename':'2024-03-20_D_UHfit','freq':20e3,
			 'Vpp':0.7,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

D_0320_f40 = {'name':'D_0320_f40', 'filename':'2024-03-20_D_UHfit','freq':40e3,
			 'Vpp':0.5,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

D_0320_f60 = {'name':'D_0320_f60', 'filename':'2024-03-20_D_UHfit','freq':60e3,
			 'Vpp':0.3,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

runs_0320 = [D_0320_f20, D_0320_f40, D_0320_f60]

### 202.1 Mar 19 varying freq using Will's code 
L_0319_f5 = {'name':'L_0319_f5', 'filename':'2024-03-19_L_UHfit','freq':5e3,
			 'Vpp':0.9,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

L_0319_f20 = {'name':'L_0319_f20', 'filename':'2024-03-19_L_UHfit','freq':20e3,
			  'Vpp':0.7,'B':202.14,'Ni':48997,'Ti':0.388, 
			  'GTi':0.276, 'trapfreq':mean_trapfreq}

L_0319_f40 = {'name':'L_0319_f40', 'filename':'2024-03-19_L_UHfit','freq':40e3,
			  'Vpp':0.5,'B':202.14,'Ni':48997,'Ti':0.388, 
			  'GTi':0.276, 'trapfreq':mean_trapfreq}

runs_0319 = [L_0319_f5, L_0319_f20, L_0319_f40]

### 202.1 Feb 09 varying freq using Kevin's code 
F_0209_f5 = {'name':'H_0209_f5','filename':'2024-02-09_F_UHfit','freq':5e3,
			 'Vpp':1.8/2,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

F_0209_f10 = {'name':'H_0209_f10','filename':'2024-02-09_F_UHfit','freq':10e3,
			  'Vpp':1.5/2,'B':202.14,'Ni':48997,'Ti':0.388, 
			  'GTi':0.276, 'trapfreq':mean_trapfreq}

F_0209_f30 = {'name':'H_0209_f30','filename':'2024-02-09_F_UHfit','freq':30e3,
			  'Vpp':1.286/2,'B':202.14,'Ni':48997,'Ti':0.388, 
			  'GTi':0.276, 'trapfreq':mean_trapfreq}

F_0209_f50 = {'name':'H_0209_f50','filename':'2024-02-09_F_UHfit','freq':50e3,
			  'Vpp':0.7/2,'B':202.14,'Ni':48997,'Ti':0.388, 
			  'GTi':0.276, 'trapfreq':mean_trapfreq}

F_0209_f150 = {'name':'H_0209_f150','filename':'2024-02-09_F_UHfit','freq':150e3,
			   'Vpp':0.54/2,'B':202.14,'Ni':48997,'Ti':0.388, 
			   'GTi':0.276, 'trapfreq':mean_trapfreq}

runs_0209 = [F_0209_f5,
			 F_0209_f10,
			 F_0209_f30,
			 F_0209_f50,
			 F_0209_f150]

### 202.1 Feb 07 varying freq using Kevin's code 

H_0207_f15 = {'name':'H_0207_f15', 'filename':'2024-02-07_H_e','freq':15e3,
			  'Vpp':1.8/2,'B':202.14,'Ni':48997,'Ti':0.388, 
			  'GTi':0.276, 'trapfreq':mean_trapfreq}

I_0207_f0 = {'name':'I_0207_f0','filename':'2024-02-07_I_e','freq':0,
			 'Vpp':1.8/2,'B':202.14,'Ni':48997,'Ti':0.388, 
			 'GTi':0.276, 'trapfreq':mean_trapfreq}

runs_0207 = [I_0207_f0,
			 H_0207_f15]

H_0207_Willfilename = "2024-02-07_H_UHfit"
I_0207_Willfilename = "2024-02-07_I_UHfit"


###
### Tilman PRL data
###
bulkT58 = np.loadtxt("zetaomega_T0.58.txt", comments="#", delimiter=" ", unpack=False)
bulkT25 = np.loadtxt("zetaomega_T0.25.txt", comments="#", delimiter=" ", unpack=False)
bulkavg = np.loadtxt("zetaavg.txt", comments="#", delimiter=" ", unpack=False)
# guess the 2.00 line :P
bulkT2p0 = np.array([[0.01, 0.014],[0.4,0.014],[1,0.0135],[1.4,0.012],
					 [2, 0.01],[4,0.005],[7,0.003],[10,0.0014]])
bulk_uni_DC = np.array([[0.16,1.8],[0.25,bulkT25[0,1]/12],
						[0.58,bulkT58[0,1]/12],[2.00,bulkT2p0[0,1]]])


#########################
##### OBSOLETE DATA #####
#########################

###
### Jump Heating Rate Data
###
data_folder = "data\\heating\\"
file = data_folder + "Nov09_ToTF0p65_203G.txt"
omegaoEF203, bulkmeas203, bulk203error = np.loadtxt(file,
								   delimiter=',', unpack=True)
file = data_folder + "Nov07_ToTF0p6_202p1G.txt"
omegaoEF202p1, bulkmeas202p1, bulkerror202p1 = np.loadtxt(file,
								  delimiter=',', unpack=True)
file = data_folder + "Nov18_bg202p1G.txt"
omegaoEFbg202p1, bulkmeasbg202p1, bulkerrorbg202p1 = np.loadtxt(file,
								  delimiter=',', unpack=True)
file = data_folder + "Nov15_ToTF1p4_202p1G.txt"
omegaoEFhot, bulkhot, bulkhoter = np.loadtxt(file,
								   delimiter=',', unpack=True)

#####
##### Jump to 209G data
#####

###
### scan time
###

C_1010 = {'filename':'2023-10-10_C_e.dat','freq':10e3,'Bamp':0.1,'B':202.1,
		  'Ni':34578,'Ti':0.528}

C_1018 = {'filename':'2023-10-18_C_e.dat','freq':30e3,'Bamp':0.1,'B':202.1,
		  'Ni':36439,'Ti':0.521}
D_1018 = {'filename':'2023-10-18_D_e.dat','freq':100e3,'Bamp':0.1,'B':202.1,
		  'Ni':36439,'Ti':0.521}
E_1018 = {'filename':'2023-10-18_E_e.dat','freq':100e3,'Bamp':0.05,'B':202.1,
		  'Ni':36439,'Ti':0.521}
### 209 G
B_1113 = {'filename':'2023-11-13_B_e.dat','freq':15e3,'Bamp':0.054*1.8,'B':209,
		  'Ni':36913,'Ti':0.437, 'GTi':0.563}


### 202p1G, ToTF ~ 0.60
G_1107 = {'filename':'2023-11-07_G_e.dat','freq':15e3,'Bamp':0.054*1.8,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

H_1107 = {'filename':'2023-11-07_H_e.dat','freq':5e3,'Bamp':0.07,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

J_1107 = {'filename':'2023-11-07_J_e.dat','freq':50e3,'Bamp':0.054*0.7,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

K_1107 = {'filename':'2023-11-07_K_e.dat','freq':150e3,'Bamp':0.054*0.54,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

M_1107 = {'filename':'2023-11-07_M_e.dat','freq':30e3,'Bamp':0.054*1.286,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

R_1107 = {'filename':'2023-11-07_R_e.dat','freq':10e3,'Bamp':0.054*1.5,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

Nov07_runs = [G_1107,J_1107,K_1107,R_1107,M_1107]
Nov07label = "Nov07_ToTF0p6_202p1G"

### 203G, ToTF ~ 0.58
B_1109 = {'filename':'2023-11-09_B_e.dat','freq':15e3,'Bamp':0.054*1.8,'B':203,
		  'Ni':49429,'Ti':0.48, 'GTi':0.553}

C_1109 = {'filename':'2023-11-09_C_e.dat','freq':50e3,'Bamp':0.054*0.7,'B':203,
		  'Ni':49429,'Ti':0.48, 'GTi':0.553}

D_1109 = {'filename':'2023-11-09_D_e.dat','freq':15e3,'Bamp':0.054*1.8,'B':203,
		  'Ni':28230,'Ti':0.662, 'GTi':0.693}

F_1109 = {'filename':'2023-11-09_F_e.dat','freq':5e3,'Bamp':0.07,'B':203,
		  'Ni':28230,'Ti':0.662, 'GTi':0.693} 

G_1109 = {'filename':'2023-11-09_G_e.dat','freq':50e3,'Bamp':0.054*0.7,'B':203,
		  'Ni':28230,'Ti':0.662, 'GTi':0.693}

I_1109 = {'filename':'2023-11-09_I_e.dat','freq':150e3,'Bamp':0.054*0.35,'B':203,
		  'Ni':27603,'Ti':0.582, 'GTi':0.637}

Nov09_runs = [D_1109,F_1109,G_1109,I_1109]
Nov09label = "Nov09_ToTF0p65_203G"

### 202p1G, ToTF ~ 1.4
E_1115_10 = {'filename':'2023-11-15_E_e_freq=10.dat','freq':10e3,'Bamp':0.054*1.8,'B':202.1,
		  'Ni':32888,'Ti':1.08, 'GTi':1.41}

E_1115_20 = {'filename':'2023-11-15_E_e_freq=20.dat','freq':20e3,'Bamp':0.054*1.8,'B':202.1,
		  'Ni':32888,'Ti':1.08, 'GTi':1.41}

E_1115_50 = {'filename':'2023-11-15_E_e_freq=50.dat','freq':50e3,'Bamp':0.054*1.8,'B':202.1,
		  'Ni':32888,'Ti':1.08, 'GTi':1.41}

E_1115_100 = {'filename':'2023-11-15_E_e_freq=100.dat','freq':100e3,'Bamp':0.054*1.8,'B':202.1,
		  'Ni':32888,'Ti':1.08, 'GTi':1.41}

F_1115 = {'filename':'2023-11-15_F_e.dat','freq':150e3,'Bamp':0.054*1.8,'B':202.1,
		  'Ni':32888,'Ti':1.08, 'GTi':1.41}

Nov15_runs = [E_1115_10,E_1115_20,E_1115_50,E_1115_100,F_1115]
Nov15label = "Nov15_ToTF1p4_202p1G"

F_1115 = {'filename':'2023-11-15_F_e.dat','freq':150e3,'Bamp':0.054*1.8,'B':202.1,
		  'Ni':32888,'Ti':1.08, 'GTi':1.41}

Feb09_runs = [E_1115_10,E_1115_20,E_1115_50,E_1115_100,F_1115]
Feb09label = "Feb09_ToTF1p4_202p1G"

### 202p1G, ToTF = 0.6, no wiggle bg, split data into different freqs
E_1118_f15 = {'filename':'2023-11-18_E_e.dat','freq':15e3,'Bamp':0.054*1.8,'B':202.1,
			  'Ni':27603,'Ti':0.582, 'GTi':0.637}
E_1118_f5 = {'filename':'2023-11-18_E_e.dat','freq':5e3,'Bamp':0.07,'B':202.1,
			  'Ni':27603,'Ti':0.582, 'GTi':0.637}
E_1118_f50 = {'filename':'2023-11-18_E_e.dat','freq':50e3,'Bamp':0.054*0.7,'B':202.1,
			  'Ni':27603,'Ti':0.582, 'GTi':0.637}
E_1118_f150 = {'filename':'2023-11-18_E_e.dat','freq':150e3,'Bamp':0.054*0.54,'B':202.1,
			  'Ni':27603,'Ti':0.582, 'GTi':0.637}
E_1118_f30 = {'filename':'2023-11-18_E_e.dat','freq':30e3,'Bamp':0.054*1.286,'B':202.1,
			  'Ni':27603,'Ti':0.582, 'GTi':0.637}
E_1118_f10 = {'filename':'2023-11-18_E_e.dat','freq':10e3,'Bamp':0.054*1.5,'B':202.1,
			  'Ni':27603,'Ti':0.582, 'GTi':0.637}

bg202p1_runs = [E_1118_f5, E_1118_f10, E_1118_f15, E_1118_f30, E_1118_f50, E_1118_f150]
bg202p1label = "Nov18_bg202p1G"

###
### scan amp
###
D_1010 = {'filename':'2023-10-10_D_e.dat','freq':10e3,'time':5e-3,'B':202.1,
		  'Ni':34578,'Ti':0.528}

J_1017 = {'filename':'2023-10-17_J_e.dat','freq':30e3,'time':5e-3,'B':202.1,
		  'Ni':32517,'Ti':0.529}

F_1018 = {'filename':'2023-10-18_F_e.dat','freq':100e3,'time':2e-3,'B':202.1,
		  'Ni':36439,'Ti':0.521}

###
### scan freq
###
E_1010 = {'filename':'2023-10-10_E_e.dat','Bamp':0.1,'time':5e-3,'B':202.1,
		  'Ni':34578,'Ti':0.528}

G_1017 = {'filename':'2023-10-17_G_e.dat','Bamp':0.1,'time':10e-3,'B':209,
		  'Ni':39573,'Ti':0.317}
I_1017 = {'filename':'2023-10-17_I_e.dat','Bamp':0.1,'time':5e-3,'B':202.1,
		  'Ni':32517,'Ti':0.529}
K_1017 = {'filename':'2023-10-17_K_e.dat','Bamp':0.1,'time':2e-3,'B':202.1,
		  'Ni':32517,'Ti':0.529}

H_1018 = {'filename':'2023-10-18_H_e.dat','Bamp':0.05,'time':1e-3,'B':202.1,
		  'Ni':36439,'Ti':0.521}
I_1018 = {'filename':'2023-10-18_I_e.dat','Bamp':0.1,'time':1e-3,'B':202.1,
		  'Ni':36439,'Ti':0.521}

S_1031 = {'filename':'2023-10-31_S_e.dat','Bamp':0.05,'time':1e-3,'B':202.1,
		  'Ni':34804,'Ti':0.568, 'GTi':0.604}

