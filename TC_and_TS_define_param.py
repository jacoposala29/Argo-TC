import pandas as pd
import numpy as np

# var2use       = 'Salinity'
var2use       = 'Temperature'
# var2use       = 'Potential_density'
#var2use       = 'MLS'
# var2use       = 'MLT'
# var2use       = 'MLD'
#var2use       = 'MLPD'
# var2use       = 'chla'
# var2use       = 'doxy'

# tag_case = 'only_hur_all_inc_dec'
# tag_case = 'all_comb_tstd_hur'

# for ML_incr_decr function used in B06 and B10
ML_delta = 50 # 1/2 1/3 2/3 1 10 20 30 40 50 60 70 
mode_tag = 'absolute' # 'fraction' # 'absolute'
mask_tag = 'plus50m' # 1ML halfML 1thirdML 2thirdML plus10m plus20m plus30m plus40m plus50m plus60m plus70m

mode_tags = ['absolute', 'fraction']
absolute_mask_tags = ['plus10m', 'plus20m', 'plus30m', 'plus40m', 'plus50m', 'plus60m', 'plus70m']
fraction_mask_tags = ['1ML', 'halfML', '1thirdML', '2thirdML']
absolute_ML_delta = [10, 20, 30, 40, 50, 60, 70]
fraction_ML_delta = [1/2, 1/3, 2/3, 1]


year_start_TC = 2004
year_end_TC   = 2021
year_pairs = tuple([n for n in range(year_start_TC, year_end_TC, 1)])

months_all = '010203040506070809101112'
months_string = tuple([(months_all[i:i+2]) for i in range(0, len(months_all), 2) ])

# Minimum wind speed in knots
min_TCwind = 64
n_level_min_mask = 5
h_B06 = 0.2
k_factor = 2 # Used in B21 and B36. We recommenend not to change, as it may be still hard coded in somewhere

# Create grid for output
# bounds_x1 = (-3, +3) # will be x-axis in 2d plots (space, in degrees)
bounds_x1 = (-5, +5) # will be x-axis in 2d plots (space, in degrees) - we use this one only (no 3 or 8)
# bounds_x1 = (-8, +8) # will be x-axis in 2d plots (space, in degrees)
bounds_x2 = (-2, +20)  # will be y-axis in 2d plots in B24 (time, in days)
bounds_x3 = (10, 210) # will be the y-axis in 2d plots in B25 (pressure, in dbar)

# Number of points of the grid
shape = (100, 400) 
# shape_ = (67, 400)

# from B35
if bounds_x1 == (-5, +5):
    train_knots_x1 = 21
    degrees_tag = '5deg'
elif bounds_x1 == (-3, +3):
    train_knots_x1 = 13
    degrees_tag = '3deg'
elif bounds_x1 == (-8, +8):
    train_knots_x1 = 33
    degrees_tag = '8deg'
if bounds_x2 == (-2, +20):
    train_knots_x2 = 45

# Define lambda values to test
a0 = np.arange(0,5,.025).ravel()
a1 = np.arange(5,100,2.5).ravel()
a2 = np.arange(100,1005,5).ravel()
LAMB = np.concatenate((a0,a1,a2))

# lamb_choice = 'Custom_diff_perc_2'
# lamb_choice = 'Custom_diff_perc_5'
# lamb_choice = 'Custom_diff_perc_0.5'
lamb_choice = 'Custom_increase_perc_reverse'


# Define lambda values to test
# LAMB = np.arange(0,1000,2).ravel()

# Old way to select lambda
# a0 = np.arange(0,10,.05).ravel()
# a = np.arange(10,100,2).ravel()
# b = np.arange(100,2000,20).ravel()
# LAMB = np.concatenate((a0,a,b))

# tag_folder_run = '_test_Feb14_0_5_100_1005'
# tag_folder_run = '_test_Mar14_0_5_100_1005_' + degrees_tag 
tag_folder_run = '_test_Mar14_0_5_100_1005_' + degrees_tag  + '_Dec2023'
# tag_folder_run = '_test_Mar14_0_5_100_1005_' + degrees_tag  + '_Dec2023_adhoc' # for Temperature ad hoc
# tag_folder_run = '_test_Feb14_0_5_100_1005_'  + degrees_tag + '_Dec2023'
# tag_folder_run = '_test_Feb14_0_5_100_1005_'  + degrees_tag + '_Dec2023_adhoc' # for MLT ad hoc
# tag_folder_run = '_test_Feb14_0_5_100_1005_'  + degrees_tag + '_Dec2023_adhoc2' # for MLT ad hoc

# add note: USE LAMBDA FROM INCREASING
# tag_lambda_2use_case = 'increasing'

# Outputs of B05 for T, S, and PD
# Note: all three outputs must be at the same, finest depth resolution (5 dbar)
folder2use_S_ML_incr_decr_B06_B10 = '../SALINITY_B05_output'
folder2use_T_ML_incr_decr_B06_B10 = '../TEMPERATURE_B05_output'
folder2use_PD_ML_incr_decr_B06_B10 = '../POTENTIAL_DENSITY_B05_output'

grid_lower_forML  = '10'
grid_upper_forML  = '210'
grid_stride_forML = '5'

if var2use == 'Salinity':
    folder2use = '../SALINITY' + tag_folder_run
    grid_lower  = '10'
    grid_upper  = '210'
    grid_stride = '5'
    depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
    clim = 0.25 #0.5
    clim_mf = 0.1
    iso_b27 = 0.07
    unit = 'psu'
    var_tag = '_jg'
    var_min_B10 = -7
elif var2use == 'Temperature': 
    folder2use = '../TEMPERATURE' + tag_folder_run
    grid_lower  = '10'
    grid_upper  = '210'
    grid_stride = '10' # change back to 10
    depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
    # clim = 1.25
    clim = 2
    # clim_mf = 0.5
    clim_mf = 1
    iso_b27 = 0.4
    unit = '°C'
    var_tag = '_jg'
    var_min_B10 = -4.5
elif var2use == 'Potential_density':
    folder2use  = '../POTENTIAL_DENSITY' + tag_folder_run
    grid_lower  = '10'
    grid_upper  = '210'
    grid_stride = '5'
    depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
    clim = .8
    clim_mf = 0.6
    iso_b27 = 0.2
    unit = 'kg/m$^3$'
    var_tag = '_jg'
    var_min_B10 = -6
elif var2use == 'MLS':
    # folder2use  = '../MLS'
    folder2use = '../MLS' + tag_folder_run
    grid_lower  = '0'
    grid_upper  = '0'
    grid_stride = '1'
    depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
    clim = 0.25 # 0.5
    clim_mf = 0.1
    unit = 'psu'
    var_tag = '_MLS'
    var_min_B10 = -7
    folder2use_ML_incr_decr_B06_B10 = '../SALINITY'
elif var2use == 'MLT': 
    folder2use  = '../MLT' + tag_folder_run
    grid_lower  = '0'
    grid_upper  = '0'
    grid_stride = '1'
    depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
    clim = 2
    clim_mf = 1.2
    unit = '°C'
    var_tag = '_MLT'
    var_min_B10 = -4.5
elif var2use == 'MLPD':
    folder2use  = '../MLPD' + tag_folder_run
    grid_lower  = '0'
    grid_upper  = '0'
    grid_stride = '1'
    depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
    clim = .8 # 1
    clim_mf = 0.5
    unit = 'kg/m$^3$'
    var_tag = '_MLPD'
    var_min_B10 = -6
elif var2use == 'MLD':
    folder2use  = '../MLD' + tag_folder_run
    grid_lower  = '0'
    grid_upper  = '0'
    grid_stride = '1'
    depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
    clim = 25
    clim_mf = 20
    unit = 'm'
    var_tag = '_MLD'
    var_min_B10 = 4
# elif var2use == 'MLT_sal': 
#     folder2use  = '../MLT_sal'
#     grid_lower  = '0'
#     grid_upper  = '0'
#     grid_stride = '1'
#     depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
#     clim = 1.2
#     unit = '°C'
#     var_tag = '_MLT'
#     var_min_B10 = -4.5
#     folder2use_ML_incr_decr_B06_B10 = '../SALINITY'
# elif var2use == 'chla': 
#     #folder2use = '../CLOROPHYLL_qcto3'
#     grid_lower  = '0'
#     grid_upper  = '0'
#     grid_stride = '1'
#     depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
#     clim = 2.0
#     unit = 'tbd'
#     #var_tag = '_chla_QCto4'
#     #var_tag = '_chla_QCto3'
#     var_tag = '_chla_QCto2'
#     var_min_B10 = -4.5
# elif var2use == 'doxy': 
#     #folder2use = '../OXYGEN_QCto2'
#     grid_lower  = '0'
#     grid_upper  = '0'
#     grid_stride = '1'
#     depth_layers= len(np.arange(int(grid_lower),int(grid_upper)+1,int(grid_stride)))
#     clim = 2.0
#     unit = 'tbd'
#     #var_tag = '_chla_QCto4'
#     #var_tag = '_chla_QCto3'
#     var_tag = '_doxy_QCto2'
#     var_min_B10 = -4.5
# elif var2use == 'SSS':
#     clim = 0.4
#     unit = 'psu'
#     var_tag = '_SSSatArgo'
#     var_min_B10 = -7.7
    
    
window_size = '8'
windowSizeGP='5'
min_num_obs = '20'
center_month= '9'

matlab_path = '/Applications/MATLAB_R2021a.app/bin/matlab'
n_parpool   = '1' #'8'

# matlab_path = 'matlab'
# n_parpool   = '16' #'8'

OB = [
    ('_AllBasins',     'meshgrid(linspace(-89.5,89.5,180),linspace(20.5,379.5,360))'),
]

# OB = [
#     ('_NorthAtlantic', 'meshgrid(0.5:70.5,261.5:360.5)'),
#     ('_WestPacific',   'meshgrid(0.5:65.5,105.5:187.5)'),
#     ('_AllBasins',     'meshgrid(linspace(-89.5,89.5,180),linspace(20.5,379.5,360))'),
# ]

# Tracks to include in the analysis
track_dir = '../Inputs'
Basins = [
        ('AL', 'HURDAT_ATLANTIC'),
        ('EP', 'HURDAT_PACIFIC'),
        ('WP', 'JTWC_WESTPACIFIC'),
        ('IO', 'JTWC_INDIANOCEAN'),
        ('SH', 'JTWC_SOUTHERNHEMISPHERE'),
        ]

AL = pd.read_csv(f'{track_dir}/HURDAT_ATLANTIC.csv')
del AL['Unnamed: 0']
EP = pd.read_csv(f'{track_dir}/HURDAT_PACIFIC.csv')
del EP['Unnamed: 0']
WP = pd.read_csv(f'{track_dir}/JTWC_WESTPACIFIC.csv')
del WP['Unnamed: 0']
SH = pd.read_csv(f'{track_dir}/JTWC_SOUTHERNHEMISPHERE.csv')
del SH['Unnamed: 0']
IO = pd.read_csv(f'{track_dir}/JTWC_INDIANOCEAN.csv')
del IO['Unnamed: 0']
Hurricanes_ALL = pd.concat([AL, EP, WP, SH, IO])
