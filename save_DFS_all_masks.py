import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sps
import datetime
import multiprocessing
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time 
import sys
import matplotlib.pyplot as plt
from scipy.io import savemat
from holteandtalley import HolteAndTalley

from functools import partial
from itertools import product

sys.path.append('./implementations/')
from implementation_tools import (ML_incr_decr)
from tools import covariance_matrix
from TC_and_TS_define_param import folder2use, \
    depth_layers, var_min_B10, n_level_min_mask, min_TCwind, \
    folder2use_S_ML_incr_decr_B06_B10, \
    folder2use_T_ML_incr_decr_B06_B10, \
    folder2use_PD_ML_incr_decr_B06_B10, var2use, \
    bounds_x1, bounds_x2, degrees_tag, \
    ML_delta, mode_tag
from TC_and_TS_define_param import mode_tags, absolute_mask_tags, absolute_ML_delta, fraction_ML_delta, fraction_mask_tags, folder2use, depth_layers, var_min_B10, n_level_min_mask, min_TCwind, var2use, grid_lower, grid_upper, grid_stride, grid_lower_forML, grid_upper_forML, grid_stride_forML

CPU_COUNT = multiprocessing.cpu_count() # 4

MLDused_tag  ='MLDdensity' # 'MLDdensity' MLDsalinity
mode_tags = ['absolute', 'fraction']
absolute_mask_tags = ['plus10m', 'plus20m', 'plus30m', 'plus40m', 'plus50m', 'plus60m', 'plus70m']
fraction_mask_tags = ['1ML', 'halfML', '1thirdML', '2thirdML']
absolute_ML_delta = [10, 20, 30, 40, 50, 60, 70]
fraction_ML_delta = [1/2, 1/3, 2/3, 1]

# Import data from B09 output 
data_dir = folder2use + '/Outputs'
plot_dir = '../Figures/B10_check_plots/'
df = pkl.load(open(f'{data_dir}/HurricaneAdjRawVariableCovDF.pkl', 'rb'))
df = df[abs(df.standard_signed_angle)<=bounds_x1[1]] 
df = df.dropna()
df.shape # (16674, 21)
df_shape_0 = str(df.shape[0])

# Plot to check outlier removal
fig, axs = plt.subplots(2)
#fig.suptitle('var for ' + var2use + ', var_min_B10 = ' + str(var_min_B10) + ', # pairs = ' +str(df.shape[0])) 
axs[0].plot(np.log(df['var'].apply(lambda x: np.min(x))))
#fig.suptitle('var for ' + var2use + ', var_min_B10 = ' + str(var_min_B10) + ', # pairs = ' +str(df.shape[0]))
fig.suptitle('var for ' + var2use + ', var_min_B10 = ' + str(var_min_B10) + 
             ', # pairs before = ' + df_shape_0 , fontsize = 14) 
axs[1].hist(np.log(df['var'].apply(lambda x: np.min(x))), bins=np.arange(min(np.log(df['var'].apply(lambda x: np.min(x)))), 
                                                                         max(np.log(df['var'].apply(lambda x: np.min(x)))), 
                                                                         0.1*abs(var_min_B10))) 
plt.savefig(f'{plot_dir}/B10_check_plot_before_screening_{var2use}_{degrees_tag}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)


# Remove profiles based on outlier of variance
df_excluded = df[np.log(df['var'].apply(lambda x: np.min(x))) < var_min_B10].sort_values([
    'before_pid',
    'after_pid',
])

df = df[np.log(df['var'].apply(lambda x: np.min(x))) >= var_min_B10].sort_values([
    'before_pid',
    'after_pid',
]) 

# Plot after outliers are removed  to make sure everything went OK 
fig, axs = plt.subplots(2)
df_shape_0 = str(df.shape[0])
axs[0].plot(np.log(df['var'].apply(lambda x: np.min(x))))
fig.suptitle('var for ' + var2use + ', var_min_B10 = ' + str(var_min_B10) + 
             ', # pairs after = ' + df_shape_0, fontsize = 14)
axs[1].hist(np.log(df['var'].apply(lambda x: np.min(x))), bins=np.arange(min(np.log(df['var'].apply(lambda x: np.min(x)))), 
                                                                         max(np.log(df['var'].apply(lambda x: np.min(x)))), 
                                                                         0.1*abs(var_min_B10)))
plt.savefig(f'{plot_dir}/B10_check_plot_after_screening_{var2use}_{degrees_tag}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)

# Load output from B05 necessary for MLD calculation for mask incr vs decr
df_S_forML = pkl.load(open(f'{folder2use_S_ML_incr_decr_B06_B10}/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))
df_T_forML = pkl.load(open(f'{folder2use_T_ML_incr_decr_B06_B10}/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))
df_PD_forML = pkl.load(open(f'{folder2use_PD_ML_incr_decr_B06_B10}/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))

# _________________________________________________________________________________________________________

# Combine before profile ID and after profile ID in a single string
df["pid_combined"] = df['before_pid'] + '_' + df['after_pid']
df_S_forML["pid_combined"] = df_S_forML['before_pid'] + '_' + df_S_forML['after_pid']
df_T_forML["pid_combined"] = df_T_forML['before_pid'] + '_' + df_T_forML['after_pid']
df_PD_forML["pid_combined"] = df_PD_forML['before_pid'] + '_' + df_PD_forML['after_pid']

# Keep only rows where we have all the data to calculate MLD
df_main_common_ids = df[df['pid_combined'].isin(df_S_forML['pid_combined']) &
                        df['pid_combined'].isin(df_T_forML['pid_combined']) &
                        df['pid_combined'].isin(df_PD_forML['pid_combined'])]
df_S_forML_common_ids = df_S_forML[df_S_forML['pid_combined'].isin(df_main_common_ids['pid_combined'])]
df_T_forML_common_ids = df_T_forML[df_T_forML['pid_combined'].isin(df_main_common_ids['pid_combined'])]
df_PD_forML_common_ids = df_PD_forML[df_PD_forML['pid_combined'].isin(df_main_common_ids['pid_combined'])]

# Delete column pid_combined from the df being considered, to make it the same as the old format
df_main_common_ids = df_main_common_ids.drop(columns=['pid_combined'])

# Remove bad profile
df_main_common_ids = df_main_common_ids[~((df_main_common_ids['before_pid'] == '0004901765_00086') & (df_main_common_ids['after_pid'] == '0004901765_00087'))]

    
# Calculate MLD
salinityMLD = []
temperatureMLD = []
densityMLD = []

# Initialize empty mask lists for all masks        
for mode_tag in mode_tags:
    if mode_tag == 'absolute':
        mask_tags = absolute_mask_tags
    elif mode_tag == 'fraction':
        mask_tags = fraction_mask_tags
    else:
        print(f"Unknown mode_tag: {mode_tag}")
        continue

    for mask_tag in mask_tags:
        # Dynamically create an empty list with a variable name
        mask_incr_name = f"mask_incr_{mask_tag}"
        locals()[mask_incr_name] = []
        mask_decr_name = f"mask_decr_{mask_tag}"
        locals()[mask_decr_name] = []

for i_t in np.arange(df_main_common_ids.shape[0]):
    conservative_temperature_nonans = df_T_forML_common_ids.iloc[i_t].raw_before_variable
    absolute_salinity_nonans = df_S_forML_common_ids.iloc[i_t].raw_before_variable
    potential_density_nonans = df_PD_forML_common_ids.iloc[i_t].raw_before_variable

    pressure_levels = np.arange(int(grid_lower_forML),int(grid_upper_forML)+1,int(grid_stride_forML))            
    # Apply function
    h = HolteAndTalley(pressures=pressure_levels, \
                   temperatures=conservative_temperature_nonans, \
                   salinities=absolute_salinity_nonans, \
                   densities=potential_density_nonans)            

    # The salinity algorithm mixed layer depth
    temperatureMLD.append(h.tempMLD)
    salinityMLD.append(h.salinityMLD)
    densityMLD.append(h.densityMLD)
    
    # Find depth index to insert ML
    i_depth_MLD = np.searchsorted(pressure_levels, densityMLD[i_t])   

    # Calculate increasing vs decreasing mask
    
    # MIXED LAYER
    # Create new depth axis which includes MLD
    depth_MLD = np.insert(pressure_levels, i_depth_MLD, densityMLD[i_t])
    # Calculate salinity at MLD via interpolation
    salinity_abs_upper = np.interp(depth_MLD, pressure_levels, absolute_salinity_nonans)

    # Calculate weighted average of salinity within the ML (from surface to MLD)
    # First, need to calculate dz (depth of each layer associated with a salinity value)
    depth_MLD_upper = np.insert(depth_MLD, i_depth_MLD, densityMLD[i_t])# add another MLD value next to the other one
    depth_MLD_upper = np.insert(depth_MLD_upper, 0, depth_MLD_upper[0]) # add another zero at the beginning of the depth

    dz = (depth_MLD_upper[2:i_depth_MLD+3]-depth_MLD_upper[0:i_depth_MLD+1])/2 # calculate dz as half of the difference between depth value above and below each depth
    MLS_abs_wgt = (np.sum(dz * salinity_abs_upper[0:i_depth_MLD+1]))/np.sum(dz) # weighted average from surface to MLD

    for mode_tag in mode_tags:
        if mode_tag == 'absolute':
            mask_tags = absolute_mask_tags
            ML_deltas = absolute_ML_delta
        elif mode_tag == 'fraction':
            mask_tags = fraction_mask_tags
            ML_deltas = fraction_ML_delta
        else:
            print(f"Unknown mode_tag: {mode_tag}")
            continue
    
        for mask_tag, ML_delta in zip(mask_tags, ML_deltas):
    
            # LAYER BELOW ML
            # Calculate bottom limit of layer to consider
            if mode_tag == 'fraction': # e.g., MLD + one third of it
                bottom_limit = densityMLD[i_t] + densityMLD[i_t] * ML_delta
            elif mode_tag == 'absolute': # e.g., 10 m below MLD
                bottom_limit = densityMLD[i_t] + ML_delta     
            # Create new depth axis which includes meters_below_ML below MLD
            depth_below_ML = np.insert(depth_MLD, i_depth_MLD, densityMLD[i_t])  # add another MLD value next to the other one
            # Find index of bottom limit
            i_depth_meters_below_MLD = np.searchsorted(depth_below_ML, bottom_limit)
            # Create new depth axis which includes bottom_limit
            depth_below_ML = np.insert(depth_below_ML, i_depth_meters_below_MLD, bottom_limit)  # add bottom_limit
            depth_below_ML = np.insert(depth_below_ML, i_depth_meters_below_MLD, bottom_limit)  # add bottom_limit
            # Calculate salinity at bottom_limit via interpolation
            salinity_abs_lower = np.interp(depth_below_ML, pressure_levels, absolute_salinity_nonans)
            
            dz = (depth_below_ML[i_depth_MLD+2:i_depth_meters_below_MLD+2] - depth_below_ML[i_depth_MLD:i_depth_meters_below_MLD])/2  # calculate dz as half of the difference between depth value above and below each depth
            salinity_abs_wgt_lower = (np.sum(dz * salinity_abs_lower[i_depth_MLD+1:i_depth_meters_below_MLD+1])) / np.sum(dz)  # weighted average from MLD to bottom_limit
            
            # Calculate difference between salinity in layer below ML and weighted average of salinity in the ML
            mean_salinity_diff_below = np.mean(salinity_abs_wgt_lower - MLS_abs_wgt)
            
            # If mean_salinity_diff_below > 0 --> increasing
            mask_incr_name = f"mask_incr_{mask_tag}"
            mask_decr_name = f"mask_decr_{mask_tag}"
            if mean_salinity_diff_below > 0: 
                locals()[mask_incr_name].append(1)
                locals()[mask_decr_name].append(0)
            # If mean_salinity_diff_below < 0 --> decreasing
            elif mean_salinity_diff_below < 0:
                locals()[mask_incr_name].append(0)
                locals()[mask_decr_name].append(1)
            else: # Nans or other weird behavior
                locals()[mask_incr_name].append(0)
                locals()[mask_decr_name].append(0)                    

# Create Pandas Series or DataFrames from lists with proper index alignment
temperature_MLD_series = pd.Series(temperatureMLD, index=df_main_common_ids['wind'])
salinity_MLD_series = pd.Series(salinityMLD, index=df_main_common_ids['wind'])
density_MLD_series = pd.Series(densityMLD, index=df_main_common_ids['wind'])

# Create a list of tuples containing conditions and their corresponding names
DFS = [
    (df_main_common_ids[df_main_common_ids['wind'] >= min_TCwind], 'all_hurricanes'),
]


for mode_tag in mode_tags:
    if mode_tag == 'absolute':
        mask_tags = absolute_mask_tags
    elif mode_tag == 'fraction':
        mask_tags = fraction_mask_tags
    else:
        print(f"Unknown mode_tag: {mode_tag}")
        continue

    for mask_tag in mask_tags:
        mask_incr_name = f"mask_incr_{mask_tag}"
        mask_decr_name = f"mask_decr_{mask_tag}"
        # Check if dynamically created variables exist
        incr_condition = locals()[mask_incr_name] & (df_main_common_ids['wind'] >= min_TCwind)
        decr_condition = locals()[mask_decr_name] & (df_main_common_ids['wind'] >= min_TCwind)

        # Save for hurricane cases only
        DFS.append((incr_condition[df_main_common_ids['wind'] >= min_TCwind], f"{mask_incr_name} & (wind >= {min_TCwind})"))
        DFS.append((decr_condition[df_main_common_ids['wind'] >= min_TCwind], f"{mask_decr_name} & (wind >= {min_TCwind})"))

# Use Pandas Series in the indexing operation
hurricane_mask = df_main_common_ids['wind'] >= min_TCwind
temperature_MLD_condition = [temp for temp, m in zip(temperatureMLD, hurricane_mask) if m]
salinity_MLD_condition = [temp for temp, m in zip(salinityMLD, hurricane_mask) if m]
density_MLD_condition = [temp for temp, m in zip(densityMLD, hurricane_mask) if m]

DFS += [
    (temperature_MLD_condition, 'temperatureMLD'),
    (salinity_MLD_condition, 'salinityMLD'),
    (density_MLD_condition, 'densityMLD')
]

# Save DFS with the appropriate name
pkl.dump(DFS, open(f'{data_dir}/DFS_allmasks_{var2use}_{degrees_tag}_{MLDused_tag}.pkl', 'wb'))
