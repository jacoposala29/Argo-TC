#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 21:09:33 2021

@author: jacoposala
"""
'''
author: addison@stat.cmu.edu

Construct block diagonal covariance matrices.
'''
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

CPU_COUNT = multiprocessing.cpu_count() # 4

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

# Recall ML_incr_decr function
DFS, df_main, mask_incr, mask_decr = ML_incr_decr(depth_layers, df, df_S_forML, df_T_forML, df_PD_forML, min_TCwind, ML_delta, mode_tag)

# Save DFS with the appropriate name
pkl.dump(DFS, open(f'{data_dir}/DFS_{var2use}_{degrees_tag}.pkl', 'wb'))




# # Uncomment this part if .mat DFS is needed
# df = pd.read_pickle('../SALINITY_40levels_oct14/Outputs/DFS_Salinity.pkl')
# df_T = pd.read_pickle('../TEMPERATURE_test/Outputs/DFS_Temperature.pkl')

# decreasing_hurricanes = df[0][0]
# decreasing_hurricanes['before_t'] = decreasing_hurricanes['before_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# decreasing_hurricanes['after_t'] = decreasing_hurricanes['after_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# decreasing_hurricanes['proj_t'] = decreasing_hurricanes['proj_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# increasing_hurricanes = df[1][0]
# increasing_hurricanes['before_t'] = increasing_hurricanes['before_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# increasing_hurricanes['after_t'] = increasing_hurricanes['after_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# increasing_hurricanes['proj_t'] = increasing_hurricanes['proj_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# all_hurricanes = df[2][0]
# all_hurricanes['before_t'] = all_hurricanes['before_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# all_hurricanes['after_t'] = all_hurricanes['after_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# all_hurricanes['proj_t'] = all_hurricanes['proj_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# savemat("../SALINITY_40levels_oct14/Outputs/DFS_decr_hur_salinity.mat", {name: col.values for name, col in decreasing_hurricanes.items()})
# savemat("../SALINITY_40levels_oct14/Outputs/DFS_incr_hur_salinity.mat", {name: col.values for name, col in increasing_hurricanes.items()})
# savemat("../SALINITY_40levels_oct14/Outputs/DFS_all_hur_salinity.mat", {name: col.values for name, col in all_hurricanes.items()})

# decreasing_hurricanes = df_T[0][0]
# decreasing_hurricanes['before_t'] = decreasing_hurricanes['before_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# decreasing_hurricanes['after_t'] = decreasing_hurricanes['after_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# decreasing_hurricanes['proj_t'] = decreasing_hurricanes['proj_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# increasing_hurricanes = df_T[1][0]
# increasing_hurricanes['before_t'] = increasing_hurricanes['before_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# increasing_hurricanes['after_t'] = increasing_hurricanes['after_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# increasing_hurricanes['proj_t'] = increasing_hurricanes['proj_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# all_hurricanes = df_T[2][0]
# all_hurricanes['before_t'] = all_hurricanes['before_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# all_hurricanes['after_t'] = all_hurricanes['after_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# all_hurricanes['proj_t'] = all_hurricanes['proj_t'].dt.strftime("%d-%b-%Y %H:%M:%S")
# savemat("../TEMPERATURE_test/Outputs/DFS_decr_hur_temperature.mat", {name: col.values for name, col in decreasing_hurricanes.items()})
# savemat("../TEMPERATURE_test/Outputs/DFS_incr_hur_temperature.mat", {name: col.values for name, col in increasing_hurricanes.items()})
# savemat("../TEMPERATURE_test/Outputs/DFS_all_hur_temperature.mat", {name: col.values for name, col in all_hurricanes.items()})

# # Save df_excluded
# pkl.dump(df_excluded, open(f'{data_dir}/df_excluded__B10_{var2use}.pkl', 'wb'))

# # PLOT ON A MAP THE LOCATIONS OF THE EXCLUDED PROFILES BASED ON VARIANCE
# argo_lat = df_excluded['argo_lat']
# argo_lon = df_excluded['argo_lon']
# ## Plot global map using cartopy
# f=plt.figure(figsize=(10,8))  ## set the figure size
# ax=plt.axes(projection=ccrs.PlateCarree(central_longitude=180));
# CS=ax.plot(argo_lon, argo_lat, 'x', transform=ccrs.PlateCarree(), zorder = -1);
# ax.coastlines(); 
# ax.add_feature(cfeature.LAND)
# plt.title('Position of excluded profiles (based on variance) ' + var2use + ' - B10',fontsize=14);
# plt.savefig(f'{plot_dir}/B10_excluded_profiles_' + f'{var2use}' + '.png', bbox_inches='tight', dpi=300, format='png')

if __name__ == '__main__':  
    for (df, sub) in DFS:
        print(sub)
        # Saves file
        pkl.dump(df, open(f'{data_dir}/HurricaneVariableCovDF_Subset{sub}_{degrees_tag}.pkl', 'wb'))
        # df = df[abs(df.standard_signed_angle)<=5].reset_index(drop=True) # Changed on Oct 13
        # df = df[df.before_t > '2007-01-01 00:00:00'].reset_index(drop=True) # Changed on Oct 22 (to replicate Hu analysis (from 2007 to 2018))
        # df = df[df.after_t < '2018-12-31 23:59:59'].reset_index(drop=True) 
        # Import directions
        results_dir = folder2use + '/Outputs'
        window_size_gp = 5
        
        before_pids = np.sort(df['before_pid'].unique())
        
        MLE = pkl.load(open(f'{results_dir}/MleCoefDF_{window_size_gp}.pkl', 'rb'))
        
        df_list = [df[df['before_pid']==bp] for bp in before_pids]
        for depth_idx in range(depth_layers):
            print(depth_idx)
            cov = partial(covariance_matrix, df_param=MLE, depth_idx=depth_idx)
        
            with multiprocessing.Pool(processes=CPU_COUNT) as pool:
                covmat_list = pool.map(cov, df_list)
        
            with multiprocessing.Pool(processes=CPU_COUNT) as pool:
                premat_list = pool.map(np.linalg.inv, covmat_list)
        
            C = sps.block_diag(covmat_list)
            P = sps.block_diag(premat_list)
            pkl.dump(C, open(f'{results_dir}/BlockCovmat_{window_size_gp}_'
                             f'{(depth_idx+1)*10:03d}{sub}_{degrees_tag}.pkl', 'wb'))
            pkl.dump(P, open(f'{results_dir}/BlockPremat_{window_size_gp}_'
                             f'{(depth_idx+1)*10:03d}{sub}_{degrees_tag}.pkl', 'wb'))
