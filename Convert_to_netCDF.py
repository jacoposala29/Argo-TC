#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:42:06 2023

@author: jacoposala
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 09:09:07 2021

@author: jacoposala
"""
from itertools import product
import numpy as np
import os
import pandas as pd
import pickle as pkl
import sys
from TC_and_TS_define_param import (depth_layers, folder2use, h_B06, perc_points_min, 
                                    min_TCwind, folder2use_ML_incr_decr_B06_B10, 
                                    bounds_x1, bounds_x2, shape)
import netCDF4 as nc
import xarray as xr

sys.path.append('./implementations/')
from Regressors import KernelSmoother
from implementation_tools import (
    grid,
    variable_diff,
    ML_incr_decr
)

# Set global variables
output_dir = str(folder2use) + '/Outputs/'
STAGE = ('adj', 'raw', 'mf')

# Salinity case, output from B05
df_var = pkl.load(open(f'{folder2use_ML_incr_decr_B06_B10}/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))

# define data with variable attributes
data_vars = {'before_pid':(['index'], df_var.before_pid),
             'before_t':(['index'], df_var.before_t),
             'after_pid':(['index'], df_var.after_pid),
             'after_t':(['index'], df_var.after_t),
             'angle':(['index'], df_var.angle),
             'wind':(['index'], df_var.wind),
             'proj_t':(['index'], df_var.proj_t),
             'sign':(['index'], df_var.sign),
             'argo_lat':(['index'], df_var.argo_lat),
             'argo_lon':(['index'], df_var.argo_lon),
             'HurricaneID':(['index'], df_var.HurricaneID),
             'profile_dt':(['index'], df_var.profile_dt),
             'hurricane_dt':(['index'], df_var.hurricane_dt),
             'signed_angle':(['index'], df_var.signed_angle),
             'standard_signed_angle':(['index'], df_var.standard_signed_angle),
             'hurricane_dtd':(['index'], df_var.hurricane_dtd),
             'profile_dtd':(['index'], df_var.profile_dtd),
             'hurricane_id':(['index'], df_var.hurricane_id),
             'adj_before_variable':(['index','depth'], np.stack(df_var.adj_before_variable.values)),
             'adj_after_variable':(['index','depth'], np.stack(df_var.adj_after_variable.values)),
             'raw_before_variable':(['index','depth'], np.stack(df_var.raw_before_variable.values)),
             'raw_after_variable':(['index','depth'], np.stack(df_var.raw_after_variable.values)),
             }

# define coordinates
coords = {'index': (['index'], df_var.index),
          'depth': (['depth'], np.arange(0,41))}

# create dataset
ds = xr.Dataset(data_vars=data_vars, 
                coords=coords)

# Save as netCDF file
ds.to_netcdf('Donata.nc')

# Check if it worked
ds_loaded = nc.Dataset('Donata.nc')

ds_loaded['hurricane_id'][:]
ds_loaded['adj_after_variable'][:]










