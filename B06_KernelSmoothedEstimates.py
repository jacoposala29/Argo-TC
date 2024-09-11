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
from TC_and_TS_define_param import (depth_layers, folder2use, h_B06, n_level_min_mask, 
                                    min_TCwind,
                                    folder2use_S_ML_incr_decr_B06_B10,
                                    folder2use_T_ML_incr_decr_B06_B10,
                                    folder2use_PD_ML_incr_decr_B06_B10,
                                    bounds_x1, bounds_x2, shape,
                                    var2use, grid_lower, grid_upper, grid_stride,
                                    ML_delta, mode_tag)

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

# Load utput from B05 necessary for MLD calculation for mask incr vs decr
df_S_forML = pkl.load(open(f'{folder2use_S_ML_incr_decr_B06_B10}/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))
df_T_forML = pkl.load(open(f'{folder2use_T_ML_incr_decr_B06_B10}/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))
df_PD_forML = pkl.load(open(f'{folder2use_PD_ML_incr_decr_B06_B10}/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))
# Read in dataframe from B05
df = pkl.load(open(str(folder2use) + '/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))
df = df.dropna()
# Call ML_incr_decr function
DFS, df_main, mask_incr, mask_decr = ML_incr_decr(depth_layers, df, df_S_forML, df_T_forML, df_PD_forML, min_TCwind, ML_delta, mode_tag)

# Grid over which the 2d plot will be made, and estimates calculated in this script
test_X = grid(bounds_x1, bounds_x2, *shape).copy()
n_test = test_X.shape[0]
# Save grid
pkl.dump(test_X, open(f'{output_dir}/test_X.pkl', 'wb'))


# Loop across stages ('adj', 'raw', 'mf') and categories (TSTD, hurricanes, combined)
for stage, (df, subset) in product(STAGE, DFS):
    fn_out = f'{output_dir}/KernelSmoothedMatx_{h_B06}_{stage}_{subset}.pkl'
    #if os.path.exists(fn_out):
    #    continue
    estimates = np.zeros((n_test, depth_layers))
    # Loop across depth levels
    for depth_idx in range(depth_layers):
        print(stage, subset, depth_idx)
        ks = KernelSmoother(h=h_B06)
        # Creates space-time training set from observations
        # standard_signed_angle is distance from TS track in degrees
        # hurricane_dtd is delta time in days
        train_X = np.array(df[['standard_signed_angle', 'hurricane_dtd']]).copy()
        if stage == 'mf': # Effect on mean field
            raw = variable_diff(df, 'raw', depth_idx) # raw after - raw before
            adj = variable_diff(df, 'adj', depth_idx) # adj after - adj before
            train_y = (raw - adj).copy() # (raw after - adj after) - (raw before - adj before) = (mf after - mf before)
        else:
            # Calculates post-pre TS temperature/salinity for the depth we're considering now
            train_y = variable_diff(df, stage, depth_idx) # (raw after - raw before) or (adj after - adj before)
        # Fit model on observations of post-pre TS differences of Temp/Salinity
        ks.fit(train_X, train_y)
        # Use the fitted model to predict values on a regular grid
        estimates[:, depth_idx] = ks.predict(test_X)
    # Writes output
    pkl.dump(estimates, open(fn_out, 'wb'))
    
    
    