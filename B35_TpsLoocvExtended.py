

from itertools import product
import argparse
import numpy as np
import os
import pandas as pd
import pickle as pkl
import sys
sys.path.append('./implementations/')

from Regressors import ThinPlateSpline
from implementation_tools import (
    grid,
    variable_diff,
)
from TC_and_TS_define_param import (grid_lower, grid_upper, grid_stride, folder2use, 
                                    var2use, bounds_x1, bounds_x2, shape, 
                                    train_knots_x1, train_knots_x2,
                                    LAMB, degrees_tag)

parser = argparse.ArgumentParser(
        description='Plot thin plate spline estimates (three panel)')
parser.add_argument('--integrated', dest='mode', action='store_const',
                    const='integrated', default='gridded',
                    help='process assuming integrated heat content (default: gridded)')
args = parser.parse_args()

# Set global variables
output_dir = folder2use + '/Outputs'

# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

data_dir = folder2use + '/Outputs'
results_dir = folder2use + '/Outputs'
window_size_gp = 5
stage = 'adj'

# Define various cases
if DEPTH_IDX > 1 or var2use.startswith("ML"):
    sub_list = (
        # 'all_combined',
        # 'all_hurricanes',
        # 'all_tstd',
        # 'increasing_combined',
        # 'decreasing_combined',
        'increasing_hurricanes',
        'decreasing_hurricanes',
        # 'increasing_tstd',
        # 'decreasing_tstd',
    )
else: # single level
    sub_list = (
    'all_combined',
    'all_hurricanes',
    'all_tstd',
)
    
# Loop through all cases
for sub in sub_list:
    print(sub)
    # Load output from B10 (Argo profile data)
    df = pkl.load(open(f'{data_dir}/HurricaneVariableCovDF_Subset{sub}_{degrees_tag}.pkl', 'rb')).reset_index(drop=True)
    # df = df[abs(df.standard_signed_angle)<=5].reset_index(drop=True) # Changed on Oct 13
    # df = df[df.before_t > '2007-01-01 00:00:00'].reset_index(drop=True) # Changed on Oct 22 (to replicate Hu analysis (from 2007 to 2018))
    # df = df[df.after_t < '2018-12-31 23:59:59'].reset_index(drop=True) # Changed on Oct 22
    X = np.array(df[['standard_signed_angle', 'hurricane_dtd']]).copy()
    
    # ############# Change bounds (-5, +5) - bounds_x1/2 and shape must be the same also in B21
    # bounds_x1 = (-5, +5) #(-8, +8)
    # bounds_x2 = (-2, +20)
    # train_knots = grid(bounds_x1, bounds_x2, 21, 45) #grid(bounds_x1, bounds_x2, 33, 45) 21x45= 945 (shape train_knots)
    # n_param = train_knots.shape[0] + 3
    # shape = (67, 400)  ########## (100, 400)
    # test_X = grid(bounds_x1, bounds_x2, *shape)
    # n_test = test_X.shape[0]
    
    train_knots = grid(bounds_x1, bounds_x2, train_knots_x1, train_knots_x2) #grid(bounds_x1, bounds_x2, 21, 45) 21x45= 945 (shape train_knots)
    n_param = train_knots.shape[0] + 3
    test_X = grid(bounds_x1, bounds_x2, *shape)
    n_test = test_X.shape[0]
    
    y = variable_diff(df, stage, 0)
    # Initialize array for LOOCV metrics
    LOOCV = np.zeros((len(LAMB), DEPTH_IDX, 4, len(y)))

    for depth_idx in range(DEPTH_IDX):
        print(depth_idx)
        y = variable_diff(df, stage, depth_idx)
        # I don't think these two lines are needed...
        #S = df['var'].apply(lambda x: x[depth_idx]).values
        #W = 1 / S
        # Load output from B10 (block covariance script)
        block_S = pkl.load(open(f'{results_dir}/BlockCovmat_{window_size_gp}_'
                                f'{(depth_idx+1)*10:03d}{sub}_{degrees_tag}.pkl', 'rb'))
        block_W = pkl.load(open(f'{results_dir}/BlockPremat_{window_size_gp}_'
                                f'{(depth_idx+1)*10:03d}{sub}_{degrees_tag}.pkl', 'rb'))
        
        # Loop through lambda values
        for idx, lamb in enumerate(LAMB):
            print(lamb)
            # Perform the thin plate spline
            tps_block = ThinPlateSpline(lamb=lamb, knots=train_knots)
            tps_block.fit(X, y, W=block_W)
            ret = tps_block.predict(X, sd='diag', S=block_S, k=2, diag_H=True)
            
            # Calculate metrics needed to compute the LOOCV score
            var_inv = (1/ret[1])**2
            norm_var_inv = var_inv / var_inv.sum()
            resid = (y - ret[0])
            diag_H = ret[3]
            
            # Calculate LOOCV score (eq 44 in paper)
            LOOCV[idx, depth_idx, 0, :] = norm_var_inv * (
                    (resid / (1 - diag_H)) ** 2)
            # Store other parameters from the thin plate spline...
            LOOCV[idx, depth_idx, 1, :] = ret[0]
            LOOCV[idx, depth_idx, 2, :] = ret[1]
            LOOCV[idx, depth_idx, 3, :] = ret[3]

    # Save lambda values and LOOCV scores
    loocv_data = (LAMB, LOOCV)
    pkl.dump(loocv_data, open(f'{output_dir}/B35_LOOCV_Data_{sub}_{degrees_tag}.pkl', 'wb'))

