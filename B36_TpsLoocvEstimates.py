from itertools import product
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import sys

sys.path.append('./implementations/')
from Regressors import ThinPlateSpline
from implementation_tools import (
    grid,
    variable_diff,
)
             
from TC_and_TS_define_param import (grid_lower, grid_upper, grid_stride, folder2use, var2use, k_factor,
                                    bounds_x1, bounds_x2, shape, train_knots_x1, train_knots_x2, lamb_choice, degrees_tag)

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
else: 
    sub_list = (
    'all_combined',
    'all_hurricanes',
    'all_tstd',
)
    



# Loop through all cases
for sub in sub_list:
    print(sub)
    # Load output from B10 (Argo profile data)
    df = pkl.load(open(f'{data_dir}/HurricaneVariableCovDF_Subset{sub}_{degrees_tag}.pkl', 'rb'))
    # df = df[abs(df.standard_signed_angle)<=5].reset_index(drop=True) # Changed on Oct 13
    # df = df[df.before_t > '2007-01-01 00:00:00'].reset_index(drop=True) # Changed on Oct 22 (to replicate Hu analysis (from 2007 to 2018))
    # df = df[df.after_t < '2018-12-31 23:59:59'].reset_index(drop=True) 

    X = np.array(df[['standard_signed_angle', 'hurricane_dtd']]).copy()
        
    ############# Change bounds (-5, +5) - bounds_x1/2 and shape must be the same also in B21
    train_knots = grid(bounds_x1, bounds_x2, train_knots_x1, train_knots_x2) #grid(bounds_x1, bounds_x2, 21, 45) 21x45= 945 (shape train_knots)
    n_param = train_knots.shape[0] + 3
    test_X = grid(bounds_x1, bounds_x2, *shape)
    n_test = test_X.shape[0]
    
    y = variable_diff(df, stage, 0)
    ymat = np.zeros((1, DEPTH_IDX, len(y)))
    
    for depth_idx in range(DEPTH_IDX):
        ymat[0, depth_idx, :] = variable_diff(df, stage, depth_idx)
    
    # Load results from B35 (lambda values and LOOCV scores)
    LAMB_LARGE, LOOCV_LARGE = pkl.load(open(f'{data_dir}/B35_LOOCV_Data_{sub}_{degrees_tag}.pkl', 'rb')) 
    
    # Combine the two sets
    LAMB = LAMB_LARGE[1:]
    LOOCV = LOOCV_LARGE[1:, :, :, :]
    varmat = np.vstack(df['var']).T
    varmat = varmat.reshape(1, *varmat.shape)
    
    # LOOCV metric (eq 44 in paper)
    loocv_estimates = ((1/varmat) * (
        (LOOCV[:,:,1,:] - ymat) / (1-LOOCV[:,:,3,:]))**2)
    error_estimates = loocv_estimates.sum(axis=2)
    # Set NaN error to infinity
    error_estimates_nonan = error_estimates.copy()
    error_estimates_nonan[np.isnan(error_estimates)] = np.inf
    # Select lambas that minimize the cost function
    #lhat = error_estimates_nonan.argmin(axis=0)
    #argmin_lamb = LAMB[lhat] 
    #print(argmin_lamb)
    
    # error_estimates_region_select = error_estimates
    
    # Select the actual lambda to use
    inflation_factor=1.01
    # if lamb_choice == 'Adapt':
    #     lamb_seq = argmin_lamb
    # elif lamb_choice == 'Mean':
    #     lamb_seq = [np.mean(argmin_lamb) for _ in range(DEPTH_IDX)]
    # elif lamb_choice == 'Median':
    #     lamb_seq = [np.median(argmin_lamb) for _ in range(DEPTH_IDX)]
    # elif lamb_choice == 'AdaptSignal':
    #     signal_region = (df.angle <= 3).values * (0 <= df.hurricane_dtd).values * (df.hurricane_dtd <= 12).values
    #     error_estimates_signal = loocv_estimates[:,:,signal_region].sum(axis=2)
    #     error_estimates_nonsignal = loocv_estimates[:,:,~signal_region].sum(axis=2)
    #     lhat_signal = error_estimates_signal.argmin(axis=0)
    #     lhat_nonsignal = error_estimates_nonsignal.argmin(axis=0)
    #     lamb_seq = LAMB[lhat_signal]
    # elif lamb_choice == 'AdaptSignal2':
    #     signal_region = (df.angle <= 2).values * (0 <= df.hurricane_dtd).values * (df.hurricane_dtd <= 12).values
    #     error_estimates_signal = loocv_estimates[:,:,signal_region].sum(axis=2)
    #     error_estimates_nonsignal = loocv_estimates[:,:,~signal_region].sum(axis=2)
    #     lhat_signal = error_estimates_signal.argmin(axis=0)
    #     lhat_nonsignal = error_estimates_nonsignal.argmin(axis=0)
    #     lamb_seq = LAMB[lhat_signal]
    # elif lamb_choice == 'AdaptSignalInflate':
    #     signal_region = ((df.angle <= 2).values * (0 <= df.hurricane_dtd).values
    #             * (df.hurricane_dtd <= 12).values)
    #     error_estimates_signal = loocv_estimates[:,:,signal_region].sum(axis=2)
    #     min_error = error_estimates_signal.min(axis=0)
    #     lhat_inflated_signal = np.argmax(error_estimates_signal < min_error *
    # 	    inflation_factor, axis=0)
    #     lamb_seq = LAMB[lhat_inflated_signal]
    # elif lamb_choice == 'AdaptInflate': # Choice of the smallest lambda with a score = 1% more than the minimum score
    #     min_error = error_estimates.min(axis=0)
    #     inflation_factor = 1.01
    #     lhat_inflated_all = np.argmax(error_estimates < min_error * inflation_factor, axis=0)
    #     lamb_seq = LAMB[lhat_inflated_all] 
    lamb_seq = np.zeros(DEPTH_IDX)
    lhat = np.zeros(DEPTH_IDX)
    if 'Custom_diff_perc_' in lamb_choice:
        lamb_choice_val = np.float(lamb_choice.split('_')[-1])
        for depth_idx in range(DEPTH_IDX):
            bfr = error_estimates[:,depth_idx]
            diff1 = bfr[0] - np.min(bfr)
            diff2 = bfr - np.min(bfr)
            diff_ratio = diff2/diff1
            lhat_all = np.where(diff_ratio <= lamb_choice_val/100)[0] 
            lhat[depth_idx] = lhat_all[0]
            argmin_lamb = LAMB[int(lhat[depth_idx])]
            lamb_seq[depth_idx] = argmin_lamb #round(argmin_lamb)
        print(lamb_seq)
    elif 'Custom_increase_perc_reverse' in lamb_choice:
        # lamb_choice_val = np.float(lamb_choice.split('_')[-1])
        for depth_idx in range(DEPTH_IDX):
            bfr = error_estimates[:,depth_idx]
            bfr_last = bfr[-1]
            bfr_last_ratio = np.flip(bfr)/bfr_last
            # bfr_last_ratio_flipped = np.flip(bfr_last_ratio)
            lhat_all = len(bfr) - np.where(bfr_last_ratio >= 1.1)[0][0] - 1 
            lhat[depth_idx] = lhat_all
            argmin_lamb = LAMB[int(lhat[depth_idx])]
            lamb_seq[depth_idx] = argmin_lamb #round(argmin_lamb)
        print(lamb_seq)  
        
        
# ADD B34 AS A FUNCTION TO ONLY MAKE PLOTS
        
        
        
        
        
# OLD METHODS TO SELECT LAMBDA
    # elif lamb_choice == 'Custom':
    #     print('Case to revisit if needed (see changes in Custom2')
    #     for depth_idx in range(DEPTH_IDX):
    #         delta_percent = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/ \
    #             error_estimates[0:-1, depth_idx]/(LAMB[1:]-(LAMB[0:-1]))
    #         #ciao = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/error_estimates[0:-1, depth_idx]/(LAMB[1:]-(LAMB[0:-1]))
    #         #ciao = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/(LAMB[1:]-(LAMB[0:-1]))
    #         d_error = np.concatenate(([delta_percent[0]], delta_percent))
    #         lhat_all = np.where(d_error >= -.001)[0] 
    #         lhat[depth_idx] = lhat_all[0]
    #         argmin_lamb = LAMB[int(lhat[depth_idx])]
    #         lamb_seq[depth_idx] = argmin_lamb #round(argmin_lamb)
    #     print(lamb_seq)
    # elif lamb_choice == 'AdaptInflate': # Choice of the smallest lambda with a score = 1% more than the minimum score
    #     min_error = error_estimates.min(axis=0)
    #     inflation_factor = 1.01
    #     lhat_inflated_all = np.argmax(error_estimates < min_error * inflation_factor, axis=0)
    #     lamb_seq = LAMB[lhat_inflated_all]
    #     print(lamb_seq)
    # elif lamb_choice == 'Custom_find_min':
    #     for depth_idx in range(DEPTH_IDX):
    #         # Scores for this depth level
    #         bfr = error_estimates[:,depth_idx]
    #         # Find index which corresponds to min of error
    #         lhat[depth_idx] = min(range(len(bfr)), key=bfr.__getitem__)
    #         # Find correspondent lambda
    #         argmin_lamb = LAMB[int(lhat[depth_idx])]
    #         lamb_seq[depth_idx] = argmin_lamb #round(argmin_lamb)
    #     print(lamb_seq)
    # elif lamb_choice == 'Custom2_0.005':
    #     for depth_idx in range(DEPTH_IDX):
    #         Custom2_delta_percent = abs((error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/ \
    #             (error_estimates[0:-1, depth_idx]))
    #         d_error = np.concatenate(([Custom2_delta_percent[0]], Custom2_delta_percent))
    #         lhat_all = np.where(d_error <= .005)[0] 
    #         lhat[depth_idx] = lhat_all[0]
    #         argmin_lamb = LAMB[int(lhat[depth_idx])]
    #         lamb_seq[depth_idx] = argmin_lamb #round(argmin_lamb)
    #     print(lamb_seq)
    # elif lamb_choice == '0.1_Custom':
    #     for depth_idx in range(DEPTH_IDX):
    #         delta_percent = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/ \
    #             error_estimates[0:-1, depth_idx]/(LAMB[1:]-(LAMB[0:-1]))
    #         #ciao = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/error_estimates[0:-1, depth_idx]/(LAMB[1:]-(LAMB[0:-1]))
    #         #ciao = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/(LAMB[1:]-(LAMB[0:-1]))
    #         d_error = np.concatenate(([delta_percent[0]], delta_percent))
    #         lhat_all = np.where(d_error >= -.001)[0] 
    #         lhat[depth_idx] = lhat_all[0]
    #         argmin_lamb = LAMB[int(lhat[depth_idx])]
    #         lamb_seq[depth_idx] = 0.1*argmin_lamb #round(argmin_lamb)
            
    #         lhat_all_factor = np.where(LAMB <= lamb_seq[depth_idx])[0]
    #         lhat[depth_idx] = lhat_all_factor[-1]
    #     print(lamb_seq)
    # else:
    #     lamb_seq = [lamb_choice for _ in range(DEPTH_IDX)]
        
    # Perform thin plate splines, as in old B11
    PREDS_NOREW = np.zeros((n_test, DEPTH_IDX))
    PREDS_BLOCK = np.zeros((n_test, DEPTH_IDX))
    STDEV_BLOCK = np.zeros((n_test, DEPTH_IDX))
    MASK_BLOCK = np.zeros((n_test, DEPTH_IDX))
    
    THETA_BLOCK = np.zeros((n_param, DEPTH_IDX))
    BASIS_BLOCK = np.zeros((n_test, n_param)) # Invariant to depth
    COV_THETA_BLOCK = np.zeros((n_param, n_param, DEPTH_IDX))
    
    for depth_idx, lamb in zip(range(DEPTH_IDX), lamb_seq):
        print(depth_idx)
        y = variable_diff(df, stage, depth_idx)
        S = df['var'].apply(lambda x: x[depth_idx]).values
        W = 1 / S
        if sub == 'All':
            block_S = pkl.load(open(f'{data_dir}/BlockCovmat_{window_size_gp}_'
                                    f'{(depth_idx+1)*10:03d}_{degrees_tag}.pkl', 'rb'))
            block_W = pkl.load(open(f'{data_dir}/BlockPremat_{window_size_gp}_'
                                    f'{(depth_idx+1)*10:03d}_{degrees_tag}.pkl', 'rb'))
        else:
            block_S = pkl.load(open(f'{data_dir}/BlockCovmat_{window_size_gp}_'
                                    f'{(depth_idx+1)*10:03d}{sub}_{degrees_tag}.pkl', 'rb'))
            block_W = pkl.load(open(f'{data_dir}/BlockPremat_{window_size_gp}_'
                                    f'{(depth_idx+1)*10:03d}{sub}_{degrees_tag}.pkl', 'rb'))
          
        tps_norew = ThinPlateSpline(lamb=lamb, knots=train_knots)
        tps_block = ThinPlateSpline(lamb=lamb, knots=train_knots)
        tps_norew.fit(X, y)
        tps_block.fit(X, y, W=block_W)
        ret0 = tps_norew.predict(test_X)
        ret2 = tps_block.predict(test_X, sd='diag', S=block_S, k=k_factor) ##### REMOVE k=2 from here -> in TC_TS
        print(ret0)
        PREDS_NOREW[:, depth_idx] = ret0
        PREDS_BLOCK[:, depth_idx] = ret2[0]
        STDEV_BLOCK[:, depth_idx] = ret2[1]
        MASK_BLOCK[ :, depth_idx] = ret2[2]
        THETA_BLOCK[:, depth_idx] = tps_block.theta
        BASIS_BLOCK[:, :] =      tps_block._test_basis
        COV_THETA_BLOCK[:, :, depth_idx] =  tps_block._cov_theta
    
    # Save TPS estimates outputs
    pkl.dump(PREDS_NOREW,
            open(f'{output_dir}/TPS_LOOCV_Preds_NoRew_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'wb'))
    pkl.dump(PREDS_BLOCK,
            open(f'{output_dir}/TPS_LOOCV_Preds_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'wb'))
    pkl.dump(STDEV_BLOCK,
            open(f'{output_dir}/TPS_LOOCV_Stdev_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'wb'))
    pkl.dump(MASK_BLOCK,
            open(f'{output_dir}/TPS_LOOCV_Mask_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'wb'))
    
    pkl.dump(THETA_BLOCK,
            open(f'{output_dir}/TPS_LOOCV_Theta_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'wb'))
    pkl.dump(BASIS_BLOCK,
            open(f'{output_dir}/TPS_LOOCV_Basis_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'wb'))
    pkl.dump(COV_THETA_BLOCK,
            open(f'{output_dir}/TPS_LOOCV_CovTheta_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'wb'))
    
