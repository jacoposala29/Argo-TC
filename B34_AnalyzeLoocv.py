import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import scipy.io as scipy
from scipy.optimize import curve_fit
import sys
sys.path.append('./implementations/')
from implementation_tools import (
    variable_diff,
)

from TC_and_TS_define_param import grid_lower, grid_upper, grid_stride, folder2use, var2use, lamb_choice, degrees_tag

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
plot_dir = '../Figures/B34_lambda/'



# if var2use == 'MLT':
#     # lamb_choice = 'Custom_find_min'
#     lamb_choice = 'Custom_2perc'
# else:
#     lamb_choice = 'Custom_2perc'
    



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
else:  # single depth
    sub_list = (
    'all_combined',
    'all_hurricanes',
    'all_tstd',
)


# data_dir = '/Users/jacoposala/Desktop/CU/3.RESEARCH/ARGO_analysis/MLS_test_Feb9/Outputs/'

    
# Loop through all cases
for sub in sub_list:
    print(sub)
    # Load output from B10 (Argo profile data)
    df = pkl.load(open(f'{data_dir}/HurricaneVariableCovDF_Subset{sub}_{degrees_tag}.pkl', 'rb'))

    y = variable_diff(df, stage, 0)
    ymat = np.zeros((1, DEPTH_IDX, len(y)))

    for depth_idx in range(DEPTH_IDX):
        ymat[0, depth_idx, :] = variable_diff(df, stage, depth_idx)
     
    # Load results from B35 (lambda values and LOOCV scores)
    LAMB_LARGE, LOOCV_LARGE = pkl.load(open(f'{data_dir}/B35_LOOCV_Data_{sub}_{degrees_tag}.pkl', 'rb'))
    # Combine the two sets
    LAMB = LAMB_LARGE[1:]
    LOOCV = LOOCV_LARGE[1:, :, :, :] 
    n = LOOCV.shape[3]
    
    # 0 - ignored
    # 1 - predictions
    # 2 - estimated standard deviations
    # 3 - diag_H
    
    varmat = np.vstack(df['var']).T
    varmat = varmat.reshape(1, *varmat.shape)
    
    # LOOCV metric (eq 44 in paper)
    loocv_estimates = ((1/varmat) * (
        (LOOCV[:,:,1,:] - ymat) / (1-LOOCV[:,:,3,:]))**2)
    
    error_estimates = loocv_estimates.sum(axis=2)
    # Set NaN error to infinity
    error_estimates_nonan = error_estimates.copy()
    error_estimates_nonan[np.isnan(error_estimates)] = np.inf
    # Select lambda based on the Jacopo-Donata approach
    # lamb_seq = np.zeros(DEPTH_IDX)
    # lhat = np.zeros(DEPTH_IDX)
    # for depth_idx in range(DEPTH_IDX):
        # delta_percent = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/error_estimates[0:-1, depth_idx]/(LAMB[1:]-(LAMB[0:-1]))
        # d_error = np.concatenate(([delta_percent[0]], delta_percent))
        # lhat_all = np.where(d_error >= -.001)[0] 
        # lhat[depth_idx] = lhat_all[0]
        # lamb_seq[depth_idx] = LAMB[lhat_all[0]]
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
        
        
# OLD METHODS TO SELECT LAMBDA
    elif lamb_choice == 'Custom':
        print('Case to revisit if needed (see changes in Custom2')
        for depth_idx in range(DEPTH_IDX):
            delta_percent = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/ \
                error_estimates[0:-1, depth_idx]/(LAMB[1:]-(LAMB[0:-1]))
            #ciao = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/error_estimates[0:-1, depth_idx]/(LAMB[1:]-(LAMB[0:-1]))
            #ciao = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/(LAMB[1:]-(LAMB[0:-1]))
            d_error = np.concatenate(([delta_percent[0]], delta_percent))
            lhat_all = np.where(d_error >= -.001)[0] 
            lhat[depth_idx] = lhat_all[0]
            argmin_lamb = LAMB[int(lhat[depth_idx])]
            lamb_seq[depth_idx] = argmin_lamb #round(argmin_lamb)
        print(lamb_seq)
    elif lamb_choice == 'AdaptInflate': # Choice of the smallest lambda with a score = 1% more than the minimum score
        min_error = error_estimates.min(axis=0)
        inflation_factor = 1.01
        lhat_inflated_all = np.argmax(error_estimates < min_error * inflation_factor, axis=0)
        lamb_seq = LAMB[lhat_inflated_all]
        print(lamb_seq)
    elif lamb_choice == 'Custom_find_min':
        for depth_idx in range(DEPTH_IDX):
            # Scores for this depth level
            bfr = error_estimates[:,depth_idx]
            # Find index which corresponds to min of error
            lhat[depth_idx] = min(range(len(bfr)), key=bfr.__getitem__)
            # Find correspondent lambda
            argmin_lamb = LAMB[int(lhat[depth_idx])]
            lamb_seq[depth_idx] = argmin_lamb #round(argmin_lamb)
        print(lamb_seq)
    elif lamb_choice == 'Custom2_0.005':
        for depth_idx in range(DEPTH_IDX):
            Custom2_delta_percent = abs((error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/ \
                (error_estimates[0:-1, depth_idx]))
            d_error = np.concatenate(([Custom2_delta_percent[0]], Custom2_delta_percent))
            lhat_all = np.where(d_error <= .005)[0] 
            lhat[depth_idx] = lhat_all[0]
            argmin_lamb = LAMB[int(lhat[depth_idx])]
            lamb_seq[depth_idx] = argmin_lamb #round(argmin_lamb)
        print(lamb_seq)
    elif lamb_choice == '0.1_Custom':
        for depth_idx in range(DEPTH_IDX):
            delta_percent = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/ \
                error_estimates[0:-1, depth_idx]/(LAMB[1:]-(LAMB[0:-1]))
            #ciao = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/error_estimates[0:-1, depth_idx]/(LAMB[1:]-(LAMB[0:-1]))
            #ciao = (error_estimates[1:, depth_idx] - error_estimates[0:-1, depth_idx])/(LAMB[1:]-(LAMB[0:-1]))
            d_error = np.concatenate(([delta_percent[0]], delta_percent))
            lhat_all = np.where(d_error >= -.001)[0] 
            lhat[depth_idx] = lhat_all[0]
            argmin_lamb = LAMB[int(lhat[depth_idx])]
            lamb_seq[depth_idx] = 0.1*argmin_lamb #round(argmin_lamb)
            
            lhat_all_factor = np.where(LAMB <= lamb_seq[depth_idx])[0]
            lhat[depth_idx] = lhat_all_factor[-1]
        print(lamb_seq)
    else:
        lamb_seq = [lamb_choice for _ in range(DEPTH_IDX)]
    # Select lambas that minimize the cost function
    #lhat = error_estimates_nonan.argmin(axis=0)
    #print(LAMB[lhat])
    
    
    if DEPTH_IDX == 1:
        depth_range_2_plot = np.arange(0,DEPTH_IDX,1)
    else:
        depth_range_2_plot = np.arange(0,DEPTH_IDX-1,1)

    #Plot 13b in paper
    plt.figure()
    ax1 = plt.subplot()
    plt.scatter(lamb_seq, np.linspace(grid_start,grid_end,DEPTH_IDX), marker='x')
    plt.plot(lamb_seq, np.linspace(grid_start,grid_end,DEPTH_IDX))
    plt.ylim(0,200)
    plt.gca().invert_yaxis()
    plt.xscale('log')
    plt.xlim(min(LAMB), max(LAMB)*1.1)
    plt.xlabel('Regularization parameter, $\lambda$')
    plt.ylabel('Pressure, dbars ($z$)')
    # plt.show()
    plt.savefig(f'{plot_dir}/lambda_plot13b_{var2use}_{sub}_{lamb_choice}_{degrees_tag}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    print(lamb_seq)
    
    # Plot 13a in paper
    plt.figure()
    ax1 = plt.subplot()
    # for depth_idx in range(DEPTH_IDX):
        # Plot error function vs lambda
        #if depth_idx < 10:
        # _ = plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}')
        #else:
        #    _ = plt.plot(LAMB[1:-1], (error_estimates[1:-1, depth_idx] - error_estimates[0:-2, depth_idx])/(LAMB[1:-1]-(LAMB[0:-2])), label=f'{(depth_idx+1)*10}')#LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}', linestyle='dotted')
        # Highlist selected lambda
        # _ = plt.scatter(LAMB[int(lhat[depth_idx])], error_estimates[int(lhat[depth_idx]), depth_idx],
                # marker='x')
        # add plot 
    
    # for depth_idx in np.arange(0,DEPTH_IDX-1,1):
 
    for depth_idx in depth_range_2_plot:
        plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}', zorder = -1) 
        ax1.set_xlim(0.01,1000)
        # ax1.set_ylim(6000,25000)
        
    plt.xscale('log')
    plt.yscale('log')
    for depth_idx in depth_range_2_plot:
        plt.scatter(LAMB[int(lhat[depth_idx])], error_estimates[int(lhat[depth_idx]), depth_idx],  marker='x') #zorder = 0,    
    plt.xlabel('Regularization parameter, $\lambda$')
    plt.ylabel('LOOCV Error')
    plt.savefig(f'{plot_dir}/lambda_plot13a_{var2use}_{sub}_{lamb_choice}_logscale_{degrees_tag}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    
    
    # Plot 13a in paper - without log-scale on x-axis
    plt.figure()
    ax1 = plt.subplot()
    # for depth_idx in range(DEPTH_IDX):
        # Plot error function vs lambda
        #if depth_idx < 10:
        # _ = plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}')
        #else:
        #    _ = plt.plot(LAMB[1:-1], (error_estimates[1:-1, depth_idx] - error_estimates[0:-2, depth_idx])/(LAMB[1:-1]-(LAMB[0:-2])), label=f'{(depth_idx+1)*10}')#LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}', linestyle='dotted')
        # Highlist selected lambda
        # _ = plt.scatter(LAMB[int(lhat[depth_idx])], error_estimates[int(lhat[depth_idx]), depth_idx],
                # marker='x')
        # add plot 
    for depth_idx in depth_range_2_plot:
        plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}') #zorder = -1, 
        # ax1.set_xlim(0,100)
    # plt.xscale('log')
    plt.yscale('log')
    for depth_idx in depth_range_2_plot:
        plt.scatter(LAMB[int(lhat[depth_idx])], error_estimates[int(lhat[depth_idx]), depth_idx],  marker='x') #zorder = 0,    
    plt.xlabel('Regularization parameter, $\lambda$')
    plt.ylabel('LOOCV Error')
    plt.savefig(f'{plot_dir}/lambda_plot13a_{var2use}_{sub}_{lamb_choice}_nologscale_xaxis_{degrees_tag}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)

    # Plot 13a in paper - without log-scale on y-axis
    plt.figure()
    ax1 = plt.subplot()
    # for depth_idx in range(DEPTH_IDX):
        # Plot error function vs lambda
        #if depth_idx < 10:
        # _ = plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}')
        #else:
        #    _ = plt.plot(LAMB[1:-1], (error_estimates[1:-1, depth_idx] - error_estimates[0:-2, depth_idx])/(LAMB[1:-1]-(LAMB[0:-2])), label=f'{(depth_idx+1)*10}')#LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}', linestyle='dotted')
        # Highlist selected lambda
        # _ = plt.scatter(LAMB[int(lhat[depth_idx])], error_estimates[int(lhat[depth_idx]), depth_idx],
                # marker='x')
        # add plot 
    for depth_idx in depth_range_2_plot:
        plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}') #zorder = -1, 
        # ax1.set_xlim(0,100)
    plt.xscale('log')
    # plt.yscale('log')
    for depth_idx in depth_range_2_plot:
        plt.scatter(LAMB[int(lhat[depth_idx])], error_estimates[int(lhat[depth_idx]), depth_idx],  marker='x') #zorder = 0,    
    plt.xlabel('Regularization parameter, $\lambda$')
    plt.ylabel('LOOCV Error')
    plt.savefig(f'{plot_dir}/lambda_plot13a_{var2use}_{sub}_{lamb_choice}_nologscale_yaxis_{degrees_tag}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
   
    # Plot 13a in paper - without log-scale on both axes
    plt.figure()
    ax1 = plt.subplot()
    # for depth_idx in range(DEPTH_IDX):
        # Plot error function vs lambda
        #if depth_idx < 10:
        # _ = plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}')
        #else:
        #    _ = plt.plot(LAMB[1:-1], (error_estimates[1:-1, depth_idx] - error_estimates[0:-2, depth_idx])/(LAMB[1:-1]-(LAMB[0:-2])), label=f'{(depth_idx+1)*10}')#LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}', linestyle='dotted')
        # Highlist selected lambda
        # _ = plt.scatter(LAMB[int(lhat[depth_idx])], error_estimates[int(lhat[depth_idx]), depth_idx],
                # marker='x')
        # add plot 
    for depth_idx in depth_range_2_plot:
        plt.plot(LAMB, error_estimates[:, depth_idx], zorder = -1, label=f'{(depth_idx+1)*10}')
        # ax1.set_xlim(0,10)        
        
    # plt.xscale('log')
    # plt.yscale('log')
    for depth_idx in depth_range_2_plot:
        plt.scatter(LAMB[int(lhat[depth_idx])], error_estimates[int(lhat[depth_idx]), depth_idx], zorder = 0, marker='x')
    plt.xlabel('Regularization parameter, $\lambda$')
    plt.ylabel('LOOCV Error')
    plt.savefig(f'{plot_dir}/lambda_plot13a_{var2use}_{sub}_{lamb_choice}_nologscale_{degrees_tag}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)

    
    #plt.legend()
    
    # depth_idx=2
    # _ = plt.plot(LAMB, error_estimates[:, depth_idx], label=f'{(depth_idx+1)*10}')
    # _ = plt.scatter(LAMB[int(lhat[depth_idx])], error_estimates[int(lhat[depth_idx]), depth_idx],
    #         marker='x')
    
    # plt.xscale('log')
    # #plt.legend()

    # error_mean = LOOCV.mean(axis=2)
    # error_std = LOOCV.std(axis=2) / np.sqrt(n)
    # #plt.errorbar(LAMB, error_mean[:, depth_idx], yerr=error_std[:, depth_idx])
    # #plt.errorbar(LAMB[:10], error_mean[:10, depth_idx], yerr=error_std[:10, depth_idx])
