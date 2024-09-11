import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys
sys.path.append('./implementations/')

from TC_and_TS_define_param import (grid_lower, grid_upper, grid_stride, var2use, folder2use, 
                                    depth_layers, clim, unit, bounds_x1, bounds_x2, bounds_x3, shape)

input_dir = folder2use + '/Outputs'
plot_dir = '../Figures/B25/'
window_size_gp = 5
prefix = '_LOOCV'
lamb_choice = 'Custom'
        
# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

fs = 24
df = 2
plt.rcParams['font.family'] = 'Liberation Serif'
#plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'
plt.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams.update({'font.size': fs})

# Define various cases
if DEPTH_IDX > 1 or var2use.startswith("ML"):
    sub_list = (
        # 'all_combined',
        'all_hurricanes',
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


from implementation_tools import (
    grid,
)



# Loop through all cases
for sub in sub_list:
    print(sub)
    BASIS = pkl.load(open(f'{input_dir}/TPS{prefix}_Basis_Block_{lamb_choice}_{sub}.pkl',
        'rb'))
    THETA = pkl.load(open(f'{input_dir}/TPS{prefix}_Theta_Block_{lamb_choice}_{sub}.pkl',
        'rb'))
    COV = pkl.load(open(f'{input_dir}/TPS{prefix}_CovTheta_Block_{lamb_choice}_{sub}.pkl',
        'rb'))
    
    def depthxc_estimates(d1, d2,
            BASIS=BASIS, THETA=THETA, COV=COV):
        n_params = THETA.shape[0]
    
        test_X = grid(bounds_x1, bounds_x2, *shape)
    
        xc_filt = ((test_X[:, 1] >= d1)
                  *(test_X[:, 1] < d2))
    
        D = np.sort(np.unique(test_X[:, 0]))
        n_d = len(D)
    
        predmat = np.zeros((DEPTH_IDX, n_d))
        maskmat = np.zeros((DEPTH_IDX, n_d))
    
        for tidx, d in enumerate(D):
            idxs = xc_filt * (test_X[:, 0] == d)
            x_pts = test_X[idxs, 1]  # 1 x 6
            y_pts = BASIS[idxs, :]   # 6 x 394
            # trap integration for each column
            int_basis = np.zeros(n_params)
            var = np.zeros(DEPTH_IDX)
            for pidx in range(n_params):
                int_basis[pidx] = np.trapz(y_pts[:, pidx], x_pts)
            for depth_idx in range(DEPTH_IDX):
                var[depth_idx] = np.linalg.multi_dot((int_basis,
                                    COV[:, :, depth_idx],
                                    int_basis))
            inprod = np.dot(int_basis, THETA)
            predmat[:, tidx] = inprod / (max(x_pts) - min(x_pts))
            sd = np.sqrt(var)
            maskmat[:, tidx] = (inprod > 2 * sd) | (inprod < -2 * sd)
        predmat_small = predmat
        maskmat_small = maskmat
        predmat = np.zeros((DEPTH_IDX, n_d*4))
        maskmat = np.zeros((DEPTH_IDX, n_d*4))
        xnew = np.linspace(-8, +8, n_d*4)
        for depth_idx in range(DEPTH_IDX):
            predmat[depth_idx, :] = np.interp(xnew, D, predmat_small[depth_idx, :])
            maskmat[depth_idx, :] = np.repeat(maskmat_small[depth_idx, :], 4)
        mat = predmat.copy()
        mat[~maskmat.astype(bool)] = np.nan
        return mat
    
    
    # Create mat1 mat2 files (or read them if already available)
    try:
        mat1, mat2 = pkl.load(open(f'{input_dir}/tmp_depthxc_{lamb_choice}_{sub}.pkl', 'rb'))
    except FileNotFoundError:
        mat1 = depthxc_estimates(0, 3)
        mat2 = depthxc_estimates(3, 20)
        pkl.dump((mat1, mat2), open(f'{input_dir}/tmp_depthxc_{lamb_choice}_{sub}.pkl', 'wb'))
        
    minima, maxima = -clim, +clim
    
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
    cmap = cm.bwr
    cmap.set_bad(color='gray')
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    fig = plt.figure(figsize=(20, 6))
    gs= gridspec.GridSpec(1, 2, figure=fig,
        width_ratios=[1.0, 1.0006])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    ax1.imshow(mat1,
            origin='lower',
            cmap=cmap, norm=norm,
            extent=(*bounds_x1, *bounds_x3),
            aspect='auto',
            )
    ax1.invert_yaxis()
    ax1.axvline(0, color='k', linewidth=0.5)
    ax1.set_xticks([-5, 0, 5])
    ax1.set_yticks([10, 50, 100, 150, 200])
    ax1.set_ylabel(r'Pressure, dbars ($z$)',
            fontsize=fs)
    '''
    ax1.set_xlabel(r'Cross-track angle, degrees ($d$)',
            fontsize=fs)
    '''
    ypos = -0.35
    ax1.set_title(r'(a) $\tau \in [0, +3)$ days',
            y=ypos,
            fontsize=fs+df)
    
    
    ax2.imshow(mat2,
            origin='lower',
            cmap=cmap, norm=norm,
            extent=(*bounds_x1, *bounds_x3),
            aspect='auto',
            )
    ax2.invert_yaxis()
    ax2.axvline(0, color='k', linewidth=0.5)
    ax2.set_xticks([-5, 0, 5])
    ax2.set_yticks([])
    #ax2.xaxis.set_ticks_position('top') 
    '''
    ax2.set_xlabel(r'Cross-track angle, degrees ($d$)',
            fontsize=fs)
    '''
    ax2.set_title(r'(b) $\tau \in [+3, +20)$ days',
            y=ypos,
            fontsize=fs+df)
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mapper, ax=ax2, cax=cax)
    cbar.set_label(f'{var2use} difference ({unit})',
            fontsize=fs)
    
    fig.text(0.5, -0.03, r'Cross-track angle, degrees ($d$)',
            ha='center',
            fontsize=fs)
    
    plt.savefig(f'{plot_dir}/TPS_ThreePanel_DepthXC_{var2use}_{DEPTH_IDX}_{lamb_choice}_{sub}.pdf',
            bbox_inches='tight', pad_inches=0, dpi=300)
    plt.savefig(f'{plot_dir}/TPS_ThreePanel_DepthXC_{var2use}_{DEPTH_IDX}_{lamb_choice}_{sub}.jpg',
            bbox_inches='tight', pad_inches=0, dpi=300)

