#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:07:06 2021

@author: jacoposala
"""

# 2 Jan 2023: added case for central panel only, capped at 150 db

# CANCEL ALL THE FILES PRODUCED BY THIS SCRIPT BEFORE RE-RUNNING IT

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('./implementations/')
from implementation_tools import grid

from TC_and_TS_define_param import (grid_lower, grid_upper, grid_stride, var2use, folder2use, 
                                    depth_layers, clim, unit, k_factor, bounds_x1, bounds_x2, bounds_x3, shape,
                                    lamb_choice, degrees_tag)

input_dir = folder2use + '/Outputs'
plot_dir = '../Figures/B24/'

window_size_gp = 5
prefix = '_LOOCV'

# if var2use == 'MLT':
#     lamb_choice = 'Custom_find_min'
# else:
#     lamb_choice = 'Custom3'

# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

fs = 21.5
df = 2
plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['font.family'] = 'Liberation Serif'
# plt.rcParams['mathtext.rm'] = 'serif'
# plt.rcParams['mathtext.it'] = 'serif:italic'
# plt.rcParams['mathtext.bf'] = 'serif:bold'
# plt.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams.update({'font.size': fs})
ypos = -0.4

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
    sub_list_comparison = (
    # '',
    # '',
    # '',
    # '',
    # '',
    '',
    'increasing_hurricanes',
    # '',
    # '',
)
else: # single level
    sub_list = (
    'all_combined',
    'all_hurricanes',
    'all_tstd',
)


# Loop through all cases
for sub, sub_comp in zip(sub_list, sub_list_comparison):
    print(sub)
    BASIS = pkl.load(open(f'{input_dir}/TPS{prefix}_Basis_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl',
        'rb'))
    THETA = pkl.load(open(f'{input_dir}/TPS{prefix}_Theta_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl',
        'rb'))
    COV = pkl.load(open(f'{input_dir}/TPS{prefix}_CovTheta_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl',
        'rb'))

    if sub_comp:
        BASIS_comp = pkl.load(open(f'{input_dir}/TPS{prefix}_Basis_Block_{lamb_choice}_{sub_comp}_{degrees_tag}.pkl',
        'rb'))
        THETA_comp = pkl.load(open(f'{input_dir}/TPS{prefix}_Theta_Block_{lamb_choice}_{sub_comp}_{degrees_tag}.pkl',
        'rb'))
        COV_comp = pkl.load(open(f'{input_dir}/TPS{prefix}_CovTheta_Block_{lamb_choice}_{sub_comp}_{degrees_tag}.pkl',
        'rb'))
        
    # Define function to create mat1 mat2 mat3
    def depthtime_estimates(center, ws=0.5,
            BASIS=BASIS, THETA=THETA, COV=COV):
        ws = 0.5
        n_params = THETA.shape[0]
    
        test_X = grid(bounds_x1, bounds_x2, *shape)
    
        xc_filt = ((test_X[:, 0] > center - ws)
                  *(test_X[:, 0] < center + ws))
    
        TAU = np.sort(np.unique(test_X[:, 1]))
        n_tau = len(TAU)
    
        predmat = np.zeros((DEPTH_IDX, n_tau))
        maskmat = np.zeros((DEPTH_IDX, n_tau))  
        sd_mat = np.zeros((DEPTH_IDX, n_tau)) 
    
        for tidx, tau in enumerate(TAU):
            idxs = xc_filt * (test_X[:, 1] == tau)
            x_pts = test_X[idxs, 0]  # 1 x 6
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
            sd_mat[:, tidx] = np.sqrt(var)
            maskmat[:, tidx] = (inprod > 2 * sd) | (inprod < -2 * sd)
        mat = predmat.copy()
        mat[~maskmat.astype(bool)] = np.nan
        return mat, predmat, sd_mat
    
    # Create mat1 mat2 mat3 files (or read them if already available)
    try:
        mat1, mat2, mat3 = pkl.load(open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{sub}_{degrees_tag}_mat.pkl', 'rb'))
        predmat1, predmat2, predmat3 = pkl.load(open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{sub}_{degrees_tag}_predmat.pkl', 'rb'))
        sd_mat1, sd_mat2, sd_mat3 = pkl.load(open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{sub}_{degrees_tag}_sdmat.pkl', 'rb'))
    except FileNotFoundError:
        mat1, predmat1, sd_mat1 = depthtime_estimates(-2)
        mat2, predmat2, sd_mat2 = depthtime_estimates(0)
        mat3, predmat3, sd_mat3 = depthtime_estimates(2)
        pkl.dump((mat1, mat2, mat3), open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{sub}_{degrees_tag}_mat.pkl', 'wb'))
        pkl.dump((predmat1, predmat2, predmat3), open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{sub}_{degrees_tag}_predmat.pkl', 'wb'))
        pkl.dump((sd_mat1, sd_mat2, sd_mat3), open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{sub}_{degrees_tag}_sdmat.pkl', 'wb'))


    if sub_comp:
        try:
            mat1_comp, mat2_comp, mat3_comp = pkl.load(open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{degrees_tag}_mat_comp.pkl', 'rb'))
            predmat1_comp, predmat2_comp, predmat3_comp = pkl.load(open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{sub}_{degrees_tag}_predmat_comp.pkl', 'rb'))
            sd_mat1_comp, sd_mat2_comp, sd_mat3_comp = pkl.load(open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{sub}_{degrees_tag}_sdmat_comp.pkl', 'rb'))
        except FileNotFoundError:
            mat1_comp, predmat1_comp, sd_mat1_comp = depthtime_estimates(-2, BASIS=BASIS_comp, THETA=THETA_comp, COV=COV_comp)
            mat2_comp, predmat2_comp, sd_mat2_comp = depthtime_estimates(0, BASIS=BASIS_comp, THETA=THETA_comp, COV=COV_comp)
            mat3_comp, predmat3_comp, sd_mat3_comp = depthtime_estimates(2, BASIS=BASIS_comp, THETA=THETA_comp, COV=COV_comp)
            pkl.dump((mat1_comp, mat2_comp, mat3_comp), open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{degrees_tag}_mat_comp.pkl', 'wb'))
            pkl.dump((predmat1_comp, predmat2_comp, predmat3_comp), open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{degrees_tag}_predmat_comp.pkl', 'wb'))
            pkl.dump((sd_mat1_comp, sd_mat2_comp, sd_mat3_comp), open(f'{input_dir}/tmp_depthtime_{lamb_choice}_{degrees_tag}_sdmat_comp.pkl', 'wb'))

        predmat1_diff = - predmat1 + predmat1_comp
        predmat1_diff_std = np.sqrt(np.square(sd_mat1_comp) + np.square(sd_mat1))
        ge = predmat1_diff - k_factor*predmat1_diff_std > 0
        le = predmat1_diff + k_factor*predmat1_diff_std < 0
        predmat1_diff_mask = ge | le
        predmat1_diff_masked = predmat1_diff.copy()
        predmat1_diff_masked[~predmat1_diff_mask.astype(bool)] = np.nan

        
        predmat2_diff = - predmat2 + predmat2_comp
        predmat2_diff_std = np.sqrt(np.square(sd_mat2_comp) + np.square(sd_mat2))
        ge = predmat2_diff - k_factor*predmat2_diff_std > 0
        le = predmat2_diff + k_factor*predmat2_diff_std < 0
        predmat2_diff_mask = ge | le
        predmat2_diff_masked = predmat2_diff.copy()
        predmat2_diff_masked[~predmat2_diff_mask.astype(bool)] = np.nan

        
        predmat3_diff = - predmat3 + predmat3_comp
        predmat3_diff_std = np.sqrt(np.square(sd_mat3_comp) + np.square(sd_mat3))
        ge = predmat3_diff - k_factor*predmat3_diff_std > 0
        le = predmat3_diff + k_factor*predmat3_diff_std < 0
        predmat3_diff_mask = ge | le
        predmat3_diff_masked = predmat3_diff.copy()
        predmat3_diff_masked[~predmat3_diff_mask.astype(bool)] = np.nan

    


    # minima, maxima = -clim, +clim
    # norm = mcolors.Normalize(vmin=minima, vmax=maxima, clip=False)
    # cmap = plt.get_cmap('bwr', 29)  # Choose an appropriate number of discrete colors
    # cmap.set_bad(color='grey')
    # mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # fig = plt.figure(figsize=(12, 6))
    # ax1 = plt.subplot(1, 1, 1)
    
    # # First panel (around -2 deg)
    # img1 = ax1.imshow(mat2,
    #                   origin='lower',
    #                   cmap=cmap, norm=norm,
    #                   extent=(*bounds_x2, *bounds_x3),
    #                   aspect='auto',
    #                   )
    # ax1.axvline(0, color='k', linewidth=0.5)
    # ax1.set_xticks([-2, 0, 2, 4, 6, 8, 10, 12, 14])
    # ax1.set_yticks([10, 50, 100, 150])
    # ax1.set_ylabel(r'Pressure, dbars ($z$)', fontsize=fs)
    # ax1.set_xlabel(r'Time difference, days ($\tau$)', fontsize=fs)
    # ax1.set_xlim(-2, 14)
    # ax1.set_ylim(10, 150)
    # ax1.invert_yaxis()
    
    # divider = make_axes_locatable(ax1)
    # cax1 = divider.append_axes("right", size="3%", pad=.3)
    # cbar1 = plt.colorbar(mapper, cax=cax1)
    # cbar1.ax.tick_params(labelsize=20)
    # cbar1.set_label(f'{var2use} difference ({unit})', fontsize=fs)
    
    # # Save the first plot
    # plt.figure(fig.number)
    # plt.tight_layout()
    # fig.savefig(f'{plot_dir}/TPS_ThreePanel_DepthTime_{var2use}_{DEPTH_IDX}_{lamb_choice}_{sub}_{degrees_tag}_CentralPanel.jpg',
    #             bbox_inches='tight', pad_inches=0, dpi=1000)
    # # plt.show()
    
    # if sub_comp:
    #     fig = plt.figure(figsize=(12, 6))
    #     ax1 = plt.subplot(1, 1, 1)
    
    #     # Second panel (around -2 deg)
    #     img2 = ax1.imshow(predmat2_diff_masked,
    #                       origin='lower',
    #                       cmap=cmap, norm=norm,
    #                       extent=(*bounds_x2, *bounds_x3),
    #                       aspect='auto',
    #                       )
    #     ax1.axvline(0, color='k', linewidth=0.5)
    #     ax1.set_xticks([-2, 0, 2, 4, 6, 8, 10, 12, 14])
    #     ax1.set_yticks([10, 50, 100, 150])
    #     ax1.set_ylabel(r'Pressure, dbars ($z$)', fontsize=fs)
    #     ax1.set_xlabel(r'Time difference, days ($\tau$)', fontsize=fs)
    #     ax1.set_xlim(-2, 14)
    #     ax1.set_ylim(10, 150)
    #     ax1.invert_yaxis()
    
    #     divider = make_axes_locatable(ax1)
    #     cax2 = divider.append_axes("right", size="3%", pad=.3)
    #     cbar2 = plt.colorbar(mapper, cax=cax2)
    #     cbar2.ax.tick_params(labelsize=20)
    #     cbar2.set_label(f'{var2use} difference ({unit})', fontsize=fs)
    
    #     # Save the second plot
    #     plt.figure(fig.number)
    #     plt.tight_layout()
    #     fig.savefig(f'{plot_dir}/TPS_ThreePanel_DepthTime_{var2use}_{DEPTH_IDX}_{sub}_{lamb_choice}_diff_with_{sub_comp}_{degrees_tag}_CentralPanel.jpg',
    #                 bbox_inches='tight', pad_inches=0, dpi=1000)

    
    

    #     # plt.show()



    #     ciao



    # ____________________
    # Make three-panel plot
    # ____________________
    minima, maxima = -clim, +clim
    
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
    cmap = cm.bwr
    cmap = plt.get_cmap('bwr', 21)  # Choose an appropriate number of discrete colors
    cmap.set_bad(color='grey')
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # fig = plt.figure(figsize=(20, 6))
    # gs= gridspec.GridSpec(1, 3, figure=fig,
    #     width_ratios=[1.0, 1.0, 1.0666])
    # ax1 = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1])
    # ax3 = plt.subplot(gs[2])
    
    # # First panel (around -2 deg)
    # ax1.imshow(mat1,
    #         origin='lower',
    #         cmap=cmap, norm=norm,
    #         extent=(*bounds_x2, *bounds_x3),
    #         aspect='auto',
    #         )
    # ax1.invert_yaxis()
    # ax1.axvline(0, color='k', linewidth=0.5)
    # ax1.set_xticks([0, 5, 10, 15, 20])
    # ax1.set_yticks([10, 50, 100, 150, 200])
    # ax1.set_ylabel(r'Pressure, dbars ($z$)',
    #         fontsize=fs)
    # ax1.set_title(r'(a) $d \in [-2.5, -1.5]$',
    #         y=ypos,
    #         fontsize=fs+df)
    
    # # Second panel (around 0 deg)
    # ax2.imshow(mat2,
    #         origin='lower',
    #         cmap=cmap, norm=norm,
    #         extent=(*bounds_x2, *bounds_x3),
    #         aspect='auto',
    #         )
    # ax2.invert_yaxis()
    # ax2.axvline(0, color='k', linewidth=0.5)
    # ax2.set_xticks([0, 5, 10, 15, 20])
    # ax2.set_yticks([])
    # #ax2.xaxis.set_ticks_position('top') 
    # #ax2.set_xlabel(r'Time difference, days ($\tau$)',
    #         #fontsize=fs)
    # ax2.set_title(r'(b) $d \in [-0.5, +0.5]$',
    #         y=ypos,
    #         fontsize=fs+df)
    
    # # Third panel (around +2 deg)
    # ax3.imshow(mat3,
    #         origin='lower',
    #         cmap=cmap, norm=norm,
    #         extent=(*bounds_x2, *bounds_x3),
    #         aspect='auto',
    #         )
    # ax3.invert_yaxis()
    # ax3.axvline(0, color='k', linewidth=0.5)
    # ax3.set_xticks([0, 5, 10, 15, 20])
    # #ax3.xaxis.set_ticks_position('top') 
    # ax3.set_title(r'(c) $d \in [+1.5, +2.5]$',
    #         y=ypos,
    #         fontsize=fs+df)
    
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = plt.colorbar(mapper, ax=ax3, cax=cax)
    # cbar.set_label(f'{var2use} difference ({unit})',
    #         fontsize=fs)
    # ax3.set_yticks([])
    
    # fig.text(0.5, -0.03, r'Time difference, days ($\tau$)',
    #         ha='center',
    #         fontsize=fs)
    
    # plt.savefig(f'{plot_dir}/TPS_ThreePanel_DepthTime_{var2use}_{DEPTH_IDX}_{lamb_choice}_{sub}_{degrees_tag}.jpg',
    #         bbox_inches='tight', pad_inches=0, dpi=300)


    # ____________________
    # Central panel only plot
    # ____________________
    # var2use = 'Pot. density'
    fig = plt.figure(figsize=(12,6))
    # fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot(1,1,1)

    # First panel (around -2 deg)
    ax1.imshow(mat2,
            origin='lower',
            cmap=cmap, norm=norm,
            extent=(*bounds_x2, *bounds_x3),
            aspect='auto',
            )
    ax1.axvline(0, color='k', linewidth=0.5)
    # ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.set_xticks([-2, 0, 2, 4, 6, 8, 10, 12, 14])
    ax1.set_yticks([10, 50, 100, 150])
    ax1.set_ylabel(r'Pressure (dbar)',
            fontsize=fs)
    ax1.set_xlabel(r'Time since the event (days)',
            fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    # Change xlim from -2 days to 14 days
    ax1.set_xlim(-2, 14)

    ax1.set_ylim(10,150) # y limit
    ax1.invert_yaxis()
    
    divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mapper, ax=ax1)#, cax=cax)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(f'{var2use} change ({unit})',
            fontsize=fs)
    
    plt.savefig(f'{plot_dir}/TPS_ThreePanel_DepthTime_{var2use}_{DEPTH_IDX}_{lamb_choice}_{sub}_{degrees_tag}_CentralPanel.jpg',
            bbox_inches='tight', pad_inches=0, dpi=1000)    
    
    
    
    if sub_comp:   
    #     # Make three-panel plot with differences - decreasing minus increasing        
    #     minima, maxima = -clim, +clim
        
    #     norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
    #     cmap = cm.bwr
    #     cmap.set_bad(color='gray')
    #     mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        
    #     fig = plt.figure(figsize=(20, 6))
    #     gs= gridspec.GridSpec(1, 3, figure=fig,
    #         width_ratios=[1.0, 1.0, 1.0666])
    #     ax1 = plt.subplot(gs[0])
    #     ax2 = plt.subplot(gs[1])
    #     ax3 = plt.subplot(gs[2])
        
    #     # First panel (around -2 deg)
    #     ax1.imshow(predmat1_diff_masked,
    #             origin='lower',
    #             cmap=cmap, norm=norm,
    #             extent=(*bounds_x2, *bounds_x3),
    #             aspect='auto',
    #             )
    #     ax1.invert_yaxis()
    #     ax1.axvline(0, color='k', linewidth=0.5)
    #     ax1.set_xticks([0, 5, 10, 15, 20])
    #     #ax1.set_yticks([10, 50, 100, 150, 200])
    #     ax1.set_ylabel(r'Pressure, dbars ($z$)',
    #             fontsize=fs)
    #     ax1.set_title(r'(a) $d \in [-2.5, -1.5]$',
    #             y=ypos,
    #             fontsize=fs+df)
        
    #     # Second panel (around 0 deg)
    #     ax2.imshow(predmat2_diff_masked,
    #             origin='lower',
    #             cmap=cmap, norm=norm,
    #             extent=(*bounds_x2, *bounds_x3),
    #             aspect='auto',
    #             )
    #     ax2.invert_yaxis()
    #     ax2.axvline(0, color='k', linewidth=0.5)
    #     ax2.set_xticks([0, 5, 10, 15, 20])
    #     ax2.set_yticks([])
    #     #ax2.xaxis.set_ticks_position('top') 
    #     #ax2.set_xlabel(r'Time difference, days ($\tau$)',
    #             #fontsize=fs)
    #     ax2.set_title(r'(b) $d \in [-0.5, +0.5]$',
    #             y=ypos,
    #             fontsize=fs+df)
        
    #     # Third panel (around +2 deg)
    #     ax3.imshow(predmat3_diff_masked,
    #             origin='lower',
    #             cmap=cmap, norm=norm,
    #             extent=(*bounds_x2, *bounds_x3),
    #             aspect='auto',
    #             )
    #     ax3.invert_yaxis()
    #     ax3.axvline(0, color='k', linewidth=0.5)
    #     ax3.set_xticks([0, 5, 10, 15, 20])
    #     #ax3.xaxis.set_ticks_position('top') 
    #     ax3.set_title(r'(c) $d \in [+1.5, +2.5]$',
    #             y=ypos,
    #             fontsize=fs+df)
        
    #     divider = make_axes_locatable(ax3)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     cbar = plt.colorbar(mapper, ax=ax3, cax=cax)
    #     cbar.set_label(f'{var2use} difference ({unit})',
    #             fontsize=fs)
    #     ax3.set_yticks([])
        
    #     fig.text(0.5, -0.03, r'Time difference, days ($\tau$)',
    #             ha='center',
    #             fontsize=fs)
        
    #     plt.savefig(f'{plot_dir}/TPS_ThreePanel_DepthTime_{var2use}_{DEPTH_IDX}_{sub}_{lamb_choice}_diff_with_{sub_comp}_{degrees_tag}.jpg',
    #             bbox_inches='tight', pad_inches=0, dpi=300)


        # ____________________
        # Central panel only plot
        # ____________________
        
        fig = plt.figure(figsize=(12, 6))
        # fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(1,1,1)
        
        # First panel (around -2 deg)
        ax1.imshow(predmat2_diff_masked,
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x2, *bounds_x3),
                aspect='auto',
                )
        ax1.axvline(0, color='k', linewidth=0.5)
        # ax1.set_xticks([0, 5, 10, 15, 20])
        ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax1.set_yticks([10, 50, 100, 150])
        ax1.set_ylabel(r'Pressure (dbar)',
                fontsize=fs)
        ax1.set_xlabel(r'Time since the event (days)',
                fontsize=fs)
        # Change xlim from -2 days to 14 days
        ax1.set_xlim(-2, 14)
        
        ax1.set_ylim(10,150) # y limit
        ax1.invert_yaxis()
        ax1.tick_params(axis='both', which='major', labelsize=18)

        
        divider = make_axes_locatable(ax1)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mapper, ax=ax1)#, cax=cax)
        cbar.set_label(f'{var2use} change ({unit})',
                fontsize=fs)
        cbar.ax.tick_params(labelsize=18)
        
        plt.savefig(f'{plot_dir}/TPS_ThreePanel_DepthTime_{var2use}_{DEPTH_IDX}_{sub}_{lamb_choice}_diff_with_{sub_comp}_{degrees_tag}_CentralPanel.jpg',
                bbox_inches='tight', pad_inches=0, dpi=1000) 






