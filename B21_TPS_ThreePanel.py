# This script is the same as B21, but also plots differences between cases


import argparse
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

from implementation_tools import (
    grid,
    variable_diff,
)
from TC_and_TS_define_param import (grid_lower, grid_upper, grid_stride, folder2use, 
                                    depth_layers, var2use, clim, unit, k_factor, clim_mf, 
                                    bounds_x1, bounds_x2, shape)

#from plot_config import plot_config

parser = argparse.ArgumentParser(
        description='Plot kernel smoothed estimates (three panel)')
parser.add_argument('--integrated', dest='mode', action='store_const',
                    const='integrated', default='gridded',
                    help='process assuming integrated heat content (default: gridded)')
args = parser.parse_args()

input_dir = folder2use + '/Outputs'
plot_dir = '../Figures/B21/'
window_size_gp = 5
stage = 'adj'
DEPTH_IDX = depth_layers
STAGE = ('adj', 'raw', 'mf')
h = 0.2 #bandwidth

fs = 16
df = 2
plt.rcParams['font.family'] = 'Liberation Serif'
#plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'
plt.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams.update({'font.size': fs})

# Set dimensions of the grid
grid_start = int(grid_lower)
grid_end = int(grid_upper)
grid_stride = int(grid_stride)
DEPTH_IDX = int((grid_end - grid_start)/ grid_stride+1)

ypos = -0.3

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
    sub_list_comparison = (
        # '',
        '',
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
    prefix = '_LOOCV'
    
    # NEED TO CHANGE IN FUTURE
    # lamb_choice = 'AdaptInflate'
    lamb_choice = 'Custom'

    # Adjusted case
    norew = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_NoRew_{lamb_choice}_{sub}.pkl', 'rb'))
    block = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_Block_{lamb_choice}_{sub}.pkl', 'rb'))
    bmask = pkl.load(open(f'{input_dir}/TPS{prefix}_Mask_Block_{lamb_choice}_{sub}.pkl', 'rb'))
    bstd = pkl.load(open(f'{input_dir}/TPS{prefix}_Stdev_Block_{lamb_choice}_{sub}.pkl', 'rb'))

    # Raw case
    norew_raw = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_NoRew_{lamb_choice}_{sub}_raw.pkl', 'rb'))
    block_raw = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_Block_{lamb_choice}_{sub}_raw.pkl', 'rb'))
    bmask_raw = pkl.load(open(f'{input_dir}/TPS{prefix}_Mask_Block_{lamb_choice}_{sub}_raw.pkl', 'rb'))
    bstd_raw = pkl.load(open(f'{input_dir}/TPS{prefix}_Stdev_Block_{lamb_choice}_{sub}_raw.pkl', 'rb'))

    n, k = norew.shape
    
    idx = 0
    
    for idx in range(k):
        # Adjusted
        preds_norew = norew[:, idx]
        preds_block = block[:, idx]
        preds_std = bstd[:, idx]
        preds_mask = preds_block.copy()
        preds_mask[~bmask[:, idx].astype(bool)] = np.nan
        # Raw
        preds_norew_raw = norew_raw[:, idx]
        preds_block_raw = block_raw[:, idx]
        preds_std_raw = bstd_raw[:, idx]
        preds_mask_raw = preds_block_raw.copy()
        preds_mask_raw[~bmask_raw[:, idx].astype(bool)] = np.nan
        # Mean field
        preds_norew_mf = - preds_norew_raw + preds_norew
        preds_block_mf = - preds_block_raw + preds_block
        
        # preds_norew_mf = preds_norew_raw - preds_norew
        # preds_block_mf = preds_block_raw - preds_block
        preds_mask_mf = preds_block_mf.copy()
        # preds_mask_mf[~bmask[:, idx].astype(bool)] = np.nan
        # preds_mask_mf[~bmask_raw[:, idx].astype(bool)] = np.nan
        
        if sub_comp:
            block_comp = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_Block_{lamb_choice}_{sub_comp}.pkl', 'rb'))
            bstd_comp = pkl.load(open(f'{input_dir}/TPS{prefix}_Stdev_Block_{lamb_choice}_{sub_comp}.pkl', 'rb'))
            # IF NEEDED FOR RAW CASE
            # block_comp_raw = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_Block_{lamb_choice}_{sub_comp}_raw.pkl', 'rb'))
            # bstd_comp_raw = pkl.load(open(f'{input_dir}/TPS{prefix}_Stdev_Block_{lamb_choice}_{sub_comp}_raw.pkl', 'rb'))
            preds_block_comp = block_comp[:, idx]
            preds_std_comp = bstd_comp[:, idx]
            preds_case_comp_diff = preds_block - preds_block_comp
            
            preds_case_comp_diff_std = np.sqrt(np.square(preds_std) + np.square(preds_std_comp))
            
            ge = preds_case_comp_diff - k_factor*preds_case_comp_diff_std > 0
            le = preds_case_comp_diff + k_factor*preds_case_comp_diff_std < 0
            
            preds_case_comp_diff_mask = ge | le
            preds_case_comp_diff_masked = preds_case_comp_diff.copy()
            preds_case_comp_diff_masked[~preds_case_comp_diff_mask.astype(bool)] = np.nan
            
            
        # Plot settings        
        minima, maxima = -clim, +clim
        depth = int(grid_lower) + idx*int(grid_stride)
        
        norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
        cmap = cm.bwr
        cmap.set_bad(color='gray')
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        # THREE PANEL PLOT FOR ADJUSTED
        fig = plt.figure(figsize=(20, 8))
        gs= gridspec.GridSpec(1, 3, figure=fig,
            width_ratios=[1.0, 1.0, 1.0666])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
                
        # Left panel
        ax1.imshow(preds_norew.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax1.invert_yaxis()
        ax1.axhline(0, color='k', linewidth=0.5)
        ax1.axvline(0, color='k', linewidth=0.5)
        ax1.set_xticks([-5, 0, +5])
        ax1.set_xlim(-5,5)
        ax1.set_yticks([0, 5, 10, 15, 20])
        ax1.set_ylabel(r'Days since TC passage ($\tau$)',
                fontsize=fs)
        ax1.set_title(r'(a) No variance reweighting',
                y=ypos,
                fontsize=fs+df)
    
        # Central panel
        ax2.imshow(preds_block.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax2.invert_yaxis()
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.axvline(0, color='k', linewidth=0.5)
        ax2.set_xlim(-5,5)
        ax2.set_xticks([-5, 0, +5])
        ax2.set_yticks([])
        #ax2.xaxis.set_ticks_position('top') 
        ax2.set_xlabel(r'Cross-track angle, degrees ($d$)',
                fontsize=fs)
        ax2.set_title(r'(b) Block covariance',
                y=ypos,
                fontsize=fs+df)
    
        # Right panel
        ax3.imshow(preds_mask.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax3.invert_yaxis()
        ax3.axhline(0, color='k', linewidth=0.5)
        ax3.axvline(0, color='k', linewidth=0.5)
        ax3.set_xticks([-5, 0, +5])
        ax3.set_xlim(-5,5)
        #ax3.xaxis.set_ticks_position('top') 
        ax3.set_title(r'(c) Pointwise $\alpha=0.05$ test',
                y=ypos,
                fontsize=fs+df)
    
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mapper, ax=ax3, cax=cax)
        cbar.set_label(f'\u0394 {var2use} ({unit})',
                fontsize=fs)
        ax3.set_yticks([])
                
        plt.suptitle({sub, f'{depth:03d}'} , fontsize = 16)
        plt.subplots_adjust(hspace=0.1)

        plt.savefig(f'{plot_dir}/TPS_ThreePanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.savefig(f'{plot_dir}/TPS_ThreePanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


        # ADJUSTED, right plot only
        fig = plt.figure(figsize=(8,8))
        ax1 = plt.subplot(1,1,1)
        ax1.imshow(preds_mask.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax1.invert_yaxis()
        ax1.axhline(0, color='k', linewidth=0.5)
        ax1.axvline(0, color='k', linewidth=0.5)
        ax1.set_xlim(-5,5)
        ax1.set_xticklabels([-4, -2, 0, 2, +4], fontsize=18)
        ax1.set_yticklabels([0, 5, 10, 15, 20], fontsize=18)
        ax1.set_xticks([-4, -2, 0, 2, +4])
        ax1.set_yticks([0, 5, 10, 15, 20])
        ax1.set_ylabel(r'Days since TC passage ($\tau$)',
                 fontsize=fs)
        ax1.set_xlabel(r'Cross-track angle, degrees ($d$)',
                 fontsize=fs)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(mapper, ax=ax1, cax=cax)
        cbar.set_label(f'\u0394 {var2use} ({unit})',
                fontsize=fs)

        plt.savefig(f'{plot_dir}/TPS_RightPanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.savefig(f'{plot_dir}/TPS_RightPanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)

        # Comparison if of interest with product in sub_list_comparison
        if sub_comp:
            preds_case_comp_diff_masked
            fig = plt.figure(figsize=(8,8))
            ax1 = plt.subplot(1,1,1)
            ax1.imshow(preds_case_comp_diff_masked.reshape(*(shape[::-1])),
                    origin='lower',
                    cmap=cmap, norm=norm,
                    extent=(*bounds_x1, *bounds_x2),
                    )
            ax1.invert_yaxis()
            ax1.axhline(0, color='k', linewidth=0.5)
            ax1.axvline(0, color='k', linewidth=0.5)
            ax1.set_xlim(-5,5)
            ax1.set_xticklabels([-4, -2, 0, 2, +4], fontsize=18)
            ax1.set_yticklabels([0, 5, 10, 15, 20], fontsize=18)
            ax1.set_xticks([-4, -2, 0, 2, +4])
            ax1.set_yticks([0, 5, 10, 15, 20])
            ax1.set_ylabel(r'Days since TC passage ($\tau$)',
                     fontsize=fs)
            ax1.set_xlabel(r'Cross-track angle, degrees ($d$)',
                     fontsize=fs)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            cbar = plt.colorbar(mapper, ax=ax1, cax=cax)
            cbar.set_label(f'\u0394 {var2use} ({unit})',
                    fontsize=fs)
    
            plt.savefig(f'{plot_dir}/TPS_RightPanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}_diff_with_{sub_comp}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
            plt.savefig(f'{plot_dir}/TPS_RightPanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}_diff_with_{sub_comp}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


        # ADJUSTED, central plot only
        fig = plt.figure(figsize=(8,8))
        ax2 = plt.subplot(1,1,1)
        
        ax2.imshow(preds_block.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax2.invert_yaxis()
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.axvline(0, color='k', linewidth=0.5)
        ax2.set_xlim(-5,5)
        ax2.set_xticklabels([-4, -2, 0, 2, +4], fontsize=18)
        ax2.set_yticklabels([0, 5, 10, 15, 20], fontsize=18)
        ax2.set_xticks([-4, -2, 0, 2, +4])
        ax2.set_yticks([0, 5, 10, 15, 20])
        ax2.set_ylabel(r'Days since TC passage ($\tau$)',
                 fontsize=fs)
        ax2.set_xlabel(r'Cross-track angle, degrees ($d$)',
                 fontsize=fs)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(mapper, ax=ax2, cax=cax)
        cbar.set_label(f'\u0394 {var2use} ({unit})',
                fontsize=fs)

        plt.savefig(f'{plot_dir}/TPS_CentralPanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.savefig(f'{plot_dir}/TPS_CentralPanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
        
        
        
        
        # RAW, central plot only
        fig = plt.figure(figsize=(8,8))
        ax2 = plt.subplot(1,1,1)
        
        ax2.imshow(preds_block_raw.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap, norm=norm,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax2.invert_yaxis()
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.axvline(0, color='k', linewidth=0.5)
        ax2.set_xlim(-5,5)
        ax2.set_xticklabels([-4, -2, 0, 2, +4], fontsize=18)
        ax2.set_yticklabels([0, 5, 10, 15, 20], fontsize=18)
        ax2.set_xticks([-4, -2, 0, 2, +4])
        ax2.set_yticks([0, 5, 10, 15, 20])
        ax2.set_ylabel(r'Days since TC passage ($\tau$)',
                  fontsize=fs)
        ax2.set_xlabel(r'Cross-track angle, degrees ($d$)',
                  fontsize=fs)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(mapper, ax=ax2, cax=cax)
        cbar.set_label(f'\u0394 {var2use} ({unit})',
                fontsize=fs)

        plt.savefig(f'{plot_dir}/TPS_CentralPanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}_raw.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.savefig(f'{plot_dir}/TPS_CentralPanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}_raw.jpg', bbox_inches='tight', pad_inches=0, dpi=300)


        
        # MEAN FIELD, CENTRAL PLOT ONLY
        minima, maxima = -clim_mf, +clim_mf        
        norm_mf = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
        cmap_mf = cm.bwr
        cmap.set_bad(color='gray')
        mapper_mf = cm.ScalarMappable(norm=norm_mf, cmap=cmap_mf)
        
        fig = plt.figure(figsize=(8,8))
        ax2 = plt.subplot(1,1,1)
        
        ax2.imshow(preds_block_mf.reshape(*(shape[::-1])),
                origin='lower',
                cmap=cmap_mf, norm=norm_mf,
                extent=(*bounds_x1, *bounds_x2),
                )
        ax2.invert_yaxis()
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.axvline(0, color='k', linewidth=0.5)
        ax2.set_xlim(-5,5)
        ax2.set_xticklabels([-4, -2, 0, 2, +4], fontsize=18)
        ax2.set_yticklabels([0, 5, 10, 15, 20], fontsize=18)
        ax2.set_xticks([-4, -2, 0, 2, +4])
        ax2.set_yticks([0, 5, 10, 15, 20])
        ax2.set_ylabel(r'Days since TC passage ($\tau$)',
                  fontsize=fs)
        ax2.set_xlabel(r'Cross-track angle, degrees ($d$)',
                  fontsize=fs)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(mapper_mf, ax=ax2, cax=cax)
        cbar.set_label(f'\u0394 {var2use} ({unit})',
                fontsize=fs)

        plt.savefig(f'{plot_dir}/TPS_CentralPanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}_mf_adj_minus_raw.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.savefig(f'{plot_dir}/TPS_CentralPanel_{var2use}_{depth:03d}_{sub}_{lamb_choice}_mf_adj_minus_raw.jpg', bbox_inches='tight', pad_inches=0, dpi=300)

