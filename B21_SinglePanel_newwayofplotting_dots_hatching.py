#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:10:00 2022

@author: jacoposala
"""

# This script is the same as B21, but also plots differences between cases
# Note: Jan 2, 2023: changed to plot +-2 days only

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
                                    bounds_x1, bounds_x2, shape, lamb_choice, degrees_tag)

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

fs = 24
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

    
def plotB21_B22_hatch(ax, variable, data_xaxis, data_yaxis, msk_data, bounds_x1, bounds_x2, minima, maxima, xlabel, ylabel, axcolorbar, legend):
    
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
    cmap = cm.bwr
    cmap = plt.get_cmap('bwr', 21)  # Choose an appropriate number of discrete colors
    cmap.set_bad(color='grey')
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    ax.imshow(variable.reshape(*(shape[::-1])),
    origin='lower',
    cmap=cmap, norm=norm,
    extent=(*bounds_x1, *bounds_x2),
    aspect = 'auto', # adapt size of panel in imshow to figsize
    )
    plt.pcolor(data_xaxis, data_yaxis, msk_data, hatch='.', alpha=0.)
    ax.invert_yaxis()
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xticks([-2, -1, 0, +1, +2]) # x ticks
    ax.set_xlim(-2,2) # x limit
    # ax.set_yticks([0, 5, 10, 15, 20]) # y ticks

    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_ylim(-2, 14)  # y limit

    ax.invert_yaxis()

    if ylabel: #set ylabel option
        ax.set_ylabel(r'Days since TC passage ($\tau$)',
            fontsize=fs)
    if xlabel: #set xlabel option
        ax.set_xlabel(r'Cross-track angle, degrees ($d$)',
         fontsize=fs)
    ax.set_title(legend,
            y=ypos,
            fontsize=fs+df)   
    if axcolorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mapper, ax=ax, cax=cax)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(f'\u0394 {var2use} ({unit})',
                    fontsize=fs)
        
        
    
# Loop through all cases
for sub, sub_comp in zip(sub_list, sub_list_comparison):
    print(sub)
    prefix = '_LOOCV'

    # Adjusted case
    norew = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_NoRew_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'rb'))
    block = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'rb'))
    bmask = pkl.load(open(f'{input_dir}/TPS{prefix}_Mask_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'rb'))
    bstd = pkl.load(open(f'{input_dir}/TPS{prefix}_Stdev_Block_{lamb_choice}_{sub}_{degrees_tag}.pkl', 'rb'))
    
    # Raw case
    norew_raw = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_NoRew_{lamb_choice}_{sub}_raw_{degrees_tag}.pkl', 'rb'))
    block_raw = pkl.load(open(f'{input_dir}/TPS{prefix}_Preds_Block_{lamb_choice}_{sub}_raw_{degrees_tag}.pkl', 'rb'))
    bmask_raw = pkl.load(open(f'{input_dir}/TPS{prefix}_Mask_Block_{lamb_choice}_{sub}_raw_{degrees_tag}.pkl', 'rb'))
    bstd_raw = pkl.load(open(f'{input_dir}/TPS{prefix}_Stdev_Block_{lamb_choice}_{sub}_raw_{degrees_tag}.pkl', 'rb'))
    
    n, k = norew.shape
    
    idx = 0
    
    for idx in range(k):
        # Adjusted
        preds_norew = norew[:, idx]
        preds_block = block[:, idx]
        preds_std = bstd[:, idx]
        #preds_mask = preds_block.copy()
        preds_block_mask = bmask[:, idx]
        #preds_mask[~bmask[:, idx].astype(bool)] = np.nan
        # Raw
        preds_norew_raw = norew_raw[:, idx]
        preds_block_raw = block_raw[:, idx]
        preds_std_raw = bstd_raw[:, idx]
        #preds_mask_raw = preds_block_raw.copy()
        preds_mask_raw = bmask_raw[:, idx]
        #preds_mask_raw[~bmask_raw[:, idx].astype(bool)] = np.nan
        # Mean field
        preds_norew_mf = - preds_norew_raw + preds_norew
        preds_block_mf = - preds_block_raw + preds_block

        depth = int(grid_lower) + idx*int(grid_stride)
        
        # SINGLE PANEL PLOTS  - mask panel
        d2plot_list = ['preds_block']
        casetag_list = ['Right_Panel_preds_mask']
        for d2plot, case_tag in zip(d2plot_list,casetag_list):
            
            if np.array_equal(eval(d2plot), preds_block_mf):
                minima, maxima = -clim_mf, +clim_mf
            else:
                minima, maxima = -clim, +clim
                
            var_to_mask = eval(d2plot)
            mask = eval(d2plot + "_mask")
            msk_data = np.ma.masked_where(~mask.astype(bool), var_to_mask).reshape(*(shape[::-1]))
            data_xaxis = np.linspace(bounds_x1[0], bounds_x1[1], var_to_mask.reshape(*(shape[::-1])).shape[1])
            data_yaxis = np.linspace(bounds_x2[0], bounds_x2[1], var_to_mask.reshape(*(shape[::-1])).shape[0])

            fig = plt.figure(figsize=(5,8))
            ax1 = plt.subplot(1,1,1)
            
            plotB21_B22_hatch(ax1, eval(d2plot), data_xaxis, data_yaxis, msk_data, bounds_x1, bounds_x2, minima, maxima, ylabel = True, xlabel = True, axcolorbar = True, legend = '')
            plt.savefig(f'{plot_dir}/TPS_{var2use}_{depth:03d}_{sub}_{lamb_choice}_{case_tag}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
       
