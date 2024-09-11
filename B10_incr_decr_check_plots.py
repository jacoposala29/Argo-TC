#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:31:54 2023

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
import matplotlib.pyplot as plt

from TC_and_TS_define_param import folder2use, \
    depth_layers, var_min_B10, n_level_min_mask, min_TCwind, \
    folder2use_S_ML_incr_decr_B06_B10, \
    folder2use_T_ML_incr_decr_B06_B10, \
    folder2use_PD_ML_incr_decr_B06_B10, var2use, \
    bounds_x1, bounds_x2, degrees_tag, \
    ML_delta, mode_tag, mask_tag

sys.path.append('./implementations/')
from implementation_tools import (
    grid,
    variable_diff,
    ML_incr_decr
)

data_dir = folder2use + '/Outputs'
plot_dir = '../Figures/B10_check_plots/'

# ________
# Part copied from B10

# Import data from B09 output 
data_dir = folder2use + '/Outputs'
plot_dir = '../Figures/B10_check_plots/'
df = pkl.load(open(f'{data_dir}/HurricaneAdjRawVariableCovDF.pkl', 'rb'))
df = df[abs(df.standard_signed_angle)<=bounds_x1[1]] 
df = df.dropna()
df.shape # (16674, 21)
df_shape_0 = str(df.shape[0])

# Remove profiles based on outlier of variance
df_excluded = df[np.log(df['var'].apply(lambda x: np.min(x))) < var_min_B10].sort_values([
    'before_pid',
    'after_pid',
])

df = df[np.log(df['var'].apply(lambda x: np.min(x))) >= var_min_B10].sort_values([
    'before_pid',
    'after_pid',
]) 

# Load utput from B05 necessary for MLD calculation for mask incr vs decr
df_S_forML = pkl.load(open(f'{folder2use_S_ML_incr_decr_B06_B10}/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))
df_T_forML = pkl.load(open(f'{folder2use_T_ML_incr_decr_B06_B10}/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))
df_PD_forML = pkl.load(open(f'{folder2use_PD_ML_incr_decr_B06_B10}/Outputs/HurricaneAdjRawVariableDF.pkl', 'rb'))

# Recall ML_incr_decr function
DFS, df_main, mask_incr, mask_decr = ML_incr_decr(depth_layers, df, df_S_forML, df_T_forML, df_PD_forML, min_TCwind, ML_delta, mode_tag)
# ________

# Change format to make it consistent with old version of the code
mask_incr = np.ravel(mask_incr).astype(bool)
mask_decr = np.ravel(mask_decr).astype(bool)

# df with only either incr or decr (otherwise plots are messy)
df_incr_or_decr = df_main[(mask_incr) | (mask_decr)]

# Create mask_incr_smaller
mask_incr_smaller = mask_incr[np.logical_or(mask_incr, mask_decr)]
mask_decr_smaller = mask_decr[np.logical_or(mask_incr, mask_decr)]


# INCREASING

# Define the number of subplots
num_subplots = 5

# Define mask
mask = mask_incr_smaller  # Replace with your actual mask

salinity_minus2 = np.array(df_incr_or_decr.raw_before_variable.tolist())

# Define the total number of events
total_events = salinity_minus2.shape[0]
total_depth_levels = salinity_minus2.shape[1]

# Set up the figure with subplots
fig, axs = plt.subplots(num_subplots, 1, figsize=(35, 20), sharex=False)

# Invert the y-axis for all subplots
for ax in axs:
    ax.invert_yaxis()

# Plot circles for each depth level in each event
event_counter = 0  # Track the current event index
for j, ax in enumerate(axs):
    # Calculate the number of events to display in this subplot
    events_per_subplot = (total_events - event_counter) // (num_subplots - j)
    # Loop through the remaining events
    for _ in range(events_per_subplot):
        # Check if the current event is masked
        if mask[event_counter]:
            # Plot profile for this event
            x_values = np.full(total_depth_levels, event_counter)
            y_values = np.arange(total_depth_levels)
            sizes = 25

            # Sort indices based on salinity_minus2 values for this event
            sorted_indices_one = np.argsort(salinity_minus2[event_counter, :])
            sorted_indices = np.argsort(sorted_indices_one)
            
            # Scatter plot with color based on sorted salinity_minus2 values
            sc = ax.scatter(x_values, y_values, s=sizes, c=sorted_indices, cmap='viridis', edgecolors='k', linewidth=0.2)

        # Increment event counter
        event_counter += 1

    # Set labels for each subplot
    ax.set_ylabel(f'Depth Index (Subplot {j + 1})', fontsize=18)

# Add colorbar to the last subplot
cbar = plt.colorbar(sc, ax=axs[-1], label='Sorted Salinity Index')

# Set labels and title for the whole figure
axs[-1].set_xlabel('Event Number', fontsize=18)
fig.suptitle(f'Sorted Salinity Index (Divided Among 5 Subplots) - increasing - mask {mask_tag}', fontsize=18)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Show the plot
plt.savefig(f"{plot_dir}/Sanity_check_{mask_tag}_incr_all_events.png", dpi=200)
plt.show()




# DECREASING

# Define the number of subplots
num_subplots = 5

# Define mask
mask = mask_decr_smaller  # Replace with your actual mask

salinity_minus2 = np.array(df_incr_or_decr.raw_before_variable.tolist())

# Define the total number of events
total_events = salinity_minus2.shape[0]
total_depth_levels = salinity_minus2.shape[1]

# Set up the figure with subplots
fig, axs = plt.subplots(num_subplots, 1, figsize=(35, 20), sharex=False)

# Invert the y-axis for all subplots
for ax in axs:
    ax.invert_yaxis()

# Plot circles for each depth level in each event
event_counter = 0  # Track the current event index
for j, ax in enumerate(axs):
    # Calculate the number of events to display in this subplot
    events_per_subplot = (total_events - event_counter) // (num_subplots - j)
    # Loop through the remaining events
    for _ in range(events_per_subplot):
        # Check if the current event is masked
        if mask[event_counter]:
            # Plot profile for this event
            x_values = np.full(total_depth_levels, event_counter)
            y_values = np.arange(total_depth_levels)
            sizes = 25

            # Sort indices based on salinity_minus2 values for this event
            sorted_indices_one = np.argsort(salinity_minus2[event_counter, :])
            sorted_indices = np.argsort(sorted_indices_one)
            
            # Scatter plot with color based on sorted salinity_minus2 values
            sc = ax.scatter(x_values, y_values, s=sizes, c=sorted_indices, cmap='viridis', edgecolors='k', linewidth=0.2)

        # Increment event counter
        event_counter += 1

    # Set labels for each subplot
    ax.set_ylabel(f'Depth Index (Subplot {j + 1})', fontsize=18)

# Add colorbar to the last subplot
cbar = plt.colorbar(sc, ax=axs[-1], label='Sorted Salinity Index')

# Set labels and title for the whole figure
axs[-1].set_xlabel('Event Number', fontsize=18)
fig.suptitle(f'Sorted Salinity Index (Divided Among 5 Subplots) - decreasing - mask {mask_tag}', fontsize=18)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Show the plot
plt.savefig(f"{plot_dir}/Sanity_check_{mask_tag}_decr_all_events.png", dpi=200)
plt.show()










