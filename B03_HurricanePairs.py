# _____________________________________________________________________________
# This script creates profile pairs (before after a TC passage)
#  
# 
# _____________________________________________________________________________ 

import h5py
import numpy as np
import pandas as pd
import pickle as pkl
from processing import Processor
import scipy.io

# Import specific functions written by Donata in file TC_and_TS_define_param.py 
from TC_and_TS_define_param import (
        Basins,year_pairs,year_start_TC, year_end_TC,  # Basins, ((2007, 2008), (2009, 2010)), 2007
        folder2use, months_string, var2use
        )

# Define process_hurricanes function
def process_hurricanes(fn, prefix, Basins, year_pairs, year_start_TC, months_string, year_end_TC): # Define 
    # the function and its arguments: 
    # Read hurricane track data
    dataframe = pd.read_csv(fn) # Read data and create a variable named 'dataframe'
    del dataframe['Unnamed: 0'] # Delete column 'unnamed:0' from dataframe variable 
    df_lst = []
    f = h5py.File(prefix + f'gridArgoProf{var2use}.mat', 'r')
    # Specify hurricane to examine: TC from 2007 to 2010; and counts the 
    # number of hurricanes 
    hurricanes = list(set(np.array(dataframe[
            (dataframe['SEASON'] >= year_start_TC) &  # IDs for hurricanes after 2007 
            (dataframe['SEASON'] <= year_end_TC)
            ]['ID'])))
    hurricanes.sort() # Sort hurricanes list 
    n = len(hurricanes) # n = 66 (number of hurricanes in the list) 
    # This cycle applies before_floats and add_after_floats functions 
    # from file Processor.py
    for idx, h_id in enumerate(hurricanes):
        hurricane_df = dataframe[dataframe['ID'] == h_id] # 54 hurricanes 
        name = np.array(hurricane_df['NAME'])[0] # TOMAS
        season = np.array(hurricane_df['SEASON'])[0] # 2010
        print(f'Processing {idx+1} of {n}: {name} of {season} ({h_id}).')
        P = Processor(hurricane_df, f)
        P.generate_before_floats()
        if P.float_df.shape[0] == 0:
            print('No before floats')
            continue
        P.add_after_floats()
        pair_df = P.create_pair_df()
        if pair_df is not None:
            df_lst.append(pair_df.assign(HurricaneID=h_id))
            
    if len(df_lst) > 0:
        df = pd.concat(df_lst
                       ).sort_values('before_t', ascending=False
                                     ).drop_duplicates('after_t').reset_index(drop=True)
        df['profile_dt'] = df['after_t'] - df['before_t']
        df['hurricane_dt'] = df['after_t'] - df['proj_t']
        df = df.assign(signed_angle=lambda r: - r.sign * r.angle)
    else:
        df = pd.DataFrame()
    return df

# Use process_hurricanes function  
df_list = []
for bs, fi in Basins: # bs: AL, EP, WP, IO, SH. fi: HURDAT_ATLANTIC, 
    # HURDAT_PACIFIC, JTWC_WESTPACIFIC, JTWC_INDIANOCEAN, JTWC_SOUTHERNHEMISPHERE
    df = process_hurricanes(f'../Inputs/{fi}.csv', # Run process_hurricanes 
                                                  # function and assign output to df
            str(folder2use) + '/Outputs/',
            Basins,year_pairs,year_start_TC,months_string,year_end_TC)
    pkl.dump(df, open(str(folder2use) + f'/Outputs/{bs}_PairDF.pkl', 'wb')) # Use df to write a new 
                                # file (saved in Data) for every single basin 
                                # named {bs}_PairDF.pkl  
    df_list.append(df)

pkl.dump(pd.concat(df_list), open(str(folder2use) + '/Outputs/AllBasin_PairDF.pkl', 'wb'))
