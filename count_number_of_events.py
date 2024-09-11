#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:54:23 2023

@author: jacoposala
"""
import pandas as pd
import numpy as np

# Tracks to include in the analysis
track_dir = '../Inputs'
Basins = [
        ('AL', 'HURDAT_ATLANTIC'),
        ('EP', 'HURDAT_PACIFIC'),
        ('WP', 'JTWC_WESTPACIFIC'),
        ('IO', 'JTWC_INDIANOCEAN'),
        ('SH', 'JTWC_SOUTHERNHEMISPHERE'),
        ]

AL = pd.read_csv(f'{track_dir}/HURDAT_ATLANTIC.csv')
del AL['Unnamed: 0']
EP = pd.read_csv(f'{track_dir}/HURDAT_PACIFIC.csv')
del EP['Unnamed: 0']
WP = pd.read_csv(f'{track_dir}/JTWC_WESTPACIFIC.csv')
del WP['Unnamed: 0']
SH = pd.read_csv(f'{track_dir}/JTWC_SOUTHERNHEMISPHERE.csv')
del SH['Unnamed: 0']
IO = pd.read_csv(f'{track_dir}/JTWC_INDIANOCEAN.csv')
del IO['Unnamed: 0']
Hurricanes_ALL = pd.concat([AL, EP, WP, SH, IO])

# Select years
HU_years = Hurricanes_ALL[(pd.to_datetime(Hurricanes_ALL.DATE).dt.year > 2003) & (pd.to_datetime(Hurricanes_ALL.DATE).dt.year < 2021)]

# Select unique events
HU_unique = HU_years.ID.unique().shape

# Select hurricane events
HU_years[HU_years.WIND > 64].ID.unique().shape

