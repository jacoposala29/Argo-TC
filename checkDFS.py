#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:01:48 2024

@author: jacoposala
"""

import pickle as pkl


dfs_new = pkl.load(open('/Users/jacoposala/Desktop/CU/3.RESEARCH/ARGO_analysis/POTENTIAL_DENSITY_test_Mar14_0_5_100_1005_5deg_Dec2023/DFS_Potential_density_5deg.pkl', 'rb'))


dfs_old = pkl.load(open('/Users/jacoposala/Desktop/CU/3.RESEARCH/ARGO_analysis/POTENTIAL_DENSITY_test_Mar14_0_5_100_1005_5deg_Dec2023/Outputs/DFS_Potential_density_5deg.pkl', 'rb'))


dfs_all_mask_sal = pkl.load(open('/Users/jacoposala/Desktop/CU/3.RESEARCH/ARGO_analysis/SALINITY_test_Mar14_0_5_100_1005_5deg_Dec2023/Outputs/DFS_allmasks_Salinity_5deg.pkl', 'rb'))

