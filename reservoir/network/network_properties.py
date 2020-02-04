# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:27:07 2019

@author: Estefany Suarez
"""

import os
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bct.algorithms import(centrality, clustering, degree, distance, modularity, core, similarity)
from scipy.spatial.distance import cdist

#%% --------------------------------------------------------------------------------------------------------------------
# NETWORKS PROPERTIES
# ----------------------------------------------------------------------------------------------------------------------
def get_network_properties(conn, cortical, class_mapping, property_list=None, include_subctx=True):

    if property_list == None:
        property_list = ['node_degree',
                         'node_strength',
                         'wei_clustering_coeff',
                         'bin_clustering_coeff',
                         'wei_centrality',
                         'bin_centrality',
                         'participation_coeff'
                         ]

    if include_subctx:
        conn_wei = conn.copy()
        conn_bin = conn_wei.copy().astype(bool).astype(int)

    else:
        ctx = np.where(cortical==1)[0]
        conn_wei = conn.copy()[np.ix_(ctx, ctx)]
        conn_bin = conn_wei.copy().astype(bool).astype(int)

    properties = []
    if 'participation_coeff' in property_list:
        properties.append(centrality.participation_coef(conn_wei, ci=class_mapping))

    if 'node_degree' in property_list:
        properties.append(degree.degrees_und(conn_bin))

    if 'node_strength' in property_list:
        properties.append(degree.strengths_und(conn_wei))

    if 'wei_clustering_coeff' in property_list:
        properties.append(clustering.clustering_coef_wu(conn_wei))

    if 'bin_clustering_coeff' in property_list:
        properties.append(clustering.clustering_coef_bu(conn_bin))

    if 'wei_centrality' in property_list:
        N = len(conn)
        properties.append(centrality.betweenness_wei(conn_wei)/((N-1)*(N-2)))

    if 'bin_centrality' in property_list:
        N = len(conn)
        properties.append(centrality.betweenness_bin(conn_bin)/((N-1)*(N-2)))


    df = pd.DataFrame(np.column_stack(properties),
                      columns=property_list)

    return df
