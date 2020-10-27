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
def get_default_property_list():
    property_list = ['node_strength',
                     'node_degree',
                     'wei_clustering_coeff',
                     'bin_clustering_coeff',
                     'wei_centrality',
                     'bin_centrality',
                     'wei_participation_coeff',
                     'bin_participation_coeff',
                     'wei_diversity_coeff',
                     ]

    return property_list


def get_local_network_properties(conn, cortical, class_mapping, property_list=None, include_subctx=True):
    """
        Given a weighted connectivity matrix, this methods estimates the local
        properties given by property_list.
    """
    if property_list is None: property_list = get_default_property_list()

    #REMOVE SUBCTX AD HOC
    # if include_subctx:
    #     conn_wei = conn.copy()
    #     conn_bin = conn_wei.copy().astype(bool).astype(int)
    #
    # else:
    #     ctx = np.where(cortical==1)[0]
    #     conn_wei = conn.copy()[np.ix_(ctx, ctx)]
    #     conn_bin = conn_wei.copy().astype(bool).astype(int)

    conn_wei = conn.copy()
    conn_bin = conn_wei.copy().astype(bool).astype(int)

    properties = []
    if 'node_strength' in property_list: #***
        properties.append(degree.strengths_und(conn_wei))

    if 'node_degree' in property_list: #***
        properties.append(degree.degrees_und(conn_bin))

    if 'wei_clustering_coeff' in property_list: #***
        properties.append(clustering.clustering_coef_wu(conn_wei))

    if 'bin_clustering_coeff' in property_list: #***
        properties.append(clustering.clustering_coef_bu(conn_bin))

    if 'wei_centrality' in property_list:
        N = len(conn)
        properties.append(centrality.betweenness_wei(1/conn_wei)/((N-1)*(N-2)))

    if 'bin_centrality' in property_list:
        N = len(conn)
        properties.append(centrality.betweenness_bin(conn_bin)/((N-1)*(N-2)))

    if 'wei_participation_coeff' in property_list: #***
        properties.append(centrality.participation_coef(conn_wei, ci=class_mapping))

    if 'bin_participation_coeff' in property_list: #***
        properties.append(centrality.participation_coef(conn_bin, ci=class_mapping))

    if 'wei_diversity_coeff' in property_list: #***
        pos, _ = centrality.diversity_coef_sign(conn_wei, ci=class_mapping)
        properties.append(pos)


    #REMOVE SUBCTX POST-HOC
    if not include_subctx: properties = [prop[cortical==1] for prop in properties]

    df = pd.DataFrame(np.column_stack(properties),
                      columns=property_list)

    return df


def get_global_network_properties(conn, cortical, class_mapping, property_list=None):

    if property_list is None: property_list = [
                                               'path_length',
                                               'clustering',
                                               'modularity',
                                               'assortativity_wei',
                                               # 'assortativity_bin'
                                               ]

    conn_wei = conn.copy()

    properties = []
    if 'path_length' in property_list:
        dist, _ = distance.distance_wei(1/conn_wei)
        char_path, _, _, _, _, = distance.charpath(dist, include_infinite=False)
        properties.append(char_path)

    if 'clustering' in property_list:
        properties.append(clustering.transitivity_wu(conn_wei))

    if 'modularity' in property_list:
        _, q = modularity.modularity_und(conn_wei, kci=class_mapping)
        properties.append(q)

    if 'assortativity_wei' in property_list:
        properties.append(core.assortativity_wei(conn_wei, flag=0))

    if 'assortativity_bin' in property_list:
        properties.append(core.assortativity_bin(conn_wei.astype(bool).astype(int), flag=0))

    return properties, property_list
