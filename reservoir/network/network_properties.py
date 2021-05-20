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

import networkx as nx
from networkx.algorithms import clique

from netneurotools import modularity as lmodularity

#%% --------------------------------------------------------------------------------------------------------------------
# NETWORKS PROPERTIES
# ----------------------------------------------------------------------------------------------------------------------
def get_local_network_properties(conn, cortical, class_mapping, property_list=None, include_subctx=True):
    """
        Given a weighted connectivity matrix, this methods estimates the local
        properties given by property_list.
    """
    def get_default_property_list():
        return ['node_strength',
                'node_degree',
                'wei_clustering_coeff',
                # 'bin_clustering_coeff',
                'wei_centrality',
                # 'bin_centrality',
                'wei_participation_coeff',
                # 'bin_participation_coeff',
                'wei_diversity_coeff',
                ]
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
                                               'assortativity_bin'
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


def get_modular_network_properties(conn, class_mapping, classes_sorted=None, property_list=None):

    if property_list is None: property_list = [
                                               'modularity',
                                               # 'rel_density',
                                               'segregation'
                                               ]

    conn_wei = conn.copy()

    properties = []
    if 'modularity' in property_list:
        if type(class_mapping) != int:
            class_mapping_int = np.array([np.where(classes_sorted == mapp)[0][0] for mapp in class_mapping]).astype(int)
        else:
            class_mapping_int = class_mapping
        q = lmodularity.get_modularity(conn_wei, class_mapping_int)
        properties.append(q)

    if 'rel_density' in property_list:
        pass

    if 'segregation' in property_list:

        def segregation(conn, clase):

           within = conn.copy()[np.ix_(np.where(class_mapping == clase)[0], np.where(class_mapping == clase)[0])]
           within = within[np.tril_indices_from(within, -1)]

           between = conn.copy()[np.ix_(np.where(class_mapping == clase)[0], np.where(class_mapping != clase)[0])]

           z_within = (within-np.mean(within[np.nonzero(within)]))/np.std(within[np.nonzero(within)])
           z_between = (between-np.mean(between[np.nonzero(between)]))/np.std(between[np.nonzero(between)])

           return (np.mean(z_within[np.nonzero(z_within)])-np.mean(z_between[np.nonzero(z_between)]))/np.mean(z_within[np.nonzero(z_within)])
        s = np.array([segregation(conn, clase) for clase in classes_sorted])
        properties.append(s)

    return properties, property_list


def get_cliques_local(conn):

    # convert to Graph
    G = nx.from_numpy_array(conn)

    # find cliques
    cliques = list(clique.enumerate_all_cliques(G))
    max_degree = len(cliques[-1])

    v_counts = []
    for degree in range(3, max_degree+1):

        cliques_ = [clique for clique in cliques if len(clique) == degree]
        v, counts = np.unique(np.row_stack(cliques_), return_counts=True)

        if len(v) != len(conn):
            new_counts = np.zeros(len(conn))
            new_counts[v] = counts
        else: new_counts = counts

        v_counts.append(new_counts)

    clique_names = [f'{k}-clique' for k in range(3, max_degree+1)]

    return v_counts, clique_names


def get_cliques_modular(conn, class_mapping, classes_sorted):

    degree_frequency = []
    max_degree = 0
    for clase in classes_sorted:

        # select rsn
        idx = np.where(class_mapping == clase)[0]
        tmp_conn = conn.copy()[np.ix_(idx, idx)]

        # convert to Graph
        G = nx.from_numpy_array(tmp_conn)

        # find cliques
        cliques = list(clique.enumerate_all_cliques(G))
        clique_degrees = [len(clique) for clique in cliques]

        degrees, counts = np.unique(clique_degrees, return_counts=True)
        degree_frequency.append(counts)

        if np.max(degrees) > max_degree: max_degree = np.max(degrees)

    # all frequencies of the same size
    ext_degree_frequency = []
    for f in degree_frequency:
        if len(f) != max_degree:
            new_f = np.zeros(max_degree)
            new_f[:len(f)] = f
            ext_degree_frequency.append(new_f)
        else:
            ext_degree_frequency.append(f)

    degree_names = [f'{k}-clique' for k in range(3, max_degree+1)]

    return ext_degree_frequency, degree_names
