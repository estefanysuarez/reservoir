# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:27:07 2019

@author: Estefany Suarez
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

from bct import reference

from scipy.spatial.distance import cdist
import scipy.stats as st


#%% --------------------------------------------------------------------------------------------------------------------
# GENERAL METHODS
# ----------------------------------------------------------------------------------------------------------------------
def check_symmetric(a, tol=1e-16):
    return np.allclose(a, a.T, atol=tol)


def construct_network_model(conn, type, **kwargs):

    if type == 'rand_mio':
        new_conn = rand_mio(conn, **kwargs)

    elif type == 'watts_and_strogatz':
        new_conn = watts_and_strogatz(conn, **kwargs)

    return new_conn


#%% --------------------------------------------------------------------------------------------------------------------
# NULL NETWORK MODELS
# ----------------------------------------------------------------------------------------------------------------------

def watts_and_strogatz(conn, p_conn=[0.1]):

    # scale conn data
    conn_vec = conn[np.tril_indices_from(conn, -1)]
    data = pd.Series(conn_vec[np.nonzero(conn_vec)])

    # generate data given a distribution
    def get_pdf(data, dist, size):

        # fit dist to data
        params = dist.fit(data)

        # separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # get same start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end   = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        pdf = dist.pdf(x, loc=loc, scale=scale, *arg)

        return pdf

    # binarize conn data
    conn_bin = conn.astype(bool).astype(int)
    deg = int(np.mean(np.sum(conn_bin, axis=0)))
    N = len(conn_bin)

    networks = []
    for p in p_conn:

        # create watts_strogatz graph
        G = nx.watts_strogatz_graph(N, deg, p)
        network = nx.to_numpy_array(G)
        mask = np.nonzero(network)

        # assign weights to conns
        actual_conns = conn[mask]
        new_conns = get_pdf(data, st.powerlognorm, len(mask[0]))
        network[mask] = new_conns[np.argsort(actual_conns)]

        # save weighted network
        networks.append(network)

    return np.dstack(networks)


def rand_mio(conn, swaps=1):
    """
        Parameters
        ----------
        directed binary/weighted connectome: nodes x nodes binary connectivity
        matrix

        Returns
        -------
        conn_mat: network model
    """

    if check_symmetric(conn):
        conn_mat, _ = reference.randmio_und_connected(conn, swaps)

    else:
        conn_mat, _ = reference.randmio_dir_connected(conn, swaps)

    return conn_mat
