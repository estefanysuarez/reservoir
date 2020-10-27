# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:27:07 2019

@author: Estefany Suarez
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

from bct import (clustering, reference)

from scipy.spatial.distance import cdist
import scipy.stats as st

from ..plotting import plot_tasks

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
def watts_and_strogatz(conn, p_conn=[0.1], bin=False):

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

    # create networks
    networks = []
    for p in p_conn:

        # create watts_strogatz graph
        G = nx.watts_strogatz_graph(N, deg, p)
        network = nx.to_numpy_array(G)

        if not bin:
            # assign weights to conns
            mask = np.nonzero(network)
            actual_conns = conn[mask]
            new_conns = get_pdf(data, st.powerlognorm, len(mask[0]))
            network[mask] = new_conns[np.argsort(actual_conns)]

        # save weighted network
        networks.append(network)

    return np.dstack(networks)


def rand_mio(conn, swaps=10):
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


#%% --------------------------------------------------------------------------------------------------------------------
# NULL NETWORK MODELS
# ----------------------------------------------------------------------------------------------------------------------
def increase_modularity(conn, class_mapping, swaps=10, max_attempts=10):

    new_conn = (conn.copy()-conn.min())/(conn.max()-conn.min())
    n = len(new_conn)

    eff = 0
    for clase in np.unique(class_mapping):
        print(f'\n-------------------{clase}---------------------------------')

        swaps = int((swaps/100)*len(np.where(class_mapping == clase)[0]))
        print(f'\t number of swaps: {swaps}')

        for swap in range(swaps):
            print(f'\t------------------- swap: {swap}---------------------------------')

            # get pairs of nodes of existent edges across all the network
            i,j = np.where(new_conn)

            # filter edges
            k = [e for e, (i,j) in enumerate(zip(i,j)) if ((class_mapping[i] == clase) and (class_mapping[i] != class_mapping[j]))]
#            k = [e for e, (i,j) in enumerate(zip(i,j)) if ((class_mapping[i] == clase) and (class_mapping[j] != clase))]

            # get and bin edges' weights
            weights = new_conn[(i[k], j[k])]
            categories = pd.qcut(pd.Series(weights),
                                 q=50,
                                 labels=False,
                                 retbins=False,
                                 precision=8
                                )

            categories = np.array(categories).astype(int)

            # rewiring
            att = 0
            while att <= max_attempts:
                print(f'\t\t-------------------attempt: {att}')

                while True:
                    # select 2 random different conenctions
                    e1, e2 = np.random.choice(k, size=2, replace=False)

                    # nodes of e1 and e1
                    a = i[e1]
                    b = j[e1]
                    c = i[e2]
                    d = j[e2]

                    if (a != c and a != d and b != c and b != d) and (class_mapping[b] != class_mapping[d]) and (categories[k == e1][0] == categories[k == e2][0]):
                        break  # all 4 vertices must be different

                # rewiring condition
                rewire = True
                if not (new_conn[a, c] or new_conn[b, d]):

                   # connectedness condition
                   R = new_conn.copy()

                   # e1
                   R[a, c] = R[a, b]
                   R[a, b] = 0
                   R[c, a] = R[b, a]
                   R[b, a] = 0

                   # e2
                   R[b, d] = R[c, d]
                   R[c, d] = 0
                   R[d, b] = R[d, c]
                   R[d, c] = 0

                   _, n_comp = clustering.get_components(R, no_depend=False)
                   if n_comp[0] != n:
                       rewire = False
                   # end of connectedness condition

                   if rewire:
                       new_conn = R.copy()
                       eff += 1
                       break

                att += 1

    return new_conn, eff


def decrease_modularity(conn, class_mapping, swaps=50, max_attempts=10):

    new_conn = (conn.copy()-conn.min())/(conn.max()-conn.min())
    n = len(new_conn)

    eff = 0
    for clase in np.unique(class_mapping):
        print(f'\n-------------------{clase}---------------------------------')

        swaps = int((swaps/100)*len(np.where(class_mapping == clase)[0]))
        print(f'\t number of swaps: {swaps}')

        for swap in range(swaps):
            print(f'\t------------------- swap: {swap}---------------------------------')

            # get pairs of nodes of existent edges across all the network
            i,j = np.where(new_conn)

            # filter edges
            k1 = [e for e, (i,j) in enumerate(zip(i,j)) if (class_mapping[i] == class_mapping[j] == clase)]
            k2 = [e for e, (i,j) in enumerate(zip(i,j)) if ((class_mapping[i] != clase) and (class_mapping[j] != clase) and (class_mapping[i] != class_mapping[j]))]

            k = k1 + k2

            # get and bin edges' weights
            weights = new_conn[(i[k], j[k])]
            categories = pd.qcut(pd.Series(weights),
                                 q=50,
                                 labels=False,
                                 retbins=False,
                                 precision=8
                                )

            categories = np.array(categories).astype(int)

            # rewiring
            att = 0
            while att <= max_attempts:

                print(f'\t\t-------------------attempt: {att}')

                while True:
                    # select 2 random different conenctions
                    e1 = np.random.choice(k1, size=1, replace=False)
                    e2 = np.random.choice(k2, size=1, replace=False)

                    # nodes of e1 and e1
                    a = i[e1]
                    b = j[e1]
                    c = i[e2]
                    d = j[e2]

                    if (a != c and a != d and b != c and b != d) and (categories[k == e1][0] == categories[k == e2][0]):
                        break  # all 4 vertices must be different

                if np.random.rand() > .5:
                    i[e2] = d
                    j[e2] = c  # flip edge c-d with 50% probability
                    c = i[e2]
                    d = j[e2]  # to explore all potential rewirings

                # rewiring condition
                rewire = True
                if not (new_conn[a, d] or new_conn[c, b]):
                    if not (new_conn[a, c] or new_conn[b, d]):

                       # connectedness condition
                       R = new_conn.copy()

                       # e1
                       R[a, d] = R[a, b]
                       R[a, b] = 0
                       R[d, a] = R[b, a]
                       R[b, a] = 0

                       # e2
                       R[c, b] = R[c, d]
                       R[c, d] = 0
                       R[b, c] = R[d, c]
                       R[d, c] = 0

                       _, n_comp = clustering.get_components(R, no_depend=False)
                       if n_comp[0] != n:
                           rewire = False
                       # end of connectedness condition

                       if rewire:
                           new_conn = R.copy()
                           eff += 1
                           break

                att += 1

    return new_conn, eff
