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

#%% --------------------------------------------------------------------------------------------------------------------
# GENERAL METHODS
# ----------------------------------------------------------------------------------------------------------------------
def check_symmetric(a, tol=1e-16):
    return np.allclose(a, a.T, atol=tol)


def construct_null_model(type, **kwargs):

    if type == 'rand_mio':
        new_conn = rand_mio(**kwargs)

    elif type == 'watts_and_strogatz':
        new_conn = watts_and_strogatz(**kwargs)

    elif type == 'randmio_one_unperturbed':
        new_conn = randmio_but_unperturbed(**kwargs)

    elif type == 'erdos_renyi':
        new_conn = erdos_renyi(**kwargs)

    return new_conn


#%% --------------------------------------------------------------------------------------------------------------------
# NULL NETWORK MODELS
# ----------------------------------------------------------------------------------------------------------------------
def erdos_renyi(conn=None, density=0.025, **kwargs):

    if conn is not None: density = np.sum(conn.astype(bool).astype(int))/(len(conn)**2)

    new_conn = nx.to_numpy_array(nx.fast_gnp_random_graph(p=density, directed=False, **kwargs))
    new_conn = new_conn*np.random.uniform(-1, 1, new_conn.shape)
    new_conn[np.where(abs(new_conn) <= 0.00001)] = 0

    upper_diag = new_conn.copy()[np.triu_indices_from(new_conn, 1)]
    new_conn = new_conn.T
    new_conn[np.triu_indices_from(new_conn, 1)] = upper_diag
    np.fill_diagonal(new_conn,0)

    return new_conn


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
def randmio_but_unperturbed(conn, class_mapping, swaps=10, unperturbed=None): #max_attempts=3,

    conn = conn.copy()
    n = len(conn)
    i, j = np.where(np.tril(conn))

    k = [e for e, (i,j) in enumerate(zip(i,j)) if not (class_mapping[i] == unperturbed and class_mapping[j] == unperturbed)]

    swaps *= len(k)

    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * len(k) / (n * (n - 1)))

    # actual number of successful rewirings
    eff = 0

    for it in range(int(swaps)):
        att = 0
        while att <= max_attempts:  # while not rewired
            rewire = True

            while True:
                e1, e2 = np.random.choice(k, size=2, replace=False)

                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if (a != c and a != d and b != c and b != d) and not((unperturbed in class_mapping[[a,b]]) and (unperturbed in class_mapping[[c,d]])):
                    break  # all 4 vertices must be different and edges must not belong both to unperturbed

            if np.random.rand() > .5:

                i.setflags(write=True)
                j.setflags(write=True)
                i[e2] = d
                j[e2] = c  # flip edge c-d with 50% probability
                c = i[e2]
                d = j[e2]  # to explore all potential rewirings

            # rewiring condition
            if not (conn[a, d] or conn[c, b]):
                # connectedness condition
                if not (conn[a, c] or conn[b, d]):
                    P = conn[(a, d), :].copy()
                    P[0, b] = 0
                    P[1, c] = 0
                    PN = P.copy()
                    PN[:, d] = 1
                    PN[:, a] = 1
                    while True:
                        P[0, :] = np.any(conn[P[0, :] != 0, :], axis=0)
                        P[1, :] = np.any(conn[P[1, :] != 0, :], axis=0)
                        P *= np.logical_not(PN)
                        if not np.all(np.any(P, axis=1)):
                            rewire = False
                            break
                        elif np.any(P[:, (b, c)]):
                            break
                        PN += P
                # end connectedness testing

                if rewire:
                    conn[a, d] = conn[a, b]
                    conn[a, b] = 0
                    conn[d, a] = conn[b, a]
                    conn[b, a] = 0
                    conn[c, b] = conn[c, d]
                    conn[c, d] = 0
                    conn[b, c] = conn[d, c]
                    conn[d, c] = 0

                    j.setflags(write=True)
                    j[e1] = d
                    j[e2] = b  # reassign edge indices
                    eff += 1
                    break

            att += 1

    return conn#, eff


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
