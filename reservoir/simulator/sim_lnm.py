# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:40:33 2019

@author: Estefany Suarez
"""
import numpy as np
import matplotlib.pyplot as plt
from ..tasks import utils

#%% --------------------------------------------------------------------------------------------------------------------
# NETWORK SIMULATION
# ----------------------------------------------------------------------------------------------------------------------
def run_sim(w, w_in, stimulus, ic=None):

    timesteps = range(1, len(stimulus))
    N = len(w)

    synap_input = np.zeros((len(timesteps)+1, N))
    x = np.zeros((len(timesteps)+1, N))
    if ic is not None: x[0,:] = ic

    for t in timesteps:
        synap_input[t,:] = np.dot(w, x[t-1,:]) + np.dot(w_in, stimulus[t-1,:])
        x[t,:] = np.tanh(synap_input[t,:])

    return x, synap_input, timesteps


def run_multiple_sim(conn, input_nodes, inputs, factor, alphas=None, **kwargs):
    """
        Given a connectivity matrix, an input sequence, and a set of input
        nodes, this method simulates the reservoir network for multiple values
        of ALPHA, and returns the reservoir states of all the nodes in the
        network.
    """

    # create input stimulus
    if type(inputs) == list: inputs = np.vstack(inputs)

    # create input connectivity matrix
    conn_input = np.zeros_like(conn)
    conn_input[:,input_nodes] = factor

    # simulate network for different alpha values
    if alphas is None: alphas = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    res_states = []
    for alpha in alphas:
        new_conn = alpha * conn.copy()
        x, _, _ = run_sim(w = new_conn,
                          w_in = conn_input,
                          stimulus=inputs)

        res_states.append(x.astype(np.float32))

    return res_states
