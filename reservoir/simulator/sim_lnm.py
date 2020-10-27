# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:40:33 2019

@author: Estefany Suarez
"""
import numpy as np
import matplotlib.pyplot as plt
from ..tasks import tasks

#%% --------------------------------------------------------------------------------------------------------------------
# NETWORK SIMULATION
# ----------------------------------------------------------------------------------------------------------------------
def sim(w, w_in, stimulus, ic=None):

    timesteps = range(1, len(stimulus))
    N = len(w)

    synap_input = np.zeros((len(timesteps)+1, N))
    x = np.zeros((len(timesteps)+1, N))
    if ic is not None: x[0,:] = ic

    for t in timesteps:
        synap_input[t,:] = np.dot(w, x[t-1,:]) + np.dot(w_in, stimulus[t-1,:])
        x[t,:] = np.tanh(synap_input[t,:])

    return x, synap_input, timesteps


def run_sim(conn, input_nodes, inputs, factor, task=None, alphas=None, **kwargs):
    """
        Given a connectivity matrix conn, an input sequence (inputs), and a set of input_
        nodes, this method simulates the dynamics of the network for multiple values
        of alpha (alphas), and returns the reservoir states of ALL the nodes in the
        network.
    """

    # create input stimulus
    if type(inputs) == list: inputs = np.vstack(inputs)

    # create input connectivity matrix
    conn_input = np.zeros_like(conn)
    conn_input[:,input_nodes] = factor

    # simulate network for different alpha values
    if alphas is None: alphas = tasks.get_default_alpha_values(task)

    res_states = []
    for alpha in alphas:
        new_conn = alpha * conn.copy()
        x, _, _ = sim(w = new_conn,
                      w_in = conn_input,
                      stimulus=inputs
                      )

        res_states.append(x.astype(np.float32))

    return res_states
