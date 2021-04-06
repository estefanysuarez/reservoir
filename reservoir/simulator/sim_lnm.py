# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:40:33 2019

@author: Estefany Suarez
"""

import numpy as np
import matplotlib.pyplot as plt

import Oger
import mdp
from NeuroTools import stgen
from scipy.linalg import eigh


from ..tasks import tasks

#%% --------------------------------------------------------------------------------------------------------------------
# NETWORK SIMULATION
# ----------------------------------------------------------------------------------------------------------------------
def sim(w, w_in, stimulus, ic=None, activation='tanh', threshold=0.5, add_perturb=False, t_perturb=200):
    """
        Simulates the dynamics of the network for provided inputs.

        Parameters
        ----------
        w : (N_source, N_target) numpy.ndarray
            Network connectivity matrix
            N_source: number of source nodes
            N_target: number of target nodes
        w_in : (N_ext_inputs, N_inputs) numpy.ndarray
            Input connectivity matrix
            N_ext_inputs: number of external input nodes
            N_inputs: number of inputs nodes in the network
        stimulus : (t, N_ext_inputs) numpy.ndarray
            External stimuli
            t : number ot time steps
            N_ext_inputs : number of external input nodes
        ic : (N,) numpy.ndarray
            Initial conditions
            N: total number of nodes in the network
        activation : {'tanh', 'piecewise'}
            Activation function for network's units
        threshold : float
            Threshold for piecewise activation function
        add_perturb : bool
            If True, adds a perturbation in network states at the time indicated
            by the parameter t_perturb

        Returns
        -------
        x : (t, N) numpy.darray
            Reservoir states
            time : number ot time steps
            N : total number of nodes in the network
    """

    # vector of timesteps
    timesteps = range(1, len(stimulus))

    # number of nodes in the network
    N = len(w)

    # create reservoir states matrix
    x = np.zeros((len(timesteps)+1, N))

    # set initial conditions
    if ic is not None: x[0,:] = ic

    # simulation of the dynamics
    if activation == 'tanh':
        for t in timesteps:
            synap_input = np.dot(x[t-1,:], w) + np.dot(stimulus[t-1,:], w_in)
            x[t,:] = np.tanh(synap_input)

            if add_perturb and (t == t_perturb): x[t, np.random.choice(N, 1)] = np.random.uniform(-1,1,1)[0] #np.random.rand(1)[0]

    elif activation == 'piecewise':
        for t in timesteps:
            synap_input = np.dot(x[t-1,:], w) + np.dot(stimulus[t-1,:], w_in)
            x[t,:] = np.piecewise(synap_input, [synap_input<threshold, synap_input>=threshold], [0, 1]).astype(int)

            if add_perturb and (t == t_perturb): x[t, np.random.choice(N, 1)] = np.random.uniform(-1,1,1)[0] #np.random.rand(1)[0]

    return x


def run_sim(conn, input_nodes, inputs, factor, task=None, alphas=None, **kwargs):
    """
        Simulates the dynamics of the network for a range of alpha values.

        Parameters
        ----------
        conn : (N_source, N_target) numpy.ndarray
            Network connectivity matrix
            N_source: number of source nodes
            N_target: number of target nodes
        input_nodes : (N_inputs,) numpy.ndarray
            Indices of input nodes
            N_inputs: number of inputs nodes in the network
        inputs : (t, N_ext_inputs) numpy.ndarray
            External input signal
            t : number ot time steps
            N_ext_inputs : number of external input nodes
        factor : float
            Factor that scales the input signal
        task : {'mem_cap', 'non_cap', 'pttn_recog', 'fcn_app'}
            Type of task:
            'mem_cap' : memory capacity task
            'non_cap' : nonlinear function approximation task
            'fcn_app' : nonlinear function approximation + memory capacity task
            'pttn_recog' : temporal pattern recognition task
        alphas : list
            List of alpha values to scale the connectivity matrix
            (equivalent to the spectral radii)

        Returns
        -------
        x : (n_alphas, t, N) numpy.darray
            Reservoir states
            n_alphas : number of alpha values
            t : number ot time steps
            N : total number of nodes in the network
        """

    # create input stimulus
    if type(inputs) == list: inputs = np.vstack(inputs)

    # create input connectivity matrix
    conn_input = np.zeros(len(input_nodes),len(conn))
    conn_input[:,input_nodes] = factor

    # simulate network for different alpha values
    if alphas is None: alphas = tasks.get_default_alpha_values(task)

    res_states = []
    for alpha in alphas:
        new_conn = alpha * conn.copy()
        x = sim(w = new_conn,
                w_in = conn_input,
                stimulus=inputs,
                **kwargs
                )

        res_states.append(x.astype(np.float32))

    return res_states


def run_sim_oger(conn, input_nodes, inputs, factor, task=None, alphas=None, **kwargs):
    """
        Given a connectivity matrix conn, an input sequence (inputs), and a set of input_
        nodes, this method simulates the dynamics of the network for multiple values
        of alpha (alphas), and returns the reservoir states of ALL the nodes in the
        network.
    """

    # create input stimulus
    if type(inputs) == list: inputs = np.vstack(inputs)

    # create input connectivity matrix

    #INCORRECT BUT COMPARABLE LSM2
    inputs_ = np.zeros((len(inputs), len(conn)))
    inputs_[:,input_nodes] = inputs
    inputs = inputs_.copy()

    input_dim = inputs.shape[1]
    output_dim = len(conn)

    conn_input = np.zeros((input_dim,output_dim))
    conn_input[:,input_nodes] = factor

    # CORRECT -LSM1
    # input_dim = inputs.shape[1]
    # output_dim = len(conn)
    #
    # conn_input = np.zeros((output_dim, input_dim))
    # conn_input[input_nodes,:] = factor

    # simulate network for different alpha values
    if alphas is None: alphas = tasks.get_default_alpha_values(task)
    res_states = []
    for alpha in alphas:
        reservoir = Oger.nodes.LeakyReservoirNode(input_dim=input_dim,
                                                  output_dim=output_dim,
                                                  spectral_radius=1.0,
                                                  reset_states=False,
                                                  bias_scaling=0,
                                                  input_scaling=1,
                                                  w=alpha*conn,
                                                  w_in=conn_input,
                                                  leak_rate=0.4,
                                                  **kwargs
                                                  )

        flow = mdp.Flow([reservoir])
        flow.train([inputs])

        x = []
        for stimulus in inputs:
            test = flow(stimulus[np.newaxis,:])
            x.append(test)

        ew, _ = eigh(reservoir.w)
        # print(f'\n Spectral radius: {np.max(ew)}')
        # print(f'\n W dimension: {reservoir.w.shape}')

        res_states.append(np.vstack(x).astype(np.float32))

    return res_states
