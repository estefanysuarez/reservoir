# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 13:47:40 2018

@author: labuser
"""

import os
import random
import time
import numpy as np
from scipy.linalg import eigh

from tvb.simulator import (simulator, models, coupling, integrators, monitors, noise)
from tvb.datatypes import (connectivity, surfaces, equations, patterns, region_mapping, sensors, cortex, local_connectivity, time_series)


def get_connectome(path, scaling_mode=None, eigen_scaling=True, alpha=1.0, normalize_weights=True):
    """
        Parameters
        ----------
        connectome_name: str
        conn_type: str
        scaling_mode: str ('binary', 'region', 'tract')
            region mode: Scale by a value such that the maximum absolute value
            of the cumulative input to any region is 1.0 (Global-wise scaling).
            tract mode: Scale by a value such that the maximum absolute value
            of a single connection is 1.0 (Global scaling).

        Returns
        -------

    """


    def eigen_scale(connectome, alpha):
        """
            scales connectome weights by the largest eigenvalue such that the
            largest eigenvalue of the new scaled matrix is 1.0

            Parameters
            ----------
            connectome: str

            Returns
            -------
            connectome object

        """
        conn_wei = connectome.weights
        ew, _ = eigh(conn_wei)
        conn_wei = conn_wei/np.max(ew)
        connectome.weights = conn_wei*alpha

        return connectome

    connectome =  connectivity.Connectivity.from_file(path)
    connectome.configure()

    if normalize_weights:
        conn_wei = connectome.weights
        connectome.weights = (conn_wei.copy()-conn_wei.min())/(conn_wei.max()-conn_wei.min())

    if scaling_mode == 'binary':
        connectome.weights = connectome.transform_binarize_matrix()

    if scaling_mode == 'region':
        connectome.weights = connectome.scaled_weights(mode='region')

    if scaling_mode == 'tract':
        connectome.weights = connectome.scaled_weights(mode='tract')

    if eigen_scaling: connectome = eigen_scale(connectome, alpha)

    return connectome


def get_stimulus(connectome, input_nodes, inputs, intensity=1.0):
    """

        Parameters
        ----------

        Returns
        -------

    """

    # function that creates a (spatio-temporal) stimulation pattern
    def create_stimuli(n_nodes, node_idx, node_weights, **params):
        weighting = np.zeros(n_nodes)
        weighting[node_idx] = node_weights

        eqn_t = equations.DiscreteTemporalEquation()
        eqn_t.parameters.update(params)

        stimulus = patterns.StimuliRegion(temporal=eqn_t,
                                          connectivity=connectome,
                                          weight=weighting
                                          )

        return stimulus

    n_nodes = connectome.weights.shape[0]
    stimuli_coll = list()
    for node in input_nodes:
        stimuli_coll.append(create_stimuli(n_nodes, [node], intensity, train=inputs[:, node]))

    stimuli = patterns.MultiStimuliRegion(*stimuli_coll)

    # configure spatiotemporal pattern
    stimuli.configure_space()

    n_time_steps = inputs.shape[0] #time lenght
    stimuli.configure_time(np.arange(n_time_steps))

    return stimuli


def get_NMM(nmm, **nmm_params):

    if nmm == '2d_oscillator':
        model = models.oscillator.Generic2dOscillator(**nmm_params)

    elif nmm == 'larter_breakspear':
        model = models.larter_breakspear.LarterBreakspear(**nmm_params)

    elif nmm == 'wong_wang':
        model = models.wong_wang.ReducedWongWang(**nmm_params)

    elif nmm == 'wilson_cowan':
        model = models.wilson_cowan.WilsonCowan(**nmm_params)

    return model


def get_global_params(global_coupling_factor=None, conduction_speed=4.0):

    return global_coupling_factor, conduction_speed


def run_sim(connectome, model, coupling, integrator, monitors, stimulus=None, sim_len=None):
    """
        Parameters
        ----------

        Returns
        -------

    """

    #create simulator object
    sim = simulator.Simulator(model = model,
                              connectivity = connectome,
                              coupling = coupling,
                              integrator = integrator,
                              monitors = monitors,
                              stimulus = stimulus,
                              # initial_conditions = np.zeros((157,3,83,1))
                              )
    sim.configure()

    print ('INITIATING PROCESSING TIME')
    t0_1 = time.clock()
    t0_2 = time.time()

    if stimulus is not None: sim_len = stimulus().shape[1] * integrator.dt #ms
    elif sim_len is None: sim_len = 7*60*1e3 * integrator.dt #ms

    ((raw_time, raw_data), _) = sim.run(simulation_length=sim_len)

    print ('PROCESSING TIME')
    print (time.clock()-t0_1, "seconds process time")
    print (time.time()-t0_2, "seconds wall time")

    return raw_data.squeeze().astype('float32'), raw_time.astype('float32')


def call_run_sim(connectome, input_nodes, inputs, global_params, integrator_params, **nmm_params):


    # # neural mass model
    model = get_NMM(**nmm_params)

    # integrator
    integrator = integrators.HeunDeterministic(dt=float(integrator_params['dt']))

    # monitors
    vars_to_monitor = (monitors.Raw(),
                       monitors.ProgressLogger(period=1000))

    # set global coupling and conduction speed
    global_coupling_factor, conduction_speed = get_global_params(**global_params)
    connectome.speed = conduction_speed
    coupling_eqn = coupling.Linear(a=global_coupling_factor)

    # get stimuli for simulation
    stimulus = get_stimulus(connectome, input_nodes, inputs)

    #run simulation
    print ('\n Running simulation ... ')
    states, time = run_sim(connectome=connectome,
                           model=model,
                           coupling=coupling_eqn,
                           integrator=integrator,
                           monitors=vars_to_monitor,
                           stimulus=stimulus,
                           )

    return states, time


def run_multiple_sim(path_conn, input_nodes, inputs, factor, alphas=None, path_results=None, global_params=None, integrator_params=None, **nmm_params):
    """
        Given a connectivity matrix, an input sequence, and a set of input
        nodes, this method simulates the reservoir network for multiple values
        of ALPHA, and returns the reservoir states of all the nodes in the
        network.
    """

    # simulate network for different alpha values
    if alphas is None: alphas = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    res_states = []

    for alpha in alphas:

        # connectivity
        connectome = get_connectome(path=path_conn,
                                    scaling_mode='binary',
                                    eigen_scaling=True,
                                    alpha=alpha,
                                    )

        x, time = call_run_sim(connectome = connectome,
                               input_nodes = input_nodes,
                               inputs=inputs*factor,
                               global_params=global_params,
                               integrator_params=integrator_params,
                               **nmm_params
                              )

        res_states.append(x)

    if path_results:
        np.save(path_results + '_reservoir_states', res_states)
        np.save(path_results + '_time', time)

    else:
        return res_states


def run_single_sim(path_conn, input_nodes, inputs, factor, path_results=None, global_params=None, integrator_params=None, **nmm_params):
    """
        Given a connectivity matrix, an input sequence, and a set of input
        nodes, this method simulates the reservoir network and returns the
        reservoir states of all the nodes in the network.
    """

    # connectivity
    connectome = get_connectome(path=path_conn,
                                scaling_mode='none',
                                eigen_scaling=False,
                                alpha=None,
                                )

    res_states, time = call_run_sim(connectome = connectome,
                                    input_nodes = input_nodes,
                                    inputs=inputs*factor,
                                    global_params=global_params,
                                    integrator_params=integrator_params,
                                    **nmm_params
                                    )
    if path_results:
        np.save(path_results + '_reservoir_states', res_states)
        np.save(path_results + '_time', time)

    else:
        return res_states
