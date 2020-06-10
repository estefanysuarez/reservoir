# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:40:33 2019

@author: Estefany Suarez
"""
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import tasks

#%% --------------------------------------------------------------------------------------------------------------------
# GENERAL METHODS
# ----------------------------------------------------------------------------------------------------------------------
def encoder(method='basic', **kwargs):

    if method == 'basic': scores = basic_encoder(**kwargs)
    elif method == 'avg': scores = avg_based_encoder(**kwargs)
    elif method == 'pca': scores = pca_based_encoder(**kwargs)

    return scores

#%% --------------------------------------------------------------------------------------------------------------------
# BASIC CODING MODE (V1)
# ----------------------------------------------------------------------------------------------------------------------
def basic_encoder(task, target, reservoir_states, output_nodes, **kwargs):

    # get performance (R) across task parameters and alpha values
    res, task_params, alpha = tasks.run_task(task=task,
                                             target=target,
                                             reservoir_states=reservoir_states,
                                             readout_nodes=output_nodes,
                                             **kwargs
                                             )

    # get max capacity and performance per alpha value
    performance, capacity = tasks.get_scores_per_alpha(task=task,
                                                       performance=res,
                                                       task_params=task_params,
                                                       **kwargs
                                                       )

    # create dataframe
    df_res = pd.DataFrame(data=np.column_stack((alpha, performance, capacity)),
                          columns=['alpha', 'performance', 'capacity'])
    df_res['n_nodes'] = len(output_nodes)

    return df_res


#%% --------------------------------------------------------------------------------------------------------------------
# AVG-based CODING MODE (V2)
# ----------------------------------------------------------------------------------------------------------------------
def avg_based_encoder(task, target, reservoir_states, output_nodes, **kwargs):

    # get principal components
    avg_res_states = []
    for res_states in reservoir_states:
        avg_res_states.append(np.mean(res_states.squeeze()[:,output_nodes], axis=1))

    # get performance (R) across task parameters and alpha values
    res, task_params, alpha = tasks.run_task(task=task,
                                             target=target,
                                             reservoir_states=avg_res_states,
                                             )

    # get max capacity and performance per alpha value
    performance, capacity = tasks.get_scores_per_alpha(task=task,
                                                       performance=res,
                                                       task_params=task_params,
                                                      )

    # create temporal dataframe
    df_res = pd.DataFrame(data=np.column_stack((alpha, performance, capacity)),
                          columns=['alpha', 'performance', 'capacity'])


    return df_res


#%% --------------------------------------------------------------------------------------------------------------------
# PCA-based CODING MODE (V4)
# ----------------------------------------------------------------------------------------------------------------------
def pca_decomposition(data, n_pcomponents=3, var_exp=None):

    # standardize reservoir states before PCA
    scaler = StandardScaler()
    z_data = scaler.fit_transform(data)

    # apply PCA
    pca = PCA(n_components=data.shape[1], svd_solver='full', random_state=1234)
    new_data = pca.fit_transform(z_data)

    if var_exp is not None:
        variance = pca.explained_variance_ratio_
        n_pcomponents = 0
        accum_variance = variance[n_pcomponents]

        while accum_variance < var_exp:
            n_pcomponents += 1
            accum_variance += variance[n_pcomponents]

        n_pcomponents += 1

        print('---------- variance_explained: ' + str(var_exp))

    else:
        print('---------- number of components: ' + str(n_pcomponents))

    return new_data[:,:n_pcomponents] #new_data[:,:n_pcomponents+1]


def pca_based_encoder(task, target, reservoir_states, output_nodes, **kwargs):

    try:
        # get principal components
        pca_res_states = []
        for res_states in reservoir_states:
            pca_res_states.append(pca_decomposition(res_states.squeeze()[:,output_nodes], **kwargs))

        # get performance (R) across task parameters, task params and alpha values
        res, task_params, alpha = tasks.run_task(task=task,
                                                 target=target,
                                                 reservoir_states=pca_res_states,
                                                )

        # get max capacity and performance per alpha value
        performance, capacity = tasks.get_scores_per_alpha(task=task,
                                                           performance=res,
                                                           task_params=task_params,
                                                          )

        # create temporal dataframe
        df_res = pd.DataFrame(data=np.column_stack((alpha, performance, capacity)),
                              columns=['alpha', 'performance', 'capacity'])


    except(IndexError, ValueError):
        pass

    return df_res
