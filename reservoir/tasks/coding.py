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
    # elif method == 'avg': scores = avg_based_encoder(**kwargs)
    # elif method == 'pca': scores = pca_based_encoder(**kwargs)

    return scores


def decoder(method='basic', **kwargs):

    if method == 'basic': scores = basic_decoder(**kwargs)
    # elif method == 'avg': scores = avg_based_decoder(**kwargs)
    # elif method == 'pca': scores = pca_based_decoder(**kwargs)

    return scores


#%% --------------------------------------------------------------------------------------------------------------------
# BASIC CODING MODE (V1)
# ----------------------------------------------------------------------------------------------------------------------
def coding(task, target, reservoir_states, readout_nodes=None, **kwargs):

    # get performance (R) across task parameters and alpha values
    res, task_params, alpha = tasks.run_task(task=task,
                                             target=target,
                                             reservoir_states=reservoir_states,
                                             readout_nodes=readout_nodes,
                                             **kwargs
                                             )

    # get max capacity and performance per alpha value
    performance, capacity = tasks.get_scores_per_alpha(task=task,
                                                       performance=res,
                                                       task_params=task_params,
                                                       **kwargs
                                                       )

    df_res = pd.DataFrame(data=np.column_stack((alpha, performance, capacity)),
                          columns=['alpha', 'performance', 'capacity'])

    df_res['n_nodes'] = len(readout_nodes)

    return df_res


def basic_encoder(task, target, reservoir_states, readout_modules=None, **kwargs):
    """
        Given the reservoir_states of the network and the target signal
        for a given task, this method returns the encoding capacity for a given
        set of readout_modules. If readout_modules is None, then it will return the
        encoding capacity of all the nodes in reservoir_states.
    """

    if readout_modules is None:
        df_encoding = coding(task=task,
                             target=target,
                             reservoir_states=reservoir_states,
                             **kwargs
                             )

    else:
        module_ids = np.unique(readout_modules)

        encoding = []
        for module in module_ids:
            print(f'--------------------------- Module : {module} ------------------------------')

            # get set of output nodes
            readout_nodes = np.where(readout_modules == module)[0]

            # create temporal dataframe
            tmp_df = coding(task=task,
                            target=target,
                            reservoir_states=reservoir_states,
                            readout_nodes=readout_nodes,
                            **kwargs
                            )

            tmp_df['module'] = module

            #get encoding scores
            encoding.append(tmp_df)

        df_encoding = pd.concat(encoding)

    return df_encoding


def basic_decoder(task, target, reservoir_states, readout_modules, bin_conn, \
                  exclude_within_nodes=True, **kwargs):
    """
        Given the reservoir_states of the network and the target signal
        for a given task, this method returns the decoding capacity for a given
        set of readout_modules.
    """
    module_ids = np.unique(readout_modules)

    decoding = []
    for module in module_ids:
        print(f'--------------------------- Module : {module} ------------------------------')

        nodes_within  = np.where(readout_modules == module)[0]
        nodes_outside = np.where(readout_modules != module)[0]

        bin_conn_module = bin_conn[nodes_within, :]
        bin_conn_profile_module = np.sum(bin_conn_module, axis=0).astype(bool).astype(int)

        try:
            unique_mapps, counts = np.unique(readout_modules[bin_conn_profile_module == 1], return_counts=True)

            if exclude_within_nodes:
                unique_mapps, counts = np.unique(readout_modules[nodes_outside][bin_conn_profile_module[nodes_outside] == 1], return_counts=True)

            composition = (counts/np.sum(counts))
            new_conn_profile = np.round(len(nodes_within)*composition, 0).astype(int)

            cont = 0
            while np.sum(new_conn_profile) > len(nodes_within):
                cont += 1
                new_conn_profile[np.argsort(composition)[-cont]] -= 1

            cont = 0
            while np.sum(new_conn_profile) < len(nodes_within):
                cont += 1
                new_conn_profile[np.argsort(composition)[-cont]] += 1

            # build distribution of scores for multiple neighbour nodes
            tmp = []
            for i in range(100):

                if i % 20 == 0:
                    print(f'\t---------- iter No. {i} ----------')

                # get set of output nodes
                readout_nodes = []
                for num_nodes, mapp in zip(new_conn_profile, unique_mapps):
                    tmp_set_mapp = np.logical_and((readout_modules == mapp), (bin_conn_profile_module == 1))
                    if exclude_within_nodes: tmp_set_mapp[nodes_within] = False
                    readout_nodes.extend(np.random.choice(np.where(tmp_set_mapp)[0], num_nodes, replace=False))

                tmp_df = coding(task=task,
                                target=target,
                                reservoir_states=reservoir_states,
                                readout_nodes=readout_nodes,
                                **kwargs
                                )

                tmp.append(tmp_df.values)

            tmp_df = pd.DataFrame(data=np.dstack(tmp).mean(axis=2),
                                  columns=['alpha', 'performance', 'capacity', 'n_nodes']
                                  )
            tmp_df['module'] = module

            decoding.append(tmp_df)

        except(IndexError):
            pass

    df_decoding = pd.concat(decoding)

    print(df_decoding.head)


    return df_decoding
