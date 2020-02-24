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


def decoder(method='basic', **kwargs):

    if method == 'basic': scores = basic_decoder(**kwargs)
    elif method == 'avg': scores = avg_based_decoder(**kwargs)
    elif method == 'pca': scores = pca_based_decoder(**kwargs)

    return scores


#%% --------------------------------------------------------------------------------------------------------------------
# BASIC CODING MODE (V1)
# ----------------------------------------------------------------------------------------------------------------------
def basic_encoder(task, target, reservoir_states, class_labels, class_mapping, **kwargs):

    res_encoding = []
    for idx, clase in enumerate(class_labels):

       print('---------------------------' + str(clase) + '------------------------------------------------------')

       # get set of output nodes
       output_nodes = np.where(class_mapping == clase)[0]

       # get performance (R) across task parameters, task params and alpha values
       perf, task_params, alpha = tasks.run_multiple_tasks(task=task,
                                                           target=target,
                                                           res_states=reservoir_states,
                                                           readout_nodes=output_nodes,
                                                           **kwargs
                                                           )

       # get max capacity and performance per alpha value
       performance, capacity = tasks.get_capacity_and_perf(task=task,
                                                           performance=perf,
                                                           task_params=task_params,
                                                           **kwargs
                                                           )

       # create temporal dataframe
       tmp_df = pd.DataFrame(data=np.column_stack((alpha, performance, capacity)),
                              columns=['alpha', 'performance', 'capacity'])
       tmp_df['class'] = clase
       tmp_df['n_nodes'] = len(output_nodes)

       res_encoding.append(tmp_df)

    return pd.concat(res_encoding)


def basic_decoder(task, target, reservoir_states, class_labels, class_mapping, bin_conn, exclude_within_nodes=True, **kwargs):

    res_decoding = []
    for idx, clase in enumerate(class_labels):

        print('---------------------------' + str(clase) + '------------------------------------------------------')

        nodes_within  = np.where(class_mapping == clase)[0]
        nodes_outside = np.where(class_mapping != clase)[0]

        bin_conn_class = bin_conn[nodes_within, :]
        bin_conn_profile_class = np.sum(bin_conn_class, axis=0).astype(bool).astype(int)

        try:
            unique_mapps, counts = np.unique(class_mapping[bin_conn_profile_class == 1], return_counts=True)
            if exclude_within_nodes:
                unique_mapps, counts = np.unique(class_mapping[nodes_outside][bin_conn_profile_class[nodes_outside] == 1], return_counts=True)

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


            #build distribution of performances
            tmp_perf = []
            for i in range(100):

                print('-------------------' + str(i) + '--------------------')

                output_nodes = []
                for num_nodes, mapp in zip(new_conn_profile, unique_mapps):
                    tmp_set_mapp = np.logical_and((class_mapping == mapp), (bin_conn_profile_class == 1))
                    if exclude_within_nodes: tmp_set_mapp[nodes_within] = False
                    output_nodes.extend(np.random.choice(np.where(tmp_set_mapp)[0], num_nodes, replace=False))

                # get performance (R), task params and alpha values
                perf, task_params, alpha = tasks.run_multiple_tasks(task=task,
                                                                    target=target,
                                                                    res_states=reservoir_states,
                                                                    readout_nodes=output_nodes,
                                                                    )

                # get max capacity and performance per alpha value
                tmp_performance, tmp_capacity = tasks.get_capacity_and_perf(task=task,
                                                                            performance=perf,
                                                                            task_params=task_params,
                                                                            )

                tmp_perf.append(np.column_stack((tmp_performance, tmp_capacity)))


            tmp_df = pd.DataFrame(data=np.column_stack((alpha, np.dstack(tmp_perf).mean(axis=2))),
                                  columns=['alpha', 'performance', 'capacity'])

            tmp_df['class'] = clase
            tmp_df['n_nodes'] = len(output_nodes)

            res_decoding.append(tmp_df)

        except(IndexError):
            pass

    return pd.concat(res_decoding)


#%% --------------------------------------------------------------------------------------------------------------------
# AVG-based CODING MODE (V2)
# ----------------------------------------------------------------------------------------------------------------------
def avg_based_encoder(task, target, reservoir_states, class_labels, class_mapping, **kwargs):

    res_encoding = []
    for idx, clase in enumerate(class_labels):

       print('---------------------------' + str(clase) + '------------------------------------------------------')

       # get set of nodes within class
       nodes = np.where(class_mapping == clase)[0]

       # get principal components
       avg_res_states = []
       for res_states in reservoir_states:
           avg_res_states.append(np.mean(res_states.squeeze()[:,nodes], axis=1))

       # get performance (R) across task parameters, task params and alpha values
       perf, task_params, alpha = tasks.run_multiple_tasks(task=task,
                                                           target=target,
                                                           res_states=avg_res_states,
                                                           )

       # get max capacity and performance per alpha value
       performance, capacity = tasks.get_capacity_and_perf(task=task,
                                                           performance=perf,
                                                           task_params=task_params,
                                                           )

       # create temporal dataframe
       tmp_df = pd.DataFrame(data=np.column_stack((alpha, performance, capacity)),
                             columns=['alpha', 'performance', 'capacity'])
       tmp_df['class'] = clase

       res_encoding.append(tmp_df)

    return pd.concat(res_encoding)


def avg_based_decoder(task, target, reservoir_states, class_labels, class_mapping, bin_conn, exclude_within_nodes=True, **kwargs):

    res_decoding = []
    for idx, clase in enumerate(class_labels):
       try:
           print('---------------------------' + str(clase) + '------------------------------------------------------')

           # get set of nodes in connectivity profile of class
           nodes_within  = np.where(class_mapping == clase)[0]
           nodes_outside = np.where(class_mapping != clase)[0]

           bin_conn_class = bin_conn[nodes_within, :]
           degree_class = np.sum(bin_conn_class, axis=0)

           nodes = np.where(degree_class != 0)[0]
           if exclude_within_nodes: nodes = np.intersect1d(nodes, nodes_outside)

           # get principal components
           avg_res_states = []
           for res_states in reservoir_states:
               avg_res_states.append(np.mean(res_states.squeeze()[:,nodes], axis=1))

           # get performance (R) across task parameters, task params and alpha values
           perf, task_params, alpha = tasks.run_multiple_tasks(task=task,
                                                               target=target,
                                                               res_states=avg_res_states,
                                                               )

           # get max capacity and performance per alpha value
           performance, capacity = tasks.get_capacity_and_perf(task=task,
                                                               performance=perf,
                                                               task_params=task_params,
                                                               )

           # create temporal dataframe
           tmp_df = pd.DataFrame(data=np.column_stack((alpha, performance, capacity)),
                                 columns=['alpha', 'performance', 'capacity'])
           tmp_df['class'] = clase

           res_decoding.append(tmp_df)

       except (IndexError, ValueError):
           pass

    return pd.concat(res_decoding)


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


def pca_based_encoder(task, target, reservoir_states, class_labels, class_mapping, **kwargs):

    res_encoding = []
    for idx, clase in enumerate(class_labels):
       try:
           print('---------------------------' + str(clase) + '------------------------------------------------------')

           # get set of nodes within class
           nodes = np.where(class_mapping == clase)[0]

           # get principal components
           pca_res_states = []
           for res_states in reservoir_states:
               pca_res_states.append(pca_decomposition(res_states.squeeze()[:,nodes], **kwargs))

           # get performance (R) across task parameters, task params and alpha values
           perf, task_params, alpha = tasks.run_multiple_tasks(task=task,
                                                               target=target,
                                                               res_states=pca_res_states,
                                                               )

           # get max capacity and performance per alpha value
           performance, capacity = tasks.get_capacity_and_perf(task=task,
                                                               performance=perf,
                                                               task_params=task_params,
                                                               )

           # create temporal dataframe
           tmp_df = pd.DataFrame(data=np.column_stack((alpha, performance, capacity)),
                                 columns=['alpha', 'performance', 'capacity'])
           tmp_df['class'] = clase

           res_encoding.append(tmp_df)

       except(IndexError, ValueError):
           pass

    return pd.concat(res_encoding)


def pca_based_decoder(task, target, reservoir_states, class_labels, class_mapping, bin_conn, exclude_within_nodes=True, **kwargs):

    res_decoding = []
    for idx, clase in enumerate(class_labels):
       try:
           print('---------------------------' + str(clase) + '------------------------------------------------------')

           # get set of nodes in connectivity profile of class
           nodes_within  = np.where(class_mapping == clase)[0]
           nodes_outside = np.where(class_mapping != clase)[0]

           bin_conn_class = bin_conn[nodes_within, :]
           degree_class = np.sum(bin_conn_class, axis=0)

           nodes = np.where(degree_class != 0)[0]
           if exclude_within_nodes: nodes = np.intersect1d(nodes, nodes_outside)

           # get principal components
           pca_res_states = []
           for res_states in reservoir_states:
               pca_res_states.append(pca_decomposition(res_states.squeeze()[:,nodes], **kwargs))

           # get performance (R) across task parameters, task params and alpha values
           perf, task_params, alpha = tasks.run_multiple_tasks(task=task,
                                                               target=target,
                                                               res_states=pca_res_states,
                                                               )

           # get max capacity and performance per alpha value
           performance, capacity = tasks.get_capacity_and_perf(task=task,
                                                               performance=perf,
                                                               task_params=task_params,
                                                               )

           # create temporal dataframe
           tmp_df = pd.DataFrame(data=np.column_stack((alpha, performance, capacity)),
                                 columns=['alpha', 'performance', 'capacity'])
           tmp_df['class'] = clase

           res_decoding.append(tmp_df)

       except(IndexError, ValueError):
           pass


    return pd.concat(res_decoding)
