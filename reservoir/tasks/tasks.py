# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:46:05 2019

@author: Estefany Suarez
"""

import os
import numpy as np
import pandas as pd
import scipy as sp
import mdp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing


#%% --------------------------------------------------------------------------------------------------------------------
# TASKS
# ----------------------------------------------------------------------------------------------------------------------
def plot_pred_vs_target(y_pred, y_test, label, ax):
    sns.regplot(x=y_test,
                y=y_pred,
                label=label,
                fit_reg=True,
                scatter=True,
                scatter_kws={"s": 40},
                ci=None,
                # linewidths=0.3,
                # edgecolors='dimgrey',
                ax=ax
                )


def run_mem_cap(X, Y, TAU=None, normalize=False, ax=None, **kwargs):
    """
    In this task, the linear readout is required to replay a delayed version of
    the input sequence s.
    """
    # TAU: memory capacity required by the task
    if TAU is None: TAU = get_default_task_params('mem_cap')

    # get train and test sets
    x_train, x_test = X
    y_train, y_test = Y

    # define transient
    transient = 0 #number of initial time points to discard

    if (x_train.squeeze().ndim == 2) and (x_test.ndim == 2):
        x_train = x_train.squeeze()[transient:,:]
        x_test  = x_test.squeeze()[transient:,:]

    else:
        x_train = x_train.squeeze()[transient:, np.newaxis]
        x_test  = x_test.squeeze()[transient:, np.newaxis]

    y_train = y_train.squeeze()[transient:]
    y_test  = y_test.squeeze()[transient:]

    if normalize:
        x_train = (x_train - x_train.mean(axis=1)[:,np.newaxis]).squeeze()
        x_test  = (x_test - x_test.mean(axis=1)[:,np.newaxis]).squeeze()

    res = []
    for tau in TAU:

       model = LinearRegression(fit_intercept=False, normalize=False).fit(x_train[tau:], y_train[:-tau])
       y_pred =  model.predict(x_test[tau:])

       with np.errstate(divide='ignore', invalid='ignore'):
           perf = np.abs(np.corrcoef(y_test[:-tau], y_pred)[0][1])

           # if perf > 0.80:
           #     print('\n----------------------')
           #     print(' Tau    = ' + str(tau))
           #     print(' perf   =   ' + str(perf))
           #
           #     plt.scatter(y_test[:-tau], y_pred)
           #     plt.show()
           #     plt.close()

       # save results
       res.append(perf)

    return np.array(res), TAU


def run_nonlin_cap(X, Y, OMEGA=None, normalize=False, ax=None, **kwargs):
    """
    In this task, the linear readout is required to replay a noninear version of
    the input sequence s.
    """
    # OMEGA: level of nonlinearity required by the task
    if OMEGA is None: OMEGA = get_default_task_params('nonlin_cap')

    # get train and test sets
    x_train, x_test = X
    y_train, y_test = Y

    # define transient
    transient = 0 #number of initial time points to discard

    if (x_train.squeeze().ndim == 2) and (x_test.ndim == 2):
        x_train = x_train.squeeze()[transient:,:]
        x_test  = x_test.squeeze()[transient:,:]

    else:
        x_train = x_train.squeeze()[transient:, np.newaxis]
        x_test  = x_test.squeeze()[transient:, np.newaxis]

    y_train = y_train.squeeze()[transient:]
    y_test  = y_test.squeeze()[transient:]

    if normalize:
        x_train = (x_train - x_train.mean(axis=1)[:,np.newaxis]).squeeze()
        x_test  = (x_test - x_test.mean(axis=1)[:,np.newaxis]).squeeze()

    res = []
    for omega in OMEGA:

       model = LinearRegression(fit_intercept=False, normalize=False).fit(x_train[1:], np.sin(omega*y_train[:-1]))
       y_pred =  model.predict(x_test[1:])

       with np.errstate(divide='ignore', invalid='ignore'):
           perf = np.abs(np.corrcoef(np.sin(omega*y_test[:-1]), y_pred)[0][1])
           # print(perf)

           # if perf > 0.85:
           #     print('\n----------------------')
           #     print(' Omega = ' + str(np.log10(omega)))
           #     print(' perf  =   ' + str(perf))
           #
           #     plt.scatter(np.sin(omega*y_test[:-1]), y_pred)
           #     plt.show()
           #     plt.close()

       # save results
       res.append(perf)

    return np.array(res), np.log10(OMEGA)


def run_fcn_app(X, Y, TAU=None, OMEGA=None, normalize=False, ax=None, **kwargs):

    # define nonlinearity and memory capcity degree of the fcn_approx
    if TAU is None:   TAU   = get_default_task_params('fcn_app')[0]
    if OMEGA is None: OMEGA = get_default_task_params('fcn_app')[1]

    param_space = dict(tau = TAU, omega = OMEGA)
    param_grid = list(ParameterGrid(param_space))

    # get train and test sets
    x_train, x_test = X
    y_train, y_test = Y

    # define transient
    transient = 0 #number of initial time points to discard

    if (x_train.squeeze().ndim == 2) and (x_test.ndim == 2):
        x_train = x_train.squeeze()[transient:,:]
        x_test  = x_test.squeeze()[transient:,:]

    else:
        x_train = x_train.squeeze()[transient:, np.newaxis]
        x_test  = x_test.squeeze()[transient:, np.newaxis]

    y_train = y_train.squeeze()[transient:]
    y_test  = y_test.squeeze()[transient:]

    if normalize:
        x_train = (x_train - x_train.mean(axis=1)[:,np.newaxis]).squeeze()
        x_test  = (x_test - x_test.mean(axis=1)[:,np.newaxis]).squeeze()

    res  = np.zeros((len(param_space['tau']), len(param_space['omega'])))
    for params_set in param_grid:

        # level of nonlinearity and temporal memory capacity required
        # by the fcn_approx task
        omega = params_set['omega']
        tau = params_set['tau']

        model = LinearRegression(fit_intercept=False, normalize=False).fit(x_train[tau:], np.sin(omega*y_train[:-tau]))
        y_pred =  model.predict(x_test[tau:])

        with np.errstate(divide='ignore', invalid='ignore'):
            perf = np.abs(np.corrcoef(np.sin(omega*y_test[:-tau]), y_pred)[0][1])
            # print(perf)

            # if perf > 0.85:
            #     print('\n----------------------')
            #     print(' Tau  = ' + str(tau))
            #     print(' Omega = ' + str(np.log10(omega)))
            #     print(' perf   =   ' + str(perf))
            #
            #     plt.scatter(np.sin(omega*y_test[:-tau]), y_pred)
            #     plt.show()
            #     plt.close()


        # print out results
        coord_omega = np.where(param_space['omega'] == omega)[0][0]
        coord_tau   = np.where(param_space['tau'] == tau)[0][0]
        res[coord_tau, coord_omega] = perf

    return res, (param_space['tau'], np.log10(param_space['omega']))


def run_pttn_recog(X, Y, time_lens=None, normalize=False, **kwargs):

    # get train and test sets
    x_train, x_test = X
    y_train, y_test = Y

    if not ((x_train.squeeze().ndim == 2) and (x_test.ndim == 2)):
        x_train = x_train.squeeze()[:, np.newaxis]
        x_test  = x_test.squeeze()[:, np.newaxis]

    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    if normalize:
        x_train = (x_train - x_train.mean(axis=1)[:,np.newaxis]).squeeze()
        x_test  = (x_test - x_test.mean(axis=1)[:,np.newaxis]).squeeze()

    sections = [np.sum(time_lens[:idx]) for idx in range(1, len(time_lens))]
    y_test = np.split(y_test, sections, axis=0) #Y[n_train_samples:]

    ridge_multi_regr = MultiOutputRegressor(Ridge(fit_intercept=True, normalize=False, solver='auto', alpha=0.0))
    y_pred = np.split(ridge_multi_regr.fit(x_train, y_train).predict(x_test), sections, axis=0)

    #lr_multi_regr = MultiOutputRegressor(LinearRegression(fit_intercept=True, normalize=False, copy_X = True))
    #y_pred = np.split(lr_multi_regr.fit(x_train, y_train).predict(x_test), np.array(pattern_lens[train_samples:-1]), axis=0)

    y_test_mean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in y_test])
    # print(y_test_mean)

    y_pred_mean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in y_pred])
    # print(y_pred_mean)

    with np.errstate(divide='ignore', invalid='ignore'):
        cm = metrics.confusion_matrix(y_test_mean, y_pred_mean)
        cm_norm = np.diagonal(cm)/np.sum(cm, axis=1)
        perf = len(np.where(cm_norm > 0.9)[0])
        # print(perf)

    return perf


#%% --------------------------------------------------------------------------------------------------------------------
# GRAL METHODS
# ----------------------------------------------------------------------------------------------------------------------
def get_default_alpha_values(task=None):

    if task is None:
        alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]

    elif (task == 'mem_cap') or (task == 'nonlin_cap') or (task == 'pttn_recog'):
        # alphas = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]

    elif (task == 'fcn_app'):
        alphas = [0.1, 0.3, 0.7, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    return alphas


def get_default_task_params(task):

    if task == 'mem_cap':
        params = np.arange(1, 16)

    elif task == 'nonlin_cap':
        params = np.array([10**-2.5, 10**-2.0, 10**-1.5, 10**-1.0, 10**-0.5, 10**0.0, 10**0.5, 10**1.0])

    elif task == 'fcn_app':
        param1 = np.arange(1, 16)
        param2 = np.array([10**-2.5, 10**-2.0, 10**-1.5, 10**-1.0, 10**-0.5, 10**0.0, 10**0.5, 10**1.0])
        params = (param1, param2)

    return params


def run_task(task, target, reservoir_states, readout_nodes=None, include_alpha=None, **kwargs):
    """
        Given a target and a set of reservoir states(corresponding to different
        values of ALPHA), this method performs multiple trials (one for each
        ALPHA) of the task specified by 'task', and returns a PERF estimate
        across the different alpha values.
    """
    # perform task for different reservoir states corresponding to different alpha values
    res = []
    for idx, x in enumerate(reservoir_states):

        # define x and y
        if readout_nodes is not None: x = x.squeeze()[:, :, readout_nodes]
        else: x = x.squeeze()
        y = target.squeeze()

        # perform task
        if task == 'mem_cap':
            perf, task_params = run_mem_cap(x, y, **kwargs)

        elif task == 'nonlin_cap':
            perf, task_params = run_nonlin_cap(x, y, **kwargs)

        elif task == 'fcn_app':
            perf, task_params = run_fcn_app(x, y, **kwargs)

        elif task == 'pttn_recog':
           perf = run_pttn_recog(x, y, **kwargs)
           task_params = None

        res.append(perf) # across task parameters

    # all alpha values at which the network was simulated
    if include_alpha is None: include_alpha = get_default_alpha_values(task)

    return res, task_params, include_alpha # across task parameters and alpha values


def get_scores_per_alpha(task, performance, task_params, thres=0.9, normalize=False, **kwargs):
    """
        This method returns the parameters at which the best performance across
        different alpha values occurs.
    """

    # estimate capacity across task params per alpha value
    if (task == 'mem_cap') or (task == 'nonlin_cap'):

        # estimate capacity across task params per alpha value
        if normalize: cap_per_alpha = [(task_params[perf<thres][np.argmax(perf[perf<thres])]-np.min(task_params))/(np.max(task_params)-np.min(task_params)) if (perf>thres).any() else 0 for perf in performance] #performance normalized in [0,1] range
        else: cap_per_alpha = [(task_params[perf<thres][np.argmax(perf[perf<thres])]) if (perf>thres).any() else 0 for perf in performance]


    elif task == 'fcn_app':

        param_tau, param_omega = task_params

        # estimate capacity across task params per alpha value
        cap_per_alpha = []
        for perf in performance:
            idx_tau_below_thrd, idx_omega_below_thrd = np.where(perf == np.max(perf[perf<thres]))

            if normalize:
                tmp_cap_param_tau   = (param_tau[idx_tau_below_thrd[0]]-np.min(param_tau))/(np.max(param_tau)-np.min(param_tau)) #performance normalized in [0,1] range
                tmp_cap_param_omega = (param_omega[idx_omega_below_thrd[0]]-np.min(param_omega))/(np.max(param_omega)-np.min(param_omega)) #performance normalized in [0,1] range

            else:
                tmp_cap_param_tau   = param_tau[idx_tau_below_thrd[0]]
                tmp_cap_param_omega = param_omega[idx_omega_below_thrd[0]]

            cap_per_alpha.append(tmp_cap_param_tau+tmp_cap_param_omega)

    elif task == 'pttn_recog':
        # estimate capacity across task params per alpha value. There is no capacity for the pattern recognition task
        cap_per_alpha = [np.nan for _ in performance]

    # estimate performance across task params per alpha value
    perf_per_alpha = np.array([np.sum(perf) for perf in performance])

    return perf_per_alpha, cap_per_alpha


# def run_pattrn_recog(target, reservoir_states, time_lens, normalize=False, **kwargs):
#
#     if normalize: reservoir_states = (reservoir_states - reservoir_states.mean(axis=1)[:,np.newaxis])
#
#     sections = [np.sum(time_lens[:idx]) for idx in range(1, len(time_lens))]
#     X = np.split(reservoir_states.squeeze(), sections, axis=0)
#     Y = np.split(target, sections, axis=0)
#
#     train_frac = 0.9
#     n_samples = len(X)
#     n_train_samples = int(round(n_samples * train_frac))
#     n_test_samples = int(round(n_samples * (1 - train_frac)))
#
#     x_train = np.vstack(X[:n_train_samples])
#     y_train = np.vstack(Y[:n_train_samples])
#
#     x_test = np.vstack(X[n_train_samples:])
#     y_test = Y[n_train_samples:]
#
#     ridge_multi_regr = MultiOutputRegressor(Ridge(fit_intercept=True, normalize=False, solver='auto', alpha=0.0))
#     Y_pred = np.split(ridge_multi_regr.fit(x_train, y_train).predict(x_test), np.array(sections[n_train_samples:]), axis=0)
#
#     #lr_multi_regr = MultiOutputRegressor(LinearRegression(fit_intercept=True, normalize=False, copy_X = True))
#     #Y_pred = np.split(lr_multi_regr.fit(x_train, y_train).predict(x_test), np.array(pattern_lens[train_samples:-1]), axis=0)
#
#     Y_test_mean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in y_test])
#     print(Y_test_mean)
#
#     Y_pred_mean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in Y_pred])
#     print(Y_pred_mean)
#
#     with np.errstate(divide='ignore', invalid='ignore'):
#         cm = metrics.confusion_matrix(Y_test_mean, Y_pred_mean)
#         cm_norm = np.diagonal(cm)/np.sum(cm, axis=1)
#         perf = len(np.where(cm_norm > 0.9)[0])
#
#         # print('\n---------' + str(perf))
#
#     return perf
