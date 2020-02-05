# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:46:05 2019

@author: Estefany Suarez
"""

import os
import numpy as np
import pandas as pd
import scipy as sp
# import mdp
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

    # ax.scatter(x=y_test,
    #            y=y_pred,
    #            s=15,
    #            linewidths=0.3,
    #            # edgecolors='dimgrey',
    #            label=label
    #            )

    sns.scatterplot(x=y_test,
                    y=y_pred,
                    s=15,
                    # linewidths=0.3,
                    # edgecolors='dimgrey',
                    label=label,
                    ax=ax
                    )


def run_memory_capacity(s, reservoir_states, TAU=None, normalize=False, ax=None):
    """
    In this task, the linear readout is required to replay a delayed version of
    the input sequence s.
    """
    # memory capacity required by the task
    if TAU is None: TAU = get_default_task_params('mem_cap')

    transient = 0 #number of initial time points to discard

    if reservoir_states.squeeze().ndim == 2:
        X = reservoir_states.copy().squeeze()[transient:,:]

    else:
        X = reservoir_states.copy().squeeze()[transient:][:, np.newaxis]
    Y = s[transient:]

    if normalize: X = (X - X.mean(axis=1)[:,np.newaxis]).squeeze()

    training_size = 2000
    x_train = X[:training_size]
    y_train = Y[:training_size]

    x_test = X[training_size:]
    y_test = Y[training_size:]

    res = []

    for tau in TAU:

       # print('---------------Running Linear Regression!!!---------------')
       model = LinearRegression(fit_intercept=False, normalize=False).fit(x_train[tau:], y_train[:-tau])
       y_pred =  model.predict(x_test[tau:])

       with np.errstate(divide='ignore', invalid='ignore'):
           perf = np.abs(np.corrcoef(y_test[:-tau], y_pred)[0][1])
           # perf = (np.corrcoef(y_test[:-tau], y_pred)[0][1])**2
           # perf = np.cov(y_test[:-tau], y_pred)[0][1]**2/(np.var(y_test[:-tau])*np.var(y_pred))

           # if perf > 0.85:
           #     print('\n----------------------')
           #     print(' Tau    = ' + str(tau))
           #     print(' perf   =   ' + str(perf))
           #
           #     plt.scatter(y_test[:-tau], y_pred)
           #     plt.show()
           #     plt.close()

           if ax is not None:
               plot_pred_vs_target(y_test[:-tau], y_pred, r'$\tau$'+' = ' + str(tau) + ' - ' + r'$r = %.2f $' % (np.round(perf, 2)), ax)

       # save results
       res.append(perf)

    return np.array(res), TAU


def run_nonlinearity_capacity(s, reservoir_states, OMEGA=None, normalize=False, ax=None):
    """
    In this task, the linear readout is required to replay a delayed version of
    the input sequence s.
    """

    # level of nonlinearity required by the task
    if OMEGA is None: OMEGA = get_default_task_params('nonlin_cap')

    res = []
    for omega in OMEGA:

       transient = 0 # number of initial point to discard
       if reservoir_states.squeeze().ndim == 2:
           X = reservoir_states.copy().squeeze()[transient:,:]

       else:
           X = reservoir_states.copy().squeeze()[transient:][:, np.newaxis]
       Y = np.sin(omega*s[transient:]) #2*np.pi*

       if normalize: X = (X - X.mean(axis=1)[:,np.newaxis]).squeeze()

       training_size = 2000
       x_train = X[:training_size]
       y_train = Y[:training_size]

       x_test = X[training_size:]
       y_test = Y[training_size:]

       # print('---------------Running Linear Regression!!!---------------')
       model = LinearRegression(fit_intercept=False, normalize=False).fit(x_train[1:], y_train[:-1])
       y_pred =  model.predict(x_test[1:])

       with np.errstate(divide='ignore', invalid='ignore'):
           perf = np.abs(np.corrcoef(y_test[:-1], y_pred)[0][1])
           # perf = (np.corrcoef(y_test[:-1], y_pred)[0][1])**2
           # perf = np.cov(y_test[:-1], y_pred)[0][1]**2/(np.var(y_test)*np.var(y_pred))

           if perf > 0.85:
               # print('\n----------------------')
               # print(' Omega = ' + str(np.log10(omega)))
               # print(' perf  =   ' + str(perf))

               # plt.scatter(y_test[:-1], y_pred)
               # plt.show()
               # plt.close()

               if ax is not None:
                   plot_pred_vs_target(y_test[:-1], y_pred, r'$\omega$' + (' = %.3f' % (omega)) + ' - ' + r'$r = %.2f $' % (np.round(perf, 4)), ax)


       # save results
       res.append(perf)

    return np.array(res), np.log10(OMEGA)


def run_fcn_approx(s, reservoir_states=10000, normalize=False, TAU=None, OMEGA=None, ax=None):

    # define nonlinearity and memory capcity degree of the fcn_approx
    # if TAU is None:   (TAU, _)   = get_default_task_params('fcn_app')
    # if OMEGA is None: (_, OMEGA) = get_default_task_params('fcn_app')

    if TAU is None:   TAU   = get_default_task_params('fcn_app')[0]
    if OMEGA is None: OMEGA = get_default_task_params('fcn_app')[1]

    # print(TAU)
    # print(np.log10(OMEGA))

    param_space = dict(tau = TAU, omega = OMEGA)
    param_grid = list(ParameterGrid(param_space))

    res  = np.zeros((len(param_space['tau']), len(param_space['omega'])))
    for params_set in param_grid:

        # level of nonlinearity and temporal memory capacity required
        # by the fcn_approx task
        omega = params_set['omega']
        tau = params_set['tau']

        transient = 0 # number of initial point to discard
        if reservoir_states.squeeze().ndim == 2:
            X = reservoir_states.copy().squeeze()[transient:,:]

        else:
            X = reservoir_states.copy().squeeze()[transient:][:, np.newaxis]
        Y = np.sin(2*np.pi*omega*s[transient:])

        if normalize: X = (X - X.mean(axis=1)[:,np.newaxis]).squeeze()

        training_size = 2000
        x_train = X[:training_size]
        y_train = Y[:training_size]

        x_test = X[training_size:]
        y_test = Y[training_size:]

#        print('---------------Running Linear Regression!!!---------------')
        model = LinearRegression(fit_intercept=False, normalize=False).fit(x_train[tau:], y_train[:-tau])
        y_pred =  model.predict(x_test[tau:])

        with np.errstate(divide='ignore', invalid='ignore'):
            perf = np.abs(np.corrcoef(y_test[:-tau], y_pred)[0][1])
            # perf = (np.corrcoef(y_test[:-tau], y_pred)[0][1])**2
            # perf = np.cov(y_test[:-tau], y_pred)[0][1]**2/(np.var(y_test[:-tau])*np.var(y_pred))

            if perf > 0.85:
               # print('\n----------------------')
               # print(' Tau  = ' + str(tau))
               # print(' Omega = ' + str(np.log10(omega)))
               # print(' perf   =   ' + str(perf))

               # plt.scatter(y_test[:-tau], y_pred)
               # plt.show()
               # plt.close()

               if ax is not None:
                   plot_pred_vs_target(y_test[:-tau], y_pred, r'$\tau$' + ',' + r'$\omega$' + (' = %.2f , %.2f' % (tau,omega))+ ' - ' + r'$r = %.2f $' % (np.round(perf, 2)), ax)#+ ' - ' + r'$r = %.2f $' % (np.round(perf, 2)), ax)

        # print out results
        coord_omega = np.where(param_space['omega'] == omega)[0][0]
        coord_tau   = np.where(param_space['tau'] == tau)[0][0]
        res[coord_tau, coord_omega] = perf

    return res, (param_space['tau'], np.log10(param_space['omega']))


def run_pattrn_recog(target, reservoir_states, time_lens, normalize=False):

    if normalize: reservoir_states = (reservoir_states - reservoir_states.mean(axis=1)[:,np.newaxis])

    sections = [np.sum(time_lens[:idx]) for idx in range(1, len(time_lens))]
    X = np.split(reservoir_states.squeeze(), sections, axis=0)
    Y = np.split(target, sections, axis=0)

    train_frac = 0.9
    n_samples = len(X)
    n_train_samples = int(round(n_samples * train_frac))
    n_test_samples = int(round(n_samples * (1 - train_frac)))

    x_train = np.vstack(X[:n_train_samples])
    y_train = np.vstack(Y[:n_train_samples])

    x_test = np.vstack(X[n_train_samples:])
    y_test = Y[n_train_samples:]

    ridge_multi_regr = MultiOutputRegressor(Ridge(fit_intercept=True, normalize=False, solver='auto', alpha=0.0))
    Y_pred = np.split(ridge_multi_regr.fit(x_train, y_train).predict(x_test), np.array(sections[n_train_samples:]), axis=0)

    #lr_multi_regr = MultiOutputRegressor(LinearRegression(fit_intercept=True, normalize=False, copy_X = True))
    #Y_pred = np.split(lr_multi_regr.fit(x_train, y_train).predict(x_test), np.array(pattern_lens[train_samples:-1]), axis=0)

    Y_test_mean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in y_test])
    print(Y_test_mean)

    Y_pred_mean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in Y_pred])
    print(Y_pred_mean)

    with np.errstate(divide='ignore', invalid='ignore'):
        cm = metrics.confusion_matrix(Y_test_mean, Y_pred_mean)
        cm_norm = np.diagonal(cm)/np.sum(cm, axis=1)
        perf = len(np.where(cm_norm > 0.9)[0])

        # print('\n---------' + str(perf))

    return perf


#%% --------------------------------------------------------------------------------------------------------------------
# GRAL METHODS
# ----------------------------------------------------------------------------------------------------------------------
def get_default_alpha_values(task):

    if (task == 'mem_cap') or (task == 'nonlin_cap'):
        alpha = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    elif (task == 'fcn_app'):
        alpha = [0.1, 0.3, 0.7, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    return alpha


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


def run_multiple_tasks(task, target, res_states, readout_nodes=None, include_alpha=None, **kwargs):
    """
        Given a target and a set of reservoir states(corresponding to different
        values of ALPHA), this method performs multiple trials (one for each
        ALPHA) of the task specified by 'task', and returns a PERF estimate
        across the different alpha values.
    """

    # all alpha values at which the network was simulated
    alpha = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    if include_alpha is None: include_alpha = get_default_alpha_values(task)

    # perform task on different reservoir states corresponding to different alpha values
    res = []
    for idx, x in enumerate(res_states):
        if alpha[idx] in include_alpha:
            # print('-----------------------------------------------------------')
            # print('alpha = ' + str(alpha[idx]))

            # define matrix X
            if readout_nodes is not None:
                x = x.squeeze()[:,readout_nodes]

            else:
                x = x.squeeze()

            # perform task
            if task == 'mem_cap':
               perf, task_params = run_memory_capacity(target, x, **kwargs)

            elif task == 'nonlin_cap':
               perf, task_params = run_nonlinearity_capacity(target, x, **kwargs)

            elif task == 'fcn_app':
               perf, task_params = run_fcn_approx(target, x, **kwargs)

            res.append(perf) # across task parameters

    return res, task_params, include_alpha # across task parameters and alpha values


def run_single_tasks(task, target, res_states, readout_nodes=None, **kwargs):
    """
        Given a target and reservoir states, this method performs the task
        specified by 'task', and returns a PERF estimate across alpha values.
    """

    # perform task on different reservoir states corresponding to different alpha values

    if readout_nodes is not None:
        x = res_states.squeeze()[:,readout_nodes]

    else:
        x = res_states.squeeze()

    if task == 'mem_cap':
       perf, params = run_memory_capacity(target, x, **kwargs)

    elif task == 'nonlin_cap':
       perf, params = run_nonlinearity_capacity(target, x, **kwargs)

    elif task == 'fcn_app':
       perf, params = run_fcn_approx(target, x, **kwargs)

    # elif task == 'ptn_rec':
    #     perf = run_pattrn_recog(target, x.squeeze()[:,readout_nodes], **kwargs)
    #     res.append(perf)

    return perf, params # across task parameters and alpha values


#%% --------------------------------------------------------------------------------------------------------------------
# GRAL METHODS
# ----------------------------------------------------------------------------------------------------------------------
def get_capacity_and_perf(task, performance, task_params, thres=0.9, normalize=False):
    """
        This method returns the parameters at which the best performance across
        different alpha values occurs.
    """
    # estimate capacity across task params per alpha value
    if (task == 'mem_cap') or (task == 'nonlin_cap'):
        if normalize: cap_per_alpha = [(task_params[perf<thres][np.argmax(perf[perf<thres])]-np.min(task_params))/(np.max(task_params)-np.min(task_params)) if (perf>thres).any() else 0 for perf in performance] #performance normalized in [0,1] range
        else: cap_per_alpha = [(task_params[perf<thres][np.argmax(perf[perf<thres])]) if (perf>thres).any() else 0 for perf in performance]

    elif (task == 'fcn_app'):
        param_tau, param_omega = task_params

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

    # estimate performance across task params per alpha value
    perf_per_alpha = np.array([np.sum(perf) for perf in performance])

    return perf_per_alpha, cap_per_alpha
