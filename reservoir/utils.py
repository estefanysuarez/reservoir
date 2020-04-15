# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:10:14 2019

@author: Estefany Suarez
"""

import numpy as np
from sklearn.linear_model import LinearRegression

#%% --------------------------------------------------------------------------------------------------------------------
# DATA STATISTICS
# ----------------------------------------------------------------------------------------------------------------------
def regress(X, y, regress=True):

    if regress:
        X = np.array(X)[:, np.newaxis]
        reg = LinearRegression().fit(X, y)
        return y - reg.predict(X)

    else:
        return y/X


def minmax_scale(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def log_minmax_scale(x):
    return np.log(((x-np.min(x))/(np.max(x)-np.min(x))+1))


#%% --------------------------------------------------------------------------------------------------------------------
# PLOTTING`
# ----------------------------------------------------------------------------------------------------------------------
def array2cmap(X):

    N = X.shape[0]
    r = np.linspace(0., 1., N+1)
    r = np.sort(np.concatenate((r, r)))[1:-1]
    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in range(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in range(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in range(N)])
    rd = tuple([(r[i], rd[i], rd[i]) for i in range(2 * N)])
    gr = tuple([(r[i], gr[i], gr[i]) for i in range(2 * N)])
    bl = tuple([(r[i], bl[i], bl[i]) for i in range(2 * N)])
    cdict = {'red': rd, 'green': gr, 'blue': bl}
    return mcolors.LinearSegmentedColormap('my_colormap', cdict, N)
