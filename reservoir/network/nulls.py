# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:27:07 2019

@author: Estefany Suarez
"""

import os
import numpy as np
from bct import reference
from scipy.spatial.distance import cdist

def create_null_model(connectome, swaps):
    """

    Parameters
    ----------
    directed binary/weighted connectome: nodes x nodes binary connectivity
    matrix

    Returns
    -------
    conn_mat: null model
    """

    if check_symmetric(connectome):
        conn_mat, _ = reference.randmio_und_connected(connectome, swaps)

    else:
        conn_mat, _ = reference.randmio_dir_connected(connectome, swaps)

    return conn_mat


def check_symmetric(a, tol=1e-16):
    return np.allclose(a, a.T, atol=tol)
