# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:10:14 2019

@author: Estefany Suarez
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['ps.usedistiller'] = 'xpdf'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.colors import (ListedColormap, Normalize)
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.patches as mpatches
import seaborn as sns

from . import plotting
from ..tasks import tasks

COLORS = sns.color_palette("husl", 8)
ENCODE_COL = '#E55FA3'
DECODE_COL = '#6CC8BA'


# --------------------------------------------------------------------------------------------------------------------
# GENERAL
# ----------------------------------------------------------------------------------------------------------------------
def sort_class_labels(class_labels):

    if 'subctx' in class_labels:
        rsn_labels = ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx']
        vEc_labels = ['PSS', 'PS', 'PM', 'LIM', 'AC1', 'IC', 'AC2', 'subctx']
    else:
        rsn_labels = ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN']
        vEc_labels = ['PSS', 'PS', 'PM', 'LIM', 'AC1', 'IC', 'AC2']

    if class_labels.all() in rsn_labels:
        return np.array([clase for clase in rsn_labels if clase in class_labels])

    elif class_labels.all() in vEc_labels:
        return np.array([clase for clase in vEc_labels if clase in class_labels])

    else:
        return class_labels


# --------------------------------------------------------------------------------------------------------------------
# TASKS RESULTS
# ----------------------------------------------------------------------------------------------------------------------
def concatenate_tsk_results(path, partition, coding, scores2return, include_alpha, n_samples=1000):

    df_scores = []
    df_avg_scores_per_class = []
    df_avg_scores_per_alpha = []
    for sample_id in range(n_samples):

        print(f'sample_id:  {sample_id}')
        succes_sample = True
        try:
            scores = pd.read_csv(os.path.join(path, f'{partition}_{coding}_score_{sample_id}.csv')).reset_index(drop=True)

        except:
            succes_sample = False
            print('\n Could not find sample No.  ' + str(sample_id))
            pass

        if succes_sample:

            # all scores (per alpha and per class)
            if 'scores' in scores2return:
                scores['coding'] = coding
                scores['sample_id'] = sample_id
                df_scores.append(scores[['sample_id', 'coding', 'class', 'alpha', 'performance', 'capacity', 'n_nodes']])

            # avg scores across alphas per class
            if 'avg_scores_per_class' in scores2return:
                avg_scores_per_class = get_avg_scores_per_class(scores.copy(),
                                                                include_alpha=include_alpha,
                                                                coding=coding
                                                                )

                avg_scores_per_class['coding'] = coding
                avg_scores_per_class['sample_id'] = sample_id
                df_avg_scores_per_class.append(avg_scores_per_class[['sample_id', 'coding', 'class', 'performance', 'capacity', 'n_nodes']])

            # avg scores across classes per alpha
            if 'avg_scores_per_alpha' in scores2return:
                avg_scores_per_alpha = get_avg_scores_per_alpha(scores.copy(),
                                                                include_alpha=None,
                                                                coding=coding
                                                                )

                avg_scores_per_alpha['coding'] = coding
                avg_scores_per_alpha['sample_id'] = sample_id
                df_avg_scores_per_alpha.append(avg_scores_per_alpha[['sample_id', 'coding', 'alpha', 'performance', 'capacity', 'n_nodes']])

    res_dict = {}
    if 'scores' in scores2return:
        df_scores = pd.concat(df_scores).reset_index(drop=True)
        res_dict['scores'] = df_scores

    if 'avg_scores_per_class' in scores2return:
        df_avg_scores_per_class = pd.concat(df_avg_scores_per_class).reset_index(drop=True)
        res_dict['avg_scores_per_class'] = df_avg_scores_per_class

    if 'avg_scores_per_alpha' in scores2return:
        df_avg_scores_per_alpha = pd.concat(df_avg_scores_per_alpha).reset_index(drop=True)
        res_dict['avg_scores_per_alpha'] = df_avg_scores_per_alpha

    return res_dict


def get_avg_scores_per_class(df_scores, include_alpha, coding='encoding'):

    if include_alpha is None: include_alpha = np.unique(df_scores['alpha'])

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))

    # filter scores by values of alpha
    df_scores = pd.concat([df_scores.loc[np.isclose(df_scores['alpha'], alpha), :] for alpha in include_alpha])

    # average scores across alphas per class
    avg_scores = []
    for clase in class_labels:
        tmp = df_scores.loc[df_scores['class'] == clase, ['performance', 'capacity', 'n_nodes']]\
                       .reset_index(drop=True)
        avg_scores.append(tmp.mean())
    avg_scores = pd.concat(avg_scores, axis=1).T

    # dataFrame with avg coding scores per class
    df_avg_scores = pd.DataFrame(data = np.column_stack((class_labels, avg_scores)),
                                 columns = ['class', 'performance', 'capacity', 'n_nodes'],
                                 ).reset_index(drop=True)

    df_avg_scores['performance']  = df_avg_scores['performance'].astype('float')
    df_avg_scores['capacity']     = df_avg_scores['capacity'].astype('float')
    df_avg_scores['n_nodes']      = df_avg_scores['n_nodes'].astype('float').astype('int')

    return df_avg_scores


def get_avg_scores_per_alpha(df_scores, include_alpha, coding='encoding'):

    if include_alpha is None: include_alpha = np.unique(df_scores['alpha'])

    # average scores across alphas per class
    avg_scores = []
    for alpha in include_alpha:
        tmp = df_scores.loc[np.isclose(df_scores['alpha'], alpha), ['performance', 'capacity', 'n_nodes']]\
                       .reset_index(drop=True)
        avg_scores.append(tmp.mean())
    avg_scores = pd.concat(avg_scores, axis=1).T

    # dataFrame with avg coding scores per class
    df_avg_scores = pd.DataFrame(data = np.column_stack((include_alpha, avg_scores)),
                                 columns = ['alpha', 'performance', 'capacity', 'n_nodes'],
                                ).reset_index(drop=True)

    df_avg_scores['performance']  = df_avg_scores['performance'].astype('float')
    df_avg_scores['capacity']     = df_avg_scores['capacity'].astype('float')
    df_avg_scores['n_nodes']      = df_avg_scores['n_nodes'].astype('float').astype('int')

    return df_avg_scores


# --------------------------------------------------------------------------------------------------------------------
# NETWORK PROPERTIES RESULTS
# ----------------------------------------------------------------------------------------------------------------------
