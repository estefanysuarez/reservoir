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
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import seaborn as sns

from . import plotting
from ..tasks import tasks


COLORS = sns.color_palette("husl", 8)
ENCODE_COL = '#E55FA3'
DECODE_COL = '#6CC8BA'


# --------------------------------------------------------------------------------------------------------------------
# GENERAL
# ----------------------------------------------------------------------------------------------------------------------
def concat_results(path, class_name, n_samples=1000, scores2return=None, include_alpha=None):

    df_encoding = []
    df_decoding = []
    df_scores   = []
    for sample_id in range(n_samples):

        print('\n sample_id:  ' + str(sample_id))
        succes_sample = True

        try:
            encod_scores = pd.read_csv(os.path.join(path, class_name + '_encoding_score_' + str(sample_id) + '.csv')).reset_index(drop=True)
            encod_scores['coding'] = 'encoding'
            encod_scores['sample_id'] = sample_id
            if 'cluster' in encod_scores.columns: encod_scores.rename(columns={'cluster':'class'}, inplace=True, errors='raise')

            decod_scores = pd.read_csv(os.path.join(path, class_name + '_decoding_score_' + str(sample_id) + '.csv')).reset_index(drop=True)
            decod_scores['coding'] = 'decoding'
            decod_scores['sample_id'] = sample_id
            if 'cluster' in decod_scores.columns: decod_scores.rename(columns={'cluster': 'class'}, inplace=True, errors='raise')

            if 'scores' in scores2return:
                scores = get_coding_scores_per_class(encod_scores.copy(),
                                                     decod_scores.copy(),
                                                     include_alpha=include_alpha,
                                                     )
                scores['sample_id'] = sample_id

        except:
            succes_sample = False
            print('Could not find sample No.  ' + str(sample_id))

            pass

        if succes_sample:

            if 'encoding' in scores2return:
                df_encoding.append(encod_scores[['sample_id', 'class', 'alpha', 'coding', 'performance', 'capacity', 'n_nodes']])

            if 'decoding' in scores2return:
                df_decoding.append(decod_scores[['sample_id', 'class', 'alpha', 'coding', 'performance', 'capacity', 'n_nodes']])

            if 'scores' in scores2return:
                df_scores.append(scores[['sample_id', 'class', 'coding', 'performance', 'capacity', 'n_nodes']])

    res_dict = {}
    if 'encoding' in scores2return:
        df_encoding = pd.concat(df_encoding).reset_index(drop=True)
        res_dict['encoding'] = df_encoding

    if 'decoding' in scores2return:
        df_decoding = pd.concat(df_decoding).reset_index(drop=True)
        res_dict['decoding'] = df_decoding

    if 'scores' in scores2return:
        df_scores = pd.concat(df_scores).reset_index(drop=True)
        res_dict['scores'] = df_scores

    return res_dict


def get_coding_scores_per_class(df_encoding, df_decoding, include_alpha=None):

    # get class labels
    class_labels = sort_class_labels(np.unique(df_encoding['class']))

    # estimate avg/sum of scores per class
    encode_scores = []
    decode_scores = []
    for clase in class_labels:

        # filter coding scores with alpha values
        if include_alpha is None:
            tmp_encode_scores = df_encoding.loc[df_encoding['class'] == clase, ['performance', 'capacity', 'n_nodes']]
            tmp_decode_scores = df_decoding.loc[df_decoding['class'] == clase, ['performance', 'capacity', 'n_nodes']]

        else:
            tmp_encode_scores = df_encoding.loc[(df_encoding['class'] == clase) & (df_encoding['alpha'].isin(include_alpha)), ['performance', 'capacity', 'n_nodes']]
            tmp_decode_scores = df_decoding.loc[(df_decoding['class'] == clase) & (df_decoding['alpha'].isin(include_alpha)), ['performance', 'capacity', 'n_nodes']]

        # estimate avg/sum coding scores across alpha values per class
        encode_scores.append(tmp_encode_scores.mean())
        decode_scores.append(tmp_decode_scores.mean())

    encode_scores = pd.concat(encode_scores, axis=1).T
    decode_scores = pd.concat(decode_scores, axis=1).T

    # DataFrame with avg/sum coding scores per class
    df_encode = pd.DataFrame(data = np.column_stack((class_labels, encode_scores)),
                             columns = ['class', 'performance', 'capacity', 'n_nodes'],
                             index = np.arange(len(class_labels))
                             )
    df_encode['coding'] = 'encoding'

    df_decode = pd.DataFrame(data = np.column_stack((class_labels, decode_scores)),
                             columns = ['class', 'performance', 'capacity', 'n_nodes'],
                             index = np.arange(len(class_labels))
                            )
    df_decode['coding'] = 'decoding'

    df_scores = pd.concat([df_encode, df_decode])
    df_scores['performance']  = df_scores['performance'].astype('float')
    df_scores['capacity']  = df_scores['capacity'].astype('float')
    df_scores['n_nodes']  = df_scores['n_nodes'].astype('float').astype('int')

    return df_scores


def merge_scores(df_scores):

    df_encoding_scores = df_scores.loc[df_scores['coding'] == 'encoding', :] \
                         .rename(columns={'performance':'encoding_performance', 'capacity':'encoding_capacity'}).reset_index(drop=True)

    df_encoding_scores.fillna({'encoding_performance':np.nanmean(df_encoding_scores['encoding_performance']), \
                               'encoding_capacity':np.nanmean(df_encoding_scores['encoding_capacity'])}, \
                                inplace=True)

    df_decoding_scores = df_scores.loc[df_scores['coding'] == 'decoding', :] \
                         .rename(columns={'performance':'decoding_performance', 'capacity':'decoding_capacity'}).reset_index(drop=True)

    df_decoding_scores.fillna({'decoding_performance':np.nanmean(df_decoding_scores['decoding_performance']), \
                               'decoding_capacity':np.nanmean(df_decoding_scores['decoding_capacity'])}, \
                               inplace=True)


    merge_columns   = list(np.intersect1d(df_encoding_scores.columns, df_decoding_scores.columns))
    df_merge_scores = pd.merge(df_encoding_scores, df_decoding_scores, on=merge_columns, left_index=True, right_index=True).reset_index(drop=True)
    df_merge_scores.drop(columns={'coding'})

    df_merge_scores['coding_performance'] = (df_merge_scores['encoding_performance'] - df_merge_scores['decoding_performance']).astype(float)
    df_merge_scores['coding_capacity']    = (df_merge_scores['encoding_capacity']   - df_merge_scores['decoding_capacity']).astype(float)

    df_merge_scores = df_merge_scores[['sample_id', 'class',
                                       'encoding_performance', 'decoding_performance', 'coding_performance', \
                                       'encoding_capacity', 'decoding_capacity', 'coding_capacity', \
                                       'n_nodes', 'analysis']]

    return df_merge_scores


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
# PERFORMANCE ACROSS ALPHAS
# ----------------------------------------------------------------------------------------------------------------------
def lineplot_scores_vs_alpha(df_encoding, df_decoding, score, include_alpha=None, scale=True, minmax=None, **kwargs):

    if include_alpha is not None:
        df_encoding = df_encoding.loc[df_encoding['alpha'].isin(include_alpha), :].reset_index(drop=True)
        df_decoding = df_decoding.loc[df_decoding['alpha'].isin(include_alpha), :].reset_index(drop=True)

    if scale:
        if minmax is None:
            min_score = min(np.min(df_encoding[score]), np.min(df_decoding[score]))
            max_score = max(np.max(df_encoding[score]), np.max(df_decoding[score]))
        else:
            min_score = minmax[0]
            max_score = minmax[1]

        df_encoding[score] = (df_encoding[score]-min_score)/(max_score-min_score)
        df_decoding[score] = (df_decoding[score]-min_score)/(max_score-min_score)

    # ------------------------------ AUC ---------------------------------------
    auc_enc = []
    auc_dec = []
    try:
        for clase in sort_class_labels(np.unique(df_encoding['class'])):

                # print(' clase: ' + clase)
                tmp_df_enc = df_encoding.loc[df_encoding['class'] == clase, [score]][score]
                # print('   auc_enc:  ' + str(auc(x=include_alpha, y=tmp_df_enc)))
                auc_enc.append(auc(x=include_alpha, y=tmp_df_enc))

                tmp_df_dec = df_decoding.loc[df_decoding['class'] == clase, [score]][score]
                # print('   auc_dec:  ' + str(auc(x=include_alpha, y=tmp_df_dec)))
                auc_dec.append(auc(x=include_alpha, y=tmp_df_dec))

    except:
        pass

    if 'sample_id' in df_encoding.columns:
        title_encoding = f'encoding {score}'
        title_decoding = f'decoding {score}'
    else:
        auc_enc = np.sum(auc_enc)
        auc_dec = np.sum(auc_dec)
        title_encoding = f'encoding  {score} - AUC = {auc_enc}' # + str(np.sum(auc_enc))[:5]
        title_decoding = f'decoding  {score} - AUC = {auc_dec}' # + str(np.sum(auc_dec))[:5]
    # --------------------------------------------------------------------------

    # ----------------------------------------------------
    plotting.lineplot(x='alpha', y=score,
                      df=df_encoding,
                      palette=COLORS[:-1],
                      hue='class',
                      hue_order=sort_class_labels(np.unique(df_encoding['class'])),
                      title=title_encoding,
                      # ci='sd',
                      fig_name='encod_vs_alpha',
                      **kwargs
                      )

    plotting.lineplot(x='alpha', y=score,
                      df=df_decoding,
                      palette=COLORS[:-1],
                      hue='class',
                      hue_order=sort_class_labels(np.unique(df_decoding['class'])),
                      title=title_decoding,
                      # ci='sd',
                      fig_name='decod_vs_alpha',
                      **kwargs
                      )


def boxplot_scores_vs_class_per_alpha(df_encoding, df_decoding, score, include_alpha=None, scale=True, minmax=None, **kwargs):

    if include_alpha is not None:
        df_encoding = df_encoding.loc[df_encoding['alpha'].isin(include_alpha), :].reset_index(drop=True)
        df_decoding = df_decoding.loc[df_decoding['alpha'].isin(include_alpha), :].reset_index(drop=True)

    if scale:
        if minmax is None:
            min_score = min(np.min(df_encoding[score]), np.min(df_decoding[score]))
            max_score = max(np.max(df_encoding[score]), np.max(df_decoding[score]))
        else:
            min_score = minmax[0]
            max_score = minmax[1]

        df_encoding[score] = (df_encoding[score]-min_score)/(max_score-min_score)
        df_decoding[score] = (df_decoding[score]-min_score)/(max_score-min_score)

    # ----------------------------------------------------------------------
    for alpha in np.unique(df_encoding['alpha']):

        tmp_encoding = df_encoding.loc[df_encoding['alpha'] == alpha, :].reset_index(drop=True)
        plotting.boxplot(x='class', y=score,
                         df=tmp_encoding,
                         palette=COLORS[:-1],
                         title=f'encoding {score} - alpha: {alpha}',
                         fig_name=f'enc_vs_class_{alpha}',
                         **kwargs
                         )

        tmp_decoding = df_decoding.loc[df_decoding['alpha'] == alpha, :].reset_index(drop=True)
        plotting.boxplot(x='class', y=score,
                         df=tmp_decoding,
                         palette=COLORS[:-1],
                         title=f'decoding {score} - alpha: {alpha}',
                         fig_name=f'dec_vs_class_{alpha}',
                         **kwargs
                         )


# --------------------------------------------------------------------------------------------------------------------
# PERFORMANCE ACROSS ALPHAS
# ----------------------------------------------------------------------------------------------------------------------
def boxplot_scores_vs_alpha(df_encoding, df_decoding, score, include_alpha=None, scale=True, minmax=None, **kwargs):

    if include_alpha is None: include_alpha = np.unique(df_encoding['alpha'])
    df_encoding = df_encoding.loc[df_encoding['alpha'].isin(include_alpha), :].reset_index(drop=True)
    df_decoding = df_decoding.loc[df_decoding['alpha'].isin(include_alpha), :].reset_index(drop=True)

    if minmax is None:
        min_score = min(np.min(df_encoding[score]), np.min(df_decoding[score]))
        max_score = max(np.max(df_encoding[score]), np.max(df_decoding[score]))
    else:
        min_score = minmax[0]
        max_score = minmax[1]

    if scale:
        df_encoding[score] = (df_encoding[score]-min_score)/(max_score-min_score)
        df_decoding[score] = (df_decoding[score]-min_score)/(max_score-min_score)
        ylim = (0,1.1)
    else:
        ylim=None

    # ----------------------------------------------------------------------
    for clase in sort_class_labels(np.unique(df_encoding['class'])):

        tmp_encoding = df_encoding.loc[df_encoding['class'] == clase, :].reset_index(drop=True)
        tmp_decoding = df_decoding.loc[df_decoding['class'] == clase, :].reset_index(drop=True)
        tmp = pd.concat((tmp_encoding, tmp_decoding)).reset_index(drop=True)

        plotting.boxplot(x='alpha', y=score,
                         df=tmp,
                         palette={'encoding':ENCODE_COL, 'decoding':DECODE_COL},
                         hue='coding',
                         hue_order=['encoding', 'decoding'],
                         title=clase,
                         # xlim=(-1.0, 3.5), x_major_loc=0.5,
                         ylim=ylim, y_major_loc=0.2,
                         legend=True, figsize=(12,5),
                         fig_name=f'bx_encod_decode_vs_alpha_{clase}',
                         **kwargs
                         )


def lineplot_scores_vs_alpha_per_class(df_encoding, df_decoding, score, include_alpha=None, scale=True, minmax=None, **kwargs):

    if include_alpha is not None:
        df_encoding = df_encoding.loc[df_encoding['alpha'].isin(include_alpha), :].reset_index(drop=True)
        df_decoding = df_decoding.loc[df_decoding['alpha'].isin(include_alpha), :].reset_index(drop=True)

    if minmax is None:
        min_score = min(np.min(df_encoding[score]), np.min(df_decoding[score]))
        max_score = max(np.max(df_encoding[score]), np.max(df_decoding[score]))
    else:
        min_score = minmax[0]
        max_score = minmax[1]

    if scale:
        df_encoding[score] = (df_encoding[score]-min_score)/(max_score-min_score)
        df_decoding[score] = (df_decoding[score]-min_score)/(max_score-min_score)

    # ----------------------------------------------------------------------
    for clase in sort_class_labels(np.unique(df_encoding['class'])):

        tmp_encoding = df_encoding.loc[df_encoding['class'] == clase, :].reset_index(drop=True)
        tmp_decoding = df_decoding.loc[df_decoding['class'] == clase, :].reset_index(drop=True)
        tmp = pd.concat((tmp_encoding, tmp_decoding)).reset_index(drop=True)

        plotting.lineplot(x='alpha', y=score,
                          df=tmp,
                          palette={'encoding':ENCODE_COL, 'decoding':DECODE_COL},
                          hue='coding',
                          hue_order=None,
                          title=clase,
                          ci='sd',
                          err_style='band',
                          figsize=(12,5),
                          fig_name=f'ln_encod_decode_vs_alpha_{clase}',
                          **kwargs
                          )



# --------------------------------------------------------------------------------------------------------------------
# SINGLE AND MULTIPLE SAMPLES
# ----------------------------------------------------------------------------------------------------------------------
def lineplot_enc_vs_dec(df_scores, score, hue='coding', hue_order=None, scale=True, minmax=None, **kwargs):

    if minmax is None:
        min_score = np.min(df_scores[score])
        max_score = np.max(df_scores[score])
    else:
        min_score = minmax[0]
        max_score = minmax[1]

    if scale:
        df_scores[score] = (df_scores[score]-min_score)/(max_score-min_score)
        ylim = (0,1)
    else:
        ylim = None

    if hue == 'coding':
        hue_order = ['encoding', 'decoding']
        palette = {'encoding':ENCODE_COL, 'decoding':DECODE_COL}
    elif (hue == 'rsn' or hue == 'cyt') and (hue_order is None):
        hue_order = sort_class_labels(np.unique(df_scores[hue]))
        palette = COLORS[:-1]
    elif (hue == 'rsn') or (hue == 'cyt'):
        class_labels = sort_class_labels(np.unique(df_scores[hue]))
        palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in hue_order])

    # ----------------------------------------------------
    plotting.lineplot(x='class', y=score,
                      df=df_scores,
                      palette=palette,
                      hue=hue,
                      hue_order=hue_order,
                      ylim=ylim,
                      **kwargs
                      )


def lineplot_coding_scores(df_scores, score, hue='class', hue_order=None, scale=True, minmax=None, **kwargs):

    if minmax is None:
        min_score = np.min(df_scores[score])
        max_score = np.max(df_scores[score])
    else:
        min_score = minmax[0]
        max_score = minmax[1]

    if scale:
        df_scores[score] = (df_scores[score]-min_score)/(max_score-min_score)

    df_scores = merge_coding_scores(df_scores, score)

    if hue_order is not None:
        class_labels = sort_class_labels(np.unique(df_scores['class']))
        palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in hue_order])
    else:
        palette = COLORS[:-1]

    # ----------------------------------------------------
    plotting.lineplot(x='class', y='coding_' + score,
                      df=df_scores,
                      palette=palette,
                      hue=hue,
                      hue_order=hue_order,
                      err_style='bars',
                      ci='sd',
                      legend=False,
                      **kwargs
                     )


def scatterplot_enc_vs_dec(df_scores, score, norm_score_by=None, scale=True, minmax=None, hue='class', **kwargs):
    """
        Scatter plot (avg across alphas) encoding/decoding score vs (avg across nodes) network property
    """
    df = merge_coding_scores(df_scores, score)

    if norm_score_by is not None:

        # divide coding scores by degree
        # df['encoding_' + score] = df['encoding_' + score]/df[norm_score_by]
        # df['decoding_' + score] = df['decoding_' + score]/df[norm_score_by]

        # regress out degree from coding scores
        X = np.array(df[norm_score_by])[:, np.newaxis]

        reg_enc = LinearRegression().fit(X, y=df['encoding_' + score])
        tmp_encode_scores = df['encoding_' + score] - reg_enc.predict(X)
        df['encoding_' + score] = tmp_encode_scores

        reg_dec = LinearRegression().fit(X, y=df['decoding_' + score])
        tmp_decode_scores = df['decoding_' + score] - reg_dec.predict(X)
        df['decoding_' + score] = tmp_decode_scores

    if minmax is None:
        max_score = max(np.max(df['encoding_' + score]), np.max(df['decoding_' + score]))
        min_score = min(np.min(df['encoding_' + score]), np.min(df['decoding_' + score]))
    else:
        min_score = minmax[0]
        max_score = minmax[1]

    if scale:
        df['encoding_' + score] = ((df['encoding_' + score]-min_score)/(max_score-min_score))
        df['decoding_' + score] = ((df['decoding_' + score]-min_score)/(max_score-min_score))

        xlim = (0,1)
        ylim = (0,1)
    else:
        xlim = None
        ylim = None

    # ----------------------------------------------------
    plotting.scatterplot(x='decoding_' + score,
                         y='encoding_' + score,
                         df=df,
                         palette=COLORS[:-1],
                         hue=hue,
                         **kwargs
                        )


# --------------------------------------------------------------------------------------------------------------------
# MULTIPLE SAMPLES
# --------------------------------------------------------------------------------------------------------------------
def boxplot_enc_vs_dec(df_scores, score, class_type='class', order=None, scale=True, minmax=None, **kwargs):

    if minmax is None:
        min_score = np.min(df_scores[score])
        max_score = np.max(df_scores[score])
    else:
        min_score = minmax[0]
        max_score = minmax[1]

    if scale:
        df_scores[score] = (df_scores[score]-min_score)/(max_score-min_score)
        ylim = (0,1)
    else:
        ylim=None

    # ----------------------------------------------------
    plotting.boxplot(x=class_type, y=score,
                     df=df_scores,
                     hue='coding',
                     order=order,
                     palette={'encoding':ENCODE_COL, 'decoding':DECODE_COL},
                     ylim=ylim,
                     **kwargs
                     )


def boxplot_coding_scores(df_scores, score, order=None, scale=True, minmax=None, **kwargs):

    if minmax is None:
        min_score = np.min(df_scores[score])
        max_score = np.max(df_scores[score])
    else:
        min_score = minmax[0]
        max_score = minmax[1]

    if scale:
        df_scores[score] = (df_scores[score]-min_score)/(max_score-min_score)

    df_scores = merge_coding_scores(df_scores, score)

    if order is not None:
        class_labels = sort_class_labels(np.unique(df_scores['class']))
        palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in order])
    else:
        palette = COLORS[:-1]

    # ----------------------------------------------------
    plotting.boxplot(x='class', y='coding_' + score,
                     df=df_scores,
                     order=order,
                     palette=palette,
                     legend=False,
                     **kwargs
                     )


def jointplot_enc_vs_dec(df_scores, score, hue_order=None, kind='scatter', norm_score_by=None, scale=True, minmax=None, draw_line=True, **kwargs):

    df = merge_coding_scores(df_scores, score)

    if norm_score_by is not None:

        # divide coding scores by degree
        # df['encoding_' + score] = df['encoding_' + score]/df[norm_score_by]
        # df['decoding_' + score] = df['decoding_' + score]/df[norm_score_by]

        # regress out degree from coding scores
        X = np.array(df[norm_score_by])[:, np.newaxis]

        reg_enc = LinearRegression().fit(X, y=df['encoding_' + score])
        tmp_encode_scores = df['encoding_' + score] - reg_enc.predict(X)
        df['encoding_' + score] = tmp_encode_scores

        reg_dec = LinearRegression().fit(X, y=df['decoding_' + score])
        tmp_decode_scores = df['decoding_' + score] - reg_dec.predict(X)
        df['decoding_' + score] = tmp_decode_scores

    if minmax is None:
        max_score = max(np.max(df['encoding_' + score]), np.max(df['decoding_' + score]))
        min_score = min(np.min(df['encoding_' + score]), np.min(df['decoding_' + score]))
    else:
        min_score = minmax[0]
        max_score = minmax[1]

    if scale:
        df['encoding_' + score] = ((df['encoding_' + score]-min_score)/(max_score-min_score))
        df['decoding_' + score] = ((df['decoding_' + score]-min_score)/(max_score-min_score))

        xlim = (0,1)
        ylim = (0,1)
    else:
        xlim = None
        ylim = None

    # ------------------------------
    sns.set(style="ticks", font_scale=2.0)

    class_labels = sort_class_labels(np.unique(df['class']))

    tmp = df.loc[df['class'] == class_labels[0], :]

    g = sns.JointGrid(x=tmp['decoding_' + score].values,
                      y=tmp['encoding_' + score].values,
                      dropna=True,
                      height=10,
                      ratio=7,
                      # space=,
                      xlim=xlim,
                      ylim=ylim
                      )

    if kind == 'kdeplot':
        g.plot_joint(sns.kdeplot, color=COLORS[0], shade=True, shade_lowest=False, **kwargs)#, label=class_labels[0], legend=False)
    elif kind == 'scatter':
        g.plot_joint(sns.scatterplot, color=COLORS[0], **kwargs)#, label=class_labels[0], legend=False)

    g.plot_marginals(sns.distplot, hist=False, kde=True, kde_kws={"shade": True}, color=COLORS[0], **kwargs)

    for i, clase in enumerate(class_labels[1:]):

        tmp = df.loc[df['class'] == clase, :]
        g.x = tmp['decoding_' + score].values
        g.y = tmp['encoding_' + score].values

        if kind == 'kdeplot':
            g.plot_joint(sns.kdeplot, color=COLORS[i+1], shade=True, shade_lowest=False, **kwargs)#, label=clase, legend=False)
        elif kind == 'scatter':
            g.plot_joint(sns.scatterplot, color=COLORS[i+1], **kwargs)#, label=class_labels[0], legend=False)

        g.plot_marginals(sns.distplot, hist=False, kde=True, kde_kws={"shade": True}, color=COLORS[i+1])

    g.ax_joint.set_xlabel('decoding  ' + score)
    g.ax_joint.set_ylabel('encoding  ' + score)

    #g.ax_joint.get_legend().remove()
    #g.ax_joint.legend(fontsize=10, frameon=False, ncol=1, loc='lower right', title='rsn')
    #plt.legend(fontsize=10, frameon=False, ncol=1)#, loc='lower right')

    if draw_line:
        g.x = [0.05,0.95]
        g.y = [0.05,0.95]
        g.plot_joint(sns.lineplot, color='dimgrey', dashes=(5, 1))


# --------------------------------------------------------------------------------------------------------------------
# VALIDATION: STATISTICAL TESTS
# ----------------------------------------------------------------------------------------------------------------------
def ttest(df_scores, score, fdr_correction=True, test='2samp'):

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))

    pval = []
    tstat = []
    for clase in class_labels:
        encod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'encoding'), :][score].values
        decod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'decoding'), :][score].values

        if test == '1samp': t, p = stats.ttest_1samp(encod_scores-decod_scores, popmean=0.0)
        elif test == '2samp': t, p = stats.ttest_ind(encod_scores, decod_scores, equal_var=False)
            # t, p = stats.ttest_rel(encod_scores, decod_scores)

        pval.append(p)
        tstat.append(t)

    if fdr_correction: pval = multipletests(pval, 0.05, 'bonferroni')[1]

    return tstat, pval


def effect_size(df_scores, score, test='2samp'):

    def cohen_d_1samp(x, mu=0.0):
        return (np.mean(x) - mu) / np.std(x)

    def cohen_d_2samp(x,y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))

    size_effect = []
    for clase in class_labels:
        encod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'encoding'),:][score].values
        decod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'decoding'),:][score].values

        if test == '1samp': size_effect.append(cohen_d_1samp(encod_scores-decod_scores))
        elif test == '2samp': size_effect.append(cohen_d_2samp(encod_scores, decod_scores))

    return size_effect


def barplot_eff_size(df_scores, score, test='2samp', hue_order=None):

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))
    eff_size  = effect_size(df_scores, score, test)

    df_eff_size = pd.DataFrame(data = np.column_stack((class_labels, eff_size)),
                               columns = ['class', 'effect_size'],
                               index = np.arange(len(class_labels))
                               )
    df_eff_size['effect_size'] = df_eff_size['effect_size'].astype('float')

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(10,10))
    ax = plt.subplot(111)

    if hue_order is not None:
        class_labels = sort_class_labels(np.unique(df_scores['class']))
        palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in hue_order])
    else: palette = COLORS[:-1]

    sns.barplot(x='class',
                y='effect_size',
                data=df_eff_size,
                order=hue_order,
                palette=palette,
                orient='v',
                )

    sns.despine(offset=10, trim=True)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/barplot_effect_size.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/barplot_effect_size.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


# --------------------------------------------------------------------------------------------------------------------
# OTHERS
# ----------------------------------------------------------------------------------------------------------------------
def jointplot_enc_vs_dec_indiv(df_scores, score, norm_score_by=None, scale=True, minmax=None, draw_line=True, **kwargs):

    df = merge_coding_scores(df_scores, score)

    if norm_score_by is not None:

        # divide coding scores by degree
        # df['encoding_' + score] = df['encoding_' + score]/df[norm_score_by]
        # df['decoding_' + score] = df['decoding_' + score]/df[norm_score_by]

        # regress out degree from coding scores
        X = np.array(df[norm_score_by])[:, np.newaxis]

        reg_enc = LinearRegression().fit(X, y=df['encoding_' + score])
        tmp_encode_scores = df['encoding_' + score] - reg_enc.predict(X)
        df['encoding_' + score] = tmp_encode_scores

        reg_dec = LinearRegression().fit(X, y=df['decoding_' + score])
        tmp_decode_scores = df['decoding_' + score] - reg_dec.predict(X)
        df['decoding_' + score] = tmp_decode_scores

    if minmax is None:
        max_score = max(np.max(df['encoding_' + score]), np.max(df['decoding_' + score]))
        min_score = min(np.min(df['encoding_' + score]), np.min(df['decoding_' + score]))
    else:
        min_score = minmax[0]
        max_score = minmax[1]

    if scale:
        df['encoding_' + score] = ((df['encoding_' + score]-min_score)/(max_score-min_score))
        df['decoding_' + score] = ((df['decoding_' + score]-min_score)/(max_score-min_score))

        xlim = (0,1)
        ylim = (0,1)
    else:
        xlim = None
        ylim = None

    # ------------------------------
    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))
    for i, clase in enumerate(class_labels):

        tmp = df.loc[df['class'] == clase, :]
        g = sns.JointGrid(x='decoding_' + score,
                          y='encoding_' + score,
                          data=tmp,
                          dropna=True,
                          height=8,
                          # ratio=,
                          # space=,
                          xlim=xlim,
                          ylim=ylim
                          )

        g = g.plot_joint(sns.scatterplot, color=COLORS[i])
        g = g.plot_marginals(sns.kdeplot, shade=True, color=COLORS[i])

        if draw_line:
            g.x = [0.05,0.95]
            g.y = [0.05,0.95]
            g.plot_joint(sns.lineplot, color='dimgrey', dashes=(5, 1))

    plt.show()
    plt.close()
