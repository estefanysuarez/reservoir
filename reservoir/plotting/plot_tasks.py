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


def merge_coding_scores(df_scores, score):

    if 'sample_id' in df_scores.columns:

        if ('rsn' in df_scores.columns) and ('cyt' in df_scores.columns):
            df_encoding_scores = df_scores.loc[df_scores['coding'] == 'encoding', ['sample_id', 'class', 'n_nodes', 'rsn', 'cyt', score]]\
                                 .rename(columns={score:'encoding_' + score})
            df_encoding_scores.fillna({'encoding_' + score:np.nanmean(df_encoding_scores['encoding_' + score])}, inplace=True)

            df_decoding_scores = df_scores.loc[df_scores['coding'] == 'decoding', ['sample_id', 'class', 'n_nodes', 'rsn', 'cyt', score]]\
                                 .rename(columns={score:'decoding_' + score})
            df_decoding_scores.fillna({'decoding_' + score:np.nanmean(df_decoding_scores['decoding_' + score])}, inplace=True)

            new_df_scores = pd.merge(df_encoding_scores, df_decoding_scores, on=['sample_id', 'class', 'n_nodes', 'rsn', 'cyt'])

        else:
            # get encoding - decoding scores
            df_encoding_scores = df_scores.loc[df_scores['coding'] == 'encoding', ['sample_id', 'class', 'n_nodes', score]]\
                                .rename(columns={score:'encoding_' + score})
            df_encoding_scores.fillna({'encoding_' + score:np.nanmean(df_encoding_scores['encoding_' + score])}, inplace=True)

            df_decoding_scores = df_scores.loc[df_scores['coding'] == 'decoding', ['sample_id', 'class', 'n_nodes', score]]\
                                .rename(columns={score:'decoding_' + score})
            df_decoding_scores.fillna({'decoding_' + score:np.nanmean(df_decoding_scores['decoding_' + score])}, inplace=True)

            new_df_scores = pd.merge(df_encoding_scores, df_decoding_scores, on=['sample_id', 'class', 'n_nodes'])

    else:
        if ('rsn' in df_scores.columns) and ('cyt' in df_scores.columns):
            df_encoding_scores = df_scores.loc[df_scores['coding'] == 'encoding', ['class',  'n_nodes', 'rsn', 'cyt', score]]\
                                 .rename(columns={score:'encoding_' + score})
            df_encoding_scores.fillna({'encoding_' + score:np.nanmean(df_encoding_scores['encoding_' + score])}, inplace=True)

            df_decoding_scores = df_scores.loc[df_scores['coding'] == 'decoding', ['class', 'n_nodes', 'rsn', 'cyt', score]]\
                                 .rename(columns={score:'decoding_' + score})
            df_decoding_scores.fillna({'decoding_' + score:np.nanmean(df_decoding_scores['decoding_' + score])}, inplace=True)

            new_df_scores = pd.merge(df_encoding_scores, df_decoding_scores, on=['class', 'n_nodes', 'rsn', 'cyt'])

        else:
            # get encoding - decoding scores
            df_encoding_scores = df_scores.loc[df_scores['coding'] == 'encoding', ['class', 'n_nodes', score]]\
                                .rename(columns={score:'encoding_' + score})
            df_encoding_scores.fillna({'encoding_' + score:np.nanmean(df_encoding_scores['encoding_' + score])}, inplace=True)

            df_decoding_scores = df_scores.loc[df_scores['coding'] == 'decoding', ['class', 'n_nodes', score]]\
                                .rename(columns={score:'decoding_' + score})
            df_decoding_scores.fillna({'decoding_' + score:np.nanmean(df_decoding_scores['decoding_' + score])}, inplace=True)

            new_df_scores = pd.merge(df_encoding_scores, df_decoding_scores, on=['class', 'n_nodes'])

    new_df_scores['coding_' + score] = new_df_scores['encoding_' + score] - new_df_scores['decoding_' + score]
    new_df_scores['coding_' + score] = new_df_scores['coding_' + score].astype(float)

    return new_df_scores


# --------------------------------------------------------------------------------------------------------------------
# SINGLE AND MULTIPLE SAMPLES
# ----------------------------------------------------------------------------------------------------------------------
def lineplot_enc_vs_dec(task_name, df_scores, score, hue='coding', hue_order=None, scale=True, **kwargs):

    if scale:
        # estimate global min and max to scale scores
        max_score = np.max(df_scores[score])
        min_score = np.min(df_scores[score])
        df_scores[score] = (df_scores[score]-min_score)/(max_score-min_score)

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(15,5))

    if hue == 'coding':
        hue_order = ['encoding', 'decoding']
        palette = {'encoding':ENCODE_COL, 'decoding':DECODE_COL}
    elif (hue == 'rsn' or hue == 'cyt') and (hue_order is None):
        hue_order = sort_class_labels(np.unique(df_scores[hue]))
        palette = COLORS[:-1]
    elif (hue == 'rsn') or (hue == 'cyt'):
        class_labels = sort_class_labels(np.unique(df_scores[hue]))
        palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in hue_order])

    ax = plt.subplot(111)
    sns.lineplot(x='class',
                 y=score,
                 data=df_scores,
                 hue=hue,
                 hue_order=hue_order,
                   # ci='sd',
                 # err_style='bars', #'band'
                 palette=palette,
                 marker='D',
                 markersize=12,
                 markers=True,
                 linewidth=1,
                 sort=False,
                 ax=ax,
                 **kwargs
                 )

    ax.legend(fontsize=15, frameon=True, ncol=1, loc='upper right')
    # ax.get_legend().remove()

    sns.despine(offset=10, trim=False)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/line_coding_across_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/line_coding_across_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def lineplot_coding_scores(df_scores, score, hue_order=None, scale=True):

    if scale:
        # estimate global min and max to scale scores
        max_score = np.max(df_scores[score])
        min_score = np.min(df_scores[score])
        df_scores[score] = (df_scores[score]-min_score)/(max_score-min_score)

    df_scores = merge_coding_scores(df_scores, score)

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(15,5))

    if hue_order is not None:
        class_labels = sort_class_labels(np.unique(df_scores['class']))
        palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in hue_order])
    else: palette = COLORS[:-1]

    ax = plt.subplot(111)
    sns.lineplot(x='class',
                 y='coding_' + score,
                 data=df_scores,
                 hue='class',
                 hue_order=hue_order,
                 # style=,
                 ci='sd',
                 err_style='bars', #'band'
                 palette=palette,
                 marker='D',
                 markersize=12,
                 markers=True,
                 linewidth=1,
                 sort=False,
                 ax=ax
                 )

    ax.legend(fontsize=15, frameon=True, ncol=1, loc='upper right')
    ax.get_legend().remove()

    sns.despine(offset=10, trim=False)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/lineplot_diff_coding_across_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/lineplot_diff_coding_across_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def lineplot_scores_across_alpha(task_name, df_encoding, df_decoding, score, include_alpha=None, scale=True, **kwargs):

    if include_alpha is None: include_alpha = np.unique(df_encoding['alpha'])
    df_encoding = df_encoding.loc[df_encoding['alpha'].isin(include_alpha), :]
    df_decoding = df_decoding.loc[df_decoding['alpha'].isin(include_alpha), :]

    if scale:
        # estimate global min and max to scale scores
        max_score = 16 #max(np.max(df_encoding[score]), np.max(df_decoding[score]))
        min_score = 0  #min(np.min(df_encoding[score]), np.min(df_decoding[score]))

        df_encoding[score] = (df_encoding[score]-min_score)/(max_score-min_score)
        df_decoding[score] = (df_decoding[score]-min_score)/(max_score-min_score)

    # ------------------------------ AUC ---------------------------------------
    auc_enc = []
    auc_dec = []
    for clase in sort_class_labels(np.unique(df_encoding['class'])):

        # print(' clase: ' + clase)
        tmp_df_enc = df_encoding.loc[df_encoding['class'] == clase, [score]][score]
        auc_enc.append(auc(x=include_alpha, y=tmp_df_enc))
        # print('   auc_enc:  ' + str(auc(x=include_alpha, y=tmp_df_enc)))

        tmp_df_dec = df_decoding.loc[df_decoding['class'] == clase, [score]][score]
        auc_dec.append(auc(x=include_alpha, y=tmp_df_dec))
        # print('   auc_dec:  ' + str(auc(x=include_alpha, y=tmp_df_dec)))

    # --------------------------------------------------------------------------

    # plot
    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(20,8)) #(20,7))

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    sns.lineplot(x='alpha',
                 y=score,
                 data=df_encoding,
                 hue='class',
                 hue_order=sort_class_labels(np.unique(df_encoding['class'])),
                 # style=,
                 # ci='sd',
                 # err_style='bars', #'band'
                 palette= COLORS[:-1],
                 marker='D',
                 markersize=12,
                 markers=True,
                 linewidth=1,
                 linestyle='--',
                 sort=False,
                 ax=ax1
                 )

    sns.lineplot(x='alpha',
                 y=score,
                 data=df_decoding,
                 hue='class',
                 hue_order=sort_class_labels(np.unique(df_decoding['class'])),
                 # style=,
                 # ci='sd',
                 # err_style='bars', #'band'
                 palette= COLORS[:-1],
                 marker='D',
                 markersize=12,
                 markers=True,
                 linewidth=1,
                 linestyle='--',
                 sort=False,
                 ax=ax2
                 )

    # ax1.legend(fontsize=15, frameon=True, ncol=1, loc='upper right')
    ax1.get_legend().remove()
    ax1.set_title('AUC = ' + str(np.sum(auc_enc))[:5])
    # ax1.set_xlim(0,2)
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.set_ylabel('encoding_' + score)
    ax1.set_ylim(0,1)
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    # ax1.set_xticklabels(ax1.get_xticklabels(),fontsize=17)

    sns.despine(offset=10, trim=False)

    ax2.legend(fontsize=15, frameon=True, ncol=1, loc='upper right')
    ax2.get_legend().remove()
    ax2.set_title('AUC = ' + str(np.sum(auc_dec))[:5])
    # ax2.set_xlim(0,2)
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.set_ylabel('decoding_' + score)
    ax2.set_ylim(0,1)
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    # ax2.set_xticklabels(ax2.get_xticklabels(),fontsize=17)

    sns.despine(offset=10, trim=False)

    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/coding_per_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/coding_per_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def scatterplot_enc_vs_dec(df_scores, score, norm_score_by=None, minmax_scale=True, hue='class', **kwargs):
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

    if minmax_scale:

        # -----------------------------------------------------------------------
        # estimate "global" min and max, and scale scores
        maxm = max(np.max(df['encoding_' + score]), np.max(df['decoding_' + score]))
        minm = min(np.min(df['encoding_' + score]), np.min(df['decoding_' + score]))

        # df['encoding_' + score] = np.log(((df['encoding_' + score]-minm)/(maxm-minm))+1)
        # df['decoding_' + score] = np.log(((df['decoding_' + score]-minm)/(maxm-minm))+1)

        df['encoding_' + score] = ((df['encoding_' + score]-minm)/(maxm-minm))
        df['decoding_' + score] = ((df['decoding_' + score]-minm)/(maxm-minm))

        # -----------------------------------------------------------------------
        # estimate "local" min and max, and scale scores
        # df['encoding_' + score] = np.log(((df['encoding_' + score]-np.min(df['encoding_' + score]))/(np.max(df['encoding_' + score])-np.min(df['encoding_' + score]))+1))
        # df['decoding_' + score] = np.log(((df['decoding_' + score]-np.min(df['decoding_' + score]))/(np.max(df['decoding_' + score])-np.min(df['decoding_' + score]))+1))

        # -----------------------------------------------------------------------

    # plot
    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(10,10))

    ax = plt.subplot(111)
    sns.scatterplot(
                    x='decoding_' + score,
                    y='encoding_' + score,
                    data=df,
                    hue=hue,
                    palette=COLORS[:-1],
                    # legend='full',
                    ax=ax,
                    **kwargs
                    )

    if minmax_scale:
        maxm = 1.0
        minm = 0.0
        mp = 0.2

    else:
        maxm = np.ceil(max(np.max(df['encoding_' + score]), np.max(df['decoding_' + score])))
        minm = np.floor(min(np.min(df['encoding_' + score]), np.min(df['decoding_' + score])))+1.0
        mp = 0.5

    ax.plot([minm, maxm],
            [minm, maxm],
            linestyle='--',
            linewidth=2,
            color='dimgrey'
            )

    ax.set_aspect("equal")

    # properties axis 1
    ax.set_title(r'$R: %.2f $' % (np.round(np.corrcoef(df['encoding_' + score], df['decoding_' + score])[0][1], 2)))
    # ax.set_title(r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['encoding_' + score])[0], 2), \
                                                          # np.round(stats.spearmanr(df[prop], df['encoding_' + score])[1], 2)), fontsize=13)
    ax.set_xlim(minm, maxm)
    ax.xaxis.set_major_locator(plt.MultipleLocator(mp))

    ax.set_ylim(minm, maxm)
    ax.yaxis.set_major_locator(plt.MultipleLocator(mp))

    # ax.legend(fontsize=15, frameon=True, ncol=1, loc=9) #'upper center')
    ax.get_legend().remove()

    sns.despine(offset=10, trim=False)
    # fig.savefig(fname=os.path.join(RES_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
    plt.show()
    plt.close()


# --------------------------------------------------------------------------------------------------------------------
# MULTIPLE SAMPLES
# --------------------------------------------------------------------------------------------------------------------
def boxplot_enc_vs_dec(df_scores, score, class_type='class', order=None, scale=True):

    if scale:
        # estimate global min and max to scale scores
        max_score = np.max(df_scores[score])
        min_score = np.min(df_scores[score])
        df_scores[score] = (df_scores[score]-min_score)/(max_score-min_score)

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(15,5))

    ax = plt.subplot(111)
    axis = sns.boxplot(x=class_type, #'class' 'rsn' 'cyt'
                       y=score,
                       data=df_scores,
                       hue='coding',
                       order=order,
                       palette={'encoding':ENCODE_COL, 'decoding':DECODE_COL},
                       orient='v',
                       width=0.6,
                       linewidth=0.8,
                       ax=ax
                       )

    for patch in axis.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.8))

    ax.legend(fontsize =15, frameon=False, ncol=1, loc='upper left')
    # ax.get_legend().remove()

    sns.despine(offset=10, trim=False)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/boxplot_coding_across_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/boxplot_coding_across_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def boxplot_coding_scores(df_scores, score, order=None, scale=True):

    if scale:
        # estimate global min and max to scale scores
        max_score = np.max(df_scores[score])
        min_score = np.min(df_scores[score])
        df_scores[score] = (df_scores[score]-min_score)/(max_score-min_score)

    df_scores = merge_coding_scores(df_scores, score)

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(15,5))

    # get class labels
    if order is not None:
        class_labels = sort_class_labels(np.unique(df_scores['class']))
        palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in order])
    else: palette = COLORS[:-1]

    ax = plt.subplot(111)
    axis = sns.boxplot(x='class',
                       y='coding_' + score,
                       data=df_scores,
                       order=order,
                       palette=palette,
                       orient='v',
                       width=0.5,
                       linewidth=0.8,
                       ax=ax
                       )

    for patch in axis.artists:
       r, g, b, a = patch.get_facecolor()
       patch.set_facecolor((r, g, b, 0.8))

    # ax.legend(fontsize=15, frameon=False, ncol=1, loc='lower right')
    # ax.get_legend().remove()

    sns.despine(offset=10, trim=False, bottom=False)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/boxplot_diff_coding_across_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/boxplot_diff_coding_across_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def kdeplot_enc_vs_dec(df_scores, score, norm_score_by=None, minmax_scale=True):

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

    if minmax_scale:

        # -----------------------------------------------------------------------
        # estimate "global" min and max, and scale scores
        maxm = max(np.max(df['encoding_' + score]), np.max(df['decoding_' + score]))
        minm = min(np.min(df['encoding_' + score]), np.min(df['decoding_' + score]))

        # df['encoding_' + score] = np.log(((df['encoding_' + score]-minm)/(maxm-minm))+1)
        # df['decoding_' + score] = np.log(((df['decoding_' + score]-minm)/(maxm-minm))+1)

        df['encoding_' + score] = ((df['encoding_' + score]-minm)/(maxm-minm))
        df['decoding_' + score] = ((df['decoding_' + score]-minm)/(maxm-minm))

        # -----------------------------------------------------------------------
        # estimate "local" min and max, and scale scores
        # df['encoding_' + score] = np.log(((df['encoding_' + score]-np.min(df['encoding_' + score]))/(np.max(df['encoding_' + score])-np.min(df['encoding_' + score]))+1))
        # df['decoding_' + score] = np.log(((df['decoding_' + score]-np.min(df['decoding_' + score]))/(np.max(df['decoding_' + score])-np.min(df['decoding_' + score]))+1))

        # -----------------------------------------------------------------------

    # plot
    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(10,10))

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))
    for i, clase in enumerate(class_labels):

        tmp_df_encod = df.loc[df['class'] == clase, ['encoding_' + score]]['encoding_' + score]
        tmp_df_decod = df.loc[df['class'] == clase, ['decoding_' + score]]['decoding_' + score]

        ax = sns.kdeplot(data=tmp_df_decod,
                         data2=tmp_df_encod,
                         color=COLORS[i],
                         shade=True,
                         shade_lowest=False,
                         )


    # Draw 1:1 line
    if minmax_scale:
        maxm = 1.0
        minm = 0.0
        mp = 0.2

    else:
        maxm = np.ceil(max(np.max(df['encoding_' + score]), np.max(df['decoding_' + score])))
        minm = np.floor(min(np.min(df['encoding_' + score]), np.min(df['decoding_' + score])))+1.0
        mp = 0.5

    ax.plot([minm, maxm],
            [minm, maxm],
            linestyle='--',
            linewidth=2,
            color='dimgrey'
            )

    ax.set_aspect("equal")

    ax.set_xlim(minm, maxm)
    ax.xaxis.set_major_locator(plt.MultipleLocator(mp))

    ax.set_ylim(minm, maxm)
    ax.yaxis.set_major_locator(plt.MultipleLocator(mp))

    # ax.legend(fontsize=20, frameon=False, ncol=1, loc='lower right')
    # ax.get_legend().remove()

    sns.despine(offset=10, trim=False)

    plt.show()
    plt.close()


# --------------------------------------------------------------------------------------------------------------------
# VALIDATION: STATISTICAL TESTS
# ----------------------------------------------------------------------------------------------------------------------
def ttest(df_scores, score, fdr_correction=True, test='2samp'):

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))

    pval = []
    tstat = []
    for clase in class_labels:
        encod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'encoding'), [score]][score]
        decod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'decoding'), [score]][score]

        if test == '1samp': t, p = stats.ttest_1samp(encod_scores-decod_scores, popmean=0.0)
        elif test == '2samp': t, p = stats.ttest_ind(encod_scores, decod_scores, equal_var=False)
            # t, p = stats.ttest_rel(encod_scores, decod_scores)

        pval.append(p)
        tstat.append(t)

    if fdr_correction: pval = multipletests(pval, 0.1, 'bonferroni')[1]

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
        encod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'encoding'), [score]][score]
        decod_scores = df_scores.loc[(df_scores['class'] == clase) & (df_scores['coding'] == 'decoding'), [score]][score]

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
