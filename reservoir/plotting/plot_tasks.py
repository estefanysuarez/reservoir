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
        return np.array(rsn_labels)

    elif class_labels.all() in vEc_labels:
        return np.array(vEc_labels)

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
            tmp_encode_scores = df_encoding.loc[df_encoding['class'] == clase, ['performance', 'capacity']]
            tmp_decode_scores = df_decoding.loc[df_decoding['class'] == clase, ['performance', 'capacity']]

        else:
            tmp_encode_scores = df_encoding.loc[(df_encoding['class'] == clase) & (df_encoding['alpha'].isin(include_alpha)), ['performance', 'capacity']]
            tmp_decode_scores = df_decoding.loc[(df_decoding['class'] == clase) & (df_decoding['alpha'].isin(include_alpha)), ['performance', 'capacity']]

        # estimate avg/sum coding scores across alpha values per class
        encode_scores.append(tmp_encode_scores.mean())
        decode_scores.append(tmp_decode_scores.mean())

    encode_scores = pd.concat(encode_scores, axis=1).T
    decode_scores = pd.concat(decode_scores, axis=1).T

    # DataFrame with avg/sum coding scores per class
    df_encode = pd.DataFrame(data = np.column_stack((class_labels, encode_scores)),
                             columns = ['class', 'performance', 'capacity'],
                             index = np.arange(len(class_labels))
                             )
    df_encode['coding'] = 'encoding'

    df_decode = pd.DataFrame(data = np.column_stack((class_labels, decode_scores)),
                             columns = ['class', 'performance', 'capacity'],
                             index = np.arange(len(class_labels))
                            )
    df_decode['coding'] = 'decoding'

    df_scores = pd.concat([df_encode, df_decode])
    df_scores['performance']  = df_scores['performance'].astype('float')
    df_scores['capacity']  = df_scores['capacity'].astype('float')

    return df_scores


# --------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
# AVG SCORES ACROSS ALPHA VALUES
def lineplot_enc_vs_dec(task_name, df_scores, score, hue='coding', **kwargs):

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(12,5))

    if hue == 'coding':
        hue_order = ['encoding', 'decoding']
        palette = {'encoding':ENCODE_COL, 'decoding':DECODE_COL}

    elif hue == 'rsn':
        hue_order = sort_class_labels(np.unique(df_scores['rsn']))
        palette = sns.color_palette("husl", 8)[:-1]

    ax = plt.subplot(111)
    sns.lineplot(x='class',
                 y=score,
                 hue=hue,
                 hue_order=hue_order,
                 style='coding',
                 data=df_scores,
                 palette=palette,
                 linewidth=1, #2, 1
                 markers=True, #'D'
                 markersize=8, #12, 5
                 sort=False,
                 ax=ax
                 )

    ax.legend(fontsize=15, frameon=True, ncol=1, loc='upper right')
    # ax.get_legend().remove()

    sns.despine(offset=10, trim=True, bottom=False)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/line_coding_across_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/line_coding_across_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def scatterplot_enc_vs_dec(task_name, df_scores, score, hue=None):

    # get encoding - decoding difference
    encod_scores = df_scores.loc[df_scores['coding'] == 'encoding', ['class', score]]
    decod_scores = df_scores.loc[df_scores['coding'] == 'decoding', ['class', score]]

    tmp_df = pd.DataFrame(data = np.column_stack((encod_scores['class'], encod_scores[score], decod_scores[score], encod_scores[score]-decod_scores[score])),
                          columns = ['class', 'encoding', 'decoding', 'coding score'],
                          index = None
                          )
    tmp_df['coding score'] = tmp_df['coding score'].astype(float)

    # get drop rename
    # encod_scores = df_scores.query('coding == "encoding"') \
    #                         .drop(['coding'], axis=1) \
    #                         .rename(dict(performance='encoding_performance',
    #                                      capacity='encoding_capacity'), axis=1)
    # decod_scores = df_scores.query('coding == "decoding"') \
    #                         # .get(['performance', 'capacity']) \
    #                         .rename(dict(performance='decoding_performance',
    #                                      capacity='decoding_capacity'), axis=1)
    #
    # tmp_df = pd.concat([encod_scores, decod_scores], axis=1)


    # plot
    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=3, figsize=(8,8))

    if hue is None: hue = 'class'

    ax = plt.subplot(111)
    sns.scatterplot(x='decoding',
                    y='encoding',
                    data=tmp_df,
                    hue=hue,
                    s=500,  #1000*np.abs(tmp_df['coding score']),
                    # sizes=(800,1500),
                    palette=COLORS[:-1],
                    legend='full',
                    ax=ax,
                    )

    maxm = np.ceil(max(np.max(encod_scores[score]), np.max(decod_scores[score])))+0.5
    minm = np.floor(min(np.min(encod_scores[score]), np.min(decod_scores[score])))-0.5

    ax.plot([0, maxm],
            [0, maxm],
            linestyle='--',
            linewidth=2,
            color='dimgrey'
            )

    ax.set_xlim(minm,maxm)
    # ax.set_xlabel('decoding')
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))

    ax.set_ylim(minm,maxm)
    # ax.set_ylabel('encoding')
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

    ax.legend(fontsize=20, frameon=False, ncol=1, loc='lower right')
    # ax.get_legend().remove()

    sns.despine(offset=10, trim=True)
    #fig.savefig(fname=os.path.join(RES_TASK_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
    plt.show()
    plt.close()


# SCORES PER ALPHA VALUE
def lineplot_enc_vs_dec_across_alpha(task_name, df_encoding, df_decoding, score, include_alpha=None):

    if include_alpha is None: include_alpha = np.unique(df_encoding['alpha'])
    df_encoding = df_encoding.loc[df_encoding['alpha'].isin(include_alpha), :]
    df_decoding = df_decoding.loc[df_decoding['alpha'].isin(include_alpha), :]

    # df_encoding = df_encoding.loc[df_encoding['class'] == 'SM', :]
    # df_decoding = df_decoding.loc[df_decoding['class'] == 'SM', :]

    # plot
    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(20,5)) #(20,7))

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    sns.lineplot(x='alpha',
                 y=score,
                 data=df_encoding,
                 palette=COLORS[:-1],
                 hue='class',
                 linewidth=2,
                 linestyle='--',
                 markersize=12,
                 marker='o',
                 ax=ax1
                 )

    sns.lineplot(x='alpha',
                 y=score,
                 data=df_decoding,
                 palette=COLORS[:-1],
                 hue='class',
                 linewidth=2,
                 linestyle='--',
                 markersize=12,
                 marker='o',
                 ax=ax2
                 )

    # ax1.set_ylim(4,14)
    # ax1.set_xlabel('alpha', fontsize=20)
    # ax1.set_ylabel('encoding ' + score, fontsize=20)
    # ax1.legend(fontsize=13, frameon=False, ncol=1, loc='upper right')
    ax1.get_legend().remove()
    # ax1.set_xlim(0,2)
    # ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    # ax1.set_ylim(5,15)
    # ax1.yaxis.set_major_locator(MultipleLocator(5))
    # ax1.set_xticks(np.arange(len(include_alpha)))
    # ax1.set_xticklabels(ax1.get_xticklabels(),fontsize=17)


    # ax2.set_ylim(4,14)
    # ax2.set_xlabel('alpha', fontsize=15)
    # ax2.set_ylabel('decoding ' + score, fontsize=15)
    # ax2.set_xlim(0,2)
    # ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    # ax2.set_ylim(5,15)
    # ax2.yaxis.set_major_locator(MultipleLocator(5))
    # ax2.set_xticks(np.arange(len(include_alpha)))
    # ax2.set_xticklabels(include_alpha)
    ax2.legend(fontsize=13, frameon=False, ncol=1, loc='upper right')
    # ax2.get_legend().remove()

    sns.despine(offset=10, trim=True)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/coding_per_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/coding_per_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


# --------------------------------------------------------------------------------------------------------------------
# VALIDATION: AVERAGE ACROSS ALPHA VALUES - SUBSAMPLING/BOOTSTRAP RESAMPLING
# --------------------------------------------------------------------------------------------------------------------
def boxplot_enc_vs_dec(df_scores, score, class_type, hue_order=None):

    sns.set(style="ticks")
    fig = plt.figure(num=1, figsize=(10,5))

    ax = plt.subplot(111)
    axis = sns.boxplot(x=class_type, #'class' 'rsn' 'cyt'
                       y=score,
                       orient='v',
                       hue='coding',
                       order=hue_order,
                       data=df_scores,
                       palette={'encoding':ENCODE_COL, 'decoding':DECODE_COL},
                       linewidth=1, #2, 1
                       ax=ax
                       )

    for patch in axis.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.8))

    ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper right')
    # ax.get_legend().remove()

    sns.despine(offset=10, trim=True, bottom=False)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/boxplot_coding_across_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/boxplot_coding_across_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def boxplot_coding_scores(df_scores, score, hue_order=None):

    # get encoding - decoding difference
    encod_scores = df_scores.loc[df_scores['coding'] == 'encoding', ['class', score]]
    decod_scores = df_scores.loc[df_scores['coding'] == 'decoding', ['class', score]]

    tmp_df = pd.DataFrame(data = np.column_stack((encod_scores['class'], encod_scores[score], decod_scores[score], encod_scores[score]-decod_scores[score])),
                          columns = ['class', 'encoding', 'decoding', 'enc-dec'],
                          index = None
                          )
    tmp_df['enc-dec'] = tmp_df['enc-dec'].astype(float)

    sns.set(style="ticks")
    fig = plt.figure(num=1, figsize=(10,5))

    # get class labels
    if hue_order is not None:
        class_labels = sort_class_labels(np.unique(df_scores['class']))
        palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in hue_order])
    else: palette = COLORS[:-1]

    ax = plt.subplot(111)
    axis = sns.boxplot(x=tmp_df['class'],
                       y=tmp_df['enc-dec'],
                       data=tmp_df,
                       order=hue_order,
                       palette=palette,
                       orient='v',
                       width=0.6,
                       ax=ax
                       )

    for patch in axis.artists:
       r, g, b, a = patch.get_facecolor()
       patch.set_facecolor((r, g, b, 0.8))

    ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper right')
    # ax.get_legend().remove()

    sns.despine(offset=10, trim=True, bottom=False)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/boxplot_diff_coding_across_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/boxplot_diff_coding_across_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    test_df = pd.concat([encod_scores, decod_scores], axis=1)


def lineplot_coding_scores(df_scores, score, hue_order=None):

    # get encoding - decoding difference
    encod_scores = df_scores.loc[df_scores['coding'] == 'encoding', ['class', score]]
    decod_scores = df_scores.loc[df_scores['coding'] == 'decoding', ['class', score]]

    tmp_df = pd.DataFrame(data = np.column_stack((encod_scores['class'], encod_scores[score], decod_scores[score], encod_scores[score]-decod_scores[score])),
                          columns = ['class', 'encoding', 'decoding', 'enc-dec'],
                          index = None
                          )
    tmp_df['enc-dec'] = tmp_df['enc-dec'].astype(float)

    sns.set(style="ticks")
    fig = plt.figure(num=1, figsize=(10,5))

    if hue_order is not None:
        class_labels = sort_class_labels(np.unique(df_scores['class']))
        palette = np.array([np.array(COLORS)[np.where(class_labels == clase)[0][0]] for clase in hue_order])
    else: palette = COLORS[:-1]

    ax = plt.subplot(111)
    sns.lineplot(x='class',
                 y='enc-dec',
                 data=tmp_df,
                 ci='sd',
                 err_style='bars', #'band'
                 palette=palette,
                 markersize=12,
                 marker='D',
                 markers=True,
                 hue='class',
                 hue_order=hue_order,
                 ax=ax
                 )

    ax.legend(fontsize=15, frameon=True, ncol=1, loc='upper right')
    ax.get_legend().remove()

    # sns.despine(offset=10, trim=True, bottom=False)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/lineplot_diff_coding_across_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/lineplot_diff_coding_across_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def boxplot_lineplot_coding_scores(df_scores, score, effect_size):

    # get encoding - decoding difference
    encod_scores = df_scores.loc[df_scores['coding'] == 'encoding', ['class', score]]
    decod_scores = df_scores.loc[df_scores['coding'] == 'decoding', ['class', score]]

    tmp_df = pd.DataFrame(data = np.column_stack((encod_scores['class'], encod_scores[score], decod_scores[score], encod_scores[score]-decod_scores[score])),
                          columns = ['class', 'encoding', 'decoding', 'enc-dec'],
                          index = None
                          )
    tmp_df['enc-dec'] = tmp_df['enc-dec'].astype(float)

    sns.set(style="ticks")
    fig = plt.figure(num=1, figsize=(10,5))

    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))
    print(class_labels[np.argsort(effect_size)])
    print(np.sort(effect_size))

    axis1 = sns.boxplot(x=tmp_df['class'],
                        y=tmp_df['enc-dec'],
                        data=tmp_df,
                        order=class_labels[np.argsort(effect_size)],
                        palette=np.array(COLORS)[np.argsort(effect_size)],
                        orient='v',
                        width=0.6,
                        ax=ax1
                        )

    for patch in axis1.artists:
       r, g, b, a = patch.get_facecolor()
       patch.set_facecolor((r, g, b, 0.8))

    axis2 = sns.lineplot(x=class_labels[np.argsort(effect_size)],
                         y=np.sort(effect_size),
                         palette=np.array(COLORS)[np.argsort(effect_size)],
                         markersize=12,
                         marker='D',
                         markers=True,
                         sort=False,
                         ax=ax2
                         )

    # sns.despine(offset=10, trim=True, bottom=False, right=False)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/boxplot_diff_coding_across_alpha2.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/boxplot_diff_coding_across_alpha2.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    test_df = pd.concat([encod_scores, decod_scores], axis=1)


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


def barplot_eff_size(df_scores, score, hue_order=None, test='2samp'):

    # get class labels
    class_labels = sort_class_labels(np.unique(df_scores['class']))
    eff_size  = effect_size(df_scores, score, test)

    df_eff_size = pd.DataFrame(data = np.column_stack((class_labels, eff_size)),
                                  columns = ['class', 'effect_size'],
                                  index = np.arange(len(class_labels))
                                  )
    df_eff_size['effect_size']  = df_eff_size['effect_size'].astype('float')

    sns.set(style="ticks")
    fig = plt.figure(num=1, figsize=(7,7))
    ax = plt.subplot(111)

    sns.barplot(x='class',
                y='effect_size',
                data=df_eff_size,
                order=hue_order,
                palette=np.array(COLORS)[np.argsort(eff_size)],
                orient='v',
                )

    sns.despine(offset=10, trim=True)
    plt.tight_layout()

    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/barplot_effect_size.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/barplot_effect_size.jpg', transparent=True, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()


# --------------------------------------------------------------------------------------------------------------------
# PENDING
# ----------------------------------------------------------------------------------------------------------------------
# def encod_vs_decod_per_alpha_scatterplot(task_name, class_names, df_encoding, df_decoding, var_type, include_alpha, normalize=True):
#
#     if include_alpha == None: include_alpha = np.unique(df_encoding['alpha'])
#
#     sns.set(style="ticks")
#     fig = plt.figure(num=2, figsize=(5*len(include_alpha),5))
#
#     for i, alpha in enumerate(include_alpha):
#
#         ax = plt.subplot(1, len(include_alpha), i+1)
#
#         encod_var = df_encoding.loc[df_encoding['alpha'] == alpha, ['class', var_type]]
#         encod_var = encod_var.set_index('class', drop = False)
#
#         decod_var = df_decoding.loc[df_encoding['alpha'] == alpha, ['class', var_type]]
#         decod_var = decod_var.set_index('class', drop = False)
#
#         for j, clase in enumerate(class_names):
#
#             x = [encod_var.loc[clase, var_type]]
#             y = [decod_var.loc[clase, var_type]]
#
#             if normalize:
#                 min_var = np.min(tasks.get_default_task_params(task_name))
#                 max_var = np.max(tasks.get_default_task_params(task_name))
#                 x = (x-min_var)/(max_var - min_var)
#                 y = (y-min_var)/(max_var - min_var)
#
#             sns.scatterplot(x=x,
#                             y=y,
#                             color=COLORS[j],
#                             # label=clase,
#                             ax=ax,
#                             s=300
#                             )
#
#             ax.plot([0,1], [0,1],
#                     linestyle='--',
#                     linewidth=2,
#                     color='dimgrey'
#                     )
#
#         ax.set_title('alpha: ' + str(alpha))
#
# #        ax.set_xlim(0,1)
#         ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
#
# #        ax.set_ylim(0,1)
#         ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
#
#         ax.set_xlabel('encoding')
#         if i == 0:
#             ax.set_ylabel('decoding')
#
#         # ax.legend(fontsize=20, frameon=False, ncol=7, loc='upper center',
#         #           bbox_to_anchor=(0.05,1.5)
#         #           )
#
#     sns.despine(offset=10, trim=True)
#     #fig.savefig(fname=os.path.join(RES_TASK_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
#     # fig.savefig(fname='C:/Users/User/Desktop/' + task_name + '.eps', transparent=True, bbox_inches='tight', dpi=300,)
#     plt.show()
#     plt.close()
#
#
