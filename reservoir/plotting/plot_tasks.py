# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:10:14 2019

@author: Estefany Suarez
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

from ..tasks import tasks


COLORS = sns.color_palette("husl", 8)
ENCODE_COL = '#E55FA3'
DECODE_COL = '#6CC8BA'


# --------------------------------------------------------------------------------------------------------------------
# GENERAL
# ----------------------------------------------------------------------------------------------------------------------
def sort_class_labels(class_labels):
    rsn_labels = ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN']
    vEc_labels = ['PSS', 'PS', 'PM', 'LIM', 'AC1', 'IC', 'AC2']

    if class_labels.all() in rsn_labels:
        return rsn_labels

    elif class_labels.all() in vEc_labels:
        return vEc_labels

    else:
        return class_labels

def get_coding_scores_per_class(df_encoding, df_decoding, score='capacity', stat2return='avg', include_alpha=None):

    # get class labels
    class_labels = sort_class_labels(np.unique(df_encoding['class']))

    # estimate avg/sum of scores per class
    encode_scores = []
    decode_scores = []
    for clase in class_labels:

        # filter coding scores with alpha values
        if include_alpha is None:
            tmp_encode_scores = df_encoding.loc[df_encoding['class'] == clase, [score]][score]
            tmp_decode_scores = df_decoding.loc[df_decoding['class'] == clase, [score]][score]

        else:
            tmp_encode_scores = df_encoding.loc[(df_encoding['class'] == clase) & (df_encoding['alpha'].isin(include_alpha)), [score]][score]
            tmp_decode_scores = df_decoding.loc[(df_decoding['class'] == clase) & (df_decoding['alpha'].isin(include_alpha)), [score]][score]

        # estimate avg/sum coding scores across alpha values per class
        if stat2return == 'avg':
            encode_scores.append(tmp_encode_scores.mean())
            decode_scores.append(tmp_decode_scores.mean())

        if stat2return == 'sum':
            encode_scores.append(tmp_encode_scores.sum())
            decode_scores.append(tmp_decode_scores.sum())


    # DataFrame with avg/sum coding scores per class
    df_encode = pd.DataFrame(data = np.column_stack((class_labels, encode_scores)), #stat2return + '_' + score
                             columns = ['class', score],
                             index = np.arange(len(class_labels))
                            )
    df_encode['coding'] = 'encoding'

    df_decode = pd.DataFrame(data = np.column_stack((class_labels, decode_scores)), #stat2return + '_' + score
                             columns = ['class', score],
                             index = np.arange(len(class_labels))
                            )
    df_decode['coding'] = 'decoding'

    df_scores = pd.concat([df_encode, df_decode])
    df_scores[score]  = df_scores[score].astype('float')
    df_scores['statistic'] = stat2return

    return df_scores


# --------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
def lineplot_coding_across_alpha(task_name, df_scores, score):

    sns.set(style="ticks")
    fig = plt.figure(num=1, figsize=(10,5))

    ax = plt.subplot(111)
    sns.lineplot(x='class',
                 y=score,
                 hue='coding',
                 style='coding',
                 data=df_scores,
                 palette={'encoding':ENCODE_COL, 'decoding':DECODE_COL},
                 linewidth=1, #2, 1
                 markers=True, #'D'
                 markersize=5, #12, 5
                 sort=False,
                 ax=ax
                 )

    ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper right')
    # ax.get_legend().remove()

    # sns.despine(offset=10, trim=True, bottom=False)
    # fig.savefig(fname=os.path.join(RES_TASK_DIR, 'lnplot_' + np.unique(df_scores['statistic']) + '_encoding_vs_decoding.jpg'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def scatterplot_coding_across_alpha(task_name, df_scores, score):

    # get encoding and decoding vals per class
    encod_vals = df_scores.loc[df_scores['coding'] == 'encoding', [df_scores]]
    decod_vals = df_scores.loc[df_scores['coding'] == 'decoding', [df_scores]]
    class_labels = np.unqiue(df_scores['class'])




    # plot
    sns.set(style="ticks")
    fig = plt.figure(num=3, figsize=(5,5))

    ax = plt.subplot(111)
    for i, clase in enumerate(class_names):

        sns.scatterplot(x=[encod_vals[i]],
                        y=[decod_vals[i]],
                        color=COLORS[i],
                        # label=clase,
                        ax=ax,
                        s=400
                        )

        ax.plot([0,1], [0,1],
                linestyle='--',
                linewidth=2,
                color='dimgrey'
                )

    # ax.set_title('connectome: ' + CONNECTOME)
#    ax.set_xlim(0,1)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))

#    ax.set_ylim(0,1)
    # ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    # ax.legend(fontsize=5, frameon=False, ncol=7, loc='upper center')

    sns.despine(offset=10, trim=True)
    #fig.savefig(fname=os.path.join(RES_TASK_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
    plt.show()
    plt.close()












# --------------------------------------------------------------------------------------------------------------------
# AVERAGE/SUM ACROSS ALPHA VALUES
# ----------------------------------------------------------------------------------------------------------------------
# def encod_vs_decod_across_alpha_lineplot(task_name, class_names, df_encoding, df_decoding, var_type, include_alpha=None, **kwargs):
#
#     # get encoding and decoding vals per class
#     encod_vals, decod_vals = filter_results(df_encoding,
#                                             df_decoding,
#                                             var_type,
#                                             class_names,
#                                             include_alpha,
#                                             **kwargs
#                                             )
#
#     # plot
#     sns.set(style="ticks")
#     fig = plt.figure(num=1, figsize=(10,5))
#
#     ax = plt.subplot(111)
#     sns.lineplot(x=np.arange(len(encod_vals)),
#                  y=np.array(encod_vals),
#                  color=ENCODE_COL,
#                  label='encoding',
#                  linewidth=1, #2, 1
#                  linestyle='-',
#                  markersize=5, #12, 5
#                  marker='D',
#                  ax=ax
#                  )
#
#     sns.lineplot(x=np.arange(len(decod_vals)),
#                  y=decod_vals,
#                  color=DECODE_COL,
#                  label='decoding',
#                  linewidth=1, #2, 1
#                  linestyle='-',
#                  markersize=5, #12, 5
#                  marker='o',
#                  ax=ax
#                  )
#
#     # ax.set_xlim()
#     # ax.set_xlabel()
#     ax.set_xticks(np.arange(len(class_names))[::10])
#     ax.set_xticklabels(class_names[::10])
#
# #    ax.set_ylim()
#     ax.set_ylabel(var_type)
# #    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
#
#     ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper right')
#     # ax.get_legend().remove()
#
#     sns.despine(offset=10, trim=True)
# #    fig.savefig(fname=os.path.join(RES_TASK_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
#     # fig.savefig(fname='C:/Users/User/Desktop/' + task_name + '.eps', transparent=True, bbox_inches='tight', dpi=300,)
#     plt.show()
#     plt.close()
#
#
# def encod_vs_decod_across_alpha_scatterplot(task_name, class_names, df_encoding, df_decoding, var_type, include_alpha=None, **kwargs):
#     # get encoding and decoding vals per class
#     encod_vals, decod_vals = filter_results(df_encoding,
#                                             df_decoding,
#                                             var_type,
#                                             class_names,
#                                             include_alpha,
#                                             **kwargs
#                                             )
#
#     # plot
#     sns.set(style="ticks")
#     fig = plt.figure(num=3, figsize=(5,5))
#
#     ax = plt.subplot(111)
#     for i, clase in enumerate(class_names):
#
#         sns.scatterplot(x=[encod_vals[i]],
#                         y=[decod_vals[i]],
#                         color=COLORS[i],
#                         # label=clase,
#                         ax=ax,
#                         s=400
#                         )
#
#         ax.plot([0,1], [0,1],
#                 linestyle='--',
#                 linewidth=2,
#                 color='dimgrey'
#                 )
#
#     # ax.set_title('connectome: ' + CONNECTOME)
# #    ax.set_xlim(0,1)
#     # ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
#
# #    ax.set_ylim(0,1)
#     # ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
#
#     # ax.legend(fontsize=5, frameon=False, ncol=7, loc='upper center')
#
#     sns.despine(offset=10, trim=True)
#     #fig.savefig(fname=os.path.join(RES_TASK_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
#     plt.show()
#     plt.close()


# --------------------------------------------------------------------------------------------------------------------
# PER ALPHA VALUE
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
# def encod_vs_decod_per_alpha_lineplot(task_name, class_names, df_encoding, df_decoding, var_type, include_alpha, normalize=True):
#
#     # get encoding and decoding vals per class
#     encod_vals, decod_vals = filter_results(df_encoding,
#                                             df_decoding,
#                                             var_type,
#                                             class_names,
#                                             include_alpha,
#                                             var2return='vals_per_alpha'
#                                             )
#
#     # plot
#     sns.set(style="ticks")
#     fig = plt.figure(num=1, figsize=(20,7))
#
#     ax1 = plt.subplot(121)
#     ax2 = plt.subplot(122)
#
#     for i, clase in enumerate(class_names):
#
#         sns.lineplot(x=np.arange(len(encod_vals[i])),
#                      y=np.array(encod_vals[i]),
#                      color=COLORS[i],
#                      label=clase,
#                      linewidth=2,
#                      linestyle='-',
#                      markersize=12,
#                      marker='o',
#                      ax=ax1
#                      )
#
#         sns.lineplot(x=np.arange(len(decod_vals[i])),
#                      y=decod_vals[i],
#                      color=COLORS[i],
#                      label=clase,
#                      linewidth=2,
#                      linestyle='-',
#                      markersize=12,
#                      marker='o',
#                      ax=ax2
#                      )
#
#     # ax1.set_ylim(4,14)
#     ax1.set_xlabel('alpha')
#     ax1.set_ylabel('encoding ' + var_type)
#     # ax1.set_yscale('log')
#     ax1.set_xticks(np.arange(len(include_alpha)))
#     ax1.set_xticklabels(include_alpha)
#     # ax1.legend(fontsize=15, frameon=False, ncol=7, loc='lower center')
#     # ax1.get_legend().remove()
#
#     # ax2.set_ylim(4,14)
#     ax2.set_xlabel('alpha')
#     ax2.set_ylabel('decoding ' + var_type)
#     ax2.set_xticks(np.arange(len(include_alpha)))
#     ax2.set_xticklabels(include_alpha)
#     # ax2.legend(fontsize=15, frameon=False, ncol=1, loc='upper center')
#     # ax2.get_legend().remove()
#
#     sns.despine(offset=10, trim=True)
# #    fig.savefig(fname=os.path.join(RES_TASK_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
#     # fig.savefig(fname='C:/Users/User/Desktop/' + task_name + '.eps', transparent=True, bbox_inches='tight', dpi=300,)
#     plt.show()
#     plt.close()
