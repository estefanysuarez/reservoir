import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .plot_tasks import (sort_class_labels, get_coding_scores_per_class)

from netneurotools import plotting
from netneurotools import datasets


COLORS = sns.color_palette("husl", 8)
ENCODE_COL = '#E55FA3'
DECODE_COL = '#6CC8BA'


# --------------------------------------------------------------------------------------------------------------------
# GENERAL
# ----------------------------------------------------------------------------------------------------------------------
def get_net_props_per_class(df_net_props):
    """
    Returns a DataFrame with the average network properties per class
    """

    # get class labels
    class_labels = sort_class_labels(np.unique(df_net_props['class']))

    # estimate avg/sum of scores per class
    class_avg_net_props = {clase:df_net_props.loc[df_net_props['class'] == clase, :].mean() for clase in class_labels}
    class_avg_net_props = pd.DataFrame.from_dict(class_avg_net_props, orient='index').reset_index().rename(columns={'index':'class'})
    class_avg_net_props = class_avg_net_props.loc[:,~class_avg_net_props.columns.duplicated()]

    return class_avg_net_props


# --------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
def boxplot_net_props_distribution_per_class(df_net_props, property, include_subctx=False, **kwargs):

    class_labels = sort_class_labels(np.unique(df_net_props['class']))
    print(class_labels)
    colors = {clase:COLORS[i] for i, clase in enumerate(class_labels)}

    sns.set(style="ticks")
    fig = plt.figure(num=1, figsize=(10,5))
    ax = plt.subplot(111)
    axis = sns.boxplot(x='class',
                       y=property,
                       data=df_net_props,
                       palette=colors,
                       order=class_labels,
                       orient='v',
                       width = 0.5,
                       linewidth=1, #2, 1
                       ax=ax
                       )

    for patch in axis.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .8))

    ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper right')
    # ax.get_legend().remove()
    ax.set_ylim(-0.01, 0.1)
    sns.despine(offset=10, trim=True, bottom=False)
    # fig.savefig(fname=os.path.join(RES_TASK_DIR, 'lnplot_' + np.unique(df_scores['statistic']) + '_encoding_vs_decoding.jpg'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    # plt.close()


def brain_plot(var_name, data, scale, cmap='viridis', cbar=True):

    os.environ['SUBJECTS_DIR'] = ''
    lh_annot = datasets.fetch_cammoun2012('surface')[scale][0]
    rh_annot = datasets.fetch_cammoun2012('surface')[scale][1]


    brain = plotting.plot_fsaverage(data=data,
                                    lhannot=lh_annot,
                                    rhannot=rh_annot,
                                    colormap=cmap,
                                    colorbar=cbar,
                                    alpha=1.0,
                                    views=['lat', 'med'],
#                                        center=0.0
                                    )

    for i in range(2):
        for j in range(2):
               brain._figures[i][j].scene.parallel_projection = True
               brain._figures[i][j].scene.parallel_projection = True

#    brain.save_image(os.path.join(FIG_DIR,  var_name + '_surf_brain' + '.png'), mode='rgba')

    return brain


def barplot_net_props_across_classes(df_net_props, class_mapping, include_property=None):


    property_list = list(df_net_props.columns)
    df_net_props['class'] = class_mapping

    if include_property is None: include_property = property_list

    sns.set(style="ticks")
    colors = np.array([COLORS[np.where(class_names==mapp)[0][0]] for mapp in class_mapping_ctx])

    for i, prop_name in enumerate(property_list):

        if prop_name in include_property:
            # bar graph plot
            fig = plt.figure(num=i, figsize=(20,5))
            ax = plt.subplot(111)

            prop = df_net_props.loc[:, [prop_name]].squeeze()
            sorted_idx_prop = np.argsort(prop)

            ax.bar(x=np.arange(len(prop)),
                   height=prop[sorted_idx_prop],
                   color=colors[sorted_idx_prop],
                   width=1.0
                   )

            ax.set_title('prop: ' + prop_name)

        #    plt.xticks(np.arange(len(grad)), class_mapping_ctx[sorted_idx])
            sns.despine(offset=10, trim=True, bottom=True)
            plt.show()
        #    plt.close()


# --------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------------------------------
def regplot_net_props_vs_coding_scores(df_encoding, df_decoding, score, df_net_props, properties=None, **kwargs):
    """
    Regression plot (avg across alphas) encoding/decoding score vs (avg across nodes) network property
    """

    # get coding scores per class
    df_class_scores = get_coding_scores_per_class(df_encoding,
                                                  df_decoding,
                                                  score,
                                                  **kwargs)
    df_class_encoding_scores = df_class_scores.loc[df_class_scores['coding'] == 'encoding', ['class', score]].rename(columns={score:'encoding_' + score})
    df_class_decoding_scores = df_class_scores.loc[df_class_scores['coding'] == 'decoding', ['class', score]].rename(columns={score:'decoding_' + score})

    # get network properties per class
    df_class_net_props = get_net_props_per_class(df_net_props)
    if properties is None: properties = list(df_class_net_props.columns[1:])

    # merge coding scores and network properties dataframes
    df = pd.merge(df_class_encoding_scores, df_class_decoding_scores, on='class')
    df = pd.merge(df, df_class_net_props, on='class')

    # plot
    sns.set(style="ticks")

    encoding_scores = df['encoding_' + score]
    decoding_scores = df['decoding_' + score]

    for i, prop in enumerate(properties): #

        fig = plt.figure(num=i, figsize=(18,8))
        ax1 = plt.subplot(121)
        sns.regplot(
                    x=encoding_scores,
                    y=df[prop],
                    label=r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], encoding_scores)[0][1], 2)),
                    #label=r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], encoding_scores)[0], 2), np.round(stats.spearmanr(df[prop], encoding_scores)[1], 2)),
                    fit_reg=True,
                    ax=ax1,
                    marker='o',
                    color='#E55FA3',
                    )

        ax2 = plt.subplot(122)
        sns.regplot(
                    x=decoding_scores,
                    y=df[prop],
                    label=r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], decoding_scores)[0][1], 2)),
                    #label=r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], decoding_scores)[0], 2), np.round(stats.spearmanr(df[prop], decoding_scores)[1], 2)),
                    fit_reg=True,
                    ax=ax2,
                    marker='D',
                    color='#6CC8BA',
                    )

        # ax1.set_title(CLASS)
        ax1.set_ylabel(prop, fontsize=20)
        ax1.set_xlabel('encoding ' + score, fontsize=20)
        ax1.legend(fontsize=15, frameon=True, ncol=1, loc=9)#'upper center')
    #    ax1.get_legend().remove()

        # ax2.set_title(CLASS)
        ax2.set_ylabel(prop, fontsize=20)
        ax2.set_xlabel('decoding ' + score, fontsize=20)
        ax2.legend(fontsize=18, frameon=True, ncol=1, loc=9)#'upper center')
    #    ax.get_legend().remove()


        sns.despine(offset=10, trim=True)
    #    fig.savefig(fname=os.path.join(RES_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
        plt.show()
    #    plt.close()
