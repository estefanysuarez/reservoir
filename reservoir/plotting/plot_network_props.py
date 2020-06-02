import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .plot_tasks import (sort_class_labels, get_coding_scores_per_class)
from ..network import network_properties

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

    if len(class_labels) > 8:
        return df_net_props

    else:

        class_avg_net_props = {clase: df_net_props.loc[df_net_props['class'] == clase, :].mean() for clase in class_labels}
        class_avg_net_props = pd.DataFrame.from_dict(class_avg_net_props, orient='index').reset_index().rename(columns={'index':'class'})
        class_avg_net_props = class_avg_net_props.loc[:,~class_avg_net_props.columns.duplicated()]

        return class_avg_net_props


# --------------------------------------------------------------------------------------------------------------------
# EXPLORING NETWORK PROPERTIES
# ----------------------------------------------------------------------------------------------------------------------
def boxplot_net_props_distribution_per_class(df_net_props, property, **kwargs):

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
    # ax.set_ylim(-0.01, 0.1)
    sns.despine(offset=10, trim=True, bottom=False)
    # fig.savefig(fname=os.path.join(RES_TASK_DIR, 'lnplot_' + np.unique(df_scores['statistic']) + '_encoding_vs_decoding.jpg'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def stripplot_net_props_distribution_per_class(df_net_props, property, **kwargs):

    class_labels = sort_class_labels(np.unique(df_net_props['class']))
    colors = {clase:COLORS[i] for i, clase in enumerate(class_labels)}

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(15,7))
    ax = plt.subplot(111)

    # sns.stripplot(x='class',
    #               y=property,
    #               data=df_net_props,
    #               palette=colors,
    #               order=class_labels,
    #               orient='v',
    #               jitter=0.2,
    #               edgecolor='dimgrey',
    #               # linewidth=,
    #               ax=ax
    #               )

    sns.swarmplot(x='class',
                  y=property,
                  data=df_net_props,
                  palette=colors,
                  order=class_labels,
                  orient='v',
                  edgecolor='dimgrey',
                  # linewidth=,
                  ax=ax
                  )


    # ax.legend(fontsize=15, frameon=False, ncol=1, loc='upper right')
    # ax.get_legend().remove()

    sns.despine(offset=10, trim=True, bottom=False)
    # fig.savefig(fname=os.path.join(RES_TASK_DIR, 'lnplot_' + np.unique(df_scores['statistic']) + '_encoding_vs_decoding.jpg'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


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
            plt.close()


def pairplot_net_props(df_net_props):

    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(10,10))
    ax = plt.subplot(111)

    sns.pairplot(df_net_props,
                 hue="rsn",
                 hue_order=['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN'],
                 palette=COLORS[:-1],
                 ax=ax
                 )

    sns.despine(offset=10, trim=True)
    ax.set_title('origin: ' + clase)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/barplot_effect_size.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/barplot_effect_size.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


# --------------------------------------------------------------------------------------------------------------------
# EXPLORING BINARY CONNECTIVITY PROFILE
# ----------------------------------------------------------------------------------------------------------------------
def get_conn_profile_per_class(conn, class_labels, class_mapping):

    conn_bin = conn.copy().astype(bool).astype(int)

    conn_profiles = []
    for i, clase in enumerate(class_labels):

        idx_class = np.where(class_mapping == clase)
        conn_profile = np.sum(conn_bin[idx_class], axis=0)

        conns_class = []
        for node in range(len(conn_bin)):
            conns_class.extend(np.repeat(class_mapping[node], conn_profile[node]))

        targets, counts = np.unique(conns_class, return_counts=True)
        df = pd.DataFrame(data = np.column_stack((targets, counts, 100*counts/np.sum(counts), 100*counts/(1015*1014))),
                          columns = ['target', 'n_conns', 'local_percent', 'global_percent'],
                          index = np.arange(len(class_labels))
                          )

        df['origin'] = clase
        df['n_conns'] = df['n_conns'].astype(int)
        df['local_percent'] = df['local_percent'].astype(float)
        df['global_percent'] = df['global_percent'].astype(float)
        df = df.reindex(columns=['origin', 'target', 'n_conns', 'local_percent', 'global_percent'])

        conn_profiles.append(df)

    return pd.concat(conn_profiles)


def barplot_conn_profile(df_conn_profile, class_list=None):

    if class_list is None: class_list = sort_class_labels(np.unique(df_conn_profile['target']))

    for i, clase in enumerate(class_list):

        sns.set(style="ticks", font_scale=2.0)
        fig = plt.figure(num=i, figsize=(10,10))

        tmp_df = df_conn_profile.loc[df_conn_profile['origin'] == clase, :]

        ax = plt.subplot(111)
        sns.barplot(x='target',
                    y='local_percent',
                    data=tmp_df,
                    order=sort_class_labels(np.unique(df_conn_profile['target'])),
                    palette=COLORS,
                    )

        ax.set_ylim(0, 60)
        sns.despine(offset=10, trim=True)
        ax.set_title('origin: ' + clase)
        # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/barplot_effect_size.eps', transparent=True, bbox_inches='tight', dpi=300)
        # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/barplot_effect_size.jpg', transparent=True, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


# --------------------------------------------------------------------------------------------------------------------
# NETWORK PROPERTIES VS CODING SCORES
# ----------------------------------------------------------------------------------------------------------------------
def regplot_net_props_vs_coding_scores(df, score, properties=None, norm_score_by=None, minmax_scale=True, fit_reg=True):
    """
        Regression plot (avg across alphas) encoding/decoding score vs (avg across nodes) network property
    """

    if properties is None: properties = network_properties.get_default_property_list()

    if norm_score_by is not None:

        # divide coding scores by degree
        # df['encoding_' + score] = df['encoding_' + score]/df['node_degree']
        # df['decoding_' + score] = df['decoding_' + score]/df['node_degree']

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

        df['encoding_' + score] = np.log(((df['encoding_' + score]-minm)/(maxm-minm))+1)
        df['decoding_' + score] = np.log(((df['decoding_' + score]-minm)/(maxm-minm))+1)

        # df['encoding_' + score] = ((df['encoding_' + score]-minm)/(maxm-minm))
        # df['decoding_' + score] = ((df['decoding_' + score]-minm)/(maxm-minm))

        # -----------------------------------------------------------------------
        # # estimate "local" min and max, and scale scores
        # df['encoding_' + score] = np.log(((df['encoding_' + score]-np.min(df['encoding_' + score]))/(np.max(df['encoding_' + score])-np.min(df['encoding_' + score]))+1))
        # df['decoding_' + score] = np.log(((df['decoding_' + score]-np.min(df['decoding_' + score]))/(np.max(df['decoding_' + score])-np.min(df['decoding_' + score]))+1))

        # -----------------------------------------------------------------------

        # scale network properties
        for prop in properties: #
            df[prop] = np.log((((df[prop]-np.min(df[prop]))/(np.max(df[prop])-np.min(df[prop])))+1))

    # plot
    sns.set(style="ticks", font_scale=2.0)

    for i, prop in enumerate(properties):

        fig = plt.figure(num=i, figsize=(20,8))

        ax1 = plt.subplot(121)
        sns.regplot(
                    x=prop,
                    y='encoding_' + score,
                    data=df,
                    fit_reg=fit_reg,
                    marker='o',
                    scatter_kws={'s':7},
                    color='#E55FA3',
                    ax=ax1,
                    )

        ax2 = plt.subplot(122)
        sns.regplot(
                    x=prop,
                    y='decoding_' + score,
                    data=df,
                    fit_reg=fit_reg,
                    marker='o',
                    scatter_kws={'s':7},
                    color='#6CC8BA',
                    ax=ax2,
                    )

        # properties axis 1
        ax1.set_title(r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], df['encoding_' + score])[0][1], 2)))
        # ax1.set_title(r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['encoding_' + score])[0], 2), \
                                                              # np.round(stats.spearmanr(df[prop], df['encoding_' + score])[1], 2)), fontsize=13)
        # ax1.legend(fontsize=15, frameon=True, ncol=1, loc=9)#'upper center')
        # ax1.get_legend().remove()
        ax1.set_ylim(0, 0.7)
        ax1.set_xlim(0, 0.7)

        # properties axis 2
        ax2.set_title(r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], df['decoding_' + score])[0][1], 2)))
        # ax2.set_title(r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['decoding_' + score])[0], 2),  \
                                                              # np.round(stats.spearmanr(df[prop], df['decoding_' + score])[1], 2)), fontsize=13)
        # ax2.legend(fontsize=18, frameon=True, ncol=1, loc=9)#'upper center')
        # ax2.get_legend().remove()
        ax2.set_ylim(0, 0.7)
        ax2.set_xlim(0, 0.7)

        sns.despine(offset=10, trim=False)
        # fig.savefig(fname=os.path.join(RES_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
        plt.show()
        plt.close()


def scatterplot_net_props_vs_coding_scores(df, score, properties=None, norm_score_by=None, minmax_scale=True, fit_reg=True, hue='class', **kwargs):
    """
        Scatter plot (avg across alphas) encoding/decoding score vs (avg across nodes) network property
    """

    if properties is None: properties = network_properties.get_default_property_list()

    if norm_score_by is not None:

        # divide coding scores by degree
        # df['encoding_' + score] = df['encoding_' + score]/df['node_degree']
        # df['decoding_' + score] = df['decoding_' + score]/df['node_degree']

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

        df['encoding_' + score] = np.log(((df['encoding_' + score]-minm)/(maxm-minm))+1)
        df['decoding_' + score] = np.log(((df['decoding_' + score]-minm)/(maxm-minm))+1)

        # df['encoding_' + score] = ((df['encoding_' + score]-minm)/(maxm-minm))
        # df['decoding_' + score] = ((df['decoding_' + score]-minm)/(maxm-minm))

        # -----------------------------------------------------------------------
        # # estimate "local" min and max, and scale scores
        # df['encoding_' + score] = np.log(((df['encoding_' + score]-np.min(df['encoding_' + score]))/(np.max(df['encoding_' + score])-np.min(df['encoding_' + score]))+1))
        # df['decoding_' + score] = np.log(((df['decoding_' + score]-np.min(df['decoding_' + score]))/(np.max(df['decoding_' + score])-np.min(df['decoding_' + score]))+1))

        # -----------------------------------------------------------------------

        # scale network properties
        for prop in properties: #
            df[prop] = np.log((((df[prop]-np.min(df[prop]))/(np.max(df[prop])-np.min(df[prop])))+1))

    # plot
    sns.set(style="ticks", font_scale=2.0)

    for i, prop in enumerate(properties):

        fig = plt.figure(num=i, figsize=(20,8))

        ax1 = plt.subplot(121)
        sns.scatterplot(
                        x=prop,
                        y='encoding_' + score,
                        data=df,
                        palette=COLORS[:-1],
                        hue=hue,
                        ax=ax1,
                        **kwargs
                        )

        ax2 = plt.subplot(122)
        sns.scatterplot(
                        x=prop,
                        y='decoding_' + score,
                        data=df,
                        palette=COLORS[:-1],
                        hue=hue,
                        ax=ax2,
                        **kwargs
                        )

        if fit_reg:
            sns.regplot(
                    x=prop,
                    y='encoding_' + score,
                    data=df,
                    scatter=False,
                    fit_reg=True,
                    ci=None,
                    truncate=True,
                    marker='D',
                    color='dimgrey', #'#E55FA3',
                    ax=ax1,
                    )

            sns.regplot(
                    x=prop,
                    y='decoding_' + score,
                    data=df,
                    scatter=False,
                    fit_reg=True,
                    ci=None,
                    truncate=True,
                    marker='D',
                    color='dimgrey', #'#6CC8BA',
                    ax=ax2,
                    )

        # properties axis 1
        ax1.set_title(r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], df['encoding_' + score])[0][1], 2)))
        # ax1.set_title(r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['encoding_' + score])[0], 2), \
                                                              # np.round(stats.spearmanr(df[prop], df['encoding_' + score])[1], 2)), fontsize=13)

        # ax1.legend(fontsize=15, frameon=True, ncol=1, loc=9) #'upper center')
        ax1.get_legend().remove()
        ax1.set_ylim(0, 0.7)
        ax1.set_xlim(0, 0.7)

        # properties axis 2
        ax2.set_title(r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], df['decoding_' + score])[0][1], 2)))
        # ax2.set_title(r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['decoding_' + score])[0], 2),  \
                                                              # np.round(stats.spearmanr(df[prop], df['decoding_' + score])[1], 2)), fontsize=13)
        # ax2.legend(fontsize=15, frameon=True, ncol=1, loc=9)#'upper center')
        ax2.get_legend().remove()
        ax2.set_ylim(0, 0.7)
        ax2.set_xlim(0, 0.7)

        sns.despine(offset=10, trim=False)
        # fig.savefig(fname=os.path.join(RES_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
        plt.show()
        plt.close()


# --------------------------------------------------------------------------------------------------------------------
# ENCODING VS DECODING - REGRESSING OUT NETWORK PROPERTIES
# ----------------------------------------------------------------------------------------------------------------------
def scatterplot_enc_vs_dec(df, score, norm_score_by=None, minmax_scale=True, hue='class', **kwargs):
    """
        Scatter plot (avg across alphas) encoding/decoding score vs (avg across nodes) network property
    """

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
