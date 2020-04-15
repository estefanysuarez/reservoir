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

    if len(class_labels) > 7:
        return df_net_props

    else:
        class_avg_net_props = {clase:df_net_props.loc[df_net_props['class'] == clase, :].mean() for clase in class_labels}
        class_avg_net_props = pd.DataFrame.from_dict(class_avg_net_props, orient='index').reset_index().rename(columns={'index':'class'})
        class_avg_net_props = class_avg_net_props.loc[:,~class_avg_net_props.columns.duplicated()]

        return class_avg_net_props


# --------------------------------------------------------------------------------------------------------------------
# EXPLORING NETWORK PROPERTIES
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
    # ax.set_ylim(-0.01, 0.1)
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
def get_conn_profiles(conn_wei, class_labels, class_mapping):

    conn_bin = conn_wei.copy().astype(bool).astype(int)

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
def regplot_net_props_vs_coding_scores(df_encoding, df_decoding, score, df_net_props, properties=None, norm_by_deg=False, minmax_scale=True, fit_reg=True, **kwargs):
    """
        Regression plot (avg across alphas) encoding/decoding score vs (avg across nodes) network property
    """

    # get coding scores per class
    df_class_scores = get_coding_scores_per_class(df_encoding,
                                                  df_decoding,
                                                  **kwargs)

    df_class_encoding_scores = df_class_scores.loc[df_class_scores['coding'] == 'encoding', ['class', score]].rename(columns={score:'encoding_' + score})
    df_class_encoding_scores.fillna({'encoding_' + score:np.nanmean(df_class_encoding_scores['encoding_' + score])}, inplace=True)

    df_class_decoding_scores = df_class_scores.loc[df_class_scores['coding'] == 'decoding', ['class', score]].rename(columns={score:'decoding_' + score})
    df_class_decoding_scores.fillna({'decoding_' + score:np.nanmean(df_class_decoding_scores['decoding_' + score])}, inplace=True)

    # get network properties per class
    df_net_props = get_net_props_per_class(df_net_props)
    if properties is None: properties = network_properties.get_default_property_list()

    # merge coding scores and network properties dataframes
    df = pd.merge(df_class_encoding_scores, df_class_decoding_scores, on='class')
    df = pd.merge(df, df_net_props, on='class')

    # return df

    if norm_by_deg:
        df['encoding_' + score] = df['encoding_' + score]/df['node_degree']
        df['decoding_' + score] = df['decoding_' + score]/df['node_degree']

    if minmax_scale:
        # scale coding scores
        df['encoding_' + score] = np.log(((df['encoding_' + score]-np.min(df['encoding_' + score]))/(np.max(df['encoding_' + score])-np.min(df['encoding_' + score]))+1))
        df['decoding_' + score] = np.log(((df['decoding_' + score]-np.min(df['decoding_' + score]))/(np.max(df['decoding_' + score])-np.min(df['decoding_' + score]))+1))

        # scale network properties
        for prop in properties: #
            df[prop] = np.log((((df[prop]-np.min(df[prop]))/(np.max(df[prop])-np.min(df[prop])))+1))


    # plot
    sns.set(style="ticks")

    for i, prop in enumerate(properties):

        fig = plt.figure(num=i, figsize=(11,4))
        ax1 = plt.subplot(121)
        sns.regplot(
                    x=prop,
                    y='encoding_' + score,
                    data=df,
                    # label=,
                    fit_reg=fit_reg,
                    ax=ax1,
                    marker='o',
                    scatter_kws={'s':7},
                    color='#E55FA3',
                    )

        ax2 = plt.subplot(122)
        sns.regplot(
                    x=prop,
                    y='decoding_' + score,
                    data=df,
                    # label=,
                    fit_reg=fit_reg,
                    ax=ax2,
                    marker='o',
                    scatter_kws={'s':7},
                    color='#6CC8BA',
                    )

        # properties axis 1
        ax1.set_title(r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], df['encoding_' + score])[0][1], 2)), fontsize=13)
        # ax1.set_title(r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['encoding_' + score])[0], 2), \
        #                                                       np.round(stats.spearmanr(df[prop], df['encoding_' + score])[1], 2)), fontsize=13)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=15, labelpad=7)
        ax1.set_ylabel(ax1.get_ylabel(), fontsize=15, labelpad=7)
        # ax1.legend(fontsize=15, frameon=True, ncol=1, loc=9)#'upper center')
        # ax1.get_legend().remove()


        # properties axis 2
        ax2.set_title(r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], df['decoding_' + score])[0][1], 2)), fontsize=13)
        # ax2.set_title(r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['decoding_' + score])[0], 2),  \
        #                                                       np.round(stats.spearmanr(df[prop], df['decoding_' + score])[1], 2)), fontsize=13)
        ax2.set_xlabel(ax2.get_xlabel(), fontsize=15, labelpad=7)
        ax2.set_ylabel(ax2.get_ylabel(), fontsize=15, labelpad=7)
        # ax2.legend(fontsize=18, frameon=True, ncol=1, loc=9)#'upper center')
        # ax2.get_legend().remove()

        sns.despine(offset=10, trim=True)
        # fig.savefig(fname=os.path.join(RES_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
        plt.show()
        plt.close()


def scatterplot_net_props_vs_coding_scores(df_encoding, df_decoding, score, df_net_props, hue, properties=None, norm_by_deg=False, minmax_scale=True, plot_reg=False, **kwargs):
    """
    Regression plot (avg across alphas) encoding/decoding score vs (avg across nodes) network property
    """

    # get coding scores per class
    df_class_scores = get_coding_scores_per_class(df_encoding,
                                                  df_decoding,
                                                  **kwargs)

    df_class_encoding_scores = df_class_scores.loc[df_class_scores['coding'] == 'encoding', ['class', score]].rename(columns={score:'encoding_' + score})
    df_class_encoding_scores.fillna({'encoding_' + score:np.nanmean(df_class_encoding_scores['encoding_' + score])}, inplace=True)

    df_class_decoding_scores = df_class_scores.loc[df_class_scores['coding'] == 'decoding', ['class', score]].rename(columns={score:'decoding_' + score})
    df_class_decoding_scores.fillna({'decoding_' + score:np.nanmean(df_class_decoding_scores['decoding_' + score])}, inplace=True)

    # get network properties per class
    df_net_props = get_net_props_per_class(df_net_props)
    if properties is None: properties = network_properties.get_default_property_list()

    # merge coding scores and network properties dataframes
    df = pd.merge(df_class_encoding_scores, df_class_decoding_scores, on='class')
    df = pd.merge(df, df_net_props, on='class')

    # return df

    if norm_by_deg:

        # # divide coding scores by degree
        # df['encoding_' + score] = df['encoding_' + score]/df['node_degree']
        # df['decoding_' + score] = df['decoding_' + score]/df['node_degree']

        # regress out degree from coding scores
        X = np.array(df['node_degree'])[:, np.newaxis]
        # X = np.array(df['node_strength'])[:, np.newaxis]

        reg_enc = LinearRegression().fit(X, y=df['encoding_' + score])
        tmp_encode_scores = df['encoding_' + score] - reg_enc.predict(X)
        df['encoding_' + score] = tmp_encode_scores

        reg_dec = LinearRegression().fit(X, y=df['decoding_' + score])
        tmp_decode_scores = df['decoding_' + score] - reg_dec.predict(X)
        df['decoding_' + score] = tmp_decode_scores

    if minmax_scale:
        # scale coding scores
        df['encoding_' + score] = np.log(((df['encoding_' + score]-np.min(df['encoding_' + score]))/(np.max(df['encoding_' + score])-np.min(df['encoding_' + score]))+1))
        df['decoding_' + score] = np.log(((df['decoding_' + score]-np.min(df['decoding_' + score]))/(np.max(df['decoding_' + score])-np.min(df['decoding_' + score]))+1))

        # scale network properties
        for prop in properties: #
            df[prop] = np.log((((df[prop]-np.min(df[prop]))/(np.max(df[prop])-np.min(df[prop])))+1))


    # plot
    sns.set(style="ticks")
    for i, prop in enumerate(properties):

        fig = plt.figure(num=i, figsize=(11,4))

        ax1 = plt.subplot(121)
        sns.scatterplot(
                        x=prop, #'encoding_' + score_type,
                        y='encoding_' + score, #prop,
                        # style='rsn',
                        hue=hue,
                        data=df,
                        # label=r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], df['encoding_' + score])[0][1], 2)),
                        # label=r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['encoding_' + score])[0], 2), np.round(stats.spearmanr(df[prop], df['encoding_' + score])[1], 2)),
                        ax=ax1,
                        palette=sns.color_palette("husl", 8)[:-1],
                        )

        ax2 = plt.subplot(122)
        sns.scatterplot(
                        x=prop,
                        y='decoding_' + score,
                        # style='rsn',
                        hue=hue,
                        data=df,
                        # label=r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], df['decoding_' + score])[0][1], 2)),
                        # label=r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['decoding_' + score])[0], 2), np.round(stats.spearmanr(df[prop], df['decoding_' + score])[1], 2)),
                        ax=ax2,
                        palette=sns.color_palette("husl", 8)[:-1],
                        )

        if plot_reg:
            sns.regplot(
                    x=prop,
                    y='encoding_' + score,
                    data=df,
                    scatter=False,
                    fit_reg=True,
                    ci=None,
                    truncate=True,
                    ax=ax1,
                    marker='D',
                    color='#E55FA3',
                    )
            sns.regplot(
                    x=prop,
                    y='decoding_' + score,
                    data=df,
                    scatter=False,
                    fit_reg=True,
                    ci=None,
                    truncate=True,
                    ax=ax2,
                    marker='D',
                    color='#6CC8BA',
                    )

        # properties axis 1
        ax1.set_title(r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], df['encoding_' + score])[0][1], 2)), fontsize=13)
        # ax1.set_title(r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['encoding_' + score])[0], 2), \
                                                              # np.round(stats.spearmanr(df[prop], df['encoding_' + score])[1], 2)), fontsize=13)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=15, labelpad=7)
        ax1.set_ylabel(ax1.get_ylabel(), fontsize=15, labelpad=7)
        # ax1.legend(fontsize=15, frameon=True, ncol=1, loc=9) #'upper center')
        ax1.get_legend().remove()

        # properties axis 2
        ax2.set_title(r'$R: %.2f $' % (np.round(np.corrcoef(df[prop], df['decoding_' + score])[0][1], 2)), fontsize=13)
        # ax2.set_title(r'$\rho: %.2f \;\;\; p_{val}= %.3f$' % (np.round(stats.spearmanr(df[prop], df['decoding_' + score])[0], 2),  \
                                                              # np.round(stats.spearmanr(df[prop], df['decoding_' + score])[1], 2)), fontsize=13)
        ax2.set_xlabel(ax2.get_xlabel(), fontsize=15, labelpad=7)
        ax2.set_ylabel(ax2.get_ylabel(), fontsize=15, labelpad=7)
        # ax2.legend(fontsize=15, frameon=True, ncol=1, loc=9)#'upper center')
        ax2.get_legend().remove()

        sns.despine(offset=10, trim=True)
        # fig.savefig(fname=os.path.join(RES_DIR, 'performance.jpg'), transparent=True, bbox_inches='tight', dpi=300,)
        plt.show()
        plt.close()
