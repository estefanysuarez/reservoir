import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)

import seaborn as sns
from scipy import stats
from .plot_tasks import (sort_class_labels)
from ..network import network_properties

from netneurotools import plotting
from netneurotools import datasets

COLORS = sns.color_palette("husl", 8)
ENCODE_COL = '#E55FA3'
DECODE_COL = '#6CC8BA'


# --------------------------------------------------------------------------------------------------------------------
# GENERAL
# ----------------------------------------------------------------------------------------------------------------------
def concat_results(path, class_mapping, n_samples=1000):

    df_net_props = []
    for sample_id in range(n_samples):

        print('\n sample_id:  ' + str(sample_id))
        success_sample = True

        try:
            net_props = pd.read_csv(os.path.join(path, 'net_props_' + str(sample_id) + '.csv'), index_col=0)
            net_prop_names = list(net_props.columns)
            net_props['class'] = class_mapping
            net_props = get_net_props_per_class(net_props)
            net_props['sample_id'] = sample_id

        except:
            success_sample = False
            print('\n Could not find sample No.  ' + str(sample_id))

            pass

        if success_sample:  df_net_props.append(net_props)

    # concatenate dataframes
    df_net_props = pd.concat(df_net_props)
    df_net_props = df_net_props.loc[df_net_props['class'] != 'subctx', :]
    df_net_props = df_net_props.reset_index(drop=True)
    df_net_props = df_net_props[['sample_id', 'class'] + net_prop_names]

    return df_net_props


def get_net_props_per_class(df_net_props):
    """
    Returns a DataFrame with the average network properties per class per subject
    """
    # get class labels
    class_labels = sort_class_labels(np.unique(df_net_props['class']))

    if len(class_labels) > 8:
        return df_net_props

    else:

        class_avg_net_props = {clase: df_net_props.loc[df_net_props['class'] == clase, :].mean() for clase in class_labels}
        # class_avg_net_props = {clase: df_net_props.loc[df_net_props['class'] == clase, :].median() for clase in class_labels}
        class_avg_net_props = pd.DataFrame.from_dict(class_avg_net_props, orient='index').reset_index().rename(columns={'index':'class'})
        class_avg_net_props = class_avg_net_props.loc[:,~class_avg_net_props.columns.duplicated()]

        return class_avg_net_props


# --------------------------------------------------------------------------------------------------------------------
# EXPLORING NETWORK PROPERTIES & BINARY CONNECTIVITY PROFILE
# --------------------------------- -------------------------------------------------------------------------------------
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


def distplt_net_props_per_class(df_net_props):

    if 'sample_id' in df_net_props.columns: net_props = np.array(df_net_props.columns[2:])
    else:net_props = np.array(df_net_props.columns[:-1])

    for prop in net_props:
        try:
            # -----------
            sns.set(style="ticks", font_scale=2.0)

            fig = plt.figure(figsize=(18,7))
            ax = plt.subplot(111)

            df_net_props[prop] = (df_net_props[prop]-min(df_net_props[prop]))/(max(df_net_props[prop])-min(df_net_props[prop]))

            for i, clase in enumerate(sort_class_labels(np.unique(df_net_props['class']))):
                sns.distplot(a=df_net_props.loc[df_net_props['class'] == clase, prop].values,
                             bins=50,
                             hist=False,
                             kde=True,
                             kde_kws={'shade':True, 'clip':(0,1)},
                             color=COLORS[i],
                             label=clase,
                             )

            ax.set_xlim(0,1)
            ax.xaxis.set_major_locator(MultipleLocator(0.25))

            # ax.set_ylim(0,20)
            ax.get_yaxis().set_visible(False)

            ax.get_legend().remove()

            sns.despine(offset=10, trim=True, left=True)
            plt.suptitle(' '.join(prop.split('_')))
        #    fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figures_RC/eps', f'{prop}_{CONNECTOME[-3:]}.eps'), transparent=True, bbox_inches='tight', dpi=300)
            plt.show()

        except ZeroDivisionError:
            pass



# --------------------------------------------------------------------------------------------------------------------
# PI - NETWORK PROPERTIES VS CODING SCORES
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


#%% --------------------------------------------------------------------------------------------------------------------
# PII - NETWORK PROPERTIES VS CODING SCORES
# ----------------------------------------------------------------------------------------------------------------------
def distplt_corr_net_props_and_scores_per_prop(corr, net_prop_names, dynamics):

    colors = sns.color_palette(["#2ecc71", "#3498db",  "#9b59b6"])

    for j, prop in enumerate(net_prop_names):

        sns.set(style="ticks", font_scale=2.0)
        fig = plt.figure(figsize=(10,7))
        ax = plt.subplot(111)

        for i, dyn_regime in enumerate(dynamics):
            sns.distplot(a=corr[:,j,i],
                         bins=50,
                         hist=False,
                         kde=True,
                         kde_kws={'shade':True, 'clip':(-1,1)},
                         color=colors[i],
                         label=' '.join(dyn_regime.split('_')),
                         )

        ax.set_xlim(-1,1)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        # ax.set_ylim(0,15)
        ax.get_yaxis().set_visible(False)

        # ax.get_legend().remove()

        plt.suptitle(' '.join(prop.split('_')))

        sns.despine(offset=10, trim=True, left=True)
        fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figures_RC/eps', f'{prop}.eps'), transparent=True, bbox_inches='tight', dpi=300)
        plt.show()


def distplot_corr_net_props_and_scores_per_regime(corr, net_prop_names, dynamics):

    colors = sns.color_palette("cubehelix", len(net_prop_names)+1)[1:]
    #("GnBu_d", len(net_prop_names)) ("cubehelix", len(net_prop_names)) sns.cubehelix_palette(len(net_prop_names))

    for i, dyn_regime in enumerate(dynamics):

         sns.set(style="ticks", font_scale=2.0)

         fig = plt.figure(figsize=(20,7))
         ax = plt.subplot(111)

         for j, prop in enumerate(net_prop_names):

             sns.distplot(a=corr[:,j,i],
                          bins=50,
                          hist=False,
                          kde=True,
                          kde_kws={'shade':True, 'clip':(-1,1)},
                          color=colors[j],
                          label=' '.join(prop.split('_')),
                          )

         ax.set_xlim(-1,1)
         ax.xaxis.set_major_locator(MultipleLocator(0.25))

         # ax.set_ylim(0,20)

         plt.suptitle(' '.join(dyn_regime.split('_')))
         sns.despine(offset=10, trim=True)
         plt.show()


def distplot_corr_net_props_and_scores_per_regime_sorted(corr, net_prop_names, dynamics):

    colors = sns.color_palette("coolwarm", len(net_prop_names))
    #("GnBu_d", len(net_prop_names)) ("cubehelix", len(net_prop_names)) sns.cubehelix_palette(len(net_prop_names))

    for i, dyn_regime in enumerate(dynamics):

         sns.set(style="ticks", font_scale=2.0)

         fig = plt.figure(figsize=(20,7))
         ax = plt.subplot(111)

         median = np.median(corr[:,:,i], axis=0)
         idx_sorted_props = np.argsort(median)
         sorted_net_props = np.array(net_prop_names.copy())[idx_sorted_props]  #np.array(df_net_props.columns[2:])[sorted_props]

         for j, prop in zip(idx_sorted_props, sorted_net_props):

             sns.distplot(a=corr[:,j,i],
                          bins=50,
                          hist=False,
                          kde=True,
                          kde_kws={'shade':True, 'clip':(-1,1)},
                          color=colors[np.where(idx_sorted_props==j)[0][0]],
                          label=' '.join(prop.split('_')),
                          )

         ax.set_xlim(-1,1)
    #     ax.set_ylim(0,20)
         ax.xaxis.set_major_locator(MultipleLocator(0.25))

         plt.suptitle(' '.join(dyn_regime.split('_')))

         sns.despine(offset=10, trim=True)
         plt.show()
