 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:10:14 2019

@author: Estefany Suarez
"""
import os
import scipy.io as sio
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from netneurotools import plotting

from . import (plot_tasks)
from .. import utils

#%matplotlib qt
COLORS = sns.color_palette("husl", 8)

color_cmap_div = np.loadtxt("C:/Users/User/Desktop/rc_tmp/cmap.dat")
cmap_div = utils.array2cmap(color_cmap_div)


# --------------------------------------------------------------------------------------------------------------------
# TIMESERIES
# ----------------------------------------------------------------------------------------------------------------------
def lineplot_timeseries(signal, class_mapping, timepoints=None, nodes=None, classes=None, demean=True, scale=False, vmin=-1, vmax=1, **kwargs):

    # remove mean across nodes from signal, per timepoint
    if demean: signal = (signal-signal.mean(axis=1)[:,np.newaxis])

    # create dataframe
    n_nodes = signal.shape[-1]

    tmp_class_mapping = np.repeat(class_mapping.copy()[np.newaxis,:],
                                  len(signal),
                                  axis=0).flatten()

    node_id = np.repeat(np.arange(n_nodes)[np.newaxis,:],
                        len(signal),
                        axis=0).flatten()

    timepoint = np.repeat(np.arange(len(signal))[:,np.newaxis],
                          n_nodes,
                          axis=1).flatten()

    signal = signal.copy().flatten()

    df = pd.DataFrame(data = np.column_stack((node_id, timepoint, signal, tmp_class_mapping)),
                        columns = ['node_id', 'timepoint', 'signal', 'class'],
                        )
    df['node_id'] = df['node_id'].astype(int)
    df['timepoint'] = df['timepoint'].astype(int)
    df['signal'] = df['signal'].astype(float)
    hue_order = plot_tasks.sort_class_labels(np.unique(df['class']))

    # select timepoints
    if timepoints is not None: df = df.loc[df['timepoint'].isin(timepoints), :]

    # select nodes
    if nodes is not None: df = df.loc[df['node_id'].isin(nodes), :]

    # select classes
    if classes is not None: df = df.loc[df['class'].isin(classes), :]

    # scale signal values between vmin and vmax
    if scale: df['signal'] = (((df['signal']-df['signal'].min())/(df['signal'].max()-df['signal'].min()))*(vmax-vmin)) + vmin

    # plot
    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(num=1, figsize=(20,5))

    ax = plt.subplot(111)
    sns.lineplot(x="timepoint",
                 y="signal",
                 data=df,
                 hue="class",
                 hue_order=hue_order, #plot_tasks.sort_class_labels(np.unique(df['class'])),
                 palette=COLORS[:-1], # palette,
                 ax=ax,
                 **kwargs
                 )


    # ax.legend(fontsize=15, frameon=True, ncol=1, loc='upper right')
    ax.get_legend().remove()

    ax.set_yticks([])
    ax.set_yticklabels('')
    # ax.set_ylabel('')

    sns.despine(offset=10, trim=True, left=True)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/line_coding_across_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/line_coding_across_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


# def lineplot_timeseries_per_class(signal, class_mapping, timepoints=None, nodes=None, classes=None, demean=True, scale=False, vmin=-1, vmax=1, **kwargs):
#
#     # remove mean across nodes from signal, per timepoint
#     if demean: signal = (signal-signal.mean(axis=1)[:,np.newaxis])
#
#     # create dataframe
#     n_nodes = signal.shape[-1]
#
#     tmp_class_mapping = np.repeat(class_mapping.copy()[np.newaxis,:],
#                                   len(signal),
#                                   axis=0).flatten()
#
#     node_id = np.repeat(np.arange(n_nodes)[np.newaxis,:],
#                         len(signal),
#                         axis=0).flatten()
#
#     timepoint = np.repeat(np.arange(len(signal))[:,np.newaxis],
#                           n_nodes,
#                           axis=1).flatten()
#
#     signal = signal.copy().flatten()
#
#     df = pd.DataFrame(data = np.column_stack((node_id, timepoint, signal, tmp_class_mapping)),
#                         columns = ['node_id', 'timepoint', 'signal', 'class'],
#                         )
#     df['node_id'] = df['node_id'].astype(int)
#     df['timepoint'] = df['timepoint'].astype(int)
#     df['signal'] = df['signal'].astype(float)
#     hue_order = plot_tasks.sort_class_labels(np.unique(df['class']))
#
#     # select timepoints
#     if timepoints is not None: df = df.loc[df['timepoint'].isin(timepoints), :]
#
#     # select nodes
#     if nodes is not None: df = df.loc[df['node_id'].isin(nodes), :]
#
#     # select classes
#     if classes is not None: df = df.loc[df['class'].isin(classes), :]
#
#     # scale signal values between vmin and vmax
#     if scale: df['signal'] = (((df['signal']-df['signal'].min())/(df['signal'].max()-df['signal'].min()))*(vmax-vmin)) + vmin
#
#     # plot
#     class_labels = plot_tasks.sort_class_labels(np.unique(df['class']))
#
#     sns.set(style="ticks", font_scale=2.0)
#     fig = plt.figure(num=1, figsize=(18,8*len(class_labels)))
#
#     for i, clase in enumerate(class_labels):
#
#         tmp_df = df.loc[df['class']==clase, :]
#
#         ax = plt.subplot(len(class_labels), 1, i+1)
#         sns.lineplot(x="timepoint",
#                      y="signal",
#                      data=tmp_df,
#                      hue="class",
#                      hue_order=hue_order,
#                      palette=COLORS[:-1],
#                      ax=ax,
#                      **kwargs
#                      )
#
#         # ax.legend(fontsize=15, frameon=True, ncol=1, loc='upper right')
#         ax.get_legend().remove()
#         sns.despine(offset=10, trim=True)
#
#     # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/line_coding_across_alpha.eps', transparent=True, bbox_inches='tight', dpi=300)
#     # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/line_coding_across_alpha.jpg', transparent=True, bbox_inches='tight', dpi=300)
#
#     plt.show()
#     plt.close()
#
#

# def recurrenceplot_timeseries(signal, class_mapping, timepoints=None, nodes=None, classes=None, demean=True, scale=False, vmin=-1, vmax=1, **kwargs):
#
#     # remove mean across nodes from signal, per timepoint
#     if demean: signal = (signal-signal.mean(axis=1)[:,np.newaxis])
#
#     # select timepoints
#     if timepoints is not None: signal = signal[timepoints, :]
#
#     # select nodes
#     if nodes is not None: signal = signal[:, nodes]
#
#     # select classes
#     if classes is not None:
#
#         signal = signal[:, np.where()]
#
#     # scale signal values between vmin and vmax
#     if scale: df['signal'] = (((df['signal']-df['signal'].min())/(df['signal'].max()-df['signal'].min()))*(vmax-vmin)) + vmin
#
#









def spreading(coords, res_states, input_nodes=None, apply_thr=True, thr=0.1):

    def plot_brain(ax, coords, colors=None, step=None, apply_thr=False):
        if apply_thr: cmap = None
        else: cmap = 'RdBu_r'

        # superior view
        mapp1 = ax.scatter(xs=coords[:,0],
                           ys=coords[:,1],
                           zs=coords[:,2],
                           s=180,
                           c=colors,
                           linewidths=0.5,
                           edgecolors='grey',
                           cmap=cmap)
        ax.view_init(90,270)
        ax.grid(False)
        ax.axis('off')
        ax.set(xlim=0.57 * np.array(ax.get_xlim()),
               ylim=0.57 * np.array(ax.get_ylim()),
               zlim=0.60 * np.array(ax.get_zlim()),
               aspect=1.1
               )

        # plt.colorbar(mapp1, aspect=20) #shrink=0.5,
        plt.gca().patch.set_facecolor('white')
        plt.title('Time point:  ' + str(step) + '%')

    # -------------------------------------------------------------------------
    if plot:
        c_source, c_on, c_off = '#fa6e59', '#4897d8', '#c9d1c8'
        time_points = [1, 5, 10, 15, 20, 25, 50, 75, 90, 100] #time points in percentiles

        fig = plt.figure(num=3, figsize=(0.8*5*len(time_points)*0.6,5))
        for idx, t in enumerate(time_points):
            t_idx = int(np.percentile(np.arange(len(res_states)), t))

            if apply_thr:
                colors = np.array([c_off for _ in range(res_states.shape[1])])
                colors[np.where(res_states[t_idx] >= thr)[0]] = c_on
            else:
                colors = res_states[t_idx]
            if input_nodes is not None: colors[input_nodes] = c_source

            ax = plt.subplot(1, len(time_points), idx+1, projection='3d')

            coords = (coords-np.mean(coords, axis=0))/np.std(coords, axis=0)
            plot_brain(ax, coords, colors=colors, step=t, apply_thr=apply_thr)

        sns.despine(offset=10, trim=True)
        fig.tight_layout()
        plt.show()


def spikes(res_states, class_names, class_mapping, output_nodes, thr=0.1):

    fig = plt.figure(num=4, figsize=(0.7*20,0.9*7))
    ax = plt.subplot(111)

    if output_nodes is not None:
        res_states = res_states[:, output_nodes]

    if (output_nodes is not None) and (class_mapping is not None):
        class_mapping = class_mapping[output_nodes]

    n_timesteps = len(res_states)
    n_nodes = res_states.shape[1]

    for node in range(n_nodes):
        x = np.arange(n_timesteps).astype(float)
        y = node * np.ones_like(x).astype(float)

        x[res_states[:,node] <= thr] = np.nan
        y[res_states[:,node] <= thr] = np.nan

        if class_mapping is not None:
            color = COLORS[np.where(class_names==class_mapping[node])[0][0]]
            label = class_mapping[node]

        ax.plot(x,
                y,
                markersize=3,
                c=color, #'k'
                label=label,
                )

    ax.set_xlabel('time steps')
    ax.set_ylabel('nodes')
    ax.set_ylim(0, n_nodes+10)

    sns.despine(offset=10, trim=True)
    fig.tight_layout()
    plt.show()


# def brain_dots(coords, class_names, class_mapping, nodes, view='superior', ax=None, title=None):
#
#     # z-score coordinates for better visualization
#     coords = (coords-np.mean(coords, axis=0))/np.std(coords, axis=0)
#
#     # create mask according to nodes
#     color_mask = np.array([1 if node in nodes else 0 for node in range(len(coords))]).astype(int)
#
#     # create array of colors according to class mapping
#     class_mapping_int = np.array([np.where(class_names == mapp)[0][0] for mapp in class_mapping]).astype(int)
#     colors = np.array([COLORS[np.where(class_names == mapp)[0][0]] if node in nodes else 'none' for node, mapp in enumerate(class_mapping)])
#
#     size = 300
#
#     if ax is None:
#         fig = plt.figure(num=1, figsize=(.8*5.25,.8*4))#2*5.25,2*4))
#         ax = plt.subplot(111, projection='3d')
#
#     ax.scatter(xs=coords[color_mask == 1,0],
#                ys=coords[color_mask == 1,1],
#                zs=coords[color_mask == 1,2],
#                c=colors[color_mask == 1],
#                s=size,
#                linewidths=0.8,
#                edgecolors='dimgrey',
#                alpha=0.6
#                )
#
#     ax.scatter(xs=coords[color_mask == 0,0],
#                ys=coords[color_mask == 0,1],
#                zs=coords[color_mask == 0,2],
#                s=size,
#                linewidths=0.8,
#                edgecolors='dimgrey',
#                facecolors=colors[color_mask == 0]
#                )
#
#     # ax.set_title('XXXX', fontsize=10, loc=)
#     ax.grid(False)
#     ax.axis('off')
#
#     if view == 'superior':
#         ax.view_init(90,270)
#         ax.set(xlim=0.57 *np.array(ax.get_xlim()),
#                ylim=0.57 *np.array(ax.get_ylim()),
#                zlim=0.60 *np.array(ax.get_zlim()),
#                aspect=1.1
#                )
#
#     if view == 'left':
#         ax.view_init(0,180)
#         ax.set(xlim=0.59 * np.array(ax.get_xlim()),
#                ylim=0.59 * np.array(ax.get_ylim()),
#                zlim=0.60 * np.array(ax.get_zlim()),
#                # aspect=0.55 #1.1
#                )
#
#     if view == 'right':
#         ax.view_init(0,0)
#         ax.set(xlim=0.59 * np.array(ax.get_xlim()),
#                ylim=0.59 * np.array(ax.get_ylim()),
#                zlim=0.60 * np.array(ax.get_zlim()),
#                # aspect=0.55 #1.1
#                )
#
#     if title is not None: plt.title(title, fontsize=15, loc='left') #pad=5)
#
#     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     plt.gca().patch.set_facecolor('white')
#     sns.despine()
#
#     if ax is None: plt.show()
#

#-------------------------------------------------------------------------------
# def brain_dots(coords, class_names, class_mapping, nodes, title, view='superior', ax=None):
#     # ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN', 'subctx']
#     TMP_COLORS = [COLORS[6], COLORS[6], COLORS[3], COLORS[6], COLORS[6], COLORS[6], COLORS[6], COLORS[5]]
#
#     # z-score coordinates for better visualization
#     coords = (coords-np.mean(coords, axis=0))/np.std(coords, axis=0)
#
#     # create mask according to nodes
#     color_mask = np.array([1 if node in nodes else 0 for node in range(len(coords))]).astype(int)
#
#     # create array of colors according to class mapping
#     class_mapping_int = np.array([np.where(class_names == mapp)[0][0] for mapp in class_mapping]).astype(int)
#     colors = np.array([TMP_COLORS[np.where(class_names == mapp)[0][0]] if node in nodes else 'none' for node, mapp in enumerate(class_mapping)])
#
#     size = 70
#
#     if ax is None:
#         fig = plt.figure(num=1, figsize=(.8*5.25,.8*4))#2*5.25,2*4))
#         ax = plt.subplot(111, projection='3d')
#
#     ax.scatter(xs=coords[color_mask == 1,0],
#                ys=coords[color_mask == 1,1],
#                zs=coords[color_mask == 1,2],
#                c=colors[color_mask == 1],
#                s=size,
#                linewidths=0.2,
#                edgecolors='dimgrey',
#                alpha=0.60
#                )
#
#     ax.scatter(xs=coords[color_mask == 0,0],
#                ys=coords[color_mask == 0,1],
#                zs=coords[color_mask == 0,2],
#                s=size,
#                linewidths=0.12,
#                edgecolors='dimgrey',
#                facecolors=colors[color_mask == 0]
#                )
#
#     # ax.set_title('XXXX', fontsize=10, loc=)
#     ax.grid(False)
#     ax.axis('off')
#
#     if view == 'superior':
#         ax.view_init(90,270)
#         ax.set(xlim=0.57 *np.array(ax.get_xlim()),
#                ylim=0.57 *np.array(ax.get_ylim()),
#                zlim=0.60 *np.array(ax.get_zlim()),
#                aspect=1.1
#                )
#
#     if view == 'left':
#         ax.view_init(0,180)
#         ax.set(xlim=0.59 * np.array(ax.get_xlim()),
#                ylim=0.59 * np.array(ax.get_ylim()),
#                zlim=0.60 * np.array(ax.get_zlim()),
#                # aspect=0.55 #1.1
#                )
#
#     if view == 'right':
#         ax.view_init(0,0)
#         ax.set(xlim=0.59 * np.array(ax.get_xlim()),
#                ylim=0.59 * np.array(ax.get_ylim()),
#                zlim=0.60 * np.array(ax.get_zlim()),
#                # aspect=0.55 #1.1
#                )
#
#
#     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     plt.gca().patch.set_facecolor('white')
#     sns.despine()
#
#     fig.savefig(fname='C:/Users/User/Desktop/'+title+'.eps', transparent=True, dpi=300, bbox_inches='tight')
#
#     if ax is None: plt.show()
