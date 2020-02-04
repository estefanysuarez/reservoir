 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:10:14 2019

@author: Estefany Suarez
"""
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from netneurotools import plotting
import seaborn as sns

import tasks
import utils

#%matplotlib qt

COLORS = sns.color_palette("husl", 8)

color_cmap_div = np.loadtxt("C:/Users/User/Desktop/cmap.dat")
cmap_div = utils.array2cmap(color_cmap_div)


def dynamics_and_performance(task, factor, coords, res_states, target, class_names, class_mapping, t_ini=0, t_end=100, nodes=None, alpha=None, center=False):

    sns.set(style="ticks")


    fig = plt.figure(num=1, figsize=(0.7*8*4,0.7*8))

    # brain plot
    ax1 = plt.subplot(141, projection='3d')
    if alpha is not None: title =  r'$\alpha$' + ' = ' + str(alpha)
    brain_dots(coords=coords,
               class_names=class_names,
               class_mapping=class_mapping,
               nodes=nodes,
               view='superior',
               title=title,
               ax=ax1
               )

    # plot time series
    ax2 = plt.subplot(142)
    timeseries(task=task,
               factor=factor,
               res_states=res_states.copy(),
               class_names=class_names,
               class_mapping=class_mapping.copy(),
               t_ini=t_ini,
               t_end=t_end,
               nodes=nodes,
               center=center,
               ax=ax2
               )

    # plot correlation matrix
    ax3 = plt.subplot(143)
    correlation(task=task,
                factor=factor,
                res_states=res_states.copy(),
                class_names=class_names,
                class_mapping=class_mapping.copy(),
                t_ini=t_ini,
                t_end=t_end,
                nodes=nodes,
                center=center,
                ax=ax3
                )

    # plot task performance
    ax4 = plt.subplot(144)
    perf, _ = tasks.run_single_tasks(task=task,
                                     target=target,
                                     res_states=res_states.copy(),
                                     readout_nodes=nodes,
                                     ax=ax4
                                     )

    ax4.set_xlabel('target')
    ax4.set_ylabel('predicted', labelpad=0.1)
    ax4.set()

    # plt.legend(frameon=False, loc=8, ncol=3) #fontsize=17, ncol=2, loc=9
    sns.despine(trim=True) #offset=10,

    # plt.subplots_adjust(left=0.1, right=1, top=1, bottom=0, wspace=0.3) #
    # plt.show()
    # plt.close()

    return perf


def timeseries_and_correlation(task, factor, res_states, class_names, class_mapping, t_ini=0, t_end=100, nodes=None, center=False):

    fig = plt.figure(num=1, figsize=(0.7*8*2,0.7*7))

    ax1 = plt.subplot(121)
    timeseries(task, factor, res_states, class_names, class_mapping, t_ini=0, t_end=100, nodes=None, center=False, ax=ax1)

    ax2 = plt.subplot(122)
    correlation(task, factor, res_states, class_names, class_mapping, t_ini=0, t_end=100, nodes=None, center=False, ax=ax2)

    plt.show()


def correlation(task, factor, res_states, class_names, class_mapping, t_ini=0, t_end=100, nodes=None, center=False, ax=None):

    # convert mapping from string to integer
    class_mapping = np.array([np.where(class_names == mapp)[0][0] for mapp in class_mapping]).astype(int)

    if center: res_states = res_states.copy()-res_states.mean(axis=1)[:,np.newaxis]
    if nodes is not None:
        res_states  = res_states[:, nodes]
        class_mapping = class_mapping[nodes]
        class_names = [class_names[mapp] for mapp in np.sort(np.unique(class_mapping))]
        dict_class_mapping = {mapp:idx for (idx, mapp) in enumerate(np.unique(class_mapping))}
        class_mapping = np.array([dict_class_mapping[mapp] for mapp in class_mapping]).astype(int)

    t_ini = int(np.percentile(np.arange(len(res_states)), t_ini))
    t_end = int(np.percentile(np.arange(len(res_states)), t_end))
    res_states = res_states[t_ini:t_end]

    corr_matrix = np.corrcoef(res_states, rowvar=False)

    if ax is None:
        fig = plt.figure(num=3, figsize=(7,5))
        ax = plt.subplot(111)

    plotting.plot_mod_heatmap(data=corr_matrix,
                              communities=class_mapping-np.min(class_mapping),
                              xlabels=class_names,
                              ylabels=class_names,
                              cmap=cmap_div, #'RdBu_r',
                              vmin=-1.0,
                              vmax=1.0,
                              # cbar=False,
                              # center=0.0,
                              ax=ax
                              )

    plt.title(task + ' - FC - factor: ' + str(factor), fontsize=15)
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    if ax is None: plt.show()


def timeseries(task, factor, res_states, class_names, class_mapping, t_ini =0, t_end=100, nodes=None, center=False, ax=None):

    if center: res_states = res_states.copy()-res_states.mean(axis=1)[:,np.newaxis]
    if nodes is not None:
        res_states  = res_states[:,nodes]
        class_mapping = class_mapping[nodes]

    t_ini = int(np.percentile(np.arange(len(res_states)), t_ini))
    t_end = int(np.percentile(np.arange(len(res_states)), t_end))
    res_states = res_states[t_ini:t_end]

    if ax is None:
        fig = plt.figure(num=2, figsize=(8,5))
        ax = plt.subplot(111)

    #-----------------------------------------------
    l1 = ax.plot(np.arange(t_ini, t_end), res_states, alpha=0.8)
    for idx, line in enumerate(l1):
        line.set_color(COLORS[np.where(class_names == class_mapping[idx])[0][0]])

    # sns.lineplot(data=res_states,
    #              palette=[COLORS[np.where(class_names == mapp)[0][0]] for mapp in class_mapping],
    #              dashes=False,
    #              legend=False,
    #              # col=class_mapping,
    #              ax=ax
    #              )
    #-----------------------------------------------
    # ax.get_legend().remove()
    ax.set_ylabel('signal')

    # ax.set_xticklabels(labels=np.arange(t_ini, t_end))
    ax.set_xlabel('time steps')

    plt.title(task + ' - res_states - factor: ' + str(factor), fontsize=15)
    sns.despine(offset=10, trim=True)
    plt.tight_layout()

    if ax is None: plt.show()


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


def brain_dots(coords, class_names, class_mapping, nodes, view='superior', ax=None, title=None):

    # z-score coordinates for better visualization
    coords = (coords-np.mean(coords, axis=0))/np.std(coords, axis=0)

    # create mask according to nodes
    color_mask = np.array([1 if node in nodes else 0 for node in range(len(coords))]).astype(int)

    # create array of colors according to class mapping
    class_mapping_int = np.array([np.where(class_names == mapp)[0][0] for mapp in class_mapping]).astype(int)
    colors = np.array([COLORS[np.where(class_names == mapp)[0][0]] if node in nodes else 'none' for node, mapp in enumerate(class_mapping)])

    size = 300

    if ax is None:
        fig = plt.figure(num=1, figsize=(.8*5.25,.8*4))#2*5.25,2*4))
        ax = plt.subplot(111, projection='3d')

    ax.scatter(xs=coords[color_mask == 1,0],
               ys=coords[color_mask == 1,1],
               zs=coords[color_mask == 1,2],
               c=colors[color_mask == 1],
               s=size,
               linewidths=0.8,
               edgecolors='dimgrey',
               alpha=0.6
               )

    ax.scatter(xs=coords[color_mask == 0,0],
               ys=coords[color_mask == 0,1],
               zs=coords[color_mask == 0,2],
               s=size,
               linewidths=0.8,
               edgecolors='dimgrey',
               facecolors=colors[color_mask == 0]
               )

    # ax.set_title('XXXX', fontsize=10, loc=)
    ax.grid(False)
    ax.axis('off')

    if view == 'superior':
        ax.view_init(90,270)
        ax.set(xlim=0.57 *np.array(ax.get_xlim()),
               ylim=0.57 *np.array(ax.get_ylim()),
               zlim=0.60 *np.array(ax.get_zlim()),
               aspect=1.1
               )

    if view == 'left':
        ax.view_init(0,180)
        ax.set(xlim=0.59 * np.array(ax.get_xlim()),
               ylim=0.59 * np.array(ax.get_ylim()),
               zlim=0.60 * np.array(ax.get_zlim()),
               # aspect=0.55 #1.1
               )

    if view == 'right':
        ax.view_init(0,0)
        ax.set(xlim=0.59 * np.array(ax.get_xlim()),
               ylim=0.59 * np.array(ax.get_ylim()),
               zlim=0.60 * np.array(ax.get_zlim()),
               # aspect=0.55 #1.1
               )

    if title is not None: plt.title(title, fontsize=15, loc='left') #pad=5)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.gca().patch.set_facecolor('white')
    sns.despine()

    if ax is None: plt.show()


def brain_networks(coords, class_names, class_mapping):
    """
        Dot brain plots ... RSN brain plots
    """

    colors = np.array([COLORS[np.where(class_names == mapp)[0][0]] for mapp in class_mapping])

    fig = plt.figure(num=np.random.randint(0, 100), figsize=(17,1.33*5))

    coords = (coords-np.mean(coords, axis=0))/np.std(coords, axis=0)

    # lateral view
    ax1 = plt.subplot(121, projection='3d')
    ax2 = plt.subplot(122, projection='3d')
    for clase in class_names:

        # vector of colors
        c= colors[np.where(class_mapping == clase)[0]]
        s = 300

        # lateral view - lh
        ax1.scatter(xs=coords[np.where(class_mapping == clase)[0],0],
                    ys=coords[np.where(class_mapping == clase)[0],1],
                    zs=coords[np.where(class_mapping == clase)[0],2],
                    linewidths=0.5,
                    edgecolors='grey',
                    s=s,
                    c=c)

        # lateral view - rh
        ax2.scatter(xs=coords[np.where(class_mapping == clase)[0],0],
                    ys=coords[np.where(class_mapping == clase)[0],1],
                    zs=coords[np.where(class_mapping == clase)[0],2],
                    linewidths=0.5,
                    edgecolors='grey',
                    s=s,
                    c=c,
                    label=clase)

    ax1.view_init(0,180)
    ax1.grid(False)
    ax1.axis('off')
    ax1.set(xlim=0.59 * np.array(ax1.get_xlim()),
            ylim=0.59 * np.array(ax1.get_ylim()),
            zlim=0.60 * np.array(ax1.get_zlim()),
            )
    plt.gca().patch.set_facecolor('white')

    ax2.view_init(0,0)
    ax2.grid(False)
    ax2.axis('off')
    ax2.set(xlim=0.59 * np.array(ax2.get_xlim()),
            ylim=0.59 * np.array(ax2.get_ylim()),
            zlim=0.60 * np.array(ax2.get_zlim()),
            )
    plt.gca().patch.set_facecolor('white')

    plt.figlegend(loc='lower center', ncol=8, labelspacing=0.5, fontsize=12, frameon=False)
#    fig.suptitle('Resting State Networks - Yeo', fontsize=50)
#    fig.savefig(fname=os.path.join(), transparent=False, bbox_inches='tight')

    sns.despine()
    fig.tight_layout()
    plt.show()


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
