# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:10:14 2019

@author: Estefany Suarez
"""
import os
import numpy as np
import seaborn as sns

from netneurotools import plotting
from netneurotools import datasets

import matplotlib.pyplot as plt
from matplotlib.colors import (ListedColormap, Normalize)
from mpl_toolkits.mplot3d import Axes3D

COLORS = sns.color_palette("husl", 8)


def brain_dots(coords, size, values=None, cmap='viridis', view='superior', title=None, **kwargs):

    # z-score coordinates for better visualization
    coords = (coords-np.mean(coords, axis=0))/np.std(coords, axis=0)

    # colors=['#c6e1f5', '#9e519f']
    # cmap = sns.blend_palette(colors=colors,
    #                         n_colors=6,
    #                         as_cmap=False)

    # if size is None: size = np.random.rand(len(coords))

    fig = plt.figure(num=1, figsize=(2*.8*5,2*.8*4))#2*5.25,2*4))
    ax = plt.subplot(111, projection='3d')

    mapp = ax.scatter(xs=coords[:,0],
                      ys=coords[:,1],
                      zs=coords[:,2],
                      cmap=cmap,
                      c=values, #np.ones(len(coords)),
                      # facecolors='#e6e7e8',
                      s=2000*(1/size),
                      # linewidths=0.8,
                      edgecolors='dimgrey', #'#414042',
                      alpha=0.5,
                      **kwargs
                      )

    plt.colorbar(mapp)

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

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.gca().patch.set_facecolor('white')
    sns.despine()

    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/' + title + '.eps', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/' + title + '.jpg', transparent=True, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()


def brain_subnetworks(coords, class_names, class_mapping, nodes, node_size, view='superior', ax=None, title=None, **kwargs):

    # z-score coordinates for better visualization
    coords = (coords-np.mean(coords, axis=0))/np.std(coords, axis=0)

    # create mask according to nodes
    color_mask = np.array([1 if node in nodes else 0 for node in range(len(coords))]).astype(int)

    # create array of colors according to class mapping
    class_mapping_int = np.array([np.where(class_names == mapp)[0][0] for mapp in class_mapping]).astype(int)
    colors = np.array([COLORS[np.where(class_names == mapp)[0][0]] if node in nodes else 'none' for node, mapp in enumerate(class_mapping)])

    # if size is None: size = 200

    if ax is None:
        fig = plt.figure(num=1, figsize=(2*.8*5.25,2*.8*4))#2*5.25,2*4))
        ax = plt.subplot(111, projection='3d')

    ax.scatter(xs=coords[color_mask == 1,0],
               ys=coords[color_mask == 1,1],
               zs=coords[color_mask == 1,2],
               c=colors[color_mask == 1],
               s=node_size[color_mask == 1],
               linewidths=0.5,
               edgecolors='dimgrey',
               alpha=0.7
               )

    ax.scatter(xs=coords[color_mask == 0,0],
               ys=coords[color_mask == 0,1],
               zs=coords[color_mask == 0,2],
               s=node_size[color_mask == 0],
               linewidths=0.5,
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

    # if title is not None: plt.title(title, fontsize=15, loc='left') #pad=5)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.gca().patch.set_facecolor('white')
    sns.despine()

    fig.savefig(fname='C:/Users/User/Desktop/poster/figures/' + title + '.eps', transparent=True, bbox_inches='tight', dpi=300)
    fig.savefig(fname='C:/Users/User/Desktop/poster/figures/' + title + '.jpg', transparent=True, bbox_inches='tight', dpi=300)
    plt.show()


def brain_networks(coords, class_names, class_mapping,  title=None):
    """
        Dot brain plots ... RSN brain plots
    """
    palette = np.array(COLORS)
    colors = np.array([palette[np.where(class_names == mapp)[0]] for mapp in class_mapping])

    fig = plt.figure(num=np.random.randint(0, 100), figsize=(1.1*17,1.1*1.33*5))

    coords = (coords-np.mean(coords, axis=0))/np.std(coords, axis=0)

    # lateral view
    ax1 = plt.subplot(121, projection='3d')
    ax2 = plt.subplot(122, projection='3d')
    for clase in class_names:

        # vector of colors
        c = colors[np.where(class_mapping == clase)[0][0]]
        s = 200

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


    # plt.figlegend(loc='lower center', ncol=8, labelspacing=0.5, fontsize=12, frameon=False)
#    fig.suptitle('Resting State Networks - Yeo', fontsize=50)

    sns.despine()
    fig.tight_layout()

    # fig.savefig(fname='C:/Users/User/Desktop/' + title + '.png', transparent=True, bbox_inches='tight', dpi=300)
    # fig.savefig(fname='C:/Users/User/Desktop/poster/figures/' + title + '.jpg', transparent=True, bbox_inches='tight', dpi=300)
    fig.savefig(fname='C:/Users/User/Dropbox/figures_RC/eps/' + title + '.eps', transparent=True, bbox_inches='tight', dpi=300)

    plt.show()


def brain_surf(var_name, data, scale='scale500', cmap='R_Bu', cbar=True):

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
                                    # vmin=,
                                    # vmax=,
                                    # center=0.0
                                    )

    for i in range(2):
        for j in range(2):
               brain._figures[i][j].scene.parallel_projection = True
               brain._figures[i][j].scene.parallel_projection = True

#    brain.save_image(os.path.join(FIG_DIR,  var_name + '_surf_brain' + '.png'), mode='rgba')

    return brain
