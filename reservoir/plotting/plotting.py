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

FIG_DIR = 'C:/Users/User/Dropbox/figures_RC/' # 'C:/Users/User/Desktop/'
FIG_EXT = '.jpg'

# --------------------------------------------------------------------------------------------------------------------
# GENERAL
# ----------------------------------------------------------------------------------------------------------------------
def lineplot(x, y, df, palette=None, title=None, hue=None, hue_order=None, \
             err_style='band', markers=True, marker='D', markersize=12, \
             linewidth=1, sort=False, xlim=None, x_major_loc=None, ylim=None, \
             y_major_loc=None, legend=True, figsize=(15,8), fig_name=None, \
             **kwargs):

     sns.set(style="ticks", font_scale=2.0)

     fig = plt.figure(figsize=figsize)
     ax = plt.subplot(111)

     sns.lineplot(x=x, y=y,
                  data=df,
                  palette=palette,
                  hue=hue,
                  hue_order=hue_order,
                  err_style=err_style, #'band' 'bars'
                  markers=markers,
                  marker=marker,
                  markersize=markersize,
                  linewidth=linewidth,
                  sort=sort,
                  ax=ax,
                  **kwargs
                  )

     if legend: ax.legend(fontsize=15, frameon=True, ncol=1, loc='upper right')
     else: ax.get_legend().remove()

     if title is not None: ax.set_title(title)

     if xlim is not None: ax.set_ylim(xlim)
     if x_major_loc is not None: ax.xaxis.set_major_locator(MultipleLocator(x_major_loc))

     if ylim is not None: ax.set_ylim(ylim)
     if y_major_loc is not None: ax.yaxis.set_major_locator(MultipleLocator(y_major_loc))

     sns.despine(offset=10, trim=True)

     if fig_name is not None: fig.savefig(fname=os.path.join(FIG_DIR, fig_name + FIG_EXT), transparent=True, bbox_inches='tight', dpi=300)

     plt.show()
     plt.close()


def boxplot(x, y, df, palette, title=None, hue=None, order=None, orient='v', \
            width=0.5, linewidth=1, xlim=None, ylim=None, legend=True, \
            fig_name=None, figsize=(15,5), **kwargs):

     sns.set(style="ticks", font_scale=2.0)

     fig = plt.figure(figsize=figsize)
     ax = plt.subplot(111)

     axis = sns.boxplot(x=x, y=y,
                        data=df,
                        palette=palette,
                        hue=hue,
                        order=order,
                        orient=orient,
                        width=width,
                        linewidth=linewidth,
                        ax=ax,
                        **kwargs
                        )

     for patch in axis.artists:
         r, g, b, a = patch.get_facecolor()
         patch.set_facecolor((r, g, b, 0.9))

     if legend: ax.legend(fontsize=15, frameon=True, ncol=1, loc='upper right')
     # else: ax.get_legend().remove()

     if title is not None: ax.set_title(title)

     if xlim is not None: ax.set_ylim(xlim)
     if ylim is not None: ax.set_ylim(ylim)

     sns.despine(offset=10, trim=True)

     if fig_name is not None: fig.savefig(fname=os.path.join(FIG_DIR, fig_name + FIG_EXT), transparent=True, bbox_inches='tight', dpi=300)

     plt.show()
     plt.close()


def scatterplot(x, y, df, palette, title=None, hue=None, hue_order=None, \
                hue_norm=None, markers=True, s=12, xlim=None, x_major_loc=None,
                ylim=None, y_major_loc=None, legend=True, figsize=(8,8),
                fig_name=None, draw_line=True, **kwargs):

     sns.set(style="ticks", font_scale=2.0)

     fig = plt.figure(figsize=figsize)
     ax = plt.subplot(111)

     sns.scatterplot(x=x, y=y,
                     data=df,
                     palette=palette,
                     hue=hue,
                     # legend='full',
                     ax=ax,
                     **kwargs
                     )

     if draw_line:
        ax.plot([0, 1],
                [0, 1],
                linestyle='--',
                linewidth=2,
                color='dimgrey'
                )

     ax.set_aspect("equal")

     if legend: ax.legend(fontsize=15, frameon=True, ncol=1, loc='lower right')
     else: ax.get_legend().remove()

     if title is not None: ax.set_title(title)

     if xlim is not None: ax.set_ylim(xlim)
     if x_major_loc is not None: ax.xaxis.set_major_locator(MultipleLocator(x_major_loc))

     if ylim is not None: ax.set_ylim(ylim)
     if y_major_loc is not None: ax.yaxis.set_major_locator(MultipleLocator(y_major_loc))

     sns.despine(offset=10, trim=True)

     if fig_name is not None: fig.savefig(fname=os.path.join(FIG_DIR, fig_name + FIG_EXT), transparent=True, bbox_inches='tight', dpi=300)

     plt.show()
     plt.close()
