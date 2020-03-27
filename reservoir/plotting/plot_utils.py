import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def array2cmap(X):

   """x = np.loadtxt("C:/Users/User/Desktop/rc_tmp/cmap.dat")
      cmap_div = plot_utils.array2cmap(x)"""

   N = X.shape[0]
   r = np.linspace(0., 1., N+1)
   r = np.sort(np.concatenate((r, r)))[1:-1]
   rd = np.concatenate([[X[i, 0], X[i, 0]] for i in range(N)])
   gr = np.concatenate([[X[i, 1], X[i, 1]] for i in range(N)])
   bl = np.concatenate([[X[i, 2], X[i, 2]] for i in range(N)])
   rd = tuple([(r[i], rd[i], rd[i]) for i in range(2 * N)])
   gr = tuple([(r[i], gr[i], gr[i]) for i in range(2 * N)])
   bl = tuple([(r[i], bl[i], bl[i]) for i in range(2 * N)])
   cdict = {'red': rd, 'green': gr, 'blue': bl}
   return mcolors.LinearSegmentedColormap('my_colormap', cdict, N)


def blend_colors(colors, as_cmap=True):
    #colors=['#c6e1f5', '#9e519f']

    cmap = sns.blend_palette(colors=colors,
                             n_colors=6,
                             as_cmap=as_cmap
                             )

    return cmap
