# PLOTS THE CONTRIBUTION PLOTS

import os, sys
import pickle
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.patches import Arrow
from matplotlib.gridspec import GridSpec
from astropy.table import Table, Column, vstack
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from utils import (load_plt_params, convolve_model, convolve_model_xy,
                   load_parula)

# set the matplotlib parameters
pltparams = load_plt_params()
parula = load_parula()
COLOR = pltparams[pltparams['name']=='text.color']['value'][0]

data = Table.read('../data/ts/CMADF-WASP_39b_NIRISS_transmission_spectrum_R300.csv',
                  format='csv', comment='#')

xsections = Table.read('../data/contributions/xsecs/xsections.txt', format='ascii',
                       comment='#') # units in m^2

no_path = '../data/contributions/models'
no_models = ['model_xClouds.txt',
             'model_xCO.txt', 'model_xCO2.txt', 'model_xH2O.txt',
             'model_xK.txt']

cutends = 15

fig = plt.figure(figsize=(14,12))
fig.set_facecolor('w')

# create the figure and gridspec environment
gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2,1.3])
gs01 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs0[1], hspace=0)
gs00 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=gs0[0], hspace=0)

ax1 = fig.add_subplot(gs00[0,0])
ax2 = fig.add_subplot(gs01[0, 0])
ax3 = fig.add_subplot(gs01[1, 0])

# plot the data
dcolor = '#5c5a58'
ax1.errorbar(data['wave'], data['dppm']/1e6,
             xerr=data['wave_error'],
             yerr=data['dppm_err']/1e6,
             linestyle='', marker='o',
             markeredgecolor=dcolor,
             ecolor=dcolor,
             color='w', zorder=1)


# plot the models lacking certain molecules / clouds
zorder=[2,3,4,10,5,6]
c = [0, 50, 160, 205, 240]
labels=['no clouds', 'no CO', r'no CO$_2$',
        r'no H$_2$O',  'no K']
for i,m in enumerate(no_models):
    label=m.split('_')[-1]
    x, y = convolve_model(os.path.join(no_path,m))
    ax1.plot(x[cutends:-cutends], y[cutends:-cutends], c=parula[c[i]],
             label=labels[i], lw=3, zorder=zorder[i])

# convolve the model reference and plot
x, y = convolve_model('../data/contributions/models/model_reference.txt')
ax1.plot(x[cutends:-cutends], y[cutends:-cutends], COLOR,
         lw=3, zorder=100,
         label='ref.')

# sets the species to plot for each subpanel (and their labels)
cols2 = ['xh2o','xco', 'xco2', 'xna', 'xk']
cols3 = ['xch4','xnh3', 'xso2', 'xph3', 'xh2s']
labels = [r'H$_2$O', 'CO', r'CO$_2$', 'Na', 'K',
          r'CH$_4$', r'NH$_3$', r'SO$_2$', r'PH$_3$', r'H$_2$S']

cmap = plt.get_cmap('magma')

colors = cmap(np.linspace(0,1,len(cols2)+len(cols3)+2))
inds = [0, 9, 5, 2, 7, 1, 10, 4, 6, 8, 3] # sets the orders for the colors
colors = colors[inds]

x=0

# plots the models for panel b
for col in cols2:
    y = convolve_model_xy(xsections[col], R=100)
    q = (y > -30.5) & (xsections['wavelength'] > 0.6)
    ax2.fill_between(xsections['wavelength'][q],
                     np.full(len(y[q]), -40), y[q], alpha=0.1,
                     color=colors[x], lw=2)

    ax2.plot(xsections['wavelength'][q],y[q],
             label=labels[x], color=colors[x], lw=4)

    x+=1

# plots the models for panel c
for col in cols3:
    y = convolve_model_xy(xsections[col], R=100)
    q = (y > -30.5) & (xsections['wavelength'] > 0.6)

    ax3.fill_between(xsections['wavelength'][q],
                     np.full(len(y[q]), -40), y[q], alpha=0.1,
                     color=colors[x], lw=2)
    ax3.plot(xsections['wavelength'][q],y[q],
             label=labels[x], color=colors[x], lw=4)
    x+=1

# sets all the limits properly
for a in [ax2, ax3]:
    a.set_ylim(-28, -20)
    a.set_xlim(0.6,2.86)
    a.set_yticks([-28, -25, -22])

# creates the three legends in the plot
leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=3, mode="expand", borderaxespad=0.,
                 fontsize=16)
for legobj in leg.legendHandles:
    legobj.set_linewidth(6.0)

for a in [ax2, ax3]:
    leg = a.legend(ncol=5, borderaxespad=0.0,
                   fontsize=16, loc="upper center")
    for legobj in leg.legendHandles:
        legobj.set_linewidth(6.0)

# sets the limits for the x-section plots
for a in [ax1, ax3]:
    a.set_xlim(0.6,2.86)
    a.set_xscale('log')
    xticks = np.append(np.linspace(0.6,2,6), np.linspace(2.3,2.8,2))
    xticks = np.round(xticks,2)
    a.set_xticks(xticks)
    a.set_xticklabels(np.round(xticks,2))

ax2.set_xlim(0.6,2.86)
ax2.set_xscale('log')
ax2.set_xticks([0.6])
ax2.set_xticklabels([''])

# axes labels
ax3.set_xlabel('wavelength [$\mu$m]')
ax1.set_ylabel('transit depth [%]')
ax3.set_ylabel(r'log$_{10}\sigma$ [m$^2$]', y=1)

# ytick labels
yticks = np.round(np.arange(0.020, 0.0230, 0.0005),4)
ax1.set_yticks(yticks)
labels = np.round(yticks*100,2)
labels = [format(i, '.2f') for i in labels]
ax1.set_yticklabels(labels)

# add subplot labels
ax1.text(s='(a)', x=0.61, y=0.0223, fontsize=20, fontweight='bold')
ax2.text(s='(b)', x=0.61, y=-21.9, fontsize=20, fontweight='bold')

plt.savefig('../figures/contribution.pdf',
            dpi=250, rasterize=True,
            #transparent=True,
            bbox_inches='tight')
