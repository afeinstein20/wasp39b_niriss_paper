# OVERPLOTTING NIRISS TRACES FOR ORDERS 1 AND 2

import time
import h5py
import pickle
import os, sys
import matplotlib
import numpy as np
from astropy import units
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import pipeline_dictionary, load_plt_params

# set the matplotlib parameters
load_plt_params()

# Load in colors and filenames
pipeline_dict = pipeline_dictionary()

# Create the figure environment
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,8), sharex=True)
fig.set_facecolor('w')

# Load in the median frame to plot as the background
img = np.load('../data/traces/example_integration.npy', allow_pickle=True)

vmin, vmax = 0.5, 2
# Add inset plot axes
axins1 = ax1.inset_axes([0.5, 0.05, 0.38, 0.4])
axins2 = ax2.inset_axes([0.1, 0.2, 0.47, 0.47])

for i, ax in enumerate([ax1, ax2, axins1, axins2]):
    if i == 0 or i == 2:
        factor=1.4
    else:
        factor=1.05

    # Plot the image
    im = ax.imshow(np.log10(img), aspect='auto', cmap='Greys',
                   vmin=vmin, vmax=vmax*factor)

    # Add colorbars for the main figures
    if i < 2:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        if i == 0:
            labels = [10,50,100,200,400,600]
        else:
            labels = [10,20,50,80,120]

        ticks = np.log10(labels)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        cbar.set_label('DN / s')

labels=['nirHiss', 'supreme-SPOON', 'NAMELESS', 'transitspectroscopy','iraclis']
x = 0
for i,name in enumerate(['CMADF', 'MCR', 'LPC', 'NE', 'AT']):

    tab = Table.read(os.path.join('../data/traces','{}_traces.csv'.format(name)),
                     format='csv')

    tab.sort('x')

    for ax in [ax1, axins1]:
        ax.plot(tab['x'], tab['order1'], lw=3,
                 color=pipeline_dict[name]['color'], label=labels[x])

    for ax in [ax2, axins2]:
        try:
            ax.plot(tab[(tab['order2']<255) & (tab['order2'] > 0)]['x'],
                     tab[(tab['order2']<255) & (tab['order2'] > 0)]['order2'],
                     lw=3, color=pipeline_dict[name]['color'])
        except:
            ax.plot(tab[tab['order2']>0]['x'],
                     tab[tab['order2']>0]['order2'],
                     lw=3, color=pipeline[name]['color'])
    x += 1


# Create legend and increase markersize in the legend
leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=2, mode="expand", borderaxespad=0., fontsize=16)
for legobj in leg.legendHandles:
    legobj.set_linewidth(6.0)

# Turn off the axes ticks for the inset plots
for ax in [axins1, axins2]:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

## Mark the inset axes for Order 1
x1, x2, y1, y2 = 1200, 1600, 45, 35
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)

rect = Rectangle((x1, y2), x2-x1, y1-y2, facecolor='none', edgecolor='k')
ax1.add_patch(rect)

ax1.plot(np.linspace(x1, 1025, 10), np.linspace(y1, 87, 10), lw=1, color='k')
ax1.plot(np.linspace(x2, 1800, 10), np.linspace(y1, 87, 10), lw=1, color='k')

## Mark the inset axes for Order 2
x1, x2, y1, y2 = 0, 1050, 110, 90
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)

rect = Rectangle((x1, y2), x2-x1, y1-y2, facecolor='none', edgecolor='k')
ax2.add_patch(rect)

ax2.plot(np.linspace(x1, 205, 10), np.linspace(110, 222, 10), lw=1, color='k')
ax2.plot(np.linspace(x2-x1, 1175, 10), np.linspace(110, 222, 10), lw=1,
         color='k')

# Add text labeling the subplots
ax1.text(s='(a) order 1', x=50, y=30)
ax2.text(s='(b) order 2', x=1550, y=105)

# Set the x and y limits for each subplot
ax2.set_ylim(256,80)
ax1.set_ylim(90, 20)
ax1.set_xlim(0,2048)

ax2.set_ylabel('y pixel position', y=1.01, fontsize=24)
ax2.set_xlabel('x pixel position', fontsize=24)
plt.savefig('../figures/traces.pdf',
            rasterize=True, bbox_inches='tight', dpi=250)
