# PLOTS THE CONTRIBUTIONS TO THE MODEL

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
from matplotlib.colors import LinearSegmentedColormap

from utils import load_plt_params, convolve_model, convolve_model_xy

# set the matplotlib parameters
load_plt_params()

# load in NIRISS spectrum
data = Table.read('../data/ts/CMADF-WASP_39b_NIRISS_transmission_spectrum_R300.csv',
                  format='csv', comment='#')

# reference model filename
ref_file = '../data/Main_Models/model_reference.txt'

# pulls all the different C/O models
path = '../data/model_rule_out_co/'
co_files = np.sort([os.path.join(path, i) for i in os.listdir(path) if
                    i.endswith('.txt')])

# pulls all the different metallicity models
path = '../data/model_rule_out_z/'
z_files = np.sort([os.path.join(path, i) for i in os.listdir(path) if
                   i.endswith('.txt')])
z_files = np.array(z_files)[np.array([0,2,4,5])] # selects only 4 models

# creates the figure environment
fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(14,12), sharex=True,
                              sharey=True)
fig.set_facecolor('w')

cutends = 15

# plots the data and reference model on both subplots
for i, a in enumerate([ax1, ax2]):
    a.errorbar(data['wave'], data['dppm']/1e6,
             xerr=data['wave_error'],
             yerr=data['dppm_err']/1e6,
             linestyle='', marker='o',
             markeredgecolor='#b0acaa',
             ecolor='#b0acaa',
             color='k', zorder=1)
    ref = convolve_model(ref_file)

    if i == 0:
        label = '0.20 (ref.)'
    else:
        label = '1.38 (ref.)'

    a.plot(ref[0][cutends:-cutends], ref[1][cutends:-cutends],
             label=label, lw=3,
             c='k', zorder=20)

# sets the colormap for the C/O and metallicity subplots
co_norm = matplotlib.colors.Normalize(vmin=0.35, vmax=0.8)
z_norm = matplotlib.colors.Normalize(vmin=-1.2, vmax=2.5)

# plots the C/O models
co_vals = [0.55, 0.70, 0.80]
c = ['#24abff', '#1f88c9', '#155d8a']
for i,fn in enumerate(co_files):
    model = convolve_model(fn)
    ax1.plot(model[0][cutends:-cutends], model[1][cutends:-cutends],
             label='{0:.2f}'.format(co_vals[i]), lw=2.5,
             c=cm.Blues(co_norm(co_vals[i])), zorder=10)

# plots the metallicity models
z_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.25]
z_vals = [0.0, 1.0, 2.0, 2.25]
for i,fn in enumerate(z_files):
    model = convolve_model(fn)
    ax2.plot(model[0][cutends:-cutends], model[1][cutends:-cutends],
             label='{0:.2f}'.format(z_vals[i]), lw=2,
             c=cm.YlOrRd(z_norm(z_vals[i])))

# sets the limits, ticks, and labels for the x-axis
plt.xlim(0.6,2.86)
ax2.set_xlabel('wavelength [$\mu$m]')
ax1.set_ylabel('transit depth [%]')
ax2.set_ylabel('transit depth [%]')
plt.xscale('log')
xticks = np.append(np.linspace(0.6,2,6), np.linspace(2.3,2.8,2))
xticks = np.round(xticks,2)
plt.xticks(xticks, labels=np.round(xticks,2))

# sets the limits, ticks, and labels for the y-axis
yticks = np.round(np.arange(0.0205, 0.0230, 0.0005),4)
ax1.set_yticks(yticks)
labels = np.round(yticks*100,2)
labels = [format(i, '.2f') for i in labels]
ax1.set_yticklabels(labels)

# labels the subplots
ax1.text(s='(a)', x=0.61, y=0.0222, fontsize=20)
ax2.text(s='(b)', x=0.61, y=0.0222, fontsize=20)

# creates the legends for both subplots
leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=4, mode="expand", borderaxespad=0.,
                 fontsize=16, title='carbon-to-oxygen ratio (C/O)')
for legobj in leg.legendHandles:
    legobj.set_linewidth(5.0)

leg = ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=7, mode="expand", borderaxespad=0.,
                 fontsize=16, title='metallicity [M/H]')
for legobj in leg.legendHandles:
    legobj.set_linewidth(5.0)

plt.subplots_adjust(hspace=0.43)

plt.savefig('../figures/co_metallicity.pdf',
           dpi=300, rasterize=True, bbox_inches='tight')
