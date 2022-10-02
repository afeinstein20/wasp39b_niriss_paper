# PLOT OF ALL PIPELINES' TRANSMISSION SPECTRA

import pickle
import os, sys
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

from utils import pipeline_dictionary, load_plt_params

# set the matplotlib parameters
load_plt_params()

# Load in colors and filenames
pipeline_dict = pipeline_dictionary()

# Creates the figure environment
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16,10),
                               sharex=True, sharey=True)
fig.set_facecolor('w')

# Defines the order for which pipelines will be plotted
pipelineorder = ['CMADF', 'NE', 'AT', 'MCR', 'ZR', 'LPC']

# Set alpha value, zorder, and marker shape for each plotted spectrum
alpha = np.full(len(pipelineorder), 0.6)
zorder = [5, 4, 3, 5, 4 , 3]
shapes = ['o', 's', '^', 'o', 's', '^',]

for i in range(len(pipelineorder)):
    fn = pipeline_dict[pipelineorder[i]]['filename']
    tab = Table.read(os.path.join('../data/ts', fn), format='csv', comment='#')

    color = pipeline_dict[pipelineorder[i]]['color']
    label = pipeline_dict[pipelineorder[i]]['name']

    for order in [1,2]:

        if order==2:
            q = (tab['quality'] == 0) & (tab['order'] == order)
            label=''
        else:
            q = tab['wave'] > 0.87
            label=label

        if i < 3:
            a = ax1
        else:
            a = ax2

        if shapes[i] == '^':
            ms = 10
        else:
            ms = 8

        a.errorbar(tab['wave'][q], tab['dppm'][q]/1e4,
                   yerr=tab['dppm_err'][q]/1e4,
                   xerr=tab['wave_error'][q],
                   markeredgecolor='w', color=color,
                   ecolor=color,
                   linestyle='', marker=shapes[i], ms=ms,
                   alpha=alpha[i], markeredgewidth=1)

        a.errorbar(tab['wave'][q], tab['dppm'][q]/1e4,
                   markeredgecolor='w', color=color,
                   ecolor=color, markeredgewidth=1, alpha=1,
                   linestyle='', marker=shapes[i], ms=ms, zorder=10)


axes = [ax1, ax1, ax1, ax2, ax2, ax2]
for i in range(len(pipelineorder)):
    color = pipeline_dict[pipelineorder[i]]['color']
    axes[i].errorbar(tab['wave'][q], tab['dppm'][q]/1000,
                     yerr=tab['dppm_err'][q]/1000,
                     xerr=np.full(len(tab[q]),0.002),
                     markeredgecolor='w', color=color,
                     ecolor=color,
                     linestyle='', marker=shapes[i],
                     label=pipeline_dict[pipelineorder[i]]['name'],
                     ms=12, lw=4, markeredgewidth=2)


# Sets the x and y labels
plt.xlabel('wavelength [$\mu$m]')
ax1.set_ylabel('transit depth [%]')
ax2.set_ylabel('transit depth [%]')

# Sets the x-limit, x-scale, and x-ticks
plt.xscale('log')
plt.xlim(0.6,2.86)
xticks = np.append(np.linspace(0.6,2,6), np.linspace(2.3,2.8,2))
xticks = np.round(xticks,2)
plt.xticks(xticks, labels=np.round(xticks,2))

# Sets the y-limit
plt.ylim(1.98,2.32)

# Creates the legends for each subplot
for a in [ax1, ax2]:
    leg = a.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                     ncol=4, mode="expand", borderaxespad=0.,
                     fontsize=20)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

plt.subplots_adjust(hspace=0.3)
plt.minorticks_off()

plt.savefig('../figures/transmission_spectrum_all.pdf', dpi=300, rasterize=True,
            bbox_inches='tight')
