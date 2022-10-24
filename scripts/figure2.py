# PLOT OF NIRHISS, SUPREME-SPOON, TRANSITSPECTROSCOPY TRANSMISSION SPECTRA

import pickle
import os, sys
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

from utils import pipeline_dictionary, load_plt_params

# set the matplotlib parameters
load_plt_params()

# Create the figure environment
fig = plt.figure(figsize=(16,6))
fig.set_facecolor('w')

# Load in the HST data from Wakeford + 2018
hst = Table.read('../data/ts/WASP-39b_Wakeford_2018_AJ.csv', format='csv',
                 comment='#')
hst_err = np.sqrt(2) * hst['Rp/R*'] * hst['Rp/R* error'] * 100 # propagates err

# Plots the HST data
plt.errorbar(hst['Wavelength microns'],
             hst['Rp/R*']**2*100,
             yerr=hst_err*np.sqrt(2),
             xerr=hst['Wavelength bin size microns'],
             marker='o', linestyle='', ms=7,
             markeredgecolor='#404040', ecolor='#404040',
             color='w', lw=2.5,markeredgewidth=1.5,
             zorder=100, label='HST (Wakeford+2018)')

# Load in colors and filenames
pipeline_dict = pipeline_dictionary()

# Select the pipelines to plot (nirhiss, supreme-spoon, transitspectroscopy)
pipelineorder = ['CMADF', 'MCR', 'NE']

# Set alpha value and zorder for each plotted spectrum
alpha = [0.6,1.0,0.9]
zorder=[10,9,8]

for i in range(len(pipelineorder)):

    # Reads in the transmission spectrum
    fn = pipeline_dict[pipelineorder[i]]['filename']
    tab = Table.read(os.path.join('../data/ts', fn), format='csv', comment='#')
    color = pipeline_dict[pipelineorder[i]]['color']
    label = pipeline_dict[pipelineorder[i]]['name']

    # Plots each order separately
    for order in [1,2]:

        # Assigns wuality flags for plotting per order
        if order==2:
            q = (tab['quality'] == 0) & (tab['order'] == order)
            label=''
        else:
            q = tab['wave'] > 0.87
            label=label

        # Plots the transmission spectrum
        plt.errorbar(tab['wave'][q], tab['dppm'][q]/1e4,
                     yerr=tab['dppm_err'][q]/1e4,
                     xerr=tab['wave_error'][q],
                     markeredgecolor='w', color=color,
                     ecolor=color,
                     linestyle='', marker='o', ms=6,
                     alpha=alpha[i], markeredgewidth=0.5)

        # Plots the marker in slightly larger size with no alpha (for ease of
        #   reading the figure)
        plt.errorbar(tab['wave'][q], tab['dppm'][q]/1e4,
                     markeredgecolor='w', color=color,
                     ecolor=color,
                     linestyle='', marker='o', ms=6, zorder=zorder[i])

        # Plots the same points but bigger (for ease of reading the point size
        #   in the legend)
        plt.errorbar(tab['wave'][q], tab['dppm'][q]/1,
                     yerr=tab['dppm_err'][q]/1,
                     xerr=tab['wave_error'][q],
                     markeredgecolor='w', color=color,
                     ecolor=color, linestyle='', marker='o',
                     label=label, ms=10, lw=2, markeredgewidth=2)

# Label some of the obvious features
lcolor = '#8a8988'
plt.text(s='K', color=lcolor, x=0.76, y=2.213)
x = [0.91, 1.115, 1.38, 1.81]
rmin = [0.03, 0.04, 0.09, 0.08]
rmax = [0.1, 0.13, 0.21, 0.24]
for i in range(len(x)):
    plt.text(s=r'H$_2$O', x=x[i], y=2.07, color=lcolor, backgroundcolor='w')
    plt.hlines(y=2.075, xmin=x[i]-rmin[i], xmax=x[i]+rmax[i], color=lcolor, lw=2)

# Creates the dividing 'Order 1/Order 2' vertical line and text
plt.axvline(0.855, color='#404040', zorder=1, alpha=0.4, lw=3)
plt.text(s='order 2', x=0.74, y=2.23)
plt.arrow(0.73, 2.234, -0.06, 0., head_width=0.01, head_length=0.01, fc='k', ec='k')
plt.text(s='order 1', x=0.87, y=2.23)
plt.arrow(1.0, 2.234, 0.08, 0., head_width=0.01, head_length=0.015, fc='k', ec='k')

# Sets the x and y labels
plt.xlabel('wavelength [$\mu$m]')
plt.ylabel('transit depth [%]')

# Sets the x-limit, x-scale, and x-ticks
plt.xscale('log')
plt.xlim(0.6,2.86)
xticks = np.append(np.linspace(0.6,2,6), np.linspace(2.3,2.8,2))
xticks = np.round(xticks,2)
plt.xticks(xticks, labels=np.round(xticks,2))

# Sets the y-limit
plt.ylim(2.02,2.25)

# Creates the legend
leg = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
             ncol=4, mode="expand", borderaxespad=0.,
             fontsize=20)
# Increases markersize in the legend
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)

plt.minorticks_off()


plt.savefig('../figures/transmission_spectrum.pdf', dpi=300, rasterize=True,
            bbox_inches='tight')
