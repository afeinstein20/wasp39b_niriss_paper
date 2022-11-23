# PLOTS CLOUD MODELS AND CLOUD MODELS FIT TO < 2UM

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


from utils import load_plt_params, convolve_model, load_parula

# set the matplotlib parameters
load_plt_params()
parula = load_parula()

# load in the NIRISS spectrum
data = Table.read('../data/ts/CMADF-WASP_39b_NIRISS_transmission_spectrum_R300.csv',
                  format='csv', comment='#')

# set the reference model file
ref_file = '../data/Main_Models/model_reference.txt'

# load in the short lambda models (fitted to < 2um) and sort
path = '../data/short_lam_models/'
models = np.sort([os.path.join(path, i) for i in os.listdir(path) if i.endswith('.txt')])
new = []
new.append(models[2])
new.append(models[1])
new.append(models[0])
new.append(models[-1])
new.append(models[-2])
models = np.copy(new)

# load in the full models (fitted to 0.6-2.8 um)
fullmodels = [os.path.join('../data/Main_Models/', i) for i in
              os.listdir('../data/Main_Models/')]

# set the colors and labels for each cloud model
cdict = {}
corder = [210, 0, 52, 105, 157]
names = ['gray [2.96 / 2.89]',
         'inhomogeneous droplet sedimentation [2.59 / 2.79]',
         'droplet sedimentation [2.70 / 2.91]',
         'gray + power-law [2.83 / 2.87]',
         'inhomogeneous gray + power law [2.83 / 2.88]']

for i, m in enumerate(['basic', 'inh_anm', 'anm_output', 'power_output',
                       'inh_power']):
    cdict[m] = {}
    cdict[m]['color'] = corder[i]
    cdict[m]['name'] = names[i]

fig, (ax0, ax) = plt.subplots(figsize=(14,8), nrows=2, sharex=True, sharey=True)
fig.set_facecolor('w')

cutends = 15

q = data['quality']==0
ax.errorbar(data['wave'][q], data['dppm'][q]/1e6,
         xerr=data['wave_error'][q],
         yerr=data['dppm_err'][q]/1e6,
         linestyle='', marker='o',
         markeredgecolor='#e8e3e0',
         ecolor='#e8e3e0',
         color='#e8e3e0', zorder=1)

ax0.errorbar(data['wave'][q], data['dppm'][q]/1e6,
             xerr=data['wave_error'][q],
             yerr=data['dppm_err'][q]/1e6,
             linestyle='', marker='o',
             markeredgecolor='#9c9693',
             ecolor='#9c9693',
             color='w', zorder=1, label='fitted data')

q = (data['quality']==0) & (data['wave'] < 2)
ax.errorbar(data['wave'][q], data['dppm'][q]/1e6,
         xerr=data['wave_error'][q],
         yerr=data['dppm_err'][q]/1e6,
         linestyle='', marker='o',
         markeredgecolor='#9c9693',
         ecolor='#9c9693',
         color='w', zorder=2)

ref = convolve_model(ref_file)

# plots the reference model on both axes
for a in [ax, ax0]:
    a.plot(ref[0][cutends:-cutends], ref[1][cutends:-cutends],
            label='ref. [2.59]', lw=3,
            c='k', zorder=10)

# sets the key for the color/label dictionary based on the filename
def set_key(filename):
    if filename == 'inh_anm_output_model.txt':
        k = 'inh_anm'
    elif filename == 'anm_output_model.txt':
        k = 'anm_output'
    elif filename == 'power_output_model.txt':
        k = 'power_output'
    elif filename == 'inh_power_output_model.txt':
        k = 'inh_power'
    else:
        k = 'basic'
    return k

inds = np.linspace(0,210,len(models),dtype=int)

# plot the models fitted to full wavelength coverage
for i, fn in enumerate(models):
    m = convolve_model(fn)

    k = set_key(fn.split('/')[-1])

    ax.plot(m[0][cutends:-cutends], m[1][cutends:-cutends],
             lw=2, zorder=3, c=parula[cdict[k]['color']])

# plot the models fitted to < 2um
for i,ind in enumerate([1,2,4,5,0]):
    model = convolve_model(fullmodels[ind])

    k = set_key(fullmodels[ind].split('/')[-1])

    cind = cdict[k]['color']
    name = cdict[k]['name']

    ax0.plot(model[0][cutends:-cutends],
             model[1][cutends:-cutends], lw=2, color=parula[cind],
             label=name, zorder=3)

# sets the limits, ticks, and labels for the x-axis
plt.xlim(0.6,2.86)
ax.set_xlabel('wavelength [$\mu$m]')
ax.set_ylabel('transit depth [%]')
ax0.set_ylabel('transit depth [%]')
plt.xscale('log')
xticks = np.append(np.linspace(0.6,2,6), np.linspace(2.3,2.8,2))
xticks = np.round(xticks,2)
plt.xticks(xticks, labels=np.round(xticks,2))

# sets the limits, ticks, and labels for the y-axis
yticks = np.round(np.arange(0.0205, 0.0230, 0.0005),4)
ax.set_yticks(yticks)
labels = np.round(yticks*100,2)
labels = [format(i, '.2f') for i in labels]
ax.set_yticklabels(labels)

# adds the labels for each subplot
ax0.text(s=r'(a) $\lambda$ = 0.63 - 2.80 $\mu$m', x=0.61, y=0.0222,
         fontweight='bold')
ax.text(s=r'(b) $\lambda$ < 2.00 $\mu$m', x=0.61, y=0.0222, fontweight='bold')

# creates the legend for both subplots
leg = ax0.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=2, mode="expand", borderaxespad=0.,
                 fontsize=13)
for legobj in leg.legendHandles:
    legobj.set_linewidth(5.0)

plt.savefig('../figures/shortward.jpg',
           dpi=300, rasterize=True, bbox_inches='tight')
