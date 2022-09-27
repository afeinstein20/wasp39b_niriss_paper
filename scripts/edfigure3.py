# PLOTS THE STELLAR SPECTRA FROM ALL PIPELINES

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import load_plt_params, pipeline_dictionary

# set the matplotlib parameters
load_plt_params()
color_dict = pipeline_dictionary()


path = '../data/stellar_spectra'
nirhiss = np.load(os.path.join(path, 'nirhiss_spectra.npy'))
spoon = np.load(os.path.join(path, 'supremespoon_spectra.npy'))
ts = np.load(os.path.join(path, 'transitspectroscopy_spectra.npy'),
             allow_pickle=True)
iraclis  = np.load(os.path.join(path, 'iraclis_spectra.npy'), allow_pickle=True)


fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14,8))
fig.set_facecolor('w')

axins1 = ax1.inset_axes([0.55, 0.4, 0.42, 0.5])
axins2 = ax2.inset_axes([0.55, 0.4, 0.42, 0.5])

kwargs={'linewidth':2}
labels = ['nirHiss', 'supreme-SPOON', 'transitspectroscopy', 'iraclis']
initials = ['ADF', 'MCR', 'NE', 'AT']
scaling1 = [1, 72, 1.065, 1.15]
scaling2 = [1, 72, 1.09, 2.55]

for ax in [ax1, axins1]:
    for i, data in enumerate([nirhiss, spoon, ts, iraclis]):
        ax.plot(data[0], data[1]/scaling1[i],
                color=color_dict[initials[i]]['color'],
                zorder=100,
                label=labels[i], **kwargs)

for ax in [ax2, axins2]:
    for i, data in enumerate([nirhiss, spoon, ts, iraclis]):
        ax.plot(data[2], data[3]/scaling2[i],
                color=color_dict[initials[i]]['color'],
                zorder=100,
                label=labels[i], **kwargs)


ax1.set_xlim(0.85,2.83)
ax1.set_ylim(0,8500)

axins1.set_xlim(0.98, 1.3)
axins1.set_ylim(6400,8100)

ax2.set_xlim(0.56,1.2)
ax2.set_ylim(0,3750)

axins2.set_xlim(0.67, 0.78)
axins2.set_ylim(2500,3650)

ax1.indicate_inset_zoom(axins1, edgecolor="black")
ax2.indicate_inset_zoom(axins2, edgecolor="black")

ax2.set_xlabel('Wavelength [$\mu$m]', fontsize=22)
ax2.set_ylabel('Flux [DN s$^{-1}$]', fontsize=22, y=1.1)

ax1.text(s='(a)', x=0.87, y=7300)
ax2.text(s='(b)', x=0.567, y=3200)

leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=4, mode="expand", borderaxespad=0., fontsize=16)

for legobj in leg.legendHandles:
    legobj.set_linewidth(6.0)

plt.subplots_adjust(hspace=0.25)
plt.savefig('../figures/stellar_spectra.pdf',
            rasterize=True, bbox_inches='tight', dpi=300)
