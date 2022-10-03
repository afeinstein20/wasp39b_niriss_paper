# PLOTS FORWARD GRID MODEL FITS
import numpy as np
from astropy.table import Table

import matplotlib.pyplot as plt

from utils import load_plt_params

# set the matplotlib parameters
load_plt_params()

# Load in the spectrum
obs = Table.read('../data/ts/CMADF-WASP_39b_NIRISS_transmission_spectrum_R300.csv',
                  format='csv', comment='#')

plt.rcParams["text.usetex"]=False

modelname = ['PICASO: ','ATMO: ','Phoenix: ']
bestparam  = [r'[M/H]=1.7, C/O=0.275, $\chi^2/N_{\rm obs}=$2.98', #PICASO
              r'[M/H]=1.0, C/O=0.35, $\chi^2/N_{\rm obs}=$3.24',  #ATMO
              r'[M/H]=2.0, C/O=0.389, $\chi^2/N_{\rm obs}=$3.51'  #Phoenix
             ]

#----- Plot figure
fig, (ax1, ax2) = plt.subplots(figsize=(14,8), nrows=2, sharex=True,
                               sharey=True)

# sets some aesthetic choices
kwargs = {'lw':2.5}
picaso_color = '#e55c24'
atmo_color = '#5F3996'
phoenix_color = '#f0c915'

# Plot the observations
dcolor = '#5c5a58'
for a in [ax1, ax2]:
    a.errorbar(obs['wave'], obs['dppm']/1e4,
               xerr=obs['wave_error'],
               yerr=obs['dppm_err']/1e4,
               linestyle='', marker='o',
               markeredgecolor=dcolor,
               ecolor=dcolor, ms=5,
               color='w', zorder=1)

# Loading best-fit cloudy models
# PICASO cloudy
data = np.loadtxt('../data/grid_best/PICASO_cloudy_R300.spec')
lmd_PICASO  = data[:,0]
dppm_PICASO = data[:,1]

# ATMO cloudy
data = np.loadtxt('../data/grid_best/ATMO_cloudy_R300.spec')
lmd_ATMO  = data[:,0]
dppm_ATMO = data[:,1]

# Phoenix cloudy
data = np.loadtxt('../data/grid_best/Phoenix_cloudy_R300.spec')
lmd_Phoenix  = data[:,0]
dppm_Phoenix = data[:,1]

# plotting each gas contribution as a cumulative fashion
ax1.plot(lmd_PICASO,dppm_PICASO*1e-4,
             color=picaso_color,label=modelname[0]+bestparam[0], **kwargs)
ax1.plot(lmd_ATMO,dppm_ATMO*1e-4,
             color=atmo_color,label=modelname[1]+bestparam[1], **kwargs)
ax1.plot(lmd_Phoenix,dppm_Phoenix*1e-4,
             color=phoenix_color,label=modelname[2]+bestparam[2], **kwargs)



# plotting best-fit clear models
bestparam_clear  = [r'[M/H]=2.0, C/O=0.825, $\chi^2/N_{\rm obs}=$7.02', #PICASO
                    r'[M/H]=2.0, C/O=0.35, $\chi^2/N_{\rm obs}=$4.11',  #ATMO
                    r'[M/H]=2.0, C/O=0.899, $\chi^2/N_{\rm obs}=$8.55'  #Phoenix
                   ]

#  Loading best-fit clear models
# PICASO cloudy
data = np.loadtxt('../data/grid_best/PICASO_clear_R300.spec')
lmd_PICASO_clear  = data[:,0]
dppm_PICASO_clear = data[:,1]

# ATMO cloudy
data = np.loadtxt('../data/grid_best/ATMO_clear_R300.spec')
lmd_ATMO_clear  = data[:,0]
dppm_ATMO_clear = data[:,1]

# Phoenix cloudy
data = np.loadtxt('../data/grid_best/Phoenix_clear_R300.spec')
lmd_Phoenix_clear  = data[:,0]
dppm_Phoenix_clear = data[:,1]


ax2.plot(lmd_PICASO_clear,dppm_PICASO_clear*1e-4,
             color=picaso_color,label=modelname[0]+bestparam_clear[0], **kwargs)
ax2.plot(lmd_ATMO_clear,dppm_ATMO_clear*1e-4,
             color=atmo_color,label=modelname[1]+bestparam_clear[1], **kwargs)
ax2.plot(lmd_Phoenix_clear,dppm_Phoenix_clear*1e-4,
             color=phoenix_color,label=modelname[2]+bestparam_clear[2], **kwargs)


# labels the y axis of each subplot
ax1.set_ylabel("transit depth [%]")
ax2.set_ylabel("transit depth [%]")

# labels the x axis of bottom subplot
ax2.set_xlabel("wavelength [$\mu$m]")

# labels each subplot
ax1.text(s='(a)', x=0.61, y=2.22, fontsize=20)
ax2.text(s='(b)', x=0.61, y=2.22, fontsize=20)

# create the axes legends
for a in [ax1, ax2]:
    leg = a.legend(frameon=0,fontsize=12,loc='upper left',ncol=2,
                   bbox_to_anchor=(0.05, 1))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(6.0)

# set the axes titles
ax1.set_title('Best-fit cloudy models',color='black')
ax2.set_title('Best-fit clear models',color='black')

# set the xscale, limits, and ticks
plt.xscale('log')
plt.xlim(0.6,2.86)
xticks = np.append(np.linspace(0.6,2,6), np.linspace(2.3,2.8,2))
xticks = np.round(xticks,2)
plt.xticks(xticks, labels=np.round(xticks,2))
plt.minorticks_off()

# set the y limits and ticks
plt.ylim(2.02,2.25)
plt.yticks(np.arange(2.05,2.3,0.05))

plt.savefig('../figures/transmission_grid_summary.pdf',
               dpi=300, rasterize=True, bbox_inches='tight', transparent=False)
