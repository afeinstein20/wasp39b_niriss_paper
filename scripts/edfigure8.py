# PLOTS THE POTASSIUM FEATURE
import os
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib import cm
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.special import erfcinv
import matplotlib.colors as colors
from scipy.interpolate import interp1d

from utils import (load_plt_params, convolve_model, load_parula,
                   truncate_colormap)

# set the matplotlib parameters
pltparams = load_plt_params()
COLOR = pltparams[pltparams['name']=='text.color']['value'][0]
parula = load_parula()

# grabs files in [K/O] directory
path = '../data/ko_short_lam/'
models = np.sort([os.path.join(path, i) for i in os.listdir(path) if
                  i.endswith('.txt')])

# grabs reference model and the full-resolution transmission spectrum
ref_file = '../data/Main_Models/model_reference.txt'

# loads in R=300 transmission spectrum
data = Table.read('../data/ts/CMADF-WASP_39b_NIRISS_transmission_spectrum_R300.csv',
                  format='csv', comment='#')
dat = np.loadtxt('../data/model_rule_out_ko/WASP39b_niriss.txt')

# loads in instrument resolution transmission spectrum
full = np.loadtxt('../data/model_rule_out_ko/WASP39b_niriss.txt')

# pre-computed [K/O] chi^2/N values
ko_labels = np.array([])
ko_chi2 = np.array([])
for i in range(len(models)):
    fn = models[i].split('/')[-1]
    each = fn.split('_')
    ko_labels = np.append(ko_labels, float(each[0]))
    ko_chi2 = np.append(ko_chi2, float(each[3].split('t')[0][:-1]))
argsort = np.argsort(ko_labels)
ko_labels = ko_labels[argsort]
ko_chi2 = ko_chi2[argsort]

# converts chi^2/N --> p-value --> sigma
p_value = 1 - stats.chi2.cdf(np.array(ko_chi2)*len(dat[:,0]), len(dat[:,0])-1)
sigma = np.sqrt(2)*erfcinv(p_value)

# defines the color map and scaling for the sigma-values
CMAP = plt.get_cmap('Oranges_r')
new_cmap= truncate_colormap(CMAP , 0.1, 2.0)
chi_arr=np.arange(1.3, 3.3, 0.001)
cNorm  = colors.Normalize(vmin=0, vmax=len(chi_arr))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=new_cmap)

# initializes the figure
cmap_name = sns.color_palette("rocket", as_cmap=True)

CMAP = plt.get_cmap(cmap_name)
new_cmap= truncate_colormap(CMAP , 0.1, 2.0)
chi_arr=np.arange(1., 9, 0.1)
cNorm  = colors.Normalize(vmin=1, vmax=12)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=new_cmap)


fig, ax = plt.subplots(figsize=(12,6))
fig.set_facecolor('w')

cax = ax.inset_axes([0.56, 0.73, 0.4, 0.4])

cutends = 15
R=300

q = data['quality']==0
ax.errorbar(data['wave'][q], data['dppm'][q]/1e6,
         xerr=data['wave_error'][q],
         yerr=data['dppm_err'][q]/1e6,
         linestyle='', marker='o',
         markeredgecolor=COLOR,
         ecolor=COLOR, ms=8, lw=2,
         markeredgewidth=1.5,
         color='w', zorder=200, label='R=300')

ax.errorbar(full[:,0], full[:,2],
         xerr=full[:,1],
         yerr=full[:,3],
         linestyle='', marker='o',
         markeredgecolor='#c7c3c0',
         ecolor='#c7c3c0',
         color='#c7c3c0', zorder=0, label='instrument resolution')


inds = np.linspace(0,235,len(models),dtype=int)
inds[-3] = 210
idx_y = np.array([])

rocket = sns.color_palette("rocket", 7)
r = 0

for i, fn in enumerate(models):
    if i > 0:
        m0 = convolve_model(models[i-1], R=R)

    m1 = convolve_model(models[i], R=R)

    if i < len(models)-1:
        m2 = convolve_model(models[i+1], R=R)

    idx = (np.abs(np.array(chi_arr) - (ko_chi2[i]/1.3))).argmin()
    colorVal = scalarMap.to_rgba(idx)
    idx_y = np.append(idx_y, idx)


    if i % 2 == 0:
        lw = 3
        label='[K/O] = {}'.format(np.round(ko_labels[i],1))
    else:
        lw = 0
        label=''
        r += 1

    ax.plot(m1[0][cutends:-cutends], m1[1][cutends:-cutends],
             color=parula[inds[i]],
             lw=lw, label=label, zorder=100)

    alpha=0.65
    lw=0.0

    if i > 0 and i < len(models)-1:
        ax.fill_between(m1[0][cutends:-cutends],
                         (m0[1][cutends:-cutends]+m1[1][cutends:-cutends])/2.0,
                         (m1[1][cutends:-cutends]+m2[1][cutends:-cutends])/2.0,
                         zorder=1, color=colorVal, alpha=alpha,
                         lw=lw)

    elif i == 0:
        ax.fill_between(m1[0][cutends:-cutends],
                         m1[1][cutends:-cutends]-0.00003,
                         (m1[1][cutends:-cutends]+m2[1][cutends:-cutends])/2.0,
                         zorder=1, color=colorVal, alpha=alpha,
                         lw=lw)

    elif i == len(models)-1:
        ax.fill_between(m1[0][cutends:-cutends],
                         (m0[1][cutends:-cutends]+m1[1][cutends:-cutends])/2.0,
                         m1[1][cutends:-cutends]+0.00003,
                         zorder=0, color=colorVal, alpha=alpha,
                         lw=lw)


ax.set_xlim(full[:,0][0], full[:,0][-1])
ax.set_xlabel('wavelength [$\mu$m]')
ax.set_ylabel('transit depth [%]')

yticks = np.round(np.arange(0.0205, 0.0230, 0.0005),4)
ax.set_yticks(yticks)
labels = np.round(yticks*100,2)
labels = [format(i, '.2f') for i in labels]
ax.set_yticklabels(labels)
ax.set_ylim(0.0208,0.02215)

cmin = 1.5
cmax = 8.0
cax.imshow(np.full((30,300),
                   np.linspace(cmin, cmax,300)),
           cmap=cmap_name, alpha=alpha)

sort = np.argsort(idx_y)
fit = interp1d(np.linspace(cmin, cmax, 300),
               np.arange(0,300,1))

ticks = np.arange(2, 9, 1,dtype=int)
cax.set_xticks(fit(ticks))
cax.set_xticklabels(np.round(ticks,2))
cax.set_yticks([])

cax.set_xlabel(r'$\sigma$-deviation')
cax.yaxis.set_label_position("right")

leg = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=3, mode="expand", borderaxespad=0.,
                 fontsize=16)
for legobj in leg.legendHandles:
    legobj.set_linewidth(5.0)

plt.subplots_adjust(wspace=0.05)

plt.savefig('../figures/potassium.pdf',
            #transparent=True,
            dpi=300, rasterize=True, bbox_inches='tight')
