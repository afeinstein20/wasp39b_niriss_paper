# PLOTS THE STACKED SPECTROSCOPIC LIGHT CURVES
import numpy as np
import matplotlib.pyplot as plt

from utils import load_plt_params, load_parula, avg_lightcurves, get_MAD_sigma

# set the matplotlib parameters
load_plt_params()
colors = load_parula()


# Load in the spectroscopic light curves
data1 = np.load('../data/spec_lc/NIRISS_WASP-39_ADF_order1_all_models_and_LCs.npy',
               allow_pickle=True).item()
data2 = np.load('../data/spec_lc/NIRISS_WASP-39_ADF_order2_all_models_and_LCs.npy',
               allow_pickle=True).item()

# Load in stellar spectra for errors
err1 = np.load('../data/spec_lc/ADF_wasp-39_order1_stellar_spectra_v10.npy',
               allow_pickle=True)
err2 = np.load('../data/spec_lc/ADF_wasp-39_order2_stellar_spectra_v10.npy',
               allow_pickle=True)

# Define out of transit indices
idx_oot = np.append(np.arange(0,200,1,dtype=int),
                    np.arange(400,len(data1['time']),1,dtype=int))

n = 8 # number of channels to plot
per = 9 # number of channels to average over

# initializes the figure and facecolor
fig, (ax1,ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios':[1.75,1]},
                              figsize=(18,20))
fig.set_facecolor('w')

temp_colors = colors[np.append(np.flip(np.linspace(20,210,n*2,dtype=int)[n:]),
                               np.flip(np.linspace(20,210,n*2,dtype=int)[:n])
                               )]
midpoint = (0.556726+2459787) - 2400000
offset = 0.014
c, o = 0, 0
textx = -0.15
alpha=0.25

text_kwargs = {'color':'k', 'zorder':20, 'fontsize':15,
               'fontweight':'bold'}

idx_oot = np.append(np.arange(0,200,1,dtype=int),
                    np.arange(400,len(data1['time']),1,dtype=int))
idx_in = np.arange(200,400,1,dtype=int)

# ORDER 2 LIGHT CURVES
n = 8
inds = [368, 440, 575, 640, 800, 820, 890, 960]#np.linspace(368,880,n,dtype=int)
for i in inds:
    flux, err, model, wmed, wlim, wlow, wupp,_,_ = avg_lightcurves(i, data2,
                                                                   err2,
                                                                   idx_oot,
                                                                   per=per)

    rms = get_MAD_sigma(np.median(np.abs(flux-model)[idx_oot] - 1.),
                                  np.abs(flux-model)[idx_oot] - 1.)*1e6

    ax1.errorbar(data2['time']-midpoint, flux-offset*o,
                 yerr=err, linestyle='', marker='.',
                 color=temp_colors[c], alpha=alpha)

    ax1.plot(data2['time']-midpoint, model-offset*o,
             zorder=10,
             color=temp_colors[c], lw=3)


    ax1.text(s=r'{:.3f} - {:.3f} um'.format(np.round(wlow,3),
                                            np.round(wupp,3)),
             x=textx, y=1.003-(offset*o), **text_kwargs)

    ax2.text(s=r'{} ppm'.format(int(np.round(rms,0))),
             x=textx, y=0.002-(offset*o), **text_kwargs)

    ax2.errorbar(data2['time']-midpoint, (flux-model)-offset*o,
                 yerr=err,
                 linestyle='', marker='.',
                 color=temp_colors[c], alpha=0.3)

    ax2.axhline(-offset*o, color=temp_colors[c], zorder=0, lw=2)

    o += 1
    c += 1

hline_dict = {'color':'#8a8988', 'lw':4, 'zorder':0}
ax1.axhline(1-(offset*o+0.013), **hline_dict)
ax2.axhline(0-(offset*o+0.013), **hline_dict)

ax1.text(s='order 2', x=0.10, y=1-(offset*o+0.011), color=hline_dict['color'],
         fontsize=22)
ax1.text(s='order 1', x=0.10, y=1-(offset*o+0.019), color=hline_dict['color'],
         fontsize=22)

ax2.text(s='order 2', x=0.065, y=0-(offset*o+0.011), color=hline_dict['color'],
         fontsize=22)
ax2.text(s='order 1', x=0.065, y=0-(offset*o+0.019), color=hline_dict['color'],
         fontsize=22)

add_offset = 0.023
n = 8
inds = np.linspace(25,2012-25*2,n,dtype=int)

# ORDER 1 LIGHT CURVES
for i in inds:
    flux, err, model, wmed, wlim, wlow, wupp,_,_ = avg_lightcurves(i, data1,
                                                                   err1,
                                                                   idx_oot,
                                                                   per=per)
    rms = get_MAD_sigma(np.median(np.abs(flux-model)[idx_oot] - 1.),
                                  np.abs(flux-model)[idx_oot] - 1.)*1e6

    ax1.errorbar(data1['time']-midpoint, flux-offset*o-add_offset,
                 yerr=err, linestyle='', marker='.',
                 color=temp_colors[c], alpha=alpha)

    ax1.plot(data1['time']-midpoint, model-offset*o-add_offset,
             zorder=10,
             color=temp_colors[c], lw=3)

    ax1.text(s=r'{:.3f} - {:.3f} um'.format(np.round(wlow,3),
                                      np.round(wupp,3)),
             x=textx, y=1.003-(offset*o)-add_offset, **text_kwargs)

    ax2.text(s=r'{} ppm'.format(int(np.round(rms,0))),
             x=textx, y=0.002-(offset*o)-add_offset, **text_kwargs)

    ax2.errorbar(data1['time']-midpoint, (flux-model)-offset*o-add_offset,
                 yerr=err,
                 linestyle='', marker='.',
                 color=temp_colors[c], alpha=0.3)

    ax2.axhline(-offset*o-add_offset, color=temp_colors[c], zorder=0, lw=2)

    o += 1
    c += 1


ax1.set_xlim(-0.152,0.152)
ax2.set_xlim(-0.152,0.152)

fs = 24
ax2.set_ylabel('residuals + offset'.format(i+1), fontsize=fs)
ax1.set_ylabel('relative flux + offset'.format(i+1), fontsize=fs)
ax1.set_xlabel('time from mid-transit [hr]', fontsize=fs)
ax2.set_xlabel('time from mid-transit [hr]', fontsize=fs)

ax1.set_ylim(0.735, 1.02)
ax2.set_ylim(-0.265, 0.02)
ax2.set_yticklabels([])

ax1.text(s='(a)', x=-0.145, y=1.013, fontweight='bold')
ax2.text(s='(b)', x=-0.145, y=0.013, fontweight='bold')

ax1.set_rasterized(True)
ax2.set_rasterized(True)

plt.subplots_adjust(wspace=0.15, hspace=0)

plt.savefig('../figures/spec_lcs.jpg', dpi=200, rasterize=True,
            bbox_inches='tight')
