# PLOTS THE STACKED SPECTROSCOPIC LIGHT CURVES

from utils import load_plt_params, load_parula, avg_lightcurves, get_MAD_sigma

# set the matplotlib parameters
load_plt_params()
parula = load_parula()


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

# sets the colors to use for each spectroscopic light curves
temp_colors = parula[np.append(np.flip(np.linspace(20,210,n*2,dtype=int)[n:]),
                               np.flip(np.linspace(20,210,n*2,dtype=int)[:n])
                               )]
# transit midpoint (based on chromatic_fitting fits)
midpoint = (0.556726+2459787) - 2400000
# sets offset value to separate spectroscopic light curves by
offset = 0.01
c, o = 0, 0
# sets the x value for where to write the text
textx = -0.15
# sets alpha value for background points
alpha=0.25
# creates a **kwargs dictionary for the text on each plot
text_kwargs = {'color':'k', 'zorder':20, 'fontsize':15,
               'fontweight':'bold'}

###
# CREATES AVERAGED SPECTROSCOPIC LIGHT CURVES FOR ORDER 2
###
inds = np.linspace(368,880,n,dtype=int)
for i in inds:
    flux, err, model, wmed, wlim = avg_lightcurves(i, data2, err2, per=per)
    rms = get_MAD_sigma(np.median(np.abs(flux-model)[idx_oot] - 1.),
                                  np.abs(flux-model)[idx_oot] - 1.)*1e6

    # plots the data and error bars as points for order 2
    ax1.errorbar(data2['time']-midpoint, flux-offset*o,
                 yerr=err, linestyle='', marker='.',
                 color=temp_colors[c], alpha=alpha)

    # plots the best fit transit model for order 2
    ax1.plot(data2['time']-midpoint, model-offset*o,
             zorder=10,
             color=temp_colors[c], lw=3)

    # adds text for the wavelength range of the channel for order 2
    ax1.text(s=r'{} +/- {} um'.format(np.round(wmed,3),
                                            np.round(wlim,3)),
             x=textx, y=1.003-(offset*o), **text_kwargs)

    # adds text for the ppm scatter of the channel for order 2
    ax2.text(s=r'{} ppm'.format(int(np.round(rms,0))),
             x=textx, y=0.002-(offset*o), **text_kwargs)

    # plots the residuals of the channel for order 2
    ax2.errorbar(data2['time']-midpoint, (flux-model)-offset*o,
                 yerr=err,
                 linestyle='', marker='.',
                 color=temp_colors[c], alpha=alpha)

    o += 1
    c += 1

###
# CREATES AVERAGED SPECTROSCOPIC LIGHT CURVES FOR ORDER 1
###
inds = np.linspace(20,2012-20*2,n,dtype=int)
for i in inds:
    flux, err, model, wmed, wlim = avg_lightcurves(i, data1, err1, per=per)
    rms = get_MAD_sigma(np.median(np.abs(flux-model)[idx_oot] - 1.),
                                  np.abs(flux-model)[idx_oot] - 1.)*1e6

    # plots the data and error bars as points for order 1
    ax1.errorbar(data1['time']-midpoint, flux-offset*o,
                 yerr=err, linestyle='', marker='.',
                 color=temp_colors[c], alpha=alpha)

    # plots the best fit transit model for order 1
    ax1.plot(data1['time']-midpoint, model-offset*o,
             zorder=10,
             color=temp_colors[c], lw=3)

    # adds text for the wavelength range of the channel for order 1
    ax1.text(s=r'{} +/- {} um'.format(np.round(wmed,3),
                                            np.round(wlim,3)),
             x=textx, y=1.003-(offset*o), **text_kwargs)

    # adds text for the ppm scatter of the channel for order 1
    ax2.text(s=r'{} ppm'.format(int(np.round(rms,0))),
             x=textx, y=0.002-(offset*o), **text_kwargs)

    # plots the residuals of the channel for order 1
    ax2.errorbar(data1['time']-midpoint, (flux-model)-offset*o,
                 yerr=err,
                 linestyle='', marker='.',
                 color=temp_colors[c], alpha=0.3)

    o += 1
    c += 1

# sets the x-limits for each subplot
ax1.set_xlim(-0.152,0.152)
ax2.set_xlim(-0.152,0.152)

# adds the axes labels
fs = 24
ax2.set_ylabel('residuals + offset'.format(i+1), fontsize=fs)
ax1.set_ylabel('relative flux + offset'.format(i+1), fontsize=fs)
ax1.set_xlabel('time from mid-transit [hr]', fontsize=fs)
ax2.set_xlabel('time from mid-transit [hr]', fontsize=fs)

# sets the x-limits for each subplot
ax1.set_ylim(0.82, 1.01)
ax2.set_ylim(-0.18, 0.01)
ax2.set_yticklabels([])

ax1.set_rasterized(True)
ax2.set_rasterized(True)

plt.subplots_adjust(wspace=0.15, hspace=0)
plt.savefig('../figures/spec_lcs.pdf', dpi=200, rasterize=True,
            bbox_inches='tight')
