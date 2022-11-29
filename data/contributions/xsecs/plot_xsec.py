
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import netCDF4

plot_name='xsec'
model_type='Inh. \,A\&M'
bestred=1.12
bestmet=1.375
bestco=0.2
bestko=0.1
chi_sq=2.59
xlim_input=[0.55,2.85]
ylim_input=[-30,-22]


# Set font properties
font = {'weight' : 'bold',
        'size'   : 17}
plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath} \boldmath'



#Plot model. Calls open model
def plot_model(ax, wave, model_depth, smooth_model=True, color='grey', model_label=None, zorder=0, alpha=1.0):
    if smooth_model:
        R=300.0#793. #native instrumnet resolution
        R0=3000.0 #cross-section resolution
        xker = np.arange(1000)-500
        sigma = (R0/R)/(2.* np.sqrt(2.0*np.log(2.0)))
        yker = np.exp(-0.5 * (xker / sigma)**2.0)
        yker /= yker.sum()
        model_to_plot=np.convolve(model_depth,yker,mode='same') #convolving
    else:
        model_to_plot=model_depth
    ax.plot(wave, model_to_plot, lw=2, color = color, zorder=zorder, alpha=alpha, label=r'$\mathrm{'+model_label+'}$')

#Beautify the plot
def bolden(ax, fs=15):
    [i.set_linewidth(2) for i in iter(ax.spines.values())]
    ax.tick_params(which='major', direction='in', length=8, width=2,top=True, right=True, zorder=1e6)
    ax.tick_params(which='minor', direction='in', length=4, width=2,top=True, right=True, zorder=1e6)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax.set_ylabel(r'$\mathbf{\log_{10}\sigma (m^2/mol)}$', fontsize=fs)
    ax.set_xlabel(r'$\mathbf{Wavelength} \, \, \boldsymbol{(} \boldsymbol{\mu} \mathbf{m} \boldsymbol{)}$', fontsize=fs)


#Let us create a model array
#the model array has the median, 1 sigma, and 2 sigma contours

#Create our plot
fig=plt.figure(figsize=(11, 5))
spec = plt.gca()

wave, xh2o, xco, xco2, xch4, xnh3, xh2s, xph3,xhcn,xc2h2,xoh,xna,xk,xso2=np.loadtxt('xsecs.txt').T

plot_model(spec,wave,xh2o, model_label='H2O',color='blue', smooth_model=True)
plot_model(spec,wave,xco,  model_label='CO',color='tab:blue', smooth_model=True)
plot_model(spec,wave,xco2,  model_label='CO2',color='tab:gray', smooth_model=True)
plot_model(spec,wave,xch4,  model_label='CH4',color='tab:orange', smooth_model=True)
plot_model(spec,wave,xnh3,  model_label='NH3',color='tab:pink', smooth_model=True)
plot_model(spec,wave,xh2s,  model_label='H2S',color='tab:brown', smooth_model=True)
plot_model(spec,wave,xph3,  model_label='PH3',color='tab:olive', smooth_model=True)
plot_model(spec,wave,xhcn,  model_label='HCN',color='tab:cyan', smooth_model=True)
plot_model(spec,wave,xc2h2,  model_label='C2H2',color='tab:green', smooth_model=True)
plot_model(spec,wave,xoh,  model_label='OH',color='tab:purple', smooth_model=True)
plot_model(spec,wave,xna,  model_label='Na',color='m', smooth_model=True)
plot_model(spec,wave,xk,  model_label='K',color='gold', smooth_model=True)
plot_model(spec,wave,xso2,  model_label='SO2',color='tab:red', smooth_model=True)

# plot_data(spec,'nircam_ks', name='KS',color='tab:red', alpha=1.0)
# plot_data(spec,'niriss', name='NIRISS',color='gold', alpha=1.0)
# r'$\mathrm{'+model_label+'}$'


# spec.text(0.7,0.9,r'$\mathrm{\chi^2/N_{data}='+str(round(chi_sq,2))+'}$',transform=spec.transAxes)
# spec.text(0.5,0.9, r'$\mathbf{WASP\text{-}39b}$', fontsize = 18, transform=spec.transAxes)
# spec.text(0.3,0.1,r'$\mathrm{[M/H]='+str(round(bestmet,2))+'}$',transform=spec.transAxes)
# spec.text(0.5,0.1,r'$\mathrm{[C/O]='+str(round(bestco,2))+'}$',transform=spec.transAxes)
# spec.text(0.8,0.1,r'$\mathrm{[K/O]='+str(round(bestko,2))+'}$',transform=spec.transAxes)
# spec.text(0.01,0.1,r'$\mathrm{Redistribution='+str(round(bestred,2))+'}$',transform=spec.transAxes)

# spec.errorbar(data_lam, data_depth*100, xerr=data_bin, yerr=data_error*100, marker='o',markersize=6, elinewidth=2, capsize=3, capthick=1.2, zorder=500, ls='none', color='tab:purple',markeredgecolor='black',  ecolor='black', alpha=1.0, label=r'$\mathrm{G395H}$')

spec.set_xlim(xlim_input[0],xlim_input[1])
spec.set_ylim(ylim_input[0],ylim_input[1])

# spec.set_xscale('log')


spec.legend(loc=1, ncol=4, fontsize=10, frameon=False)
bolden(spec)

plt.tight_layout()
fig.savefig(plot_name+".pdf")
