import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as colors


__all__ = ['load_plt_params', 'load_parula', 'pipeline_dictionary',
           'convolve_model', 'convolve_model_xy', 'truncate_colormap',
           'avg_lightcurves', 'get_MAD_sigma']

def load_plt_params():
    """ Load in plt.rcParams and set (based on paper defaults).
    """
    params = Table.read('rcParams.txt', format='csv')
    for i, name in enumerate(params['name']):
        try:
            plt.rcParams[name] = float(params['value'][i])
        except:
            plt.rcParams[name] = params['value'][i]
    return params

def load_parula():
    """ Load in custom parula colormap.
    """
    colors = np.load('../data/parula_colors.npy')
    return colors

def pipeline_dictionary():
    """ Loads in the custom colors for the paper figures.
    """
    pipeline_dict = {}
    pipelines = Table.read('pipelines.csv', format='csv')

    # Sets the initials key for each pipeline
    for i, name in enumerate(pipelines['initials']):
        pipeline_dict[name] = {}
        pipeline_dict[name]['color'] = pipelines['color'][i]
        pipeline_dict[name]['name'] = pipelines['name'][i]
        pipeline_dict[name]['filename'] = pipelines['filename'][i]
        pipeline_dict[name]['author'] = pipelines['author'][i]
        pipeline_dict[name]['contact'] = pipelines['contact'][i]

    return pipeline_dict

def convolve_model(filename, R=300):
    model = np.loadtxt(filename)

    R0=3000.0 #cross-section resolution
    xker = np.arange(1000)-500
    sigma = (R0/R)/(2.* np.sqrt(2.0*np.log(2.0)))
    yker = np.exp(-0.5 * (xker / sigma)**2.0)
    yker /= yker.sum()
    model_to_plot=np.convolve(model[:,1],yker,mode='same') #convolving
    return model[:,0], model_to_plot


def convolve_model_xy(y, R=300):
    R0=3000.0 #cross-section resolution
    xker = np.arange(1000)-500
    sigma = (R0/R)/(2.* np.sqrt(2.0*np.log(2.0)))
    yker = np.exp(-0.5 * (xker / sigma)**2.0)
    yker /= yker.sum()
    model_to_plot=np.convolve(y,yker,mode='same') #convolving
    return model_to_plot

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,
                                                                                            a=minval,
                                                                                            b=maxval),cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def avg_lightcurves(i, data, err, idx_oot, per=5):
    """
    Creates averaged spectroscopic light curves across 'per' number of
    channels.
    """

    flux  = np.zeros((per*2+1, len(data['time'])))
    model = np.zeros((per*2+1, len(data['time'])))
    error = np.zeros((per*2+1, len(data['time'])))
    fnorm = np.zeros((per*2+1, len(data['time'])))
    wrange = np.zeros(per*2+1)

    for j in range(i-per, i+per+1):
        flux[j-(i-per)]  = data['lc_w{}'.format(j)]
        model[j-(i-per)] = data['combined_model_w{}'.format(i)]

        eind = np.where(err[1] <=
                        data['w{}'.format(j)][0].value)[0][0]

        error[j-(i-per)] = err[3][:,eind]
        fnorm[j-(i-per)] = err[2][:,eind]
        wrange[j-(i-per)] = data['w{}'.format(j)][0].value

    wrange = np.sort(wrange)
    wmed = wrange[5]
    low,upp = wrange[5]-wrange[0], wrange[-1]-wrange[5]
    lim = np.round(np.nanmedian([low,upp]),3)

    e = (np.sqrt(np.nansum(error,axis=0))/len(error))/np.nanmax(fnorm)
    f = np.nanmean(flux, axis=0)
    m = np.nanmax(model[-2:], axis=0)

    f /= np.nanmedian(f[idx_oot])
    m /= np.nanmedian(m[idx_oot])

    return f, e, m, wmed, lim, wrange[0], wrange[-1], model, flux

def get_MAD_sigma(x, median):
    """
    Wrapper function for transitspectroscopy.utils.get_MAD_sigma to estimate
    the noise properties of the light curves.

    Parameters
    ----------
    x : np.ndarray
    median : np.ndarray
    """
    mad = np.nanmedian( np.abs ( x - median ) )

    return 1.4826*mad
