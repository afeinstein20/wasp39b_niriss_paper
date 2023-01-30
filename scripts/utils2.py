import numpy as np
from tqdm import tqdm
from astropy.table import Table
#import transitspectroscopy as ts

__all__ = ['scaling_image_regions', 'exotep_to_ers_format', 'chromatic_writer',
           'bin_at_resolution', 'get_MAD_sigma']

def scaling_image_regions(integraions, bkg, x1, x2, y1, y2,
                          vals=np.linspace(0,10,500), test=True):
    """
    Scales a given (x1, y1) to (x2, y2) region of the input integration
    to the input background. This function is meant to be used for scaling the
    STScI model background and the F277W regions to the raw Stage 2
    integrations.

    Parameters
    ----------
    integrations : np.ndarray
       The data frames to scale the background to.
    bkg : np.ndarray
       The background model to use.
    x1 : int
       The lower left x-coordinate of the region to scale.
    x2 : int
       The upper right x-coordinate of the region to scale.
    y1 : int
       The lower left y-coordinate of the region to scale.
    y2 : int
       The upper right y-coordinate of the region to scale.
    vals : np.array, optional
       The scaling values to test over. Default is `np.linspace(0,10,500)`.
       I recommend running this with `test = True` first, and then changing this
       vals array to be centered on the best fit from the test round.
    test : bool, optional
       The option to test values first before running the whole scaling routine.
       Default is `True`. If `True`, will only run the first five integrations
       and print the scaling values from that run.

    Returns
    -------
    scaling_val : float
       The median best-fit scaling value from the background to the integration.
    """

    shape = (x2-x1) * (y2-y1)
    scaling_vals = np.zeros(len(integrations))

    if test:
        length = 5
    else:
        length=len(integrations)

    for j in tqdm(range(length)):
        rms = np.zeros(len(vals))

        for i, v in enumerate(vals):
            diff = integrations[j][x1:x2,y1:y2] - (bkg[x1:x2,y1:y2]*v)
            rms[i] = np.sqrt(np.nansum(diff**2)/shape)

        scaling_vals[j] = vals[np.argmin(rms)]

    if test:
        print(scaling_vals[:5])

    return np.nanmedian(scaling_vals)


def chromatic_writer(filename, time, wavelength, flux, var):
    """Writes numpy files to read into chromatic reader for `feinstein.py`."""
    np.save(filename+'.npy', [time, wavelength, flux, var])
    return


def exotep_to_ers_format(filename1, filename2, filename):
    """
    Takes the outputs of exotep and puts it in the agreed upon format.

    Parameters
    ----------
    filename1 : str
       The filename (+ path) for the output csv for NIRISS order 1.
    filename2 : str
       The filename (+ path) for the output csv for NIRISS order 2.
    filename : str
       The output filename to save the new table to.

    Returns
    -------
    tab : astropy.table.Table
    """
    table1 = Table.read(filename1, format='csv')
    table2 = Table.read(filename2, format='csv')

    tab = Table(names=['wave', 'wave_err', 'dppm', 'dppm_err', 'order'],
                dtype=[np.float64, np.float64, np.float64, np.float64, int])

    for i in range(len(table1)):
        row = [table1['wave'][i], table1['waveMin'][i],
               table1['yval'][i], table1['yerrLow'][i], 1]
        tab.add_row(row)

    short = table2[table2['wave'] < 0.9]
    for i in range(len(short)):
        row = [short['wave'][i], short['waveMin'][i],
               short['yval'][i], short['yerrLow'][i], 2]
        tab.add_row(row)

    tab.write(filename, format='csv', overwrite=True)
    return tab

def bin_at_resolution(wavelengths, depths, depth_error, wave_error, R = 100):
    # Sort wavelengths from lowest to highest:
    idx = np.argsort(wavelengths)

    ww = wavelengths[idx]
    dd = depths[idx]
    de = depth_error[idx]
    we = wave_error[idx]

    # Prepare output arrays:
    wout, dout, derrout = np.array([]), np.array([]), np.array([])
    werrout = np.array([])

    oncall = False

    # Loop over all (ordered) wavelengths:
    for i in range(len(ww)):

        if not oncall:

            # If we are in a given bin, initialize it:
            current_wavs = np.array([ww[i]])
            current_depths = np.array(dd[i])
            current_errors = np.array(de[i])
            current_werrs = np.array(we[i])
            oncall = True

        else:
            # On a given bin, append next wavelength/depth:
            current_wavs = np.append(current_wavs, ww[i])
            current_depths = np.append(current_depths, dd[i])
            current_errors = np.append(current_errors, de[i])
            current_werrs = np.append(current_werrs, we[i])

            # Calculate current mean R:
            current_R = np.mean(current_wavs) / np.abs(current_wavs[0] - current_wavs[-1])

            # If the current set of wavs/depths is below or at the target resolution, stop and move to next bin:
            if current_R <= R:

                wout = np.append(wout, np.mean(current_wavs))
                dout = np.append(dout, np.mean(current_depths))

                #derrout = np.append(derrout, np.sqrt(np.nansum(current_errors**2)/len(current_errors)))

                # np.sqrt(2) * hst['Rp/R*'] * hst['Rp/R* error'] * 100
                derrout = np.append(derrout,
                                    np.sqrt(np.nansum((current_errors)**2))/len(current_errors))

                #derrout = np.append(derrout, np.sqrt(np.var(current_depths)) / np.sqrt(len(current_depths)))

                wmax = current_wavs[0]-current_werrs[0]
                wmin = current_wavs[-1]+current_werrs[-1]
                werrout = np.append(werrout, np.abs(wmax - wmin)/2)

                oncall = False

    return wout, dout, derrout, werrout


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
