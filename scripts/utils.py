from astropy.table import Table
import matplotlib.pyplot as plt

__all__ = ['load_plt_params', 'load_parula', 'pipeline_dictionary']

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
    colors = np.load('parula_colors.npy')
    return colors

def pipeline_dictionary():
    """ Loads in the custom colors for the paper figures.
    """
    pipeline_dict = {}
    pipelines = Table.read('../data/pipelines.csv', format='csv')

    # Sets the initials key for each pipeline
    for i, name in enumerate(pipelines['initials']):
        pipeline_dict[name] = {}
        pipeline_dict[name]['color'] = pipelines['color'][i]
        pipeline_dict[name]['name'] = pipelines['name'][i]
        pipeline_dict[name]['filename'] = pipelines['filename'][i]

    return pipeline_dict
