# PLOT OF ALL WHITE LIGHT CURVES FROM EACH PIPELINE

import pickle
import os, sys
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

from utils import pipeline_dictionary, load_plt_params

# set the matplotlib parameters
load_plt_params()

# Load in colors and filenames
pipeline_dict = pipeline_dictionary()

# Set up the figure environment
fig = plt.figure(figsize=(18, 16))
gs = GridSpec(3,2, figure=fig, height_ratios=[0.1, 2, 1.2])
fig.set_facecolor('w')

ax1 = fig.add_subplot(gs[1,0]) # white light curves (Order 1)
ax2 = fig.add_subplot(gs[1,1]) # residuals (Order 1)
ax3 = fig.add_subplot(gs[2,0]) # white light curves (Order 2)
ax4 = fig.add_subplot(gs[2,1]) # residuals (Order 2)

ax_legend = fig.add_subplot(gs[0,:]) # legend axes

axes = [[ax1, ax2], [ax3, ax4]]

# defines offsets between each light curve/residuals
offset_lc = np.full(len(names), 0.02)
offset_resid = np.full(len(names), 0.002)
