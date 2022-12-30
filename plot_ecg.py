import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import ecg_plot

import os
os.chdir("D:\\age\\04")

# load dataset
file_address = open('test_all_shuffle.pkl', 'rb')
lead = pickle.load(file_address)
file_address.close()    
print(len(lead))

# pick one sample
# this sample was already be well processed
data = lead[0][0]

# the order of 12 leads  
lead_name = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# plot 12 leads 
# without fixed range of y axis
def plot_seaborn(data, save_name):
    dpi_value =300
    plt.figure(figsize= (6,10))
    gs1 = gridspec.GridSpec(12,1)
    gs1.update(wspace=0, hspace=0.3) # set the spacing between axes. 
    font = {'size': 8}
    matplotlib.rc('font', **font)
    for i in range(len(lead_name)):
        signal = data[i]
        ax = plt.subplot(gs1[i])
        ax.plot(signal, color='black', linewidth=0.6) # control the width of line in the plot
        ax.set_ylabel(lead_name[i])
        ax.set_yticks([])
        ax.set_xticks([])
    ax.set_xticks(list(range(0,5001,1000)))
    plt.style.use('seaborn-white')
    plt.savefig("{}.png".format(save_name), dpi=dpi_value,bbox_inches="tight")
    plt.close()

plot_seaborn(data,'seaborn_white')



# plot 12 leads
# with fixed range of y axis 
def plot_cs(dat, name, tlim=5000):
    # Get Lead data, this data must be already divided by 10 (Eg. SPxml)
    fig, ax = plt.subplots(12,1,figsize=(20*3, 2*3*12))
    
    # Plot Lead data
    for i in range(12):
        ax[i].plot(dat[i][0:tlim], linewidth = 5)
        ax[i].set_ylabel(lead_name[i],fontsize=20)
        ax[i].tick_params(axis="x", labelsize=20)
        ax[i].tick_params(axis="y", labelsize=20)
        # Set axis limit
        ax[i].set_xlim(0, tlim)
        ax[i].set_ylim(-20, 20)  # Some leads will need larger limits

    # Ensure the aspect ratio to force a square grid
        ax[i].set_aspect(100/10)

    # Draw the grid
        ax[i].xaxis.set_major_locator(MultipleLocator(100))
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))

        ax[i].yaxis.set_major_locator(MultipleLocator(10))
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))

        ax[i].grid(which='major', color='#FF0000', linestyle='-')
        ax[i].grid(which='minor', color='#FF0000', linestyle=':')

    plt.savefig(name+'.pdf', dpi=1000)
    plt.close()



plot_cs(data, "cs", tlim=5000)


# plot 12 leads
# close to the format doctors usully read
# still unknow how much should data be divided first before plot on the figure
# data/15
ecg_plot.plot(data/15, sample_rate = 500, title = 'ECG 12',lead_index = lead_name)
ecg_plot.save_as_png('ecg_plot')

