from __future__ import division
import numpy as np
from astropy.io import fits
import sys, os
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if __name__ == '__main__':

    # Define dirs
    sns_plotdir = '/Users/baj/Desktop/FIGS/new_codes/seaborn_plots/'

    # Read in the stacks
    stacks = fits.open('/Users/baj/Desktop/FIGS/new_codes/coadded_coarsegrid_PEARSgrismspectra.fits')
    # find number of extensions
    totalstacks = 0
    while 1:
        try:
            if stacks[totalstacks+2]:
                totalstacks += 1
        except IndexError:
            break

    # Create grid for making grid plots
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.00)
    fig_ages = plt.figure()
    #fig_metals = plt.figure()
    #fig_tau = plt.figure()
    #fig_av = plt.figure()
    #fig_mass_wht_ages = plt.figure()
    #fig_quench = plt.figure()

    sns.set_style("ticks")

    # read in data 
    ages = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_ages.txt', usecols=range(1, int(1e4) + 1))
    metals = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_metals.txt', usecols=range(1, int(1e4) + 1))
    logtau = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_logtau.txt', usecols=range(1, int(1e4) + 1))
    tauv = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_tauv.txt', usecols=range(1, int(1e4) + 1))
    mass_wht_ages = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/mass_weighted_ages.txt', usecols=range(1, int(1e4) + 1))
    ongrid_vals = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_ages.txt', dtype=np.str, usecols=range(0,1,1))

    av = (2.5 / np.log(10)) * tauv * 10

    quenching_times = 10**ages - 10**mass_wht_ages
    quenching_times = np.log10(quenching_times)

    # Loop over all stacks
    color_step = 0.6
    mstar_step = 1.0

    count = 0
    for stackcount in range(0,totalstacks,1):
        
        flam = stacks[stackcount+2].data[0]
        ferr = stacks[stackcount+2].data[1]
        ferr = ferr + 0.05 * flam # putting in a 5% additional error bar
        ongrid = stacks[stackcount+2].header["ONGRID"]
        numspec = int(stacks[stackcount+2].header["NUMSPEC"])

        if numspec < 5:
            print ongrid, "Too few spectra in stack. Continuing to the next grid cell..."
            continue

        # for the grid plots
        row = 4 - int(float(ongrid.split(',')[0])/color_step)
        column = int((float(ongrid.split(',')[1]) - 7.0)/mstar_step)

        #### Age grid plots ####
        ax_gs_ages = fig_ages.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        sns.distplot(ages[count], bins=10, ax=ax_gs_ages)#, norm_hist=False, kde=False) # bins=len(np.unique(ages[count])),

        ax_gs_ages.set_xlim(8, 10)
        #ax_gs_ages.set_ylim(0, 1)
        ax_gs_ages.set_yscale('log')

        if (row == 3) and ((column == 0) or (column == 1) or (column == 2)):
            ax_gs_ages.get_xaxis().set_ticklabels([])

        if (row == 2):
            ax_gs_ages.get_xaxis().set_ticklabels([])

        count += 1

    plt.show()