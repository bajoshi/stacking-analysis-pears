from __future__ import division
import numpy as np
from astropy.io import fits
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def makefig_hist(qty):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(qty)
    ax.set_ylabel('N')

    return fig, ax

if __name__ == '__main__':

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
    fig_metals = plt.figure()
    fig_tau = plt.figure()
    fig_av = plt.figure()
    fig_mass_wht_ages = plt.figure()
    fig_quench = plt.figure()

    ages = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_ssp_ages.txt', usecols=range(1, int(1e4) + 1))
    metals = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_ssp_metals.txt', usecols=range(1, int(1e4) + 1))
    #mass_wht_ages = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/ssp_mass_weighted_ages.txt', usecols=range(1, int(1e4) + 1))
    ongrid_vals = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_ages.txt', dtype=np.str, usecols=range(0,1,1))

    #quenching_times = 10**ages - 10**mass_wht_ages
    #quenching_times = np.log10(quenching_times)

    """
    # mass-weighted ages
    # the previous array called ages is actually formation time i.e. age of the oldest star
    f_mass_wht_ages = open('ssp_mass_weighted_ages.txt', 'wa')
    timestep = 1e5
    mass_wht_ages = np.empty(ages.shape)
    for j in range(ages.shape[0]):
        ongrid = ongrid_vals[j]
        f_mass_wht_ages.write(ongrid + ' ')
        for i in range(ages.shape[1]):
            formtime = 10**ages[j][i]
            timearr = np.arange(timestep, formtime, timestep) # in years
            tau = 10**logtau[j][i] * 10**9 # in years
            n_arr = np.log10(formtime - timearr) * np.exp(-timearr/tau) * timestep
            d_arr = np.exp(-timearr/tau) * timestep
            
            n = np.sum(n_arr)
            d = np.sum(d_arr)
            mass_wht_ages[j][i] = n / d
            f_mass_wht_ages.write(str(mass_wht_ages[j][i]) + ' ')
        f_mass_wht_ages.write('\n')
    print np.max(mass_wht_ages), np.min(mass_wht_ages)
    f_mass_wht_ages.close()
    sys.exit(0)
    """

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
        ax_gs_ages.hist(ages[count], 10, histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_ages.set_xlim(8, 10)
        ax_gs_ages.set_yscale('log')

        ax_gs_ages.get_xaxis().set_ticklabels([])
        ax_gs_ages.get_yaxis().set_ticklabels([])
        
        if (row == 3) and (column == 0):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)

        if (row == 4) and (column == 2):
            ax_gs_ages.set_xlabel(r'$\mathrm{log(Formation\ Time\ [yr])}$', fontsize=13)
        if (row == 2) and (column == 1):
            ax_gs_ages.set_ylabel('N', fontsize=13)

        if (row == 4) and (column == 0):
            ax_gs_ages.get_xaxis().set_ticklabels(['8', '8.5', '9.0', '9.5', '10'], fontsize=10, rotation=45)
        
        if (row == 4) and (column == 1):
            ax_gs_ages.get_xaxis().set_ticklabels(['', '8.5', '9.0', '9.5'], fontsize=10, rotation=45)
        
        if (row == 4) and (column == 2):
            ax_gs_ages.get_xaxis().set_ticklabels(['8', '8.5', '9.0', '9.5', '10'], fontsize=10, rotation=45)
        
        if (row == 3) and (column == 3):
            ax_gs_ages.get_xaxis().set_ticklabels(['8', '8.5', '9.0', '9.5', '10'], fontsize=10, rotation=45)
        
        if (row == 1) and (column == 4):
            ax_gs_ages.get_xaxis().set_ticklabels(['8', '8.5', '9.0', '9.5', '10'], fontsize=10, rotation=45)

        fig_ages.savefig('/Users/baj/Desktop/FIGS/new_codes/ssp_agedist_jackknife_hist_grid_new' + '_run2.png', dpi=300)

        #### Metallicity grid plots ####
        ax_gs_metals = fig_metals.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_metals.hist(metals[count], 5, histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_metals.set_xlim(0, 0.05)
        ax_gs_metals.set_yscale('log')
        
        ax_gs_metals.get_xaxis().set_ticklabels([])
        ax_gs_metals.get_yaxis().set_ticklabels([])
        
        if (row == 3) and (column == 0):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '', '$10^4$'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)

        if (row == 4) and (column == 2):
            ax_gs_metals.set_xlabel('Metals [Z]', fontsize=13)
        if (row == 2) and (column == 1):
            ax_gs_metals.set_ylabel('N', fontsize=13)

        if (row == 4) and (column == 0):
            ax_gs_metals.get_xaxis().set_ticklabels(['0', '0.01', '0.02', '0.03', '0.04', '0.05'], fontsize=10, rotation=45)
        
        if (row == 4) and (column == 1):
            ax_gs_metals.get_xaxis().set_ticklabels(['', '0.01', '0.02', '0.03', '0.04', ''], fontsize=10, rotation=45)
        
        if (row == 4) and (column == 2):
            ax_gs_metals.get_xaxis().set_ticklabels(['0', '0.01', '0.02', '0.03', '0.04', '0.05'], fontsize=10, rotation=45)
        
        if (row == 3) and (column == 3):
            ax_gs_metals.get_xaxis().set_ticklabels(['0', '0.01', '0.02', '0.03', '0.04', '0.05'], fontsize=10, rotation=45)
        
        if (row == 1) and (column == 4):
            ax_gs_metals.get_xaxis().set_ticklabels(['0', '0.01', '0.02', '0.03', '0.04', '0.05'], fontsize=10, rotation=45)

        fig_metals.savefig('/Users/baj/Desktop/FIGS/new_codes/ssp_metalsdist_jackknife_hist_grid_new' + '_run2.png', dpi=300)

        count += 1

    #plt.show()