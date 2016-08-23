from __future__ import division
import numpy as np
from astropy.io import fits
import collections, sys
#from scipy.integrate import simps

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

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
    figdir = '/Users/baj/Desktop/FIGS/stacking-analysis-pears/figures/'
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.00)
    fig_ages = plt.figure()
    fig_metals = plt.figure()
    fig_tau = plt.figure()
    fig_av = plt.figure()
    fig_mass_wht_ages = plt.figure()
    fig_quench = plt.figure()

    ages = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_ages.txt', usecols=range(1, int(1e4) + 1))
    metals = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_metals.txt', usecols=range(1, int(1e4) + 1))
    logtau = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_logtau.txt', usecols=range(1, int(1e4) + 1))
    tauv = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_tauv.txt', usecols=range(1, int(1e4) + 1))
    mass_wht_ages = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/mass_weighted_ages.txt', usecols=range(1, int(1e4) + 1))
    ongrid_vals = np.loadtxt('/Users/baj/Desktop/FIGS/new_codes/jackknife_ages.txt', dtype=np.str, usecols=range(0,1,1))

    av = (2.5 / np.log(10)) * tauv * 10

    quenching_times = 10**ages - 10**mass_wht_ages
    quenching_times = np.log10(quenching_times)

    """
    # mass-weighted ages
    # the previous array called ages is actually formation time i.e. age of the oldest star
    f_mass_wht_ages = open('mass_weighted_ages.txt', 'wa')
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

        #ctr_ages = collections.Counter()
        #ctr_metals = collections.Counter()
        #ctr_logtau = collections.Counter()
        #ctr_av = collections.Counter()
        #
        #for i in ages[count]:
        #    ctr_ages[i] += 1
        #
        #for i in metals[count]:
        #    ctr_metals[i] += 1        
        #
        #for i in logtau[count]:
        #    ctr_logtau[i] += 1
        #
        #for i in av[count]:
        #    ctr_av[i] += 1

        """
        #### Age grid plots ####
        ax_gs_ages = fig_ages.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_ages.hist(ages[count], len(np.unique(ages[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        # k and v (keys and values) are required to be numpy arrays for the type conversion and sorting
        # the counter objects keys and values are python lists
        #k = np.asarray(ctr_ages.keys())
        #v = np.asarray(ctr_ages.values())
        #args = np.argsort(k)
        #tick_labels = k.astype('|S6')

        #bar_left_edges = [ for i in range(len(k))]
        #ax_gs_ages.bar(k[args] + 0.01, v[args], alpha=0.5, linewidth=0.3)
        #ax_gs_ages.set_xticklabels(['8', '8.5', '9', '9.5', '10'], fontsize=5, rotation=45)
        #ax_gs_ages.set_xticks(np.arange(8,10.5,0.5))

        #ymajorlocator = MultipleLocator(2000)
        #ymajorformatter = FormatStrFormatter('%1.1e')
        #ax_gs_ages.yaxis.set_major_locator(ymajorlocator)
        #ax_gs_ages.yaxis.set_major_formatter(ymajorformatter)
        #ax_gs_ages.yaxis.set_tick_params(labelsize=5)  

        ax_gs_ages.set_xlim(8, 10)
        ax_gs_ages.set_yscale('log')

        ax_gs_ages.get_xaxis().set_ticklabels([])
        ax_gs_ages.get_yaxis().set_ticklabels([])
        #ylabels = [i.get_text() for i in ax_gs_ages.yaxis.get_ticklabels()]
        #ax_gs_ages.yaxis.set_ticklabels(ylabels, fontsize=10)
        # these two lines above don't work to set the labels' fontsize
        
        if (row == 3) and (column == 0):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
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

        fig_ages.savefig(figdir + 'agedist_jackknife_hist_grid_new' + '_run2.png', dpi=300)
        """

        #### Mass weighted age grid plots ####
        ax_gs_mass_wht_ages = fig_mass_wht_ages.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        N, bins, patches = ax_gs_mass_wht_ages.hist(mass_wht_ages[count], 10, histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        # this part of the code i.e. to color the histogram based on the x value came from a stackoverflow answer.
        # I modified it for my code.
        cm = plt.cm.get_cmap('bwr')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # scale values to interval [0,1]
        low_lim = 8.4
        up_lim = 9.9
        col = (bin_centers - low_lim)/(up_lim - low_lim)

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))

        ax_gs_mass_wht_ages.set_yscale('log')
        ax_gs_mass_wht_ages.set_xlim(7.5, 10)
        ax_gs_mass_wht_ages.get_xaxis().set_ticklabels([])
        ax_gs_mass_wht_ages.get_yaxis().set_ticklabels([])
        if (row == 3) and (column == 0):
            ax_gs_mass_wht_ages.get_yaxis().set_ticklabels(['', '', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_mass_wht_ages.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_mass_wht_ages.get_yaxis().set_ticklabels(['', '', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_mass_wht_ages.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_mass_wht_ages.get_yaxis().set_ticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)

        if (row == 4) and (column == 2):
            ax_gs_mass_wht_ages.set_xlabel(r'$\mathrm{log(Mass-weighted\ Age\ [yr])}$', fontsize=13)
        if (row == 2) and (column == 1):
            ax_gs_mass_wht_ages.set_ylabel('N', fontsize=13)

        if (row == 4) and (column == 0):
            ax_gs_mass_wht_ages.get_xaxis().set_ticklabels(['7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 4) and (column == 1):
            ax_gs_mass_wht_ages.get_xaxis().set_ticklabels(['', '8', '8.5', '9.0', '9.5', ''], fontsize=8, rotation=45)
        
        if (row == 4) and (column == 2):
            ax_gs_mass_wht_ages.get_xaxis().set_ticklabels(['7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 3) and (column == 3):
            ax_gs_mass_wht_ages.get_xaxis().set_ticklabels(['7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 1) and (column == 4):
            ax_gs_mass_wht_ages.get_xaxis().set_ticklabels(['7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)

        fig_mass_wht_ages.savefig(figdir + 'mass_wht_agedist_jackknife_hist_grid_new' + '_run2_8p4.png', dpi=300)

        #### Quenching timescale grid plots ####
        ax_quench = fig_quench.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_quench.hist(quenching_times[count], 15, histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_quench.set_yscale('log')
        ax_quench.set_xlim(7, 10)

        ax_quench.get_xaxis().set_ticklabels([])
        ax_quench.get_yaxis().set_ticklabels([])
        if (row == 3) and (column == 0):
            ax_quench.get_yaxis().set_ticklabels(['', '', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_quench.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_quench.get_yaxis().set_ticklabels(['', '', '', '$10^4$'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_quench.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_quench.get_yaxis().set_ticklabels(['', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)

        if (row == 4) and (column == 2):
            ax_quench.set_xlabel(r'$\mathrm{log(Quenching\ Time\ [yr])}$', fontsize=13)
        if (row == 2) and (column == 1):
            ax_quench.set_ylabel('N', fontsize=13)

        if (row == 4) and (column == 0):
            ax_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 4) and (column == 1):
            ax_quench.get_xaxis().set_ticklabels(['', '7.5', '8', '8.5', '9.0', '9.5', ''], fontsize=8, rotation=45)
        
        if (row == 4) and (column == 2):
            ax_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 3) and (column == 3):
            ax_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 1) and (column == 4):
            ax_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)

        fig_quench.savefig(figdir + 'quenchdist_jackknife_hist_grid_new' + '_run2.png', dpi=300)

        """
        #### Metallicity grid plots ####
        ax_gs_metals = fig_metals.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_metals.hist(metals[count], len(np.unique(metals[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        #k = np.asarray(ctr_metals.keys())
        #v = np.asarray(ctr_metals.values())
        #args = np.argsort(k)
        #tick_labels = k.astype('|S6')

        #ax_gs_metals.bar(np.arange(len(k)), v[args], alpha=0.5, linewidth=0.3)
        #ax_gs_metals.set_xticklabels(tick_labels[args], fontsize=9, rotation=45)
        #ax_gs_metals.set_xticks(np.arange(0.4, len(tick_labels) + 0.4, 1))

        ax_gs_metals.set_xlim(0, 0.05)
        ax_gs_metals.set_yscale('log')

        ax_gs_metals.get_xaxis().set_ticklabels([])
        #ylabels = [i.get_text() for i in ax_gs_metals.yaxis.get_ticklabels()]
        #ax_gs_metals.yaxis.set_ticklabels(ylabels, fontsize=10)
        # these two lines above don't work to set the labels' fontsize
        ax_gs_metals.get_yaxis().set_ticklabels([])
        
        if (row == 3) and (column == 0):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '', '$10^4$'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_metals.get_yaxis().set_ticklabels(['$10^0$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '', '$10^4$'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '', '$10^3$', '$10^4$'], fontsize=10)

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

        fig_metals.savefig(figdir + 'metalsdist_jackknife_hist_grid_new' + '_run2.png', dpi=300)

        #### Tau grid plots ####
        ax_gs_tau = fig_tau.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_tau.hist(logtau[count], len(np.unique(logtau[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        #k = np.asarray(ctr_logtau.keys())
        #v = np.asarray(ctr_logtau.values())
        #args = np.argsort(k)
        #tick_labels = k.astype('|S6')
        #ax_gs_tau.bar(np.arange(len(k)), v[args], alpha=0.5, linewidth=0.3)
        #ax_gs_tau.set_xticklabels(tick_labels[args], fontsize=9, rotation=45)
        #ax_gs_tau.set_xticks(np.arange(0.4, len(tick_labels) + 0.4, 1))

        ax_gs_tau.set_xlim(-2, 2)
        ax_gs_tau.set_yscale('log')

        ax_gs_tau.get_xaxis().set_ticklabels([])
        #ylabels = [i.get_text() for i in ax_gs_tau.yaxis.get_ticklabels()]
        #ax_gs_tau.yaxis.set_ticklabels(ylabels, fontsize=10)
        # these two lines above don't work to set the labels' fontsize
        ax_gs_tau.get_yaxis().set_ticklabels([])
        
        if (row == 3) and (column == 0):
            ax_gs_tau.get_yaxis().set_ticklabels(['', '', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_tau.get_yaxis().set_ticklabels(['$10^0$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_tau.get_yaxis().set_ticklabels(['', '', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_tau.get_yaxis().set_ticklabels(['', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_tau.get_yaxis().set_ticklabels(['', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)

        if (row == 4) and (column == 2):
            ax_gs_tau.set_xlabel(r'$\mathrm{log}(\tau)$', fontsize=13)
        if (row == 2) and (column == 1):
            ax_gs_tau.set_ylabel('N', fontsize=13)

        if (row == 4) and (column == 0):
            ax_gs_tau.get_xaxis().set_ticklabels(['-2', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)
        
        if (row == 4) and (column == 1):
            ax_gs_tau.get_xaxis().set_ticklabels(['', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', ''], fontsize=10, rotation=45)
        
        if (row == 4) and (column == 2):
            ax_gs_tau.get_xaxis().set_ticklabels(['-2', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)
        
        if (row == 3) and (column == 3):
            ax_gs_tau.get_xaxis().set_ticklabels(['-2', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)
        
        if (row == 1) and (column == 4):
            ax_gs_tau.get_xaxis().set_ticklabels(['-2', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)

        fig_tau.savefig(figdir + 'logtaudist_jackknife_hist_grid_new' + '_run2.png', dpi=300)

        #### AV grid plots ####
        ax_gs_av = fig_av.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_av.hist(av[count], len(np.unique(av[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        #k = np.asarray(ctr_av.keys())
        #v = np.asarray(ctr_av.values())
        #args = np.argsort(k)
        #tick_labels = k.astype('|S6')

        #ax_gs_av.bar(np.arange(len(k)), v[args], alpha=0.5, linewidth=0.3)
        #ax_gs_av.set_xticklabels(tick_labels[args], fontsize=9, rotation=45)
        #ax_gs_av.set_xticks(np.arange(0.4, len(tick_labels) + 0.4, 1))

        ax_gs_av.set_xlim(0, 2.1)
        ax_gs_av.set_yscale('log')

        ax_gs_av.get_xaxis().set_ticklabels([])
        #ylabels = [i.get_text() for i in ax_gs_av.yaxis.get_ticklabels()]
        #ax_gs_av.yaxis.set_ticklabels(ylabels, fontsize=10)
        # these two lines above don't work to set the labels' fontsize
        ax_gs_av.get_yaxis().set_ticklabels([])
        
        if (row == 3) and (column == 0):
            ax_gs_av.get_yaxis().set_ticklabels(['', '', '$10^4$'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_av.get_yaxis().set_ticklabels(['$10^0$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_av.get_yaxis().set_ticklabels(['$10^0$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_av.get_yaxis().set_ticklabels(['', '', '$10^4$'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_av.get_yaxis().set_ticklabels(['', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)

        if (row == 4) and (column == 2):
            ax_gs_av.set_xlabel(r'$A_V$', fontsize=13)
        if (row == 2) and (column == 1):
            ax_gs_av.set_ylabel('N', fontsize=13)

        if (row == 4) and (column == 0):
            ax_gs_av.get_xaxis().set_ticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)
        
        if (row == 4) and (column == 1):
            ax_gs_av.get_xaxis().set_ticklabels(['', '0.5', '1.0', '1.5', ''], fontsize=10, rotation=45)
        
        if (row == 4) and (column == 2):
            ax_gs_av.get_xaxis().set_ticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)
        
        if (row == 3) and (column == 3):
            ax_gs_av.get_xaxis().set_ticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)
        
        if (row == 1) and (column == 4):
            ax_gs_av.get_xaxis().set_ticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)

        fig_av.savefig(figdir + 'avdist_jackknife_hist_grid_new' + '_run2.png', dpi=300)
        """

        count += 1

    #plt.show()

