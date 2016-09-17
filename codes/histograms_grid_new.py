from __future__ import division

import numpy as np
from astropy.io import fits

import collections
import sys
import os
import datetime
import time
#from scipy.integrate import simps

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from fast_chi2_jackknife import get_total_extensions

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figures_dir = stacking_analysis_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"

def makefig_hist(qty):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(qty)
    ax.set_ylabel('N')

    return fig, ax

def get_mass_weighted_ages(library, ages, logtau, ongrid_vals):

    # the previous array called ages (also called by the same name in an argument for this function)
    # is actually formation time i.e. age of the oldest star
    f_mass_wht_ages = open(stacking_analysis_dir + 'mass_weighted_ages_' + library + '.txt', 'wa')
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
    
    return None

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Read in stacks
    stacks = fits.open(home + '/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra_coarsegrid.fits')
    totalstacks = get_total_extensions(stacks)

    # Create grid for making grid plots
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.00)

    # Create figures to be used for plots for all 3 SPS libraries
    #### BC03 ####
    fig_ages_bc03 = plt.figure()
    fig_metals_bc03 = plt.figure()
    fig_tau_bc03 = plt.figure()
    fig_av_bc03 = plt.figure()
    fig_mass_wht_ages_bc03 = plt.figure()
    fig_quench_bc03 = plt.figure()

    #### MILES ####
    fig_ages_miles = plt.figure()
    fig_metals_miles = plt.figure()

    #### FSPS ####
    fig_ages_fsps = plt.figure()
    fig_metals_fsps = plt.figure()
    fig_tau_fsps = plt.figure()
    fig_mass_wht_ages_fsps = plt.figure()
    fig_quench_fsps = plt.figure()

    # Read ongrid values for subplot placement
    # It needs to be read only once because it is the same for all files
    ongrid_vals = np.loadtxt(stacking_analysis_dir + 'jackknife_ages_bc03.txt', dtype=np.str, usecols=range(0,1,1))

    num_jackknife_samps = 5e4
    # Read files with params from jackknife runs
    #### BC03 ####
    ages_bc03 = np.loadtxt(stacking_analysis_dir + 'jackknife_ages_bc03.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    metals_bc03 = np.loadtxt(stacking_analysis_dir + 'jackknife_metals_bc03.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    logtau_bc03 = np.loadtxt(stacking_analysis_dir + 'jackknife_logtau_bc03.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    tauv_bc03 = np.loadtxt(stacking_analysis_dir + 'jackknife_tauv_bc03.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    mass_wht_ages_bc03 = np.loadtxt(stacking_analysis_dir + 'mass_weighted_ages_bc03.txt', usecols=range(1, int(num_jackknife_samps) + 1))

    av_bc03 = (2.5 / np.log(10)) * tauv_bc03 * 10

    quenching_times_bc03 = 10**ages_bc03 - 10**mass_wht_ages_bc03
    quenching_times_bc03 = np.log10(quenching_times_bc03)

    #### MILES ####
    ages_miles = np.loadtxt(stacking_analysis_dir + 'jackknife_ages_miles.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    metals_miles = np.loadtxt(stacking_analysis_dir + 'jackknife_metals_miles.txt', usecols=range(1, int(num_jackknife_samps) + 1))

    #### FSPS ####
    ages_fsps = np.loadtxt(stacking_analysis_dir + 'jackknife_ages_fsps.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    metals_fsps = np.loadtxt(stacking_analysis_dir + 'jackknife_metals_fsps.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    logtau_fsps = np.loadtxt(stacking_analysis_dir + 'jackknife_logtau_fsps.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    mass_wht_ages_fsps = np.loadtxt(stacking_analysis_dir + 'mass_weighted_ages_fsps.txt', usecols=range(1, int(num_jackknife_samps) + 1))

    quenching_times_fsps = 10**ages_fsps - 10**mass_wht_ages_fsps
    quenching_times_fsps = np.log10(quenching_times_fsps)

    #get_mass_weighted_ages('bc03', ages_bc03, logtau_bc03, ongrid_vals)
    #get_mass_weighted_ages('fsps', ages_fsps, logtau_fsps, ongrid_vals)
    #sys.exit(0)

    color_step = 0.6
    mstar_step = 1.0

    count = 0
    for stackcount in range(0, totalstacks-1, 1):

        ongrid = stacks[stackcount+2].header["ONGRID"]
        numspec = int(stacks[stackcount+2].header["NUMSPEC"])
        print count, ongrid, numspec

        if numspec < 5:
            print ongrid, "Too few spectra in stack. Continuing to the next grid cell..."
            continue

        # for the grid plots
        row = 4 - int(float(ongrid.split(',')[0])/color_step)
        column = int((float(ongrid.split(',')[1]) - 7.0)/mstar_step)

        ############ BC03 ############
        #### Age grid plots ####
        ax_gs_ages = fig_ages_bc03.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_ages.hist(ages_bc03[count], len(np.unique(ages_bc03[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_ages.set_xlim(8, 10)
        ax_gs_ages.set_yscale('log')

        ax_gs_ages.get_xaxis().set_ticklabels([])
        ax_gs_ages.get_yaxis().set_ticklabels([])
        
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

        fig_ages_bc03.savefig(figures_dir + 'agedist_grid_bc03.eps', dpi=300)

        #### Mass weighted age grid plots ####
        ax_gs_mass_wht_ages = fig_mass_wht_ages_bc03.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        N, bins, patches = ax_gs_mass_wht_ages.hist(mass_wht_ages_bc03[count], 10, histtype='bar', align='mid', alpha=0.5, linewidth=0)

        # this part of the code i.e. to color the histogram based on the x value came from a stackoverflow answer.
        # I modified it for my code.
        cm = plt.cm.get_cmap('bwr')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # scale values to interval [0,1]
        low_lim = 8.7
        up_lim = 9.9
        col = (bin_centers - low_lim)/(up_lim - low_lim)

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))

        ax_gs_mass_wht_ages.set_yscale('log')
        ax_gs_mass_wht_ages.set_xlim(7.5, 10)
        ax_gs_mass_wht_ages.set_ylim(1, 1e4)
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
            ax_gs_mass_wht_ages.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)

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

        fig_mass_wht_ages_bc03.savefig(figures_dir + 'mass_wht_agedist_grid_bc03.eps', dpi=300)

        #### Quenching timescale grid plots ####
        ax_gs_quench = fig_quench_bc03.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        N, bins, patches = ax_gs_quench.hist(quenching_times_bc03[count], 15, histtype='bar', align='mid', alpha=0.5, linewidth=0)

        # this part of the code i.e. to color the histogram based on the x value came from a stackoverflow answer.
        # I modified it for my code.
        cm = plt.cm.get_cmap('bwr_r')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # scale values to interval [0,1]
        low_lim = 7.0
        up_lim = 9.9
        col = (bin_centers - low_lim)/(up_lim - low_lim)

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))

        ax_gs_quench.set_yscale('log')
        ax_gs_quench.set_xlim(7, 10)
        ax_gs_quench.set_ylim(1, 1e4)
        ax_gs_quench.get_xaxis().set_ticklabels([])
        ax_gs_quench.get_yaxis().set_ticklabels([])
        if (row == 3) and (column == 0):
            ax_gs_quench.get_yaxis().set_ticklabels(['', '', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_quench.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_quench.get_yaxis().set_ticklabels(['', '', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_quench.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_quench.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)

        if (row == 4) and (column == 2):
            ax_gs_quench.set_xlabel(r'$\mathrm{log(Quenching\ Time\ [yr])}$', fontsize=13)
        if (row == 2) and (column == 1):
            ax_gs_quench.set_ylabel('N', fontsize=13)

        if (row == 4) and (column == 0):
            ax_gs_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 4) and (column == 1):
            ax_gs_quench.get_xaxis().set_ticklabels(['', '7.5', '8', '8.5', '9.0', '9.5', ''], fontsize=8, rotation=45)
        
        if (row == 4) and (column == 2):
            ax_gs_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 3) and (column == 3):
            ax_gs_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 1) and (column == 4):
            ax_gs_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)

        fig_quench_bc03.savefig(figures_dir + 'quenchdist_grid_bc03.eps', dpi=300)

        #### Metallicity grid plots ####
        ax_gs_metals = fig_metals_bc03.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_metals.hist(metals_bc03[count], len(np.unique(metals_bc03[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_metals.set_xlim(0, 0.05)
        ax_gs_metals.set_yscale('log')

        ax_gs_metals.get_xaxis().set_ticklabels([])
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

        fig_metals_bc03.savefig(figures_dir + 'metalsdist_grid_bc03.eps', dpi=300)

        #### Tau grid plots ####
        ax_gs_tau = fig_tau_bc03.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_tau.hist(logtau_bc03[count], len(np.unique(logtau_bc03[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_tau.set_xlim(-2, 2)
        ax_gs_tau.set_yscale('log')

        ax_gs_tau.get_xaxis().set_ticklabels([])
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

        fig_tau_bc03.savefig(figures_dir + 'logtaudist_grid_bc03.eps', dpi=300)

        #### AV grid plots ####
        ax_gs_av = fig_av_bc03.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_av.hist(av_bc03[count], len(np.unique(av_bc03[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_av.set_xlim(0, 2.1)
        ax_gs_av.set_yscale('log')

        ax_gs_av.get_xaxis().set_ticklabels([])
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

        fig_av_bc03.savefig(figures_dir + 'avdist_grid_bc03.eps', dpi=300)

        del ax_gs_ages, ax_gs_metals, ax_gs_tau, ax_gs_av, ax_gs_mass_wht_ages, ax_gs_quench

        ############ MILES ############
        #### Age grid plots ####
        ax_gs_ages = fig_ages_miles.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_ages.hist(ages_miles[count], len(np.unique(ages_miles[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_ages.set_xlim(8, 10)
        ax_gs_ages.set_yscale('log')

        ax_gs_ages.get_xaxis().set_ticklabels([])
        ax_gs_ages.get_yaxis().set_ticklabels([])
        
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

        fig_ages_miles.savefig(figures_dir + 'agedist_grid_miles.eps', dpi=300)

        #### Metallicity grid plots ####
        ax_gs_metals = fig_metals_miles.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_metals.hist(metals_miles[count], len(np.unique(metals_miles[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_metals.set_xlim(0, 0.05)
        ax_gs_metals.set_yscale('log')

        ax_gs_metals.get_xaxis().set_ticklabels([])
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

        fig_metals_miles.savefig(figures_dir + 'metalsdist_grid_miles.eps', dpi=300)

        del ax_gs_ages, ax_gs_metals

        ############ FSPS ############
        #### Age grid plots ####
        ax_gs_ages = fig_ages_fsps.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_ages.hist(ages_fsps[count], len(np.unique(ages_fsps[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_ages.set_xlim(8, 10)
        ax_gs_ages.set_yscale('log')

        ax_gs_ages.get_xaxis().set_ticklabels([])
        ax_gs_ages.get_yaxis().set_ticklabels([])
        
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

        fig_ages_fsps.savefig(figures_dir + 'agedist_grid_fsps.eps', dpi=300)

        #### Metallicity grid plots ####
        ax_gs_metals = fig_metals_fsps.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_metals.hist(metals_fsps[count], len(np.unique(metals_fsps[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_metals.set_xlim(0, 0.05)
        ax_gs_metals.set_yscale('log')

        ax_gs_metals.get_xaxis().set_ticklabels([])
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

        fig_metals_fsps.savefig(figures_dir + 'metalsdist_grid_fsps.eps', dpi=300)

        #### Tau grid plots ####
        ax_gs_tau = fig_tau_fsps.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_tau.hist(logtau_fsps[count], len(np.unique(logtau_fsps[count])), histtype='bar', align='mid', alpha=0.5, linewidth=0.3)

        ax_gs_tau.set_xlim(-2, 2)
        ax_gs_tau.set_yscale('log')

        ax_gs_tau.get_xaxis().set_ticklabels([])
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

        fig_tau_fsps.savefig(figures_dir + 'logtaudist_grid_fsps.eps', dpi=300)

        #### Mass weighted age grid plots ####
        ax_gs_mass_wht_ages = fig_mass_wht_ages_fsps.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        N, bins, patches = ax_gs_mass_wht_ages.hist(mass_wht_ages_fsps[count], 10, histtype='bar', align='mid', alpha=0.5, linewidth=0)

        # this part of the code i.e. to color the histogram based on the x value came from a stackoverflow answer.
        # I modified it for my code.
        cm = plt.cm.get_cmap('bwr')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # scale values to interval [0,1]
        low_lim = 8.7
        up_lim = 9.9
        col = (bin_centers - low_lim)/(up_lim - low_lim)

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))

        ax_gs_mass_wht_ages.set_yscale('log')
        ax_gs_mass_wht_ages.set_xlim(7.5, 10)
        ax_gs_mass_wht_ages.set_ylim(1, 1e4)
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
            ax_gs_mass_wht_ages.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)

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

        fig_mass_wht_ages_fsps.savefig(figures_dir + 'mass_wht_agedist_grid_fsps.eps', dpi=300)

        #### Quenching timescale grid plots ####
        ax_gs_quench = fig_quench_fsps.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        N, bins, patches = ax_gs_quench.hist(quenching_times_fsps[count], 15, histtype='bar', align='mid', alpha=0.5, linewidth=0)

        # this part of the code i.e. to color the histogram based on the x value came from a stackoverflow answer.
        # I modified it for my code.
        cm = plt.cm.get_cmap('bwr_r')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # scale values to interval [0,1]
        low_lim = 7.0
        up_lim = 9.9
        col = (bin_centers - low_lim)/(up_lim - low_lim)

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))

        ax_gs_quench.set_yscale('log')
        ax_gs_quench.set_xlim(7, 10)
        ax_gs_quench.set_ylim(1, 1e4)
        ax_gs_quench.get_xaxis().set_ticklabels([])
        ax_gs_quench.get_yaxis().set_ticklabels([])
        if (row == 3) and (column == 0):
            ax_gs_quench.get_yaxis().set_ticklabels(['', '', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_quench.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_quench.get_yaxis().set_ticklabels(['', '', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_quench.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_quench.get_yaxis().set_ticklabels(['', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], fontsize=10)

        if (row == 4) and (column == 2):
            ax_gs_quench.set_xlabel(r'$\mathrm{log(Quenching\ Time\ [yr])}$', fontsize=13)
        if (row == 2) and (column == 1):
            ax_gs_quench.set_ylabel('N', fontsize=13)

        if (row == 4) and (column == 0):
            ax_gs_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 4) and (column == 1):
            ax_gs_quench.get_xaxis().set_ticklabels(['', '7.5', '8', '8.5', '9.0', '9.5', ''], fontsize=8, rotation=45)
        
        if (row == 4) and (column == 2):
            ax_gs_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 3) and (column == 3):
            ax_gs_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)
        
        if (row == 1) and (column == 4):
            ax_gs_quench.get_xaxis().set_ticklabels(['7.0', '7.5', '8', '8.5', '9.0', '9.5', '10'], fontsize=8, rotation=45)

        fig_quench_fsps.savefig(figures_dir + 'quenchdist_grid_fsps.eps', dpi=300)

        del ax_gs_ages, ax_gs_metals, ax_gs_tau

        count += 1

    #plt.show()

    # total run time
    print "Total time taken --", time.time() - start, "seconds."