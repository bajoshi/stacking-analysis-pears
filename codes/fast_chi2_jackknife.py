from __future__ import division
import numpy as np
import numpy.ma as ma
from astropy.io import fits

import sys, os, time, glob, datetime
import logging
import collections

import matplotlib.pyplot as plt
from matplotlib import cm 
import matplotlib.gridspec as gridspec

def makefig_hist(qty):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(qty)
    ax.set_ylabel('N')

    return fig, ax

def makefig(xlab, ylab):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    return fig, ax

def plot_spectrum_data(lam, flux, flux_err):
    
    ax.errorbar(lam, flux, yerr=flux_err, fmt='o-', color='k', linewidth=2, ecolor='r', markeredgecolor='k', capsize=0, markersize=4)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)
    #ax.set_ylim(0,2)

def plot_spectrum_bc03(lam, flux):
    
    ax.plot(lam, flux, 'o-', color='r', linewidth=3)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

if __name__ == '__main__':

    # start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # File to save distribution of best params in
    f_ages = open('jackknife_ages.txt', 'wa')
    f_metals = open('jackknife_metals.txt', 'wa')
    f_logtau = open('jackknife_logtau.txt', 'wa')
    f_tauv = open('jackknife_tauv.txt', 'wa')    

    # Get comparison spectra
    #h = fits.open('all_spectra_dist.fits', memmap=False)   
    #nexten = 0 # this is the total number of distinguishable spectra
    #while 1:
    #    try:
    #        if h[nexten+1]:
    #            nexten += 1
    #    except IndexError:
    #        break

    h = fits.open('/Users/baj/Desktop/FIGS/new_codes/all_comp_spectra.fits', memmap=False)
    nexten = 136800 # this is the total number of spectra

    lam_step = 100
    lam_lowfit = 2500
    lam_highfit = 6500
    lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)
    arg_lamlow = np.argmin(abs(lam_grid_tofit - 3000))
    arg_lamhigh = np.argmin(abs(lam_grid_tofit - 6000))

    comp_spec = np.zeros([nexten, len(lam_grid_tofit)], dtype=np.float64)
    for i in range(nexten):
        comp_spec[i] = h[i+1].data

    comp_spec = comp_spec[:,arg_lamlow:arg_lamhigh]

    # Read stacks
    stacks = fits.open('/Users/baj/Desktop/FIGS/new_codes/coadded_coarsegrid_PEARSgrismspectra.fits')
    fig_savedir = '/Users/baj/Desktop/FIGS/new_codes/jackknife_figs/coarse/'
    #stacks = fits.open('/Users/baj/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra.fits')
    #fig_savedir = '/Users/baj/Desktop/FIGS/new_codes/jackknife_figs/'

    # Create grid for making grid plots
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.00)
    fig_ages = plt.figure()
    fig_metals = plt.figure()
    fig_tau = plt.figure()
    fig_tauv = plt.figure()

    color_step = 0.6
    mstar_step = 1.0

    # find number of extensions
    totalstacks = 0
    while 1:
        try:
            if stacks[totalstacks+2]:
                totalstacks += 1
        except IndexError:
            break

    # Loop over all stacks
    num_samp_to_draw = 1e4
    print "Running over", int(num_samp_to_draw), "random jackknifed samples."
    for stackcount in range(0,totalstacks,1):
        # start time for each stack
        chi2start = time.time()
        
        flam = stacks[stackcount+2].data[0]
        ferr = stacks[stackcount+2].data[1]
        ferr = ferr + 0.05 * flam # putting in a 5% additional error bar
        ongrid = stacks[stackcount+2].header["ONGRID"]
        numspec = int(stacks[stackcount+2].header["NUMSPEC"])
        print "ONGRID", ongrid

        if numspec < 5:
            print "Too few spectra in stack. Continuing to the next grid cell..."
            continue

        # All the masks in this block only generate the mask which is applied during the loop that loops over jackknife runs.
        # mask the array where the flam value has been set to 0 by the stacking code
        if np.any(flam == 0.0):
            indices_to_be_masked = np.where(flam == 0.0)[0]
            flam_mask = np.zeros(len(flam)) # by default create a masked array where all values in the original array are assumed to be valid
            flam_mask[indices_to_be_masked] = 1 # now set the indices to be masked as True
            flam = ma.masked_array(flam, mask = flam_mask)
            ferr = ma.masked_array(ferr, mask = flam_mask)
        else:
            flam_mask = np.zeros(len(flam))
    
        # Also check if ferr might be 0 at a different index from flam... so doing this differently from the check for flam
        # mask the array where the ferr value has been set to 0 by the stacking code
        if np.any(ferr == 0.0):
            indices_to_be_masked = np.where(ferr == 0.0)[0]
            ferr_mask = np.zeros(len(ferr)) # by default create a masked array where all values in the original array are assumed to be valid
            ferr_mask[indices_to_be_masked] = 1 # now set the indices to be masked as True
            flam = ma.masked_array(flam, mask = ferr_mask)
            ferr = ma.masked_array(ferr, mask = ferr_mask)            
        else:
            ferr_mask = np.zeros(len(ferr))

        # for the blue spectra only
        # mask the region with h-beta and [OIII] emission lines
        # I'm masking the region between 4800A to 5100A because my stacks are sampled at 100A in delta_lambda
        # I will be excluding the points at 4800A, 4900A, 5000A, and 5100A
        urcol = float(ongrid.split(',')[0])
        stmass = float(ongrid.split(',')[1])
        if urcol <= 1.2:
            arg4800 = np.argmin(abs(lam_grid_tofit - 4800))
            arg4900 = np.argmin(abs(lam_grid_tofit - 4900))
            arg5000 = np.argmin(abs(lam_grid_tofit - 5000))
            arg5100 = np.argmin(abs(lam_grid_tofit - 5100)) 
            lam_mask = np.zeros(len(flam))
            lam_mask[arg4800:arg5100 + 1] = 1
            flam = ma.masked_array(flam, mask = lam_mask)
            ferr = ma.masked_array(ferr, mask = lam_mask)
        else:
            lam_mask = np.zeros(len(flam))

        # Chop off the ends of the stacked spectrum
        orig_lam_grid = np.arange(2700, 6000, lam_step)
        # redefine lam_lowfit and lam_highfit
        lam_lowfit = 3000
        lam_highfit = 6000
        lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)
        arg_lamlow = np.argmin(abs(orig_lam_grid - lam_lowfit))
        arg_lamhigh = np.argmin(abs(orig_lam_grid - lam_highfit+100))
        flam = flam[arg_lamlow:arg_lamhigh+1]
        ferr = ferr[arg_lamlow:arg_lamhigh+1]

        # Get random samples by jackknifing
        resampled_spec = ma.empty((len(flam), num_samp_to_draw))
        for i in range(len(flam)):
            if flam[i] is not ma.masked:
                resampled_spec[i] = np.random.normal(flam[i], ferr[i], num_samp_to_draw)
            else:
                resampled_spec[i] = ma.masked
        resampled_spec = resampled_spec.T

        # Actual chi2 fitting
        ages = []
        tau = []
        tauv = []
        metals = []
        totalchi2 = []    
        bestchi2index = []
        bestalpha = []
        for i in range(int(num_samp_to_draw)): # loop over jackknife runs
            #if i%1000 == 0: print i
            flam = resampled_spec[i]
            #ferr = stack_ferr
    
            """
            # Chop off the ends of the stacked spectrum
            orig_lam_grid = np.arange(2700, 6000, lam_step)
            # redefine lam_lowfit and lam_highfit
            lam_lowfit = 3000
            lam_highfit = 6000
            lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)
            arg_lamlow = np.argmin(abs(orig_lam_grid - lam_lowfit))
            arg_lamhigh = np.argmin(abs(orig_lam_grid - lam_highfit+100))
            flam = flam[arg_lamlow:arg_lamhigh+1]
            ferr = ferr[arg_lamlow:arg_lamhigh+1]
            """

            currentspec = comp_spec
        
            chi2 = np.zeros(nexten, dtype=np.float64)
            alpha = np.sum(flam*currentspec/(ferr**2), axis=1)/np.sum(currentspec**2/ferr**2, axis=1)
            chi2 = np.sum(((flam - (alpha * currentspec.T).T) / ferr)**2, axis=1)
        
            # This is to get only physical ages
            sortargs = np.argsort(chi2)
            for k in range(len(chi2)):
                best_age = float(h[sortargs[k]+1].header['LOG_AGE'])
                if (best_age < 9 + np.log10(8)) & (best_age > 9 + np.log10(0.1)):
                    tau.append(h[sortargs[k]+1].header['TAU_GYR'])
                    tauv.append(h[sortargs[k]+1].header['TAUV'])
                    ages.append(best_age)
                    totalchi2.append(chi2[sortargs[k]])
                    bestchi2index.append(sortargs[k])
                    bestalpha.append(alpha[sortargs[k]])
                    metals.append(h[sortargs[k]+1].header['METAL'])
                    #print np.min(chi2), best_metal, best_age, curr_tau, curr_tauv
                    break
        
        # total computational time
        print "Total computational time taken to get chi2 values --", time.time() - chi2start, "seconds."

        #print np.min(totalchi2)
        #print bestchi2index[np.argmin(totalchi2)]
        #print bestalpha[np.argmin(totalchi2)]

        ages = np.asarray(ages, dtype=np.float64)
        metals = np.asarray(metals, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        logtau = np.log10(tau)
        tauv = np.asarray(tauv, dtype=np.float64)
        
        print np.mean(ages), np.median(ages), np.std(ages)
        print np.mean(metals), np.median(metals), np.std(metals)
        print np.mean(tau), np.median(tau), np.std(tau)
        print np.mean(tauv), np.median(tauv), np.std(tauv)

        #print len(zip(ages, tau)), len(set(zip(ages, tau)))
        print len(np.unique(ages)), len(np.unique(metals)), len(np.unique(tau)), len(np.unique(tauv))

        # Save the data from the runs
        f_ages.write(ongrid + ' ')
        for k in range(len(ages)):
            f_ages.write(str(ages[k]) + ' ')
        f_ages.write('\n')

        f_metals.write(ongrid + ' ')
        for k in range(len(metals)):
            f_metals.write(str(metals[k]) + ' ')
        f_metals.write('\n')

        f_logtau.write(ongrid + ' ')
        for k in range(len(logtau)):
            f_logtau.write(str(logtau[k]) + ' ')
        f_logtau.write('\n')

        f_tauv.write(ongrid + ' ')
        for k in range(len(tauv)):
            f_tauv.write(str(tauv[k]) + ' ')
        f_tauv.write('\n')

        # for the grid plots
        row = 4 - int(float(ongrid.split(',')[0])/color_step)
        column = int((float(ongrid.split(',')[1]) - 7.0)/mstar_step)

        #### Age grid plots ####
        ax_gs_ages = fig_ages.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        #iqr = 1.349 * np.std(np.unique(ages), dtype=np.float64)
        #binsize = 2*iqr*np.power(len(np.unique(ages)),-1/3) # Freedman-Diaconis Rule
        #totalbins = np.floor((max(ages) - min(ages))/binsize)
        ax_gs_ages.hist(ages, 8, histtype='bar', align='mid', alpha=0.5, weights=np.ones_like(ages)/len(ages), facecolor='b')
        ax_gs_ages.set_xlim(8, 10)
        ax_gs_ages.set_ylim(0, 1)        
        ax_gs_ages.get_xaxis().set_ticklabels([])
        ax_gs_ages.get_yaxis().set_ticklabels([])

        if (row == 3) and (column == 0):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_ages.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_ages.get_yaxis().set_ticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_ages.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_ages.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

        if (row == 4) and (column == 2):
            ax_gs_ages.set_xlabel(r'$\mathrm{log(Age\ [yr])}$', fontsize=13)
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

        fig_ages.savefig('/Users/baj/Desktop/FIGS/new_codes/agedist_jackknife_grid' + '_run2.png', dpi=300)

        #### Metallicity grid plots ####
        ax_gs_metals = fig_metals.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        #iqr = 1.349 * np.std(np.unique(metals), dtype=np.float64)
        #binsize = 2*iqr*np.power(len(np.unique(metals)),-1/3) # Freedman-Diaconis Rule
        #totalbins = np.floor((max(metals) - min(metals))/binsize)
        ax_gs_metals.hist(metals, 8, histtype='bar', align='mid', alpha=0.5, weights=np.ones_like(metals)/len(metals), facecolor='b')
        ax_gs_metals.set_xlim(0, 0.05)
        ax_gs_metals.set_ylim(0, 1)        
        ax_gs_metals.get_xaxis().set_ticklabels([])
        ax_gs_metals.get_yaxis().set_ticklabels([])

        if (row == 3) and (column == 0):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_metals.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_metals.get_yaxis().set_ticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_metals.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_metals.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

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

        fig_metals.savefig('/Users/baj/Desktop/FIGS/new_codes/metalsdist_jackknife_grid' + '_run2.png', dpi=300)

        #### Tau grid plots ####
        ax_gs_tau = fig_tau.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        #iqr = 1.349 * np.std(np.unique(logtau), dtype=np.float64)
        #binsize = 2*iqr*np.power(len(np.unique(logtau)),-1/3) # Freedman-Diaconis Rule
        #totalbins = np.floor((max(logtau) - min(logtau))/binsize)
        ax_gs_tau.hist(logtau, 8, histtype='bar', align='mid', alpha=0.5, weights=np.ones_like(logtau)/len(logtau), facecolor='b')
        ax_gs_tau.set_xlim(-2, 2)
        ax_gs_tau.set_ylim(0, 1)        
        ax_gs_tau.get_xaxis().set_ticklabels([])
        ax_gs_tau.get_yaxis().set_ticklabels([])

        if (row == 3) and (column == 0):
            ax_gs_tau.get_yaxis().set_ticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_tau.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_tau.get_yaxis().set_ticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_tau.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_tau.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

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

        fig_tau.savefig('/Users/baj/Desktop/FIGS/new_codes/logtaudist_jackknife_grid' + '_run2.png', dpi=300)

        #### TauV grid plots ####
        ax_gs_tauv = fig_tauv.add_subplot(gs[row*3:row*3+3, column*3:column*3+3])
        ax_gs_tauv.hist(tauv, 8, histtype='bar', align='mid', alpha=0.5, weights=np.ones_like(logtau)/len(logtau), facecolor='b')
        ax_gs_tauv.set_xlim(0, 2)
        ax_gs_tauv.set_ylim(0, 1)        
        ax_gs_tauv.get_xaxis().set_ticklabels([])
        ax_gs_tauv.get_yaxis().set_ticklabels([])

        if (row == 3) and (column == 0):
            ax_gs_tauv.get_yaxis().set_ticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 4) and (column == 0):
            ax_gs_tauv.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 1) and (column == 1):
            ax_gs_tauv.get_yaxis().set_ticklabels(['', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 2) and (column == 1):
            ax_gs_tauv.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        if (row == 0) and (column == 3):
            ax_gs_tauv.get_yaxis().set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

        if (row == 4) and (column == 2):
            ax_gs_tauv.set_xlabel(r'$\tau_V$', fontsize=13)
        if (row == 2) and (column == 1):
            ax_gs_tauv.set_ylabel('N', fontsize=13)

        if (row == 4) and (column == 0):
            ax_gs_tauv.get_xaxis().set_ticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)
        
        if (row == 4) and (column == 1):
            ax_gs_tauv.get_xaxis().set_ticklabels(['', '0.5', '1.0', '1.5', ''], fontsize=10, rotation=45)
        
        if (row == 4) and (column == 2):
            ax_gs_tauv.get_xaxis().set_ticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)
        
        if (row == 3) and (column == 3):
            ax_gs_tauv.get_xaxis().set_ticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)
        
        if (row == 1) and (column == 4):
            ax_gs_tauv.get_xaxis().set_ticklabels(['0.0', '0.5', '1.0', '1.5', '2.0'], fontsize=10, rotation=45)

        fig_tauv.savefig('/Users/baj/Desktop/FIGS/new_codes/tauvdist_jackknife_grid' + '_run2.png', dpi=300)

        # for the individual plots
        # histograms of the best fit params
        fig, ax = makefig_hist(r'$\mathrm{log(Age\ [yr])}$')
        ax.hist(ages, 10, histtype='bar', align='mid', alpha=0.5, weights=np.ones_like(ages)/len(ages), facecolor='b')
        ax.set_xlim(8, 10)
        ax.set_ylim(0, 1)
        fig.savefig(fig_savedir + 'agedist_jackknife_' + ongrid.replace('.','p').replace(',','_') + '_run2.png', dpi=300)
   
        fig, ax = makefig_hist('Metals [Z]')
        ax.hist(metals, 10, histtype='bar', align='mid', alpha=0.5, weights=np.ones_like(metals)/len(metals), facecolor='b')
        ax.set_xlim(0, 0.05)
        ax.set_ylim(0, 1)
        fig.savefig(fig_savedir + 'metaldist_jackknife_' + ongrid.replace('.','p').replace(',','_') + '_run2.png', dpi=300)
        
        fig, ax = makefig_hist(r'$\mathrm{log}(\tau)$')
        ax.hist(logtau, 10, histtype='bar', align='mid', alpha=0.5, weights=np.ones_like(logtau)/len(logtau), facecolor='b')
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 1)
        fig.savefig(fig_savedir + 'taudist_jackknife_' + ongrid.replace('.','p').replace(',','_') + '_run2.png', dpi=300)
        
        fig, ax = makefig_hist(r'$\tau_V$')
        ax.hist(tauv, 10, histtype='bar', align='mid', alpha=0.5, weights=np.ones_like(tauv)/len(tauv), facecolor='b')
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        fig.savefig(fig_savedir + 'tauvdist_jackknife_' + ongrid.replace('.','p').replace(',','_') + '_run2.png', dpi=300)

        ########### Plots between parameters ##########
        fig, ax = makefig(r'$\tau_V$', r'$\mathrm{log(Age\ [yr])}$')
        ax.plot(tauv, ages, 'o', color='k', markeredgecolor='k', markersize=2)
        ax.set_xlim(0, 2)
        ax.set_ylim(8, 10)
        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')
        fig.savefig(fig_savedir + 'tauv_age_jackknife_' + ongrid.replace('.','p').replace(',','_') + '_run2.png', dpi=300)

        fig, ax = makefig(r'$\mathrm{log}(\tau)$', r'$\mathrm{log(Age\ [yr])}$')
        ax.plot(logtau, ages, 'o', color='k', markeredgecolor='k', markersize=2)
        ax.set_xlim(-2, 2)
        ax.set_ylim(8, 10)
        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')
        fig.savefig(fig_savedir + 'tau_age_jackknife_' + ongrid.replace('.','p').replace(',','_') + '_run2.png', dpi=300)

        fig, ax = makefig('Metals [Z]', r'$\mathrm{log(Age\ [yr])}$')
        ax.plot(metals, ages, 'o', color='k', markeredgecolor='k', markersize=2)
        ax.set_xlim(0, 0.05)
        ax.set_ylim(8, 10)
        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')
        fig.savefig(fig_savedir + 'metals_age_jackknife_' + ongrid.replace('.','p').replace(',','_') + '_run2.png', dpi=300)

        fig, ax = makefig(r'$\tau_V$', r'$\mathrm{log}(\tau)$')
        ax.plot(tauv, logtau, 'o', color='k', markeredgecolor='k', markersize=2)
        ax.set_xlim(0, 2)
        ax.set_ylim(-2, 2)
        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')
        fig.savefig(fig_savedir + 'tauv_tau_jackknife_' + ongrid.replace('.','p').replace(',','_') + '_run2.png', dpi=300)

        fig, ax = makefig(r'$\tau_V$', 'Metals [Z]')
        ax.plot(tauv, metals, 'o', color='k', markeredgecolor='k', markersize=2)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 0.05)
        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')
        fig.savefig(fig_savedir + 'tauv_metals_jackknife_' + ongrid.replace('.','p').replace(',','_') + '_run2.png', dpi=300)

        fig, ax = makefig(r'$\mathrm{log}(\tau)$', 'Metals [Z]')
        ax.plot(logtau, metals, 'o', color='k', markeredgecolor='k', markersize=2)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 0.05)
        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')
        fig.savefig(fig_savedir + 'tau_metals_jackknife_' + ongrid.replace('.','p').replace(',','_') + '_run2.png', dpi=300)

    f_ages.close()
    f_metals.close()
    f_logtau.close()
    f_tauv.close()

    #plt.show()
         
    # total run time
    print "Total time taken --", time.time() - start, "seconds."

