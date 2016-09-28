from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits

import sys
import os
import time
import glob
import datetime
import logging
import collections

import matplotlib.pyplot as plt
from matplotlib import cm 
import matplotlib.gridspec as gridspec

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

    return None

def plot_spectrum_model(lam, flux):
    
    ax.plot(lam, flux, 'o-', color='r', linewidth=3)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

    return None

def get_total_extensions(fitsfile):
    """
    This function will return the number of extensions in a fits file.
    It does not count the 0th extension.

    It takes the opened fits header data unit as its only argument.
    """

    nexten = 0 # this is the total number of extensions
    while 1:
        try:
            if fitsfile[nexten+1]:
                nexten += 1
        except IndexError:
            break

    return nexten

def fit_chi2(flam, ferr, comp_spec, nexten, resampled_spec, num_samp_to_draw, library, spec_hdu):
    """
    This is the function that does the actual chi2 fitting.
    """

    # start time for each stack
    chi2start = time.time()

    # Actual chi2 fitting
    ages = []
    tau = []
    tauv = []
    metals = []
    best_exten = []
    #totalchi2 = []    
    #bestchi2index = []
    #bestalpha = []
    for i in range(int(num_samp_to_draw)): # loop over jackknife runs
        #if i%1000 == 0: print i
        flam = resampled_spec[i]

        currentspec = comp_spec
    
        chi2 = np.zeros(nexten, dtype=np.float64)
        alpha = np.sum(flam * currentspec / (ferr**2), axis=1) / np.sum(currentspec**2 / ferr**2, axis=1)
        chi2 = np.sum(((flam - (alpha * currentspec.T).T) / ferr)**2, axis=1)
    
        if library == 'bc03':
            bc03_spec = spec_hdu
            # This is to get only physical ages
            sortargs = np.argsort(chi2)
            for k in range(len(chi2)):
                best_age = float(bc03_spec[sortargs[k] + 1].header['LOG_AGE'])
                if (best_age < 9 + np.log10(7.5)) & (best_age > 9 + np.log10(0.1)):
                    tau.append(bc03_spec[sortargs[k] + 1].header['TAU_GYR'])
                    tauv.append(bc03_spec[sortargs[k] + 1].header['TAUV'])
                    ages.append(best_age)
                    #totalchi2.append(chi2[sortargs[k]])
                    #bestchi2index.append(sortargs[k])
                    #bestalpha.append(alpha[sortargs[k]])
                    metals.append(bc03_spec[sortargs[k] + 1].header['METAL'])
                    best_exten.append(sortargs[k] + 1)
                    break

        if library == 'miles':
            miles_spec = spec_hdu
            # This is to get only physical ages
            sortargs = np.argsort(chi2)
            for k in range(len(chi2)):
                best_age = float(miles_spec[sortargs[k] + 1].header['LOG_AGE'])
                if (best_age < 9 + np.log10(7.5)) & (best_age > 9 + np.log10(0.1)):
                    ages.append(best_age)
                    metals.append(miles_spec[sortargs[k] + 1].header['METAL'])
                    best_exten.append(sortargs[k] + 1)
                    break

        if library == 'fsps':
            fsps_spec = spec_hdu
            # This is to get only physical ages
            sortargs = np.argsort(chi2)
            for k in range(len(chi2)):
                best_age = float(fsps_spec[sortargs[k] + 1].header['LOG_AGE'])
                if (best_age < 9 + np.log10(7.5)) & (best_age > 9 + np.log10(0.1)):
                    tau.append(fsps_spec[sortargs[k] + 1].header['TAU_GYR'])
                    ages.append(best_age)
                    metals.append(fsps_spec[sortargs[k] + 1].header['METAL'])
                    best_exten.append(sortargs[k] + 1)
                    break

    # total computational time
    print "\n"
    print "--------- {0} ---------".format(library)
    print "Total computational time taken to get chi2 values --", time.time() - chi2start, "seconds."

    if library == 'bc03':
        ages = np.asarray(ages, dtype=np.float64)
        metals = np.asarray(metals, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        logtau = np.log10(tau)
        tauv = np.asarray(tauv, dtype=np.float64)
        best_exten = np.asarray(best_exten, dtype=np.float64)
        
        print "Ages: Median +- std dev = ", np.median(ages), "+-", np.std(ages)
        print "Metals: Median +- std dev = ", np.median(metals), "+-", np.std(metals)
        print "Tau: Median +- std dev = ", np.median(tau), "+-", np.std(tau)
        print "Tau_v: Median +- std dev = ", np.median(tauv), "+-", np.std(tauv)
    
        print "Unique elements in jackknifed runs for BC03 --"
        print "Ages - ", len(np.unique(ages))
        print "Metals - ", len(np.unique(metals))
        print "Tau - ", len(np.unique(tau))
        print "Tau_v - ", len(np.unique(tauv))

        return ages, metals, tau, tauv, best_exten

    elif library == 'miles':
        ages = np.asarray(ages, dtype=np.float64)
        metals = np.asarray(metals, dtype=np.float64)
        best_exten = np.asarray(best_exten, dtype=np.float64)
        
        print "Ages: Median +- std dev = ", np.median(ages), "+-", np.std(ages)
        print "Metals: Median +- std dev = ", np.median(metals), "+-", np.std(metals)
    
        print "Unique elements in jackknifed runs for MILES --"
        print "Ages - ", len(np.unique(ages))
        print "Metals - ", len(np.unique(metals))

        return ages, metals, best_exten

    elif library == 'fsps':
        ages = np.asarray(ages, dtype=np.float64)
        metals = np.asarray(metals, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        logtau = np.log10(tau)
        best_exten = np.asarray(best_exten, dtype=np.float64)
        
        print "Ages: Median +- std dev = ", np.median(ages), "+-", np.std(ages)
        print "Metals: Median +- std dev = ", np.median(metals), "+-", np.std(metals)
        print "Tau: Median +- std dev = ", np.median(tau), "+-", np.std(tau)
    
        print "Unique elements in jackknifed runs for FSPS --"
        print "Ages - ", len(np.unique(ages))
        print "Metals - ", len(np.unique(metals))
        print "Tau - ", len(np.unique(tau))

        return ages, metals, tau, best_exten

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Files to save distribution of best params in
    f_ages_bc03 = open(stacking_analysis_dir + 'jackknife_ages_bc03.txt', 'wa')
    f_metals_bc03 = open(stacking_analysis_dir + 'jackknife_metals_bc03.txt', 'wa')
    f_logtau_bc03 = open(stacking_analysis_dir + 'jackknife_logtau_bc03.txt', 'wa')
    f_tauv_bc03 = open(stacking_analysis_dir + 'jackknife_tauv_bc03.txt', 'wa')
    f_exten_bc03 = open(stacking_analysis_dir + 'jackknife_exten_bc03.txt', 'wa')      

    f_ages_miles = open(stacking_analysis_dir + 'jackknife_ages_miles.txt', 'wa')
    f_metals_miles = open(stacking_analysis_dir + 'jackknife_metals_miles.txt', 'wa')
    f_exten_miles = open(stacking_analysis_dir + 'jackknife_exten_miles.txt', 'wa')

    f_ages_fsps = open(stacking_analysis_dir + 'jackknife_ages_fsps.txt', 'wa')
    f_metals_fsps = open(stacking_analysis_dir + 'jackknife_metals_fsps.txt', 'wa')
    f_logtau_fsps = open(stacking_analysis_dir + 'jackknife_logtau_fsps.txt', 'wa')
    f_exten_fsps = open(stacking_analysis_dir + 'jackknife_exten_fsps.txt', 'wa')

    # Open fits files with comparison spectra
    bc03_spec = fits.open(home + '/Desktop/FIGS/new_codes/all_comp_spectra_bc03.fits', memmap=False)
    miles_spec = fits.open(home + '/Desktop/FIGS/new_codes/all_comp_spectra_miles.fits', memmap=False)
    fsps_spec = fits.open(home + '/Desktop/FIGS/new_codes/all_comp_spectra_fsps.fits', memmap=False)

    # Find number of extensions in each
    bc03_extens = get_total_extensions(bc03_spec)
    miles_extens = get_total_extensions(miles_spec)
    fsps_extens = get_total_extensions(fsps_spec)

    # set up lambda grid
    lam_step = 100
    lam_lowfit = 3600
    lam_highfit = 6000
    lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)
    arg_lamlow = np.argmin(abs(lam_grid_tofit - 3600))
    arg_lamhigh = np.argmin(abs(lam_grid_tofit - 6000))

    # set up comparison spectra arrays for faster array computations
    comp_spec_bc03 = np.zeros([bc03_extens, len(lam_grid_tofit)], dtype=np.float64)
    for i in range(bc03_extens):
        comp_spec_bc03[i] = bc03_spec[i+1].data
    comp_spec_bc03 = comp_spec_bc03[:,arg_lamlow:arg_lamhigh+1]

    comp_spec_miles = np.zeros([miles_extens, len(lam_grid_tofit)], dtype=np.float64)
    for i in range(miles_extens):
        comp_spec_miles[i] = miles_spec[i+1].data
    comp_spec_miles = comp_spec_miles[:,arg_lamlow:arg_lamhigh+1]

    comp_spec_fsps = np.zeros([fsps_extens, len(lam_grid_tofit)], dtype=np.float64)
    for i in range(fsps_extens):
        comp_spec_fsps[i] = fsps_spec[i+1].data
    comp_spec_fsps = comp_spec_fsps[:,arg_lamlow:arg_lamhigh+1]

    # Read stacks
    stacks = fits.open(home + '/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra_coarsegrid.fits')
    fig_savedir = home + '/Desktop/FIGS/new_codes/jackknife_figs/coarse/'

    totalstacks = get_total_extensions(stacks)

    # Create grid for making grid plots
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.00)
    fig_ages = plt.figure()
    fig_metals = plt.figure()
    fig_tau = plt.figure()
    fig_tauv = plt.figure()

    color_step = 0.6
    mstar_step = 1.0

    # Loop over all stacks
    num_samp_to_draw = 5e4
    print "Running over", int(num_samp_to_draw), "random jackknifed samples."
    for stackcount in range(0, totalstacks-1, 1): # it is totalstacks-1 because the first extension is the lambda array
        
        flam = stacks[stackcount + 2].data[0]
        ferr = stacks[stackcount + 2].data[1]
        ferr = ferr + 0.05 * flam  # putting in a 5% additional error bar
        ongrid = stacks[stackcount + 2].header["ONGRID"]
        numspec = int(stacks[stackcount + 2].header["NUMSPEC"])
        print "Time right now -- ", dt.now()
        print "-------------------------------------------------------------------------------------------------------------"
        print "ONGRID", ongrid

        if numspec < 5:
            print "Too few spectra in stack. Continuing to the next grid cell..."
            continue

        ### All the masks in this block only generate the mask which is applied during the loop that loops over jackknife runs. ###

        # mask the array where the flam value has been set to 0 by the stacking code
        if np.any(flam == 0.0):
            indices_to_be_masked = np.where(flam == 0.0)[0]
            flam_mask = np.zeros(len(flam)) # by default create a masked array where all values in the original array are assumed to be valid
            flam_mask[indices_to_be_masked] = 1 # now set the indices to be masked as True
            flam = ma.masked_array(flam, mask = flam_mask)
            ferr = ma.masked_array(ferr, mask = flam_mask)
    
        # Also check if ferr might be 0 at a different index from flam... so doing this differently from the check for flam
        # mask the array where the ferr value has been set to 0 by the stacking code
        if np.any(ferr == 0.0):
            indices_to_be_masked = np.where(ferr == 0.0)[0]
            ferr_mask = np.zeros(len(ferr)) # by default create a masked array where all values in the original array are assumed to be valid
            ferr_mask[indices_to_be_masked] = 1 # now set the indices to be masked as True
            flam = ma.masked_array(flam, mask = ferr_mask)
            ferr = ma.masked_array(ferr, mask = ferr_mask)            

        # for the blue spectra only
        # mask the region with h-beta and [OIII] emission lines
        # I'm masking the region between 4800A to 5100A because my stacks are sampled at 100A in delta_lambda
        # I will be excluding the points at 4800A, 4900A, 5000A, and 5100A
        urcol = float(ongrid.split(',')[0])
        stmass = float(ongrid.split(',')[1])
        if urcol <= 1.2:
            arg4800 = np.argmin(abs(lam_grid_tofit - 4800))
            arg5100 = np.argmin(abs(lam_grid_tofit - 5100)) 
            lam_mask = np.zeros(len(flam))
            lam_mask[arg4800:arg5100 + 1] = 1
            flam = ma.masked_array(flam, mask = lam_mask)
            ferr = ma.masked_array(ferr, mask = lam_mask)
        else:
            lam_mask = np.zeros(len(flam))

        # Chop off the ends of the stacked spectrum
        orig_lam_grid = np.arange(2700, 6000, lam_step)  
        # this is the lam grid used for the stacks. 
        # it has to be defined again because it (and therefore its indices) are different from the lam grid used to resample the models.
        # redefine lam_lowfit and lam_highfit
        lam_lowfit = 3600
        lam_highfit = 6000
        lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)
        arg_lamlow = np.argmin(abs(orig_lam_grid - lam_lowfit))
        arg_lamhigh = np.argmin(abs(orig_lam_grid - lam_highfit))
        flam = flam[arg_lamlow:arg_lamhigh + 1]
        ferr = ferr[arg_lamlow:arg_lamhigh + 1]

        # Get random samples by jackknifing
        resampled_spec = ma.empty((len(flam), num_samp_to_draw))
        for i in range(len(flam)):
            if flam[i] is not ma.masked:
                resampled_spec[i] = np.random.normal(flam[i], ferr[i], num_samp_to_draw)
            else:
                resampled_spec[i] = ma.masked
        resampled_spec = resampled_spec.T

        ages_bc03, metals_bc03, tau_bc03, tauv_bc03, exten_bc03 = fit_chi2(flam, ferr, comp_spec_bc03, bc03_extens, resampled_spec, num_samp_to_draw, 'bc03', bc03_spec)
        ages_miles, metals_miles, exten_miles = fit_chi2(flam, ferr, comp_spec_miles, miles_extens, resampled_spec, num_samp_to_draw, 'miles', miles_spec)
        ages_fsps, metals_fsps, tau_fsps, exten_fsps = fit_chi2(flam, ferr, comp_spec_fsps, fsps_extens, resampled_spec, num_samp_to_draw, 'fsps', fsps_spec)

        logtau_bc03 = np.log10(tau_bc03)
        logtau_fsps = np.log10(tau_fsps)

        # Save the data from the runs
        ### BC03 ### 
        f_ages_bc03.write(ongrid + ' ')
        for k in range(len(ages_bc03)):
            f_ages_bc03.write(str(ages_bc03[k]) + ' ')
        f_ages_bc03.write('\n')

        f_metals_bc03.write(ongrid + ' ')
        for k in range(len(metals_bc03)):
            f_metals_bc03.write(str(metals_bc03[k]) + ' ')
        f_metals_bc03.write('\n')

        f_logtau_bc03.write(ongrid + ' ')
        for k in range(len(logtau_bc03)):
            f_logtau_bc03.write(str(logtau_bc03[k]) + ' ')
        f_logtau_bc03.write('\n')

        f_tauv_bc03.write(ongrid + ' ')
        for k in range(len(tauv_bc03)):
            f_tauv_bc03.write(str(tauv_bc03[k]) + ' ')
        f_tauv_bc03.write('\n')

        f_exten_bc03.write(ongrid + ' ')
        for k in range(len(exten_bc03)):
            f_exten_bc03.write(str(exten_bc03[k]) + ' ')
        f_exten_bc03.write('\n')

        ### MILES ###
        f_ages_miles.write(ongrid + ' ')
        for k in range(len(ages_miles)):
            f_ages_miles.write(str(ages_miles[k]) + ' ')
        f_ages_miles.write('\n')

        f_metals_miles.write(ongrid + ' ')
        for k in range(len(metals_miles)):
            f_metals_miles.write(str(metals_miles[k]) + ' ')
        f_metals_miles.write('\n')

        f_exten_miles.write(ongrid + ' ')
        for k in range(len(exten_miles)):
            f_exten_miles.write(str(exten_miles[k]) + ' ')
        f_exten_miles.write('\n')

        ### FSPS ###
        f_ages_fsps.write(ongrid + ' ')
        for k in range(len(ages_fsps)):
            f_ages_fsps.write(str(ages_fsps[k]) + ' ')
        f_ages_fsps.write('\n')

        f_metals_fsps.write(ongrid + ' ')
        for k in range(len(metals_fsps)):
            f_metals_fsps.write(str(metals_fsps[k]) + ' ')
        f_metals_fsps.write('\n')

        f_logtau_fsps.write(ongrid + ' ')
        for k in range(len(logtau_fsps)):
            f_logtau_fsps.write(str(logtau_fsps[k]) + ' ')
        f_logtau_fsps.write('\n')

        f_exten_fsps.write(ongrid + ' ')
        for k in range(len(exten_fsps)):
            f_exten_fsps.write(str(exten_fsps[k]) + ' ')
        f_exten_fsps.write('\n')

        ########### Plots between parameters ##########
        """
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
        """
         
    # Close all files to write them -- 
    f_ages_bc03.close()
    f_metals_bc03.close()
    f_logtau_bc03.close()
    f_tauv_bc03.close()
    f_exten_bc03.close()

    f_ages_miles.close()
    f_metals_miles.close()
    f_exten_miles.close()

    f_ages_fsps.close()
    f_metals_fsps.close()
    f_logtau_fsps.close()
    f_exten_fsps.close()

    # total run time
    print "Total time taken --", time.time() - start, "seconds."

