from __future__ import division
import numpy as np
import numpy.ma as ma
from astropy.io import fits

import sys, os, time, glob
import logging
import collections

import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D

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

if __name__ == '__main__':

    # start time
    start = time.time()

    h = fits.open('all_spectra_dist.fits', memmap=False)
    
    nexten = 1249 # this is the total number of distinguishable spectra

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

    curr_boots = fits.open('/Users/baj/Desktop/FIGS/new_codes/spectra_newbootstrap_2p1_10p5.fits')
    ongrid = curr_boots[2].header["ONGRID"]
    ages = []
    tau = []
    tauv = []
    metals = []
    totalchi2 = []
    for i in range(1000): # loop over bootstrap runs
        stack_flam = curr_boots[i+2].data[0]
        stack_ferr = curr_boots[i+2].data[1] 
        flam = stack_flam
        ferr = stack_ferr
        ferr = ferr + 0.05 * flam # putting in a 5% additional error bar

        # mask the array where the flam value has been set to 0 by the stacking code
        if np.any(flam == 0.0):
            indices_to_be_masked = np.where(flam == 0.0)[0]
            mask = np.zeros(len(flam)) # by default create a masked array where all values in the original array are assumed to be valid
            mask[indices_to_be_masked] = 1 # now set the indices to be masked as True
            flam_masked = ma.masked_array(flam, mask = mask)
            ferr_masked = ma.masked_array(ferr, mask = mask)
            flam = flam_masked
            ferr = ferr_masked

        # Also check if ferr might be 0 at a different index from flam... so doing this differently from the check for flam
        # mask the array where the ferr value has been set to 0 by the stacking code
        if np.any(ferr == 0.0):
            indices_to_be_masked = np.where(ferr == 0.0)[0]
            mask = np.zeros(len(ferr)) # by default create a masked array where all values in the original array are assumed to be valid
            mask[indices_to_be_masked] = 1 # now set the indices to be masked as True
            flam_masked = ma.masked_array(flam, mask = mask)
            ferr_masked = ma.masked_array(ferr, mask = mask)
            flam = flam_masked
            ferr = ferr_masked

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
            flam_masked = ma.masked_array(flam, mask = lam_mask)
            ferr_masked = ma.masked_array(ferr, mask = lam_mask)
            flam = flam_masked
            ferr = ferr_masked

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

        currentspec = comp_spec

        chi2 = np.zeros(nexten, dtype=np.float64)
        alpha = np.sum(flam*currentspec/(ferr**2), axis=1)/np.sum(currentspec**2/ferr**2, axis=1)
        
        chi2 = np.sum(((flam - (alpha * currentspec.T).T) / ferr)**2, axis=1)

        # This is to get only physical ages
        sortargs = np.argsort(chi2)
        for k in range(len(chi2)):
            best_age = float(h[sortargs[k]+1].header['AGE_GYR'])
            if (best_age < 8) & (best_age > 0.1):
                tau.append(h[sortargs[k]+1].header['TAU'])
                tauv.append(h[sortargs[k]+1].header['TAUV'])
                ages.append(best_age)
                totalchi2.append(chi2[sortargs[k]])
                metals.append(h[sortargs[k]+1].header['METAL'])
                #print np.min(chi2), best_metal, best_age, curr_tau, curr_tauv
                break

    # total computational time
    print "Total computational time taken to get chi2 values --", time.time() - start, "seconds."

    ages = np.asarray(ages, dtype=np.float64)
    metals = np.asarray(metals, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    logtau = np.log10(tau)
    tauv = np.asarray(tauv, dtype=np.float64)

    print np.mean(ages), np.median(ages), np.std(ages)
    print np.mean(metals), np.median(metals), np.std(metals)
    print np.mean(tau), np.median(tau), np.std(tau)
    print np.mean(tauv), np.median(tauv), np.std(tauv)

    # histograms of the best fit params
    fig, ax = makefig_hist('Age [Gyr]')
    iqr = 1.349 * np.std(ages, dtype=np.float64)
    binsize = 2*iqr*np.power(len(ages),-1/3) # Freedman-Diaconis Rule
    totalbins = np.floor((13.85 - 0.1)/binsize)
    ax.hist(ages, totalbins, histtype='bar', align='mid', alpha=0.5)
    fig.savefig('agedist_' + '2p1_10p5' + '.png', dpi=300)

    # this block of code is useful if the metals are in their usual BC03 notation
    #ctr = collections.Counter()
    #for i in metals:
    #    ctr[i] += 1
    #fig, ax = makefig_hist('Metals [Z]')
    #print ctr
    #metal_labels = ctr.keys()
    #for count, val in enumerate(ctr.keys()):
    #    if val == 'm22':
    #        metal_labels[count] = '0.0001'
    #    elif val == 'm32':
    #        metal_labels[count] = '0.0004'
    #    elif val == 'm42':
    #        metal_labels[count] = '0.004'
    #    elif val == 'm52':
    #        metal_labels[count] = '0.008'
    #    elif val == 'm62':
    #        metal_labels[count] = '0.02'
    #    elif val == 'm72':
    #        metal_labels[count] = '0.05'
    #ax.bar(np.arange(len(ctr.keys())), ctr.values(), alpha=0.5)
    #ax.set_xticklabels(metal_labels)
    #ax.set_xticks(np.arange(0.4, len(ctr.keys()) + 0.4, 1))
    #fig.savefig('metaldist_' + '2p1_10p5' + '.png', dpi=300)

    fig, ax = makefig_hist('Metals [Z]')
    iqr = 1.349 * np.std(metals, dtype=np.float64)
    binsize = 2*iqr*np.power(len(metals),-1/3) # Freedman-Diaconis Rule
    totalbins = np.floor((max(metals) - min(metals))/binsize)
    ax.hist(metals, totalbins, histtype='bar', align='mid', alpha=0.5)
    fig.savefig('metaldist_' + '2p1_10p5' + '.png', dpi=300)
    
    fig, ax = makefig_hist(r'$\mathrm{log}(\tau)$')
    iqr = 1.349 * np.std(logtau, dtype=np.float64)
    binsize = 2*iqr*np.power(len(logtau),-1/3) # Freedman-Diaconis Rule
    totalbins = np.floor((2 - (-2))/binsize)
    ax.hist(logtau, totalbins, histtype='bar', align='mid', alpha=0.5)
    fig.savefig('taudist_' + '2p1_10p5' + '.png', dpi=300)
    
    fig, ax = makefig_hist(r'$\tau_V$')
    totalbins = np.floor((1.9 - 0.1)/0.1)
    ax.hist(tauv, totalbins, histtype='bar', align='mid', alpha=0.5)
    fig.savefig('tauvdist_' + '2p1_10p5' + '.png', dpi=300)

    plt.show()

    """

    fig, ax = makefig('tauv', 'age')
    ax.plot(tauv, tau, 'o', color='k', markeredgecolor='k', markersize=2)
    #ax.set_xlim(-2.1, 0)
    #ax.set_ylim(0, 8)
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    plt.show()
    """

    # total run time
    print "Total time taken --", time.time() - start, "seconds."

