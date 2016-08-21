from __future__ import division
import numpy as np
import numpy.ma as ma
import pyfits as pf

import datetime, sys, os, time, glob
import logging

import matplotlib.pyplot as plt

def plot_spectrum_data(lam, flux, flux_err):
    
    ax.errorbar(lam, flux, yerr=flux_err, fmt='o-', color='k', linewidth=2, ecolor='r', markeredgecolor='k', capsize=0, markersize=4)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

def plot_spectrum_bc03(lam, flux):
    
    ax.plot(lam, flux, '-', linewidth=2)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

def resample(lam, spec, lam_grid):
    
    lam_em = lam
    resampled_flam = np.zeros((244, len(lam_grid)))

    for i in range(len(lam_grid)):
        new_ind = np.where((lam_em >= lam_grid[i] - lam_step/2) & (lam_em < lam_grid[i] + lam_step/2))[0]
        resampled_flam[:,i] = np.median(spec[:,[new_ind]], axis=2).reshape(244)

    lam_grid = lam_grid + 50
    return lam_grid, resampled_flam

def rescale(lam, spec):
    
    arg4450 = np.argmin(abs(lam - 4450))
    arg4650 = np.argmin(abs(lam - 4650))

    medval = np.median(spec[:,arg4450:arg4650+1], axis=1)
    medval = medval.reshape(244,1)
    return medval

def resample_indiv(lam, spec, lam_grid):
    
    lam_em = lam
    resampled_flam = np.zeros(len(lam_grid))
    for i in range(len(lam_grid)):
        
        new_ind = np.where((lam_em >= lam_grid[i] - lam_step/2) & (lam_em < lam_grid[i] + lam_step/2))[0]
        
        for j in range(len(new_ind)):
            resampled_flam[i] = np.median(spec[new_ind])

    return lam_grid, resampled_flam

def rescale_indiv(lam, spec):
    
    arg4450 = np.argmin(abs(lam - 4450))
    arg4650 = np.argmin(abs(lam - 4650))
    medval = np.median(spec[arg4450:arg4650+1])
    
    return medval

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    
    #metals = ["m22","m32","m42","m52","m62","m72"]
    metals = ["m62"]

    # read in median stacked spectra
    stacks = pf.open('/Users/baj/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra.fits')
    
    lam_step = 100
    lam_grid = np.arange(2700, 6000, lam_step)
    
    for stackcount in range(46,47,1):
        
        flam = stacks[stackcount+3].data[0]
        ferr = stacks[stackcount+3].data[1]
        ongrid = stacks[stackcount+3].header["ONGRID"]
        numspec = int(stacks[stackcount+3].header["NUMSPEC"])
        print "At ", ongrid, " with ", numspec, " spectra in stack."

        if numspec < 10:
            print "Too few spectra in stack. Continuing to the next grid cell..."
            continue
        
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

        """
        # for the blue spectra only
        # mask the region with h-beta and [OIII] emission lines
        # I'm masking the region between 4850A to 5050A because my stacks are sampled at 100A in delta_lambda
        # I will be excluding the points at 4850A, 4950A, and 5050A
        arg4850 = np.argmin(abs((lam_grid + 50) - 4850))
        arg4950 = np.argmin(abs((lam_grid + 50) - 4950))
        arg5050 = np.argmin(abs((lam_grid + 50) - 5050)) 
        lam_mask = np.zeros(len(flam))
        lam_mask[arg4850] = 1
        lam_mask[arg4950] = 1
        lam_mask[arg5050] = 1
        flam_masked = ma.masked_array(flam, mask = lam_mask)
        ferr_masked = ma.masked_array(ferr, mask = lam_mask)
        flam = flam_masked
        ferr = ferr_masked

        # test block
        testspec = pf.open('/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/m22/bc2003_hr_m22_tauV10_csp_tau100000_salp.fits')
        testspec_lam = testspec[1].data
        testspec_ages = testspec[2].data
        arg10gyr = np.argmin(abs(testspec_ages - 2e8))
        testspec_flam = testspec[arg10gyr + 3].data
        testspec_ferr = np.random.random_sample(size = len(lam_grid)) / 10
        testspec_lam, testspec_flam = resample_indiv(testspec_lam, testspec_flam, lam_grid)
        testspec_medval = rescale_indiv(testspec_lam, testspec_flam)
        testspec_flam /= testspec_medval
        flam = testspec_flam
        ferr = testspec_ferr
        """

        totalchi2 = []
        bestages= []
        besttau = []
        besttauV = []
        bestmetallicity = []
        alpha_arr = []
        file_arr = []
    
        # read in BC03 spectra
        for metallicity in metals:
            cspout = '/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/'
            metalfolder = metallicity + '/'
        
            filecount = 0
            for filename in glob.glob(cspout + metalfolder + '*.fits'):
            
                #colfilename = filename.replace('.fits', '.3color')
                #col = np.genfromtxt(colfilename, dtype=None, names=['age', 'b4_vn'], skip_header=29, usecols=(0,2))
                #ageindices = np.where((col['b4_vn'] < data_dn4000 + eps) & (col['b4_vn'] > data_dn4000 - eps))[0]
                print filename
                h = pf.open(filename, memmap=False)
                ages = h[2].data
                tau = float(filename.split('/')[-1].split('_')[5][3:])/10000
                tauV = float(filename.split('/')[-1].split('_')[3][4:])/10
                totalmodels = 245
                chi2 = np.ones(totalmodels) * 99999.0
            
                currentspec = np.zeros([244, 6900], dtype=np.float64)
                currentlam = h[1].data
                for i in range(1,totalmodels,1):
                    """
                    if np.all(h[i+3].data) == 0.0:
                        # Skip if they are all zeros
                        # This doesn't matter because currentspec is already initialized to all zeros
                        print "all 0s continuing..."
                        continue
                    """
                    currentspec[i-1] = h[i+3].data

                #print currentspec
            
                # Only consider the part of BC03 spectrum between 2700 to 6000
                arg2700 = np.argmin(abs(currentlam - 2700))
                arg6000 = np.argmin(abs(currentlam - 6000))
                currentlam = currentlam[arg2700:arg6000+1] # chopping off unrequired lambda
                currentspec = currentspec[:,arg2700:arg6000+1] # chopping off unrequired spectrum

                currentlam, currentspec = resample(currentlam, currentspec, lam_grid)
                #medval = rescale(currentlam, currentspec)
                #currentspec = np.divide(currentspec, medval)
                
                chi2 = np.zeros(244, dtype=np.float64)
                alpha = np.sum(flam*currentspec/(ferr**2), axis=1)/np.sum(currentspec**2/ferr**2, axis=1)
                chi2 = np.sum(((flam - (alpha * currentspec.T).T) / ferr)**2, axis=1)

                #print chi2
                if filecount == 5:
                    sys.exit()
                filecount += 1

                besttau.append(tau)
                besttauV.append(tauV)
                bestages.append(ages[np.argmin(chi2)])
                totalchi2.append(np.min(chi2))
                bestmetallicity.append(metallicity)
                alpha_arr.append(alpha[np.argmin(chi2)])
                file_arr.append(filename)

                print np.min(chi2), metallicity, tau, tauV, ages[np.argmin(chi2)]/1e9

        totalchi2 = np.asarray(totalchi2)
        bestages = np.asarray(bestages)
        besttau = np.asarray(besttau)
        besttauV = np.asarray(besttauV)
        alpha_arr = np.asarray(alpha_arr)

        minindex = np.argmin(totalchi2)
        print stackcount, totalchi2[minindex], bestmetallicity[minindex], bestages[minindex]/1e9, besttau[minindex], besttauV[minindex], alpha_arr[minindex], file_arr(minindex)
    
    # Total time taken
    totaltime = time.time() - start
    print "Total time taken -- ", totaltime/3600, "hours."