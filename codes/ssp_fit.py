from __future__ import division
import numpy as np
import numpy.ma as ma
from astropy.io import fits

import datetime, sys, os, time, glob

import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at -- ", dt.now()
    
    sspdir = '/Users/baj/Documents/GALAXEV_BC03/bc03/models/Padova1994/salpeter/'
    # you should do this with chabrier too

    metals = ["m22","m32","m42","m52","m62","m72"]

    # read in median stacked spectra
    stacks = fits.open('/Users/baj/Desktop/FIGS/new_codes/coadded_coarsegrid_PEARSgrismspectra.fits')
    totalstacks = 0 # this is the total number of stacked spectra
    while 1:
        try:
            if stacks[totalstacks+2]:
                totalstacks += 1
        except IndexError:
            break
    
    # Get comparison spectra
    h = fits.open('/Users/baj/Desktop/FIGS/new_codes/all_ssp_comp_spectra.fits', memmap=False) 
    nexten = 0 # this is the total number of comparison spectra
    while 1:
        try:
            if h[nexten+1]:
                nexten += 1
        except IndexError:
            break

    # define lambda grid and create comparison spectra array
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

    # Files to save the distribution of best params
    f_ages = open('jackknife_ssp_ages.txt', 'wa')
    f_metals = open('jackknife_ssp_metals.txt', 'wa')  

    # Loop over all stacks
    num_samp_to_draw = 1e4
    print "Running over", int(num_samp_to_draw), "random jackknifed samples."
    color_step = 0.6
    mstar_step = 1.0
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
                    ages.append(best_age)
                    totalchi2.append(chi2[sortargs[k]])
                    bestchi2index.append(sortargs[k])
                    bestalpha.append(alpha[sortargs[k]])
                    metals.append(h[sortargs[k]+1].header['METAL'])
                    #print np.min(chi2), best_metal, best_age
                    break
        
        # total computational time
        print "Total computational time taken to get chi2 values --", time.time() - chi2start, "seconds."

        ages = np.asarray(ages, dtype=np.float64)
        metals = np.asarray(metals, dtype=np.float64)
        
        print np.mean(ages), np.median(ages), np.std(ages)
        print np.mean(metals), np.median(metals), np.std(metals)
        print len(np.unique(ages)), len(np.unique(metals))

        # Save the data from the runs
        f_ages.write(ongrid + ' ')
        for k in range(len(ages)):
            f_ages.write(str(ages[k]) + ' ')
        f_ages.write('\n')

        f_metals.write(ongrid + ' ')
        for k in range(len(metals)):
            f_metals.write(str(metals[k]) + ' ')
        f_metals.write('\n')

    f_ages.close()
    f_metals.close()
         
    # total run time
    print "Total time taken --", time.time() - start, "seconds."