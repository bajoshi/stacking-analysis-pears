from __future__ import division
import numpy as np
import numpy.ma as ma
from astropy.io import fits

import datetime, sys, os, time, glob
import logging

import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D

def makefig():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    return ax

def plot_spectrum_data(lam, flux, flux_err):
    
    ax.errorbar(lam, flux, yerr=flux_err, fmt='o-', color='k', linewidth=2, ecolor='r', markeredgecolor='k', capsize=0, markersize=4)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

def plot_spectrum_bc03(lam, flux):
    
    ax.plot(lam, flux, 'o-', linewidth=2, color='r')
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

def plot_spectrum_bc03_ages(lam, flux, ages):
    
    totalages = totalagestoconsider
    for i in range(0,totalages,25):
        ax.plot(lam, flux[i], '-', label=str(ages[i+1]/1e9), linewidth=2)
        
        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')
        ax.set_xlim(2500, 6500)
        ax.legend(loc=0)
        #ax.set_yscale('log')

def resample(lam, spec):
    
    lam_em = lam
    resampled_flam = np.zeros((totalagestoconsider, len(lam_grid_tofit)))

    for i in range(len(lam_grid_tofit)):
        new_ind = np.where((lam_em >= lam_grid_tofit[i] - lam_step/2) & (lam_em < lam_grid_tofit[i] + lam_step/2))[0]
        resampled_flam[:,i] = np.median(spec[:,[new_ind]], axis=2).reshape(totalagestoconsider)

    return resampled_flam

def rescale(lam, spec):
    
    arg4450 = np.argmin(abs(lam - 4450))
    arg4650 = np.argmin(abs(lam - 4650))

    medval = np.median(spec[:,arg4450:arg4650+1], axis=1)
    medval = medval.reshape(totalagestoconsider,1)
    return medval

def resample_indiv(lam, spec):
    
    lam_em = lam
    resampled_flam = np.zeros(len(lam_grid_tofit))
    for i in range(len(lam_grid_tofit)):
        
        new_ind = np.where((lam_em >= lam_grid_tofit[i] - lam_step/2) & (lam_em < lam_grid_tofit[i] + lam_step/2))[0]
        resampled_flam[i] = np.median(spec[new_ind])

    return resampled_flam

def rescale_indiv(lam, spec):
    
    arg4450 = np.argmin(abs(lam - 4450))
    arg4650 = np.argmin(abs(lam - 4650))
    medval = np.median(spec[arg4450:arg4650+1])
    
    return medval

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at -- ", dt.now()
    
    #metals = ["m22","m32","m42","m52","m62","m72"]
    metals = ["m62"]

    # read in median stacked spectra
    stacks = fits.open('/Users/baj/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra.fits')
    #stacks = fits.open('/Users/baj/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra_coarsegrid.fits')

    lam_step = 100
    lam_lowfit = 3000
    lam_highfit = 6000
    lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)
    
    for stackcount in range(47,48,1):
        
        flam = stacks[stackcount+2].data[0]
        ferr = stacks[stackcount+2].data[1]
        ongrid = stacks[stackcount+2].header["ONGRID"]
        numspec = int(stacks[stackcount+2].header["NUMSPEC"])
        print "At ", ongrid, " with ", numspec, " spectra in stack."

        ferr = ferr + 0.05 * flam # putting in a 5% additional error bar

        # Chop off the ends of the stacked spectrum
        orig_lam_grid = np.arange(2700, 6000, lam_step)
        arg_lamlow = np.argmin(abs(orig_lam_grid - lam_lowfit))
        arg_lamhigh = np.argmin(abs(orig_lam_grid - lam_highfit+100))
        flam = flam[arg_lamlow:arg_lamhigh+1]
        ferr = ferr[arg_lamlow:arg_lamhigh+1]

        if numspec <= 10:
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

        """
        # test block
        testspec = fits.open('/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/m22/bc2003_hr_m22_tauV10_csp_tau100000_salp.fits')
        testspec_lam = testspec[1].data
        testspec_ages = testspec[2].data
        arg10gyr = np.argmin(abs(testspec_ages - 2e8))
        testspec_flam = testspec[arg10gyr + 3].data
        testspec_ferr = np.random.random_sample(size = len(lam_grid_tofit)) / 10
        testspec_lam, testspec_flam = resample_indiv(testspec_lam, testspec_flam, lam_grid_tofit)
        testspec_medval = rescale_indiv(testspec_lam, testspec_flam)
        testspec_flam /= testspec_medval
        flam = testspec_flam
        ferr = testspec_ferr
        """

        totalchi2 = []
        totalchi2_fit = []
        bestages= []
        besttau = []
        besttauV = []
        bestmetallicity = []
        alpha_arr = []
        file_arr = []

        # read in BC03 spectra
        tauVarr = np.arange(0.0, 3.0, 0.1)
        for metallicity in metals:
            for tauVarrval in tauVarr:
                cspout = '/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/'
                metalfolder = metallicity + '/' 
                
                chi2_meshed = np.empty((40,244))
        
                filecount = 0
                tauarr = []
                for filename in glob.glob(cspout + metalfolder + 'bc2003_hr_' + metallicity + '_tauV' + str(int(tauVarrval*10)) + '_' + '*.fits'):
            
                    #colfilename = filename.replace('.fits', '.3color')
                    #col = np.genfromtxt(colfilename, dtype=None, names=['age', 'b4_vn'], skip_header=29, usecols=(0,2))
                    #ageindices = np.where((col['b4_vn'] < data_dn4000 + eps) & (col['b4_vn'] > data_dn4000 - eps))[0]
                    #print filename
                    h = fits.open(filename, memmap=False)
                    ages = h[2].data[1:]
                    tau = float(filename.split('/')[-1].split('_')[5][3:])/10000
                    tauV = float(filename.split('/')[-1].split('_')[3][4:])/10
                    tauarr.append(tau)
        
                    totalmodels = 245
                    currentlam = h[1].data
        
                    totalagestoconsider = int(244/1) # change the denominator to 1 to fit all ages
        
                    currentspec = np.zeros([totalagestoconsider, len(currentlam)], dtype=np.float64)
                    count=0
                    for i in range(1,totalmodels,1):
                        currentspec[count] = h[i+3].data
                        count += 1
                    #print currentspec
                
                    #currentspec = np.ones([totalagestoconsider,len(currentlam)]) * 1e-18 # testing a flat model
        
                    # Only consider the part of BC03 spectrum between 2700 to 6000
                    arg2650 = np.argmin(abs(currentlam - 2650))
                    arg5950 = np.argmin(abs(currentlam - 5950))
        
                    currentlam = currentlam[arg2650:arg5950+1] # chopping off unrequired lambda
                    currentspec = currentspec[:,arg2650:arg5950+1] # chopping off unrequired spectrum
                    currentspec = resample(currentlam, currentspec)
                    currentlam = lam_grid_tofit
                    #medval = rescale(currentlam, currentspec)
                    #currentspec = np.divide(currentspec, medval)
        
                    chi2 = np.zeros(totalagestoconsider, dtype=np.float64)
                    alpha = np.sum(flam*currentspec/(ferr**2), axis=1)/np.sum(currentspec**2/ferr**2, axis=1)
        
                    chi2 = np.sum(((flam - (alpha * currentspec.T).T) / ferr)**2, axis=1)
                    chi2_meshed[filecount] = chi2
                    filecount += 1
        
                    sortargs = np.argsort(chi2)
                    for k in range(len(chi2)):
                        if (ages[sortargs[k]]/1e9 < 8) & (np.log10(ages[sortargs[k]]) > 8.0):
                            besttau.append(tau)
                            besttauV.append(tauV)
                            bestages.append(ages[sortargs[k]])
                            totalchi2.append(chi2[sortargs[k]])
                            bestmetallicity.append(metallicity)
                            alpha_arr.append(alpha[sortargs[k]])
                            file_arr.append(filename)
                            break
        
                    #print np.min(chi2), metallicity, tau, tauV, ages[np.argmin(chi2)]/1e9, alpha[np.argmin(chi2)]
    
                # You need a plot like this for each tauV
                ages = np.log10(ages)
                tauarr = np.asarray(tauarr)
                ages, tau = np.meshgrid(ages, tauarr)
        
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
        
                ax.set_xlabel('Age')
                ax.set_ylabel('Tau')
                ax.set_zlabel('Chi2' + r'$\tau$' + str(tauV))
        
                ax.plot_surface(ages, tau, chi2_meshed, rstride=2, cstride=5, alpha=0.3, cmap=cm.coolwarm, linewidth=1)
                # rstride and cstride only determine how your row and column arrays (in this case ages and tau?) are sampled on the plot
                # e.g. if rstride is 5 it will plot every 5th age and same for cstride
                #ax.plot() # Plot the min chi2 point as a big red point

                ax.set_zlim(0, 2000)
        
                cset = ax.contour(ages, tau, chi2_meshed, zdir='z', offset=-300, cmap=cm.coolwarm)
                cset = ax.contour(ages, tau, chi2_meshed, zdir='x', offset=5, cmap=cm.coolwarm)
                cset = ax.contour(ages, tau, chi2_meshed, zdir='y', offset=150, cmap=cm.coolwarm)
        
                fig.savefig('/Users/baj/Desktop/FIGS/new_codes/surf_images/' + 'chi2_surf_' + metallicity + '_' + ongrid.replace('.','p').replace(',','_') + '_' + str(tauV) + '.png', dpi=300)
                del fig, ax, ages, tau, chi2_meshed

        totalchi2 = np.asarray(totalchi2)
        bestages = np.asarray(bestages)
        besttau = np.asarray(besttau)
        besttauV = np.asarray(besttauV)
        alpha_arr = np.asarray(alpha_arr)

        minindex = np.argmin(totalchi2)
        print ongrid
        #print "Fitting range -- ", lam_lowfit, " to ", lam_highfit
        print "Minimum Chi square -- ", totalchi2[minindex]
        #print "Minimum Chi square within fitting range -- ", totalchi2_fit[minindex]
        print "Best Metallicity -- ", bestmetallicity[minindex]
        print "Best fit age [Gyr] -- ", bestages[minindex]/1e9
        print "Best fit tau (in exp SFH) -- ", besttau[minindex]
        print "Best fit tau_V (opt. depth to dust extn) -- ", besttauV[minindex]
        print "Value for vertical scaling constant that minimizes chi square -- ", alpha_arr[minindex]
        print "Filename with best fit model -- ", file_arr[minindex]
        print "\n"
        print totalchi2[minindex], bestmetallicity[minindex], bestages[minindex]/1e9, besttau[minindex], besttauV[minindex], alpha_arr[minindex], file_arr[minindex]

    # Total time taken
    totaltime = time.time() - start
    print "Total time taken -- ", totaltime/3600, "hours."