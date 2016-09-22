from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from scipy import stats

import sys
import os
import datetime
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, AnchoredText

from fast_chi2_jackknife import get_total_extensions

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figures_dir = stacking_analysis_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"

def makefig():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\lambda\ [\AA]$')
    ax.set_ylabel('$f_{\lambda}\ [\mathrm{erg/s/cm^2/\AA}] $')
    ax.axhline(y=0,linestyle='--')

    return fig, ax

def plot_spectrum_data(lam, flux, flux_err):
    
    ax.errorbar(lam, flux, yerr=flux_err, fmt='o-', color='k', linewidth=2, ecolor='r', markeredgecolor='k', capsize=0, markersize=4)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(3000, 6000)
    #ax.set_ylim(0,2)

def plot_spectrum_model(lam, flux, col):

    ax.plot(lam, flux, 'o-', color=col, markeredgecolor=None, linewidth=3)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)
    #ax.legend(loc=0, numpoints=1, prop={'size':12})

def get_boot_samps(bootname, nsamp, orig_lam_grid):
    
    h = fits.open(bootname)
    boot_samps = np.empty((nsamp, len(orig_lam_grid)))

    for i in range(nsamp):
        boot_samps[i] = h[i + 2].data[0]

    return boot_samps

def get_avg_color_stellarmass(stack_hdu, col_low, col_high, col_step, mstar_low, mstar_high, mstar_step, ur_color, stellarmass):

    h = stack_hdu

    # Find the averages of all grid cells in a particular row/column
    # While these are useful numbers to have, they are currently only used in the plotting routine.
    cellcount = 0

    avgcolarr = np.zeros(5)
    avgmassarr = np.zeros(5)
        
    avgmassarr = avgmassarr.tolist()
    for k in range(len(avgmassarr)):
        avgmassarr[k] = []

    for i in np.arange(col_low, col_high, col_step):
        colcount = 0
        for j in np.arange(mstar_low, mstar_high, mstar_step):
            
            indices = np.where((ur_color >= i) & (ur_color < i + col_step) &\
                       (stellarmass >= j) & (stellarmass < j + mstar_step))[0]
            
            row = int(i/col_step)
            column = int((j - 7.0)/mstar_step)

            if indices.size:
                avgcol = h[cellcount+2].header['AVGCOL']
                avgmass = h[cellcount+2].header['AVGMASS']
                avgcolarr[row] += float(avgcol)
                avgmassarr[column].append(float(avgmass))

                cellcount += 1
                colcount += 1
            else:
                continue

        avgcolarr[row] /= (colcount)

    for x in range(len(avgmassarr)):
        avgmassarr[x] = np.sum(avgmassarr[x]) / len(avgmassarr[x])

    return avgcolarr, avgmassarr

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Read in stacks
    stacks = fits.open(home + '/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra_coarsegrid.fits')
    totalstacks = get_total_extensions(stacks)

    # Set up lambda grid
    lam_step = 100
    lam_lowfit = 3600
    lam_highfit = 6000
    lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)

    # Parameters for the original grid
    # Used here only for getting average colors and stellar masses
    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt',
                       dtype=None, names=True, skip_header=2)

    ur_color = cat['urcol']
    stellarmass = cat['mstar']

    col_step = 0.6
    mstar_step = 1.0
    col_low = 0.0
    col_high = 3.0
    mstar_low = 7.0
    mstar_high = 12.0

    avgcol, avgmass = get_avg_color_stellarmass(stacks, col_low, col_high, col_step, mstar_low, mstar_high, mstar_step, ur_color, stellarmass)

    # Create pdf file to plot figures in
    pdfname = figures_dir + 'overplot_all_sps.pdf'
    pdf = PdfPages(pdfname)

    # Create grid for making grid plots
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.2)

    # Read ongrid values
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

    #### MILES ####
    ages_miles = np.loadtxt(stacking_analysis_dir + 'jackknife_ages_miles.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    metals_miles = np.loadtxt(stacking_analysis_dir + 'jackknife_metals_miles.txt', usecols=range(1, int(num_jackknife_samps) + 1))

    #### FSPS ####
    ages_fsps = np.loadtxt(stacking_analysis_dir + 'jackknife_ages_fsps.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    metals_fsps = np.loadtxt(stacking_analysis_dir + 'jackknife_metals_fsps.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    logtau_fsps = np.loadtxt(stacking_analysis_dir + 'jackknife_logtau_fsps.txt', usecols=range(1, int(num_jackknife_samps) + 1))
    mass_wht_ages_fsps = np.loadtxt(stacking_analysis_dir + 'mass_weighted_ages_fsps.txt', usecols=range(1, int(num_jackknife_samps) + 1))

    # Open fits files with comparison spectra
    bc03_spec = fits.open(home + '/Desktop/FIGS/new_codes/all_comp_spectra_bc03.fits', memmap=False)
    miles_spec = fits.open(home + '/Desktop/FIGS/new_codes/all_comp_spectra_miles.fits', memmap=False)
    fsps_spec = fits.open(home + '/Desktop/FIGS/new_codes/all_comp_spectra_fsps.fits', memmap=False)

    # Find number of extensions in each
    bc03_extens = get_total_extensions(bc03_spec)
    miles_extens = get_total_extensions(miles_spec)
    fsps_extens = get_total_extensions(fsps_spec)

    # Make multi-dimensional arrays of parameters for all SPS libraries 
    # This is done to make access to the fits headers easier because
    # there is no way to directly search the headers for all extensions in a fits file simultaneously.
    # Once the numpy arrays have the header values in them the arrays can easily be searched by using np.where
    bc03_params = np.empty([bc03_extens, 4])  # the 4 is because the BC03 param space is 4D
    for i in range(bc03_extens):
        age = float(bc03_spec[i+1].header['LOG_AGE'])
        z = float(bc03_spec[i+1].header['METAL'])
        tau = float(bc03_spec[i+1].header['TAU_GYR'])
        tauv = float(bc03_spec[i+1].header['TAUV'])

        bc03_params[i] = np.array([age, z, tau, tauv])

    miles_params = np.empty([miles_extens, 2])  # the 2 is because the MILES param space is 2D
    for i in range(miles_extens):
        age = float(miles_spec[i+1].header['LOG_AGE'])
        z = float(miles_spec[i+1].header['METAL'])

        miles_params[i] = np.array([age, z])

    fsps_params = np.empty([fsps_extens, 3])  # the 3 is because the FSPS param space is 3D
    for i in range(fsps_extens):
        age = float(fsps_spec[i+1].header['LOG_AGE'])
        z = float(fsps_spec[i+1].header['METAL'])
        tau = float(fsps_spec[i+1].header['TAU_GYR'])

        fsps_params[i] = np.array([age, z, tau])

    # Loop over the stacks and plot 
    count = 0
    for stackcount in range(0, totalstacks-1, 1):

        flam = stacks[stackcount + 2].data[0]
        ferr = stacks[stackcount + 2].data[1]
        ferr = ferr + 0.05 * flam  # putting in a 5% additional error bar
        ongrid = stacks[stackcount + 2].header["ONGRID"]
        numspec = int(stacks[stackcount + 2].header["NUMSPEC"])
        #print ongrid

        if numspec < 5:
            #print "Too few spectra in stack. Continuing to the next grid cell..."
            continue

        # Mask flam and ferr arrays
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

        # Chop off the ends of the stacked spectrum
        # This will cause it to chop the spectrum from 3600 to 5900
        orig_lam_grid = np.arange(2700, 6000, lam_step)
        arg_lamlow = np.argmin(abs(orig_lam_grid - lam_lowfit))
        arg_lamhigh = np.argmin(abs(orig_lam_grid - lam_highfit))
        flam = flam[arg_lamlow:arg_lamhigh+1]
        ferr = ferr[arg_lamlow:arg_lamhigh+1]

        # Make figure and axes and plot stacked spectrum with errors from bootstrap
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[:10,:])
        ax2 = fig.add_subplot(gs[10:,:])

        ax1.set_ylabel('$f_{\lambda}\ [\mathrm{erg/s/cm^2/\AA}] $')
        ax2.set_xlabel('$\lambda\ [\AA]$')

        #bootname = savefits_dir + 'bootstrap-err-stacks/' + 'spectra_bootstrap_' + ongrid.replace(',','_').replace('.','p') + '.fits'

        #if not os.path.isfile(bootname):
        #    print "Did not find bootstrapped samples file! Using default errors..."
        #    # This is a temporary step which will be removed after all stacks have their bootstrap errors
        #    ferr = ferr + 0.05 * flam # putting in a 5% additional error bar
        #    ax1.errorbar(lam_grid_tofit[:-5], flam, yerr=ferr, fmt='o-', color='k', linewidth=2, ecolor='r', markeredgecolor='k', capsize=0, markersize=4)
        #    
        #    ax1.minorticks_on()
        #    ax1.tick_params('both', width=1, length=3, which='minor')
        #    ax1.tick_params('both', width=1, length=4.7, which='major')
        #else:

        #number_boot_samps = 100
        #boot_samps = get_boot_samps(bootname, number_boot_samps, orig_lam_grid)

        #boot_samps_masked = np.empty((number_boot_samps, len(orig_lam_grid[arg_lamlow:arg_lamhigh+1])))
        #for i in range(number_boot_samps): # Not yet sure if I can get rid of this for loop
        #    boot_samps[i] = ma.masked_array(boot_samps[i], mask = flam_mask)
        #    boot_samps[i] = ma.masked_array(boot_samps[i], mask = ferr_mask)
        #    boot_samps_masked[i] = boot_samps[i][arg_lamlow:arg_lamhigh+1]

        #ax1 = sns.tsplot(data=boot_samps[:,arg_lamlow:arg_lamhigh+1], time=lam_grid_tofit, err_style='ci_band', ax=ax1)
        ax1.plot(lam_grid_tofit, flam, color='k')
        ax1.fill_between(lam_grid_tofit, flam + ferr, flam - ferr, color='lightgray')
        ax1.set_xlim(3000, 6000)
        ax1.xaxis.set_ticklabels([])
        ax1.minorticks_on()
        ax1.tick_params('both', width=1, length=3, which='minor')
        ax1.tick_params('both', width=1, length=4.7, which='major')

        """
        In the comparisons below, I'm using np.allclose instead of np.setdiff1d because 
        when I go from logtau to tau by doing 10**logtau the operation is introducing 
        some junk digits at the end of the floating point number.
        """

        #### BC03 ####
        best_age = stats.mode(ages_bc03[count])[0]
        best_metal = stats.mode(metals_bc03[count])[0]
        best_tau = 10**stats.mode(logtau_bc03[count])[0]
        best_tauv = stats.mode(tauv_bc03[count])[0]
        best_mass_wht_age = stats.mode(mass_wht_ages_bc03[count])[0]

        best_age_err = np.std(ages_bc03[count]) * 10**best_age / (1e9 * 0.434)
        # the number 0.434 is (1 / ln(10)) 
        # the research notebook has this short calculation at the end 
        best_metal_err = np.std(metals_bc03[count])
        best_tau_err = np.std(logtau_bc03[count]) * best_tau / 0.434
        best_tauv_err = np.std(tauv_bc03[count])
        best_mass_wht_age_err = np.std(mass_wht_ages_bc03[count]) * 10**best_mass_wht_age / (1e9 * 0.434)

        # Plot best fit parameters as anchored text boxes
        i = ongrid.split(',')[0]
        j = ongrid.split(',')[1]
        row = int(float(i)/col_step)
        column = int((float(j) - 7.0)/mstar_step)

        print 'bc03', best_age, best_age_err, best_tau, best_tau_err, best_mass_wht_age, best_mass_wht_age_err, avgmass[column]

        for j in range(bc03_extens):
            if np.allclose(bc03_params[j], np.array([best_age, best_metal, best_tau, best_tauv]).reshape(4)):
                currentspec = bc03_spec[j+1].data

                alpha = np.sum(flam * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)

                #ongridbox = TextArea(ongrid, textprops=dict(color='k', size=10)) # Change this to average color and average stellar mass
                #anc_ongridbox = AnchoredOffsetbox(loc=2, child=ongridbox, pad=0.0, frameon=False,\
                #                                     bbox_to_anchor=(0.03, 0.75),\
                #                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                #ax1.add_artist(anc_ongridbox)                

                currentcol = avgcol[row]
                avgcolbox = TextArea(r"$\left<\mathrm{(U-R)_{rest}}\right>$ = " + "{:.2f}".format(currentcol), textprops=dict(color='k', size=10))
                anc_avgcolbox = AnchoredOffsetbox(loc=2, child=avgcolbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.7),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_avgcolbox)  

                currentmass = avgmass[column]
                avgmassbox = TextArea(r"$\left<\mathrm{log(\frac{M_*}{M_\odot})}\right>$ = " + "{:.2f}".format(currentmass), textprops=dict(color='k', size=10))
                anc_avgmassbox = AnchoredOffsetbox(loc=2, child=avgmassbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.65),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_avgmassbox) 

                labelbox = TextArea("BC03", textprops=dict(color='r', size=8))
                anc_labelbox = AnchoredOffsetbox(loc=2, child=labelbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.95),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_labelbox)

                agebox = TextArea(r"$\left<t\right>_M$ = " + "{:.2f}".format(float(10**best_mass_wht_age/1e9)) + r" $\pm$ " + "{:.2f}".format(float(best_mass_wht_age_err)) + " Gyr",
                 textprops=dict(color='r', size=8))
                anc_agebox = AnchoredOffsetbox(loc=2, child=agebox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.9),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_agebox)

                metalbox = TextArea("Z = " + "{:.2f}".format(float(best_metal)) + r" $\pm$ " + "{:.2f}".format(float(best_metal_err)),
                 textprops=dict(color='r', size=8))
                anc_metalbox = AnchoredOffsetbox(loc=2, child=metalbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.8),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_metalbox)

                taubox = TextArea(r"$\tau$ = " + "{:.2f}".format(float(best_tau)) + r" $\pm$ " + "{:.2f}".format(float(best_tau_err)) + " Gyr",
                 textprops=dict(color='r', size=8))
                anc_taubox = AnchoredOffsetbox(loc=2, child=taubox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.85),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_taubox)

                #tauvbox = TextArea(r"$A_v$ = " + "{:.2f}".format(float(best_tauv)) + r" $\pm$ " + "{:.2f}".format(float(best_tauv_err)),
                # textprops=dict(color='r', size=8))
                #anc_tauvbox = AnchoredOffsetbox(loc=2, child=tauvbox, pad=0.0, frameon=False,\
                #                                     bbox_to_anchor=(0.03, 0.75),\
                #                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                #ax1.add_artist(anc_tauvbox)

                # Plot the best fit spectrum
                ax1.plot(lam_grid_tofit, alpha * currentspec, color='r')
                ax1.set_xlim(3000, 6000)
                ax1.xaxis.set_ticklabels([])
                ax1.yaxis.set_tick_params(labelsize=9)

                ax1.minorticks_on()
                ax1.tick_params('both', width=1, length=3, which='minor')
                ax1.tick_params('both', width=1, length=4.7, which='major')
                
                # Plot the residual
                ax2.plot(lam_grid_tofit, flam - alpha * currentspec, '-', color='r', drawstyle='steps')

                ax2.set_ylim(-1e-18, 1e-18)
                ax2.set_xlim(3000, 6000)
                ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
                ax2.xaxis.set_tick_params(labelsize=10)

                ax2.axhline(y=0.0, color='k', linestyle='--')
                ax2.grid(True)
                
                ax2.minorticks_on()
                ax2.tick_params('both', width=1, length=3, which='minor')
                ax2.tick_params('both', width=1, length=4.7, which='major')

        del best_age, best_metal, best_tau, best_tauv

        #### MILES ####
        best_age = stats.mode(ages_miles[count])[0]
        best_metal = stats.mode(metals_miles[count])[0]

        best_age_err = np.std(ages_miles[count]) * 10**best_age / (1e9 * 0.434)
        best_metal_err = np.std(metals_miles[count])

        for j in range(miles_extens):
            if np.allclose(miles_params[j], np.array([best_age, best_metal]).reshape(2)):
                currentspec = miles_spec[j+1].data

                alpha = np.sum(flam * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)

                # Plot best fit parameters as anchored text boxes
                labelbox = TextArea("MILES", textprops=dict(color='b', size=8))
                anc_labelbox = AnchoredOffsetbox(loc=2, child=labelbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.28, 0.95),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_labelbox)

                agebox = TextArea(r"$t$ = " + "{:.2f}".format(float(10**best_age/1e9)) + r" $\pm$ " + "{:.2f}".format(float(best_age_err)) + " Gyr",
                 textprops=dict(color='b', size=8))
                anc_agebox = AnchoredOffsetbox(loc=2, child=agebox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.28, 0.9),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_agebox)

                metalbox = TextArea("Z = " + "{:.2f}".format(float(best_metal)) + r" $\pm$ " + "{:.2f}".format(float(best_metal_err)),
                 textprops=dict(color='b', size=8))
                anc_metalbox = AnchoredOffsetbox(loc=2, child=metalbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.28, 0.85),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_metalbox)

                # Plot the best fit spectrum
                ax1.plot(lam_grid_tofit, alpha * currentspec, color='b')
                ax1.set_xlim(3000, 6000)
                ax1.xaxis.set_ticklabels([])
                ax1.yaxis.set_tick_params(labelsize=9)

                ax1.minorticks_on()
                ax1.tick_params('both', width=1, length=3, which='minor')
                ax1.tick_params('both', width=1, length=4.7, which='major')
                
                # Plot the residual
                ax2.plot(lam_grid_tofit, flam - alpha * currentspec, '-', color='b', drawstyle='steps')
                
                ax2.set_ylim(-1e-18, 1e-18)
                ax2.set_xlim(3000, 6000)
                ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
                ax2.xaxis.set_tick_params(labelsize=10)

                ax2.axhline(y=0.0, color='k', linestyle='--')
                ax2.grid(True)
                
                ax2.minorticks_on()
                ax2.tick_params('both', width=1, length=3, which='minor')
                ax2.tick_params('both', width=1, length=4.7, which='major')

                #break

        del best_age, best_metal

        #### FSPS ####
        best_age = stats.mode(ages_fsps[count])[0]
        best_metal = stats.mode(metals_fsps[count])[0]
        best_tau = 10**stats.mode(logtau_fsps[count])[0]
        best_mass_wht_age = stats.mode(mass_wht_ages_fsps[count])[0]

        best_age_err = np.std(ages_fsps[count]) * 10**best_age / (1e9 * 0.434)
        best_metal_err = np.std(metals_fsps[count])
        best_tau_err = np.std(logtau_fsps[count]) * best_tau / 0.434
        best_mass_wht_age_err = np.std(mass_wht_ages_fsps[count]) * 10**best_mass_wht_age / (1e9 * 0.434)

        print 'fsps', best_age, best_age_err, best_tau, best_tau_err, best_mass_wht_age, best_mass_wht_age_err, avgmass[column]

        for j in range(fsps_extens):
            if np.allclose(fsps_params[j], np.array([best_age, best_metal, best_tau]).reshape(3)):
                currentspec = fsps_spec[j+1].data

                alpha = np.sum(flam * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)

                # Plot best fit parameters as anchored text boxes
                labelbox = TextArea("FSPS", textprops=dict(color='g', size=8))
                anc_labelbox = AnchoredOffsetbox(loc=2, child=labelbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.53, 0.95),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_labelbox)

                agebox = TextArea(r"$\left<t\right>_M$ = " + "{:.2f}".format(float(10**best_mass_wht_age/1e9)) + r" $\pm$ " + "{:.2f}".format(float(best_mass_wht_age_err)) + " Gyr",
                 textprops=dict(color='g', size=8))
                anc_agebox = AnchoredOffsetbox(loc=2, child=agebox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.53, 0.9),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_agebox)

                taubox = TextArea(r"$\tau$ = " + "{:.2f}".format(float(best_tau)) + r" $\pm$ " + "{:.2f}".format(float(best_tau_err)) + " Gyr",
                 textprops=dict(color='g', size=8))
                anc_taubox = AnchoredOffsetbox(loc=2, child=taubox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.53, 0.85),\
                                                     bbox_transform=ax1.transAxes, borderpad=0.0)
                ax1.add_artist(anc_taubox)

                # Plot the best fit spectrum
                ax1.plot(lam_grid_tofit, alpha * currentspec, color='g')
                ax1.set_xlim(3000, 6000)
                ax1.xaxis.set_ticklabels([])
                ax1.yaxis.set_tick_params(labelsize=9)

                ax1.minorticks_on()
                ax1.tick_params('both', width=1, length=3, which='minor')
                ax1.tick_params('both', width=1, length=4.7, which='major')
                
                # Plot the residual
                ax2.plot(lam_grid_tofit, flam - alpha * currentspec, '-', color='g', drawstyle='steps')
                
                ax2.set_ylim(-1e-18, 1e-18)
                ax2.set_xlim(3000, 6000)
                ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
                ax2.xaxis.set_tick_params(labelsize=10)

                ax2.get_yaxis().set_ticklabels(['-1', '-0.5', '0.0', '0.5', ''], fontsize=8, rotation=45)

                ax2.axhline(y=0.0, color='k', linestyle='--')
                ax2.grid(True)

                ax2.minorticks_on()
                ax2.tick_params('both', width=1, length=3, which='minor')
                ax2.tick_params('both', width=1, length=4.7, which='major')

                # Get the residual tick label to show that the axis ticks are multiplied by 10^-18 
                resid_labelbox = TextArea(r"$\times 1 \times 10^{-18}$", textprops=dict(color='k', size=8))
                anc_resid_labelbox = AnchoredOffsetbox(loc=2, child=resid_labelbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.01, 0.98),\
                                                     bbox_transform=ax2.transAxes, borderpad=0.0)
                ax2.add_artist(anc_resid_labelbox)

        del best_age, best_metal, best_tau

        count += 1
        pdf.savefig(bbox_inches='tight')
        
    pdf.close()

    # total run time
    print "Total time taken --", time.time() - start, "seconds."
    sys.exit(0)
