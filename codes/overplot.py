from __future__ import division

import numpy as np
from astropy.io import fits
from scipy import stats

import sys
import os
import datetime
import time

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from fast_chi2_jackknife import get_total_extensions

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figures_dir = stacking_analysis_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"

sns.set()
sns.set_style("white")

def makefig():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\lambda\ [\AA]$')
    ax.set_ylabel('$f_{\lambda}\ [\mathrm{erg/s/cm^2/\AA}] $')
    ax.axhline(y=0,linestyle='--')

    return fig, ax

def plot_spectrum_data(lam, flux, flux_err, ax):
    
    ax.errorbar(lam, flux, yerr=flux_err, fmt='o-', color='k', linewidth=2, ecolor='r', markeredgecolor='k', capsize=0, markersize=4)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(3000, 6000)
    #ax.set_ylim(0,2)

def plot_spectrum_model(lam, flux, col, bestparams, ax):

    ax.plot(lam, flux, 'o-', color=col, linewidth=3, label=bestparams)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)
    ax.legend(loc=0, numpoints=1, prop={'size':12})

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Read in stacks
    stacks = fits.open(home + '/Desktop/FIGS/new_codes/coadded_coarsegrid_PEARSgrismspectra.fits')
    totalstacks = get_total_extensions(stacks)

    # Set up lambda grid
    lam_step = 100
    lam_lowfit = 3600
    lam_highfit = 6500
    lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)

    # Create pdf file to plot figures in
    pdfname = figures_dir + 'overplot_all_sps.pdf'
    pdf = PdfPages(pdfname)

    # Read ongrid values
    # It needs to be read only once because it is the same for all files
    ongrid_vals = np.loadtxt(stacking_analysis_dir + 'jackknife_ages_bc03.txt', dtype=np.str, usecols=range(0,1,1))

    # Read files with params from jackknife runs
    #### BC03 ####
    ages_bc03 = np.loadtxt(stacking_analysis_dir + 'jackknife_ages_bc03.txt', usecols=range(1, int(1e4) + 1))
    metals_bc03 = np.loadtxt(stacking_analysis_dir + 'jackknife_metals_bc03.txt', usecols=range(1, int(1e4) + 1))
    logtau_bc03 = np.loadtxt(stacking_analysis_dir + 'jackknife_logtau_bc03.txt', usecols=range(1, int(1e4) + 1))
    tauv_bc03 = np.loadtxt(stacking_analysis_dir + 'jackknife_tauv_bc03.txt', usecols=range(1, int(1e4) + 1))

    #### MILES ####
    ages_miles = np.loadtxt(stacking_analysis_dir + 'jackknife_ages_miles.txt', usecols=range(1, int(1e4) + 1))
    metals_miles = np.loadtxt(stacking_analysis_dir + 'jackknife_metals_miles.txt', usecols=range(1, int(1e4) + 1))

    #### FSPS ####
    ages_fsps = np.loadtxt(stacking_analysis_dir + 'jackknife_ages_fsps.txt', usecols=range(1, int(1e4) + 1))
    metals_fsps = np.loadtxt(stacking_analysis_dir + 'jackknife_metals_fsps.txt', usecols=range(1, int(1e4) + 1))
    logtau_fsps = np.loadtxt(stacking_analysis_dir + 'jackknife_logtau_fsps.txt', usecols=range(1, int(1e4) + 1))

    # Open fits files with comparison spectra
    bc03_spec = fits.open(home + '/Desktop/FIGS/new_codes/all_comp_spectra_bc03.fits', memmap=False)
    miles_spec = fits.open(home + '/Desktop/FIGS/new_codes/all_comp_spectra_miles.fits', memmap=False)
    fsps_spec = fits.open(home + '/Desktop/FIGS/new_codes/all_comp_spectra_fsps.fits', memmap=False)

    # Find number of extensions in each
    bc03_extens = get_total_extensions(bc03_spec)
    miles_extens = get_total_extensions(miles_spec)
    fsps_extens = get_total_extensions(fsps_spec)

    # Make multi-dimensional arrays of parameters for all SPS libraries 
    # This is done to make access to the fits headers easier because there is no way to directly search the headers for all extensions in a fits file simultaneously.
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
        ongrid = stacks[stackcount + 2].header["ONGRID"]
        numspec = int(stacks[stackcount + 2].header["NUMSPEC"])

        fig, ax = makefig()

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

        for j in range(bc03_extens):
            if np.allclose(bc03_params[j], np.array([best_age, best_metal, best_tau, best_tauv]).reshape(4)):
                currentspec = bc03_spec[j+1].data
                alpha = np.sum(flam * currentspec / ferr**2) / np.sum(currentspec**2 / ferr**2)
                plot_spectrum_model(lam_grid_tofit, alpha*bc03_spec[j+1].data, 'r', ax)

        del best_age, best_metal, best_tau, best_tauv

        #### MILES ####
        best_age = stats.mode(ages_miles[count])[0]
        best_metal = stats.mode(metals_miles[count])[0]

        for j in range(miles_extens):
            if np.allclose(miles_params[j], np.array([best_age, best_metal]).reshape(2)):
                plot_spectrum_model(lam_grid_tofit, miles_spec[j+1].data, 'b', ax)

        del best_age, best_metal

        #### FSPS ####
        best_age = stats.mode(ages_fsps[count])[0]
        best_metal = stats.mode(metals_fsps[count])[0]
        best_tau = 10**stats.mode(logtau_fsps[count])[0]

        for j in range(fsps_extens):
            if np.allclose(fsps_params[j], np.array([best_age, best_metal, best_tau]).reshape(3)):
                plot_spectrum_model(lam_grid_tofit, fsps_spec[j+1].data, 'g', ax)

        del best_age, best_metal, best_tau

        sys.exit(0)



    