from __future__ import division

from astropy.io import fits
import numpy as np
import fsps

import matplotlib.pyplot as plt

import sys
import glob
import os

from fast_chi2_jackknife import get_total_extensions

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figures_dir = stacking_analysis_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"

def resample_single(lam, spec, lam_grid_tofit, lam_step):
    
    lam_em = lam
    resampled_flam = np.zeros((len(lam_grid_tofit)))

    for i in range(len(lam_grid_tofit)):
        new_ind = np.where((lam_em >= lam_grid_tofit[i] - lam_step/2) & (lam_em < lam_grid_tofit[i] + lam_step/2))[0]
        resampled_flam[i] = np.median(spec[new_ind])

    return resampled_flam

def rescale_single(lam, spec):

    arg4400 = np.argmin(abs(lam - 4400))
    arg4600 = np.argmin(abs(lam - 4600))

    medval = np.median(spec[arg4400:arg4600+1])

    return medval

def resample(lam, spec, lam_grid_tofit, lam_step, total_ages):
    
    lam_em = lam
    resampled_flam = np.zeros((total_ages, len(lam_grid_tofit)))

    for i in range(len(lam_grid_tofit)):
        new_ind = np.where((lam_em >= lam_grid_tofit[i] - lam_step/2) & (lam_em < lam_grid_tofit[i] + lam_step/2))[0]
        resampled_flam[:,i] = np.median(spec[:,[new_ind]], axis=2).reshape(total_ages)

    return resampled_flam

def rescale(lam, spec, total_ages):
    
    arg4400 = np.argmin(abs(lam - 4400))
    arg4600 = np.argmin(abs(lam - 4600))

    medval = np.median(spec[:, arg4400:arg4600+1], axis=1)
    medval = medval.reshape(total_ages, 1)

    return medval

def plotspec(lam, flux, ax, col):

    ax.plot(lam, flux, '-', color=col, linewidth=2)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(3000, 6500)

    return None

def makefig():

    fig = plt.figure()
    ax = fig.add_subplot(111)

    return fig, ax

def create_miles_lib_main():
    """
    Creates a consolidated fits file from the 1900 or so individual fits file that
    were available for the MILES SPS models.
    It will also resample and rescale each individual model spectrum in the exact same way as done for the data stacks.

    This is done to decrease file I/O time by only reading in one large fits file
    during the program that fits models.
    """

    milesdir = os.getenv('HOME') + '/Documents/MILES_BaSTI_ku_1.30_fits/'

    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)

    currentlam = np.arange(3540.5, 7409.6 + 0.9, 0.9)
    # those numbers come from the start and final wavelengths in the header of each fits file. # length of lam array = 4300

    for file in glob.glob(milesdir + "*.fits"):

        h = fits.open(file)      

        currentspec = h[0].data
        currentspec = resample_single(currentlam, currentspec, lam_grid_tofit, lam_step)

        #currentspec = currentspec / rescale_single(lam_grid_tofit, currentspec)

        hdr = fits.Header()
        # I included my header this way for better read access when fitting the models.
        # The original fits files, however, have a copyright notice, disclaimer,
        # and acknowledgement request in them which should be looked at carefully.

        # Keep in mind that the floats in any header keyword which have spaces in them 
        # can directly be passed to float() without have to strip any leading or trailing spaces.

        hdr['Strt_Wav'] = h[0].header['COMMENT'][48].split(':')[-1].split('\'')[0].split('(')[0] + "Angstrom"
        hdr['Fin_Wav'] = h[0].header['COMMENT'][49].split(':')[-1].split('\'')[0].split('(')[0] + "Angstrom"
        hdr['Spec_smp'] = h[0].header['COMMENT'][50].split(':')[-1].split('\'')[0].split('(')[0] + "Angstrom/pixel"
        hdr['Spec_res'] = h[0].header['COMMENT'][52].split(':')[-1].split('\'')[0].split('(')[0] + "Angstrom FWHM"
        hdr['Redshift'] = h[0].header['COMMENT'][53].split(':')[-1].split('\'')[0].lstrip(' ')
        hdr['IMF_Slp'] = h[0].header['COMMENT'][39].split(':')[-1].split('\'')[0].lstrip(' ').replace('ku','Kroupa Universal')
        hdr['Age_Gyr'] = h[0].header['COMMENT'][40].split(':')[-1].split('\'')[0].lstrip(' ')
        hdr['M2H'] = h[0].header['COMMENT'][41].split(':')[-1].split('\'')[0].lstrip(' ')
        hdr['alp2Fe'] = h[0].header['COMMENT'][42].split(':')[-1].split('\'')[0].lstrip(' ')
        hdr['Isochron'] = h[0].header['COMMENT'][43].split(':')[-1].split('\'')[0].lstrip(' ')

        t = np.log10(float(h[0].header['COMMENT'][40].split(':')[-1].split('\'')[0].lstrip(' ')) * 1e9)
        m2h = h[0].header['COMMENT'][41].split(':')[-1].split('\'')[0].lstrip(' ')
        z = 10**(float(m2h) + np.log10(2 / 73)) * 0.73
        # In this conversion from m2h to Z I have assumed that the SSP has a hydrogen content that is the same as the sun.
        # They don't give me this number so I have no idea what it actually is. 

        hdr['LOG_AGE'] = str(t)
        hdr['METAL'] = str(z)

        hdulist.append(fits.ImageHDU(data=currentspec, header=hdr))

    hdulist.writeto(savefits_dir + 'all_comp_spectra_miles.fits', clobber=True)

    return None

def create_fsps_lib_main():
    """
    Creates a consolidated fits file for all the FSPS models.

    This is done to decrease file I/O time by only reading in one large fits file
    during the program that fits models.    
    """

    # Parameter array that I want the models for -
    logtauarr = np.arange(-2, 2, 0.2)
    metallicities = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])

    # Create a stellar pop with interpolation in metallicity enabled and the rest set to default values
    sps = fsps.StellarPopulation(zcontinuous=1)

    # Set params for stellar pop
    sps.params['imf_type'] = 2  # Kroupa IMF
    #sps.params['dust_type'] = 2  # Calzetti attenuation curve
    #sps.params['dust2'] = -0.7  # Not entirely sure how it fits into Calzetti's law but I think they call it the dust exponent for Calzetti law. See Conroy et al. 2009.
    sps.params['sfh'] = 1  # Tau model SFH
    # This exponential decaying SFH in here is a five param model where the parameters are -
    # tau -- this is the time scale for decay; in Gyr; default 1
    # const -- this is fraction of const SF in the SFH; default 0
    # fburst -- fraction of mass formed in an instantaneous burst; default 0
    # tburst -- the age of the universe when SF starts; default 11 Gyr
    # I think the fifth parameter is that the SF will produce a total of 1 solar mass over the total SFH.

    # Create fits file for saving all consolidated spectra
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)    

    # Loop over parameter space and generate model spectra 
    count = 0
    for metals in metallicities:
        for logtau in logtauarr:
            tau = 10**(logtau)
            sps.params['tau'] = tau  # it wants this in Gyr
            sps.params['logzsol'] = np.log10(metals / 0.02)  # it wants this in log(Z/Z_sol)
            currentlam, currentspec = sps.get_spectrum(peraa=True)
            log_ages = sps.log_age

            currentspec = resample(currentlam, currentspec, lam_grid_tofit, lam_step, len(log_ages))
            #currentlam = lam_grid_tofit

            #medvals = rescale(lam_grid_tofit, currentspec, len(log_ages))
            #currentspec = np.divide(currentspec, medvals)
            
            for j in range(len(currentspec)):
                hdr = fits.Header()
                hdr['LOG_AGE'] = str(log_ages[j])
                hdr['METAL'] = str(metals)
                hdr['TAU_GYR'] = str(tau)

                dat = currentspec[j]
                hdulist.append(fits.ImageHDU(data=dat, header=hdr))

    hdulist.writeto(savefits_dir + 'all_comp_spectra_fsps.fits', clobber=True)

    return None

if __name__ == "__main__":

    lam_step = 100
    lam_lowfit = 3600
    lam_highfit = 6000
    lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)

    #create_miles_lib_main()
    create_fsps_lib_main()
    sys.exit(0)

    ### Block used for testing model changes below --
    """
    miles_spec_orig = fits.open('/Users/baj/Desktop/FIGS/new_codes/all_comp_spectra_miles_nochanges.fits', memmap=False)
    fsps_spec_orig = fits.open('/Users/baj/Desktop/FIGS/new_codes/all_comp_spectra_fsps_nochanges.fits', memmap=False)

    miles_spec = fits.open('/Users/baj/Desktop/FIGS/new_codes/all_comp_spectra_miles.fits', memmap=False)
    fsps_spec = fits.open('/Users/baj/Desktop/FIGS/new_codes/all_comp_spectra_fsps.fits', memmap=False)

    miles_extens = get_total_extensions(miles_spec)
    fsps_extens = get_total_extensions(fsps_spec)

    fig1, ax1 = makefig()
    fig2, ax2 = makefig()

    miles_lam = np.arange(3540.5, 7409.6 + 0.9, 0.9)

    count = 0
    for spec in range(miles_extens):

        orig_spec = miles_spec_orig[spec+1].data
        plotspec(miles_lam, orig_spec, ax1, 'k')

        new_spec = miles_spec[spec+1].data 
        plotspec(lam_grid_tofit, new_spec, ax1, 'r')

        count += 1
        if count == 5: break

    count = 0
    for spec in range(fsps_extens):

        fsps_lam = fsps_spec_orig[spec+1].data[1]
        orig_spec = fsps_spec_orig[spec+1].data[0]
        plotspec(fsps_lam, orig_spec, ax2, 'k')

        new_spec = fsps_spec[spec+1].data[0]
        plotspec(lam_grid_tofit, new_spec, ax2, 'r')

        count += 1
        if count == 5: break

    plt.show()

    sys.exit(0)
    """


