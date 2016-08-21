from __future__ import division
from astropy.io import fits
import numpy as np

import glob, os, sys

def resample(lam, spec, lam_grid_tofit, lam_step, total_ages):
    
    lam_em = lam
    resampled_flam = np.zeros((total_ages, len(lam_grid_tofit)))

    for i in range(len(lam_grid_tofit)):
        new_ind = np.where((lam_em >= lam_grid_tofit[i] - lam_step/2) & (lam_em < lam_grid_tofit[i] + lam_step/2))[0]
        resampled_flam[:,i] = np.median(spec[:,[new_ind]], axis=2).reshape(total_ages)

    return resampled_flam
    
def normalize(spec):

    return np.mean(spec, axis=1)

if __name__ == '__main__':

    sspdir = '/Users/baj/Documents/GALAXEV_BC03/bc03/models/Padova1994/salpeter/'
    cspout = '/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/'
    metals = ['m22','m32','m42','m52','m62','m72']

    # lam grid
    lam_step = 100
    lam_lowfit = 2500
    lam_highfit = 6500
    lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)

    # find age indices to be used and total number of ages required
    example = fits.open('/Users/baj/Documents/GALAXEV_BC03/bc03/models/Padova1994/salpeter/bc2003_hr_m22_salp_ssp.fits')
    ages = example[2].data
    age_ind = np.where((ages/1e9 < 8) & (ages/1e9 > 0.1))[0]
    total_ages = int(len(age_ind)) # there are 57 ages here for SSP models # they have 221 ages in total

    # FITS file where the reduced number of spectra will be saved
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)

    # loop over all fits files and save them
    for filename in glob.glob(sspdir + '*.fits'):

        print filename
        h = fits.open(filename, memmap=False)
        metallicity = os.path.basename(filename).split('_')[2]
        currentlam = h[1].data
        
        currentspec = np.zeros([total_ages, len(currentlam)], dtype=np.float64)
        for i in range(total_ages):
            # this loop simply fills in all of the currentspec array
            currentspec[i] = h[age_ind[i]+3].data

        # Only consider the part of BC03 spectrum between 2500 to 6500
        arg2500 = np.argmin(abs(currentlam - 2500))
        arg6500 = np.argmin(abs(currentlam - 6500))
            
        currentlam = currentlam[arg2500:arg6500+1] # chopping off unrequired lambda
        currentspec = currentspec[:,arg2500:arg6500+1] # chopping off unrequired spectrum
        currentspec = resample(currentlam, currentspec, lam_grid_tofit, lam_step, total_ages)
        currentlam = lam_grid_tofit
            
        meanvals = normalize(currentspec)
        meanvals = meanvals.reshape(total_ages,1)
        currentspec = np.divide(currentspec, meanvals)
            
        for i in range(total_ages):
            hdr = fits.Header()
            hdr['LOG_AGE'] = str(np.log10(ages[age_ind[i]]))
            
            if metallicity == 'm22':
                metal_val = 0.0001
            elif metallicity == 'm32':
                metal_val = 0.0004
            elif metallicity == 'm42':
                metal_val = 0.004
            elif metallicity == 'm52':
                metal_val = 0.008
            elif metallicity == 'm62':
                metal_val = 0.02
            elif metallicity == 'm72':
                metal_val = 0.05
        
            hdr['METAL'] = str(metal_val)
            hdulist.append(fits.ImageHDU(data = currentspec[i], header=hdr))

    hdulist.writeto('all_ssp_comp_spectra.fits', clobber=True)











