from astropy.io import fits

import fsps

import sys
import glob
import os

def resample(lam, spec, lam_grid_tofit, lam_step, total_ages):
    
    lam_em = lam
    resampled_flam = np.zeros((total_ages, len(lam_grid_tofit)))

    for i in range(len(lam_grid_tofit)):
        new_ind = np.where((lam_em >= lam_grid_tofit[i] - lam_step/2) & (lam_em < lam_grid_tofit[i] + lam_step/2))[0]
        resampled_flam[:,i] = np.median(spec[:,[new_ind]], axis=2).reshape(total_ages)

    return resampled_flam

def normalize(spec):

    return np.mean(spec, axis=1)

def create_miles_lib_main():
    """
    Creates a consolidated fits file from the 1900 or so individual fits file that
    were available for the MILES SPS models.

    This is done to decrease file I/O time by only reading in one large fits file
    during the program that fits models.
    """

    milesdir = os.getenv('HOME') + '/Documents/MILES_BaSTI_ku_1.30_fits/'

    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)

    for file in glob.glob(milesdir + "*.fits"):

        h = fits.open(file)

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

        hdulist.append(fits.ImageHDU(data = h[0].data, header = hdr))

    hdulist.writeto(stacking_maindir + 'all_comp_spectra_miles.fits', clobber = True)

    return

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
    sps.params['dust_type'] = 2  # Calzetti attenuation curve
    sps.params['dust2'] = -0.7  # Not entirely sure how it fits into Calzetti's law but I think they call it the dust exponent for Calzetti law. See Conroy et al. 2009.
    sps.params['sfh'] = 1  # Tau model SFH
    # This exponential decaying SFH in here is a five param model where the parameters are -
    # tau -- this is the time scale for decay; in Gyr; default 1
    # const -- this is fraction of const SF in the SFH; default 0
    # fburst -- fraction of mass formed in an instantaneous burst; default 0
    # tburst -- the age of the universe when SF starts; default 11 Gyr
    # I think the fifth parameter is that the SF will produce a total of 1 solar mass over the total SFH.

    # Loop over parameter space and generate model spectra 
    count = 0
    for metals in metallicities:
        for logtau in logtauarr:
            tau = 10**(logtau)
            sps.params['tau'] = tau
            sps.params['logzsol'] = metals
            lam, spec = sps.get_spectrum(peraa=True)
            print lam, spec
            count += 1
            if count == 4: break

    return

if __name__ == "__main__":

    stacking_maindir = '/Users/baj/Desktop/FIGS/stacking-analysis-pears/'

    #create_miles_lib_main()
    create_fsps_lib_main()
    sys.exit(0)









