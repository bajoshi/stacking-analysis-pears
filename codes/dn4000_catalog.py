from __future__ import division

import numpy as np
from astropy.io import fits

import sys
import os
import glob

import matplotlib.pyplot as plt

import grid_coadd as gd

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
pears_spectra_dir = home + "/Documents/PEARS/data_spectra_only/"

"""
I think this code should also have the same contamination tolerances that grid_coadd has.
"""

def dn4000(lam, spec, spec_err):
    """
        Make sure the supplied lambda is in angstroms and the spectrum is in f_lambda.
    """

    arg3850 = np.argmin(abs(lam - 3850))
    arg3950 = np.argmin(abs(lam - 3950))
    arg4000 = np.argmin(abs(lam - 4000))
    arg4100 = np.argmin(abs(lam - 4100))

    fnu_plus = spec[arg4000:arg4100+1] * lam[arg4000:arg4100+1]**2 / 2.99792458e10
    fnu_minus = spec[arg3850:arg3950+1] * lam[arg3850:arg3950+1]**2 / 2.99792458e10

    dn4000 = np.trapz(fnu_plus, x=lam[arg4000:arg4100+1]) / np.trapz(fnu_minus, x=lam[arg3850:arg3950+1])

    delta_lam = 100
    spec_nu_err = spec_err * lam**2 / 2.99792458e10
    flux_nu_err_sqr = spec_nu_err**2
    sum_up_err = np.sqrt(delta_lam**2 * (sum(4 * flux_nu_err_sqr[arg4000+1:arg4100+1]) + flux_nu_err_sqr[arg4000] + flux_nu_err_sqr[arg4100]))
    sum_low_err = np.sqrt(delta_lam**2 * (sum(4 * flux_nu_err_sqr[arg3850+1:arg3950+1]) + flux_nu_err_sqr[arg3850] + flux_nu_err_sqr[arg3950]))
    sum_low = np.trapz(fnu_minus, x=lam[arg3850:arg3950+1])
    sum_up = np.trapz(fnu_plus, x=lam[arg4000:arg4100+1])
    dn4000_err = (1/sum_low**2) * np.sqrt(sum_up_err**2 * sum_low**2 + sum_up**2 * sum_low_err**2)
    
    return dn4000, dn4000_err

def d4000(lam, spec, spec_err):
    """
        Make sure the supplied lambda is in angstroms and the spectrum is in f_lambda.
    """

    arg3750 = np.argmin(abs(lam - 3750))
    arg3950 = np.argmin(abs(lam - 3950))
    arg4050 = np.argmin(abs(lam - 4050))
    arg4250 = np.argmin(abs(lam - 4250))

    fnu_plus = spec[arg4050:arg4250+1] * lam[arg4050:arg4250+1]**2 / 2.99792458e10
    fnu_minus = spec[arg3750:arg3950+1] * lam[arg3750:arg3950+1]**2 / 2.99792458e10

    d4000 = np.trapz(fnu_plus, x=lam[arg4050:arg4250+1]) / np.trapz(fnu_minus, x=lam[arg3750:arg3950+1])

    delta_lam = 100
    spec_nu_err = spec_err * lam**2 / 2.99792458e10
    flux_nu_err_sqr = spec_nu_err**2
    sum_up_err = np.sqrt(delta_lam**2 * (sum(4 * flux_nu_err_sqr[arg4050+1:arg4250+1]) + flux_nu_err_sqr[arg4050] + flux_nu_err_sqr[arg4250]))
    sum_low_err = np.sqrt(delta_lam**2 * (sum(4 * flux_nu_err_sqr[arg3750+1:arg3950+1]) + flux_nu_err_sqr[arg3750] + flux_nu_err_sqr[arg3950]))
    sum_low = np.trapz(fnu_minus, x=lam[arg3750:arg3950+1])
    sum_up = np.trapz(fnu_plus, x=lam[arg4050:arg4250+1])
    d4000_err = (1/sum_low**2) * np.sqrt(sum_up_err**2 * sum_low**2 + sum_up**2 * sum_low_err**2)
    
    return d4000, d4000_err

if __name__ == '__main__':
    
    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt',
                   dtype=None, names=True, skip_header=2)

    pears_id = cat['pearsid']
    photz = cat['threedzphot']

    dn4000_arr = np.zeros(len(pears_id))
    dn4000_err_arr = np.zeros(len(pears_id))
    d4000_arr = np.zeros(len(pears_id))
    d4000_err_arr = np.zeros(len(pears_id))
    pears_ra = np.zeros(len(pears_id))
    pears_dec = np.zeros(len(pears_id))

    for i in range(len(pears_id)):
        redshift = photz[i]
        lam_em, flam_em, ferr, specname = gd.fileprep(pears_id[i], redshift)

        fitsfile = fits.open(pears_spectra_dir + specname)
        pears_ra[i] = float(fitsfile[0].header['RA'])
        pears_dec[i] = float(fitsfile[0].header['DEC'])

        dn4000_arr[i], dn4000_err_arr[i] = dn4000(lam_em, flam_em, ferr)
        d4000_arr[i], d4000_err_arr[i] = d4000(lam_em, flam_em, ferr)

    data = np.array(zip(pears_id, photz, pears_ra, pears_dec, dn4000_arr, dn4000_err_arr, d4000_arr, d4000_err_arr),\
                dtype=[('pears_id', int), ('photz', float), ('pears_ra', float), ('pears_dec', float), ('dn4000_arr', float), ('dn4000_err_arr', float), ('d4000_arr', float), ('d4000_err_arr', float)])
    np.savetxt(stacking_analysis_dir + 'pears_4000break_catalog.txt', data, fmt=['%d', '%.3f', '%.6f', '%.6f', '%.4f', '%.4f', '%.4f', '%.4f'], delimiter=' ',\
               header='Catalog for all galaxies that matched between 3DHST and PEARS. \n' +
               'pears_id redshift ra dec dn4000 dn4000_err d4000 d4000_err')