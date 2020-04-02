from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
figs_dir = home + '/Desktop/FIGS/'

massive_galaxies_dir = figs_dir + "massive-galaxies/"
stacking_analysis_dir = figs_dir + "stacking-analysis-pears/"

def get_figs_cats():
    """
    For all FIGS fields: GN1, GN2, GS1, GS2
    Will return FIGS ID, RA, DEC, 
    F105W flux and err,
    F125W flux and err,
    F160W flux and err.
    """

    # Read in FIGS catalogs # latest version v1.2
    # All fluxes and flux errors are in nJy
    gn1cat = np.genfromtxt(massive_galaxies_dir + 'GN1_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec','f105w_flux','f105w_ferr','f125w_flux','f125w_ferr','f160w_flux','f160w_ferr'], \
                           usecols=([2,3,4,17,18,19,20,23,24]), skip_header=25)
    gn2cat = np.genfromtxt(massive_galaxies_dir + 'GN2_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec','f105w_flux','f105w_ferr','f125w_flux','f125w_ferr','f160w_flux','f160w_ferr'], \
                           usecols=([2,3,4,17,18,19,20,23,24]), skip_header=25)

    gs1cat = np.genfromtxt(massive_galaxies_dir + 'GS1_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec','f105w_flux','f105w_ferr','f125w_flux','f125w_ferr','f160w_flux','f160w_ferr'], \
                           usecols=([2,3,4,17,18,19,20,23,24]), skip_header=25)
    gs2cat = np.genfromtxt(massive_galaxies_dir + 'GS2_prelim_science_v1.2.cat', dtype=None,\
                           names=['id','ra','dec','f105w_flux','f105w_ferr','f125w_flux','f125w_ferr','f160w_flux','f160w_ferr'], \
                           usecols=([2,3,4,13,14,15,16,17,18]), skip_header=19)  # GS2 has fewer photometric measurements

    return gn1cat, gn2cat, gs1cat, gs2cat

def main():

    # Read in FIGS catalogs
    gn1cat, gn2cat, gs1cat, gs2cat = get_figs_cats()

    # Read in PEARS results
    pearscat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    # Now match!
    for i in range(len(pearscat)):

        # find PEARS ra, dec
        current_pears_ra = float(pearscat['RA'][i])
        current_pears_dec = float(pearscat['DEC'][i])

        # Matching radius # arcseconds in degrees
        ra_lim = 0.3/3600
        dec_lim = 0.3/3600

        # ------------------------- Find FIGS idx ------------------------- #
        if current_pears_dec > 0.0:
            figs_gn1_idx = np.where((gn1cat['ra'] >= current_pears_ra - ra_lim) & (gn1cat['ra'] <= current_pears_ra + ra_lim) & \
                (gn1cat['dec'] >= current_pears_dec - dec_lim) & (gn1cat['dec'] <= current_pears_dec + dec_lim))[0]
            figs_gn2_idx = np.where((gn2cat['ra'] >= current_pears_ra - ra_lim) & (gn2cat['ra'] <= current_pears_ra + ra_lim) & \
                (gn2cat['dec'] >= current_pears_dec - dec_lim) & (gn2cat['dec'] <= current_pears_dec + dec_lim))[0]

            if figs_gn1_idx.size:
                print(figs_gn1_idx)
                sys.exit(0)

            #if (len(figs_gn1_idx) > 1):
            #    print("Multiple matches GN1:", figs_gn1_idx)
            #if (len(figs_gn2_idx) > 1):
            #    print("Multiple matches GN2:", figs_gn2_idx)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)