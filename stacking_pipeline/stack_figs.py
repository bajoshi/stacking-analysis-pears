import numpy as np
from astropy.io import fits

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
figs_dir = home + "/Documents/pears_figs_data/"
stacking_utils = home + "/Documents/GitHub/stacking-analysis-pears/util_codes/"

sys.path.append(stacking_utils)
from get_total_extensions import get_total_extensions

def gen_figs_matches():

    # Read in catalog from Santini et al.
    names_header=['id', 'ra', 'dec', 'zbest', 'zphot', 'zphot_l68', 'zphot_u68', 'Mmed', 'smed', 'Mdeltau']
    santini_cat = np.genfromtxt(home + '/Documents/GitHub/massive-galaxies/santini_candels_cat.txt',\
                  names=names_header, usecols=(0,1,2,9,13,14,15,19,20,40), skip_header=187)

    # Set mathcing tolerances
    ra_tol = 0.3 / 3600  # arcseconds expressed in deg
    dec_tol = 0.3 / 3600  # arcseconds expressed in deg

    search_ra = santini_cat['ra']
    search_dec = santini_cat['dec']

    # Read in FIGS catalogs
    # Only GOODS-S for now because 
    # Santini et al only have GOODS-S masses
    figs_gs1 = fits.open(figs_dir + 'GS1_G102_2.combSPC.fits')
    figs_gs2 = fits.open(figs_dir + 'GS2_G102_2.combSPC.fits')

    # Now try and match every figs object to
    # the candels catalog and then stack.
    # Empty file for saving results
    # fh = open(figs_dir + 'figs_candels_goodss_matches.txt', 'w')
    # Write header
    # 

    checkplot = True
    for figs_cat in [figs_gs1, figs_gs2]:

        nobj = get_total_extensions(figs_cat)
        print("# extensions:", nobj)

        for i in range(1, nobj+1):

            print(figs_cat[i].header['EXTNAME'])

            wav = figs_cat[i].data['LAMBDA']
            avg_wflux = figs_cat[i].data['AVG_WFLUX']
            std_wflux = figs_cat[i].data['STD_WFLUX']

            # Get SNR

            # PLot to check
            if checkplot:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            
                ax.plot(wav, avg_wflux, color='k')
                ax.fill_between(wav, avg_wflux - std_wflux, avg_wflux + std_wflux, color='gray')

                plt.show()

            # Now match
            #obj_ra = 
            #obj_dec =

            if i > 5: sys.exit(0)

    # fh.close()

    return None

def main():

    gen_figs_matches()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)