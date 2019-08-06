from __future__ import division

import numpy as np

import os
import sys
import glob

import matplotlib.pyplot as plt

home = os.getenv('HOME')
figs_dir = home + '/Desktop/FIGS/'

massive_figures_dir = figs_dir + 'massive-galaxies-figures/'
full_pears_results_dir = massive_figures_dir + 'full_pears_results/'
stacking_figures_dir = figs_dir + 'stacking-analysis-figures/'

# Get correct directory for 3D-HST data
if 'firstlight' in os.uname()[1]:
    threedhst_datadir = '/Users/baj/Desktop/3dhst_data/'
else:
    threedhst_datadir = '/Volumes/Bhavins_backup/3dhst_data/'

def make_stellar_mass_hist():

    # Define empty list for stellar mass
    stellar_mass = []

    # Loop over all results and store stellar mass values
    for fl in glob.glob(full_pears_results_dir + 'redshift_fitting_results_*.txt'):
        f = np.genfromtxt(fl, dtype=None, names=True, skip_header=1)
        stellar_mass.append(f['zp_ms'])

    # Convert to numpy array to make histogram
    stellar_mass = np.asarray(stellar_mass)
    ms = np.log10(stellar_mass)  # plot log of stellar mass to make it easier to see

    # Now make plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\rm log(M_s)\ [M_\odot]$', fontsize=15)
    ax.set_ylabel(r'$\rm \# objects$', fontsize=15)

    binsize = 0.5
    total_bins = int((13.0 - 4.0)/binsize)

    ax.hist(ms, total_bins, range=(4.0, 13.0), histtype='step', linewidth=1.2, color='k')

    ax.minorticks_on()
    ax.set_xticks(np.arange(4.0, 13.5, 1.0))

    # Other info on plot
    num = len(stellar_mass)
    ax.text(0.04, 0.95, r'$\rm N\, =\, $' + str(num), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=15)
    
    fig.savefig(stacking_figures_dir + 'stellar_mass_hist.pdf', \
        dpi=300, bbox_inches='tight')

    return None

def compare_with_threedhst():

    # Define empty lists for id, field, ra, dec, 
    # and stellar mass from my code and from 3dhst
    pears_id = []
    pears_field = []
    pears_ra = []
    pears_dec = []
    pears_ms = []

    # Loop over all results and store values from my pipeline
    for fl in glob.glob(full_pears_results_dir + 'redshift_fitting_results_*.txt'):

        f = np.genfromtxt(fl, dtype=None, names=True, skip_header=1)

        current_id = f['PearsID']
        current_field = f['Field']
        current_ra = f['RA']
        current_dec = f['DEC']
        current_ms = f['zp_ms']

        pears_id.append(current_id)
        pears_field.append(current_field)
        pears_ra.append(current_ra)
        pears_dec.append(current_dec)
        pears_ms.append(current_ms)

    # Convert to numpy arrays
    pears_id = np.asarray(pears_id)
    pears_field = np.asarray(pears_field)
    pears_ra = np.asarray(pears_ra)
    pears_dec = np.asarray(pears_dec)
    pears_ms = np.asarray(pears_ms)

    pears_ms = np.log10(pears_ms)  # convert to log of stellar mass

    # Now match with 3D-HST and get their stellar mass values
    # Read in 3D-HST stellar mass and photometry catalogs
    # The photometry catalogs contain the ra and dec for matching
    # The 3D-HST v4.1 ID is a unique identifier which is common 
    # for both the photometry and the stellar mass catalogs.
    # ------------------------------- Read in photometry catalogs ------------------------------- #
    # GOODS photometry catalogs from 3DHST
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
    'f_U', 'e_U', 'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
    'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), \
        skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), \
        skip_header=3)

    # ------------------------------- Read in stellar mass catalogs ------------------------------- #
    # These are output from the FAST code (Kriek et al.)
    goodsn_ms_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.fout', dtype=None, names=True, skip_header=17)
    goodss_ms_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.fout', dtype=None, names=True, skip_header=17)

    # Match astrometry and get stellar mass from 3D-HST 
    threed_ms = match_get_threed_ms(goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, \
        goodsn_ms_cat_3dhst, goodss_ms_cat_3dhst, pears_id, pears_field, pears_ra, pears_dec)

    # Now plot both vs each other
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\rm log(M^{this\, work}_s)\ [M_\odot]$', fontsize=15)
    ax.set_ylabel(r'$\rm log(M^{3D-HST}_s)\ [M_\odot]$', fontsize=15)

    ax.scatter(pears_ms, threed_ms, s=1.0, color='k')
    ax.plot(np.arange(4.0, 13.1, 0.1), np.arange(4.0, 13.1, 0.1), '--', color='r')

    ax.set_xlim(4.0, 13.0)
    ax.set_ylim(4.0, 13.0)

    ax.minorticks_on()

    fig.savefig(stacking_figures_dir + 'pears_vs_3d_ms_comparison.pdf', \
        dpi=300, bbox_inches='tight')

    return None

def match_get_threed_ms(goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, \
    goodsn_ms_cat_3dhst, goodss_ms_cat_3dhst, pears_id, pears_field, pears_ra, pears_dec):

    # Define empty list for stellar mass
    threed_ms = []

    for i in range(len(pears_id)):

        current_id = pears_id[i]
        current_field = pears_field[i]
        current_ra = pears_ra[i]
        current_dec = pears_dec[i]

        # ------------------------------- Set field ------------------------------- #
        # Assign catalogs 
        if current_field == 'GOODS-N':
            phot_cat_3dhst = goodsn_phot_cat_3dhst
            ms_cat_3dhst = goodsn_ms_cat_3dhst
        elif current_field == 'GOODS-S':
            phot_cat_3dhst = goodss_phot_cat_3dhst
            ms_cat_3dhst = goodss_ms_cat_3dhst

        threed_ra = phot_cat_3dhst['ra']
        threed_dec = phot_cat_3dhst['dec']

        # Now match
        ra_lim = 0.3/3600  # arcseconds in degrees
        dec_lim = 0.3/3600
        threed_phot_idx = np.where((threed_ra >= current_ra - ra_lim) & (threed_ra <= current_ra + ra_lim) & \
            (threed_dec >= current_dec - dec_lim) & (threed_dec <= current_dec + dec_lim))[0]

        """
        If there are multiple matches with the photometry catalog 
        within 0.3 arseconds then choose the closest one.
        """
        if len(threed_phot_idx) > 1:
            print "Multiple matches found in photmetry catalog. Choosing the closest one."

            ra_two = current_ra
            dec_two = current_dec

            dist_list = []
            for v in range(len(threed_phot_idx)):

                ra_one = threed_ra[threed_phot_idx][v]
                dec_one = threed_dec[threed_phot_idx][v]

                dist = np.arccos(np.cos(dec_one*np.pi/180) * \
                    np.cos(dec_two*np.pi/180) * np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + \
                    np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
                dist_list.append(dist)

            dist_list = np.asarray(dist_list)
            dist_idx = np.argmin(dist_list)
            threed_phot_idx = threed_phot_idx[dist_idx]

        elif len(threed_phot_idx) == 0:  
            # Raise IndexError if match not found.
            # Because the code that generated the full pears sample
            # already matched with 3dhst and required a match to
            # be found to be included in the final full pears sample.
            print "Match not found. This should not have happened. Exiting."
            print "At ID and Field:", current_id, current_field
            raise IndexError
            sys.exit(1)

        # Now get 3D-HST stellar mass
        threed_id = phot_cat_3dhst['id'][threed_phot_idx]
        threed_mscat_idx = int(np.where(ms_cat_3dhst['id'] == threed_id)[0])  # There should only be one match
        threed_ms.append(ms_cat_3dhst['lmass'][threed_mscat_idx])

    # Convert to numpy array and return
    threed_ms = np.asarray(threed_ms)

    return threed_ms

def main():

    #make_stellar_mass_hist()
    compare_with_threedhst()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

