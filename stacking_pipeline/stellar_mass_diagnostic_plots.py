from __future__ import division

import numpy as np

import os
import sys
import glob

import matplotlib.pyplot as plt

home = os.getenv('HOME')
figs_dir = home + '/Desktop/FIGS/'

massive_figures_dir = figs_dir + 'massive-galaxies-figures/'
stacking_analysis_dir = figs_dir + 'stacking-analysis-pears/'
stacking_figures_dir = figs_dir + 'stacking-analysis-figures/'
full_pears_results_dir = massive_figures_dir + 'full_pears_results/'

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

    # Number of objects with believable stellar masses
    ms_val_idx = np.where((ms >= 8.0) & (ms <= 12.0))[0]
    print "Number of galaxies with stellar masses between 10^8 and 10^12 M_sol:",
    print len(ms_val_idx)

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

def compare_prep(selection):

    if selection == 'all_salp':
        selected_results_dir = full_pears_results_dir
        final_file_name = stacking_analysis_dir + 'full_pears_results.txt'
    elif selection == 'all_salp_no_irac_ch3_ch4':
        selected_results_dir = full_pears_results_dir.replace('full_pears_results', 'full_pears_results_no_irac_ch3_ch4')
        final_file_name = stacking_analysis_dir + 'full_pears_results_no_irac_ch3_ch4.txt'
    elif selection == 'all_salp_no_irac':
        selected_results_dir = full_pears_results_dir.replace('full_pears_results', 'full_pears_results_no_irac')
        final_file_name = stacking_analysis_dir + 'full_pears_results_no_irac.txt'
    elif selection == 'all_chab':
        selected_results_dir = full_pears_results_dir.replace('full_pears_results', 'full_pears_results_chabrier')
        final_file_name = stacking_analysis_dir + 'full_pears_results_chabrier.txt'
    elif selection == 'all_chab_no_irac_ch3_ch4':
        selected_results_dir = full_pears_results_dir.replace('full_pears_results', 'full_pears_results_chabrier_no_irac_ch3_ch4')
        final_file_name = stacking_analysis_dir + 'full_pears_results_chabrier_no_irac_ch3_ch4.txt'
    elif selection == 'all_chab_no_irac':
        selected_results_dir = full_pears_results_dir.replace('full_pears_results', 'full_pears_results_chabrier_no_irac')
        final_file_name = stacking_analysis_dir + 'full_pears_results_chabrier_no_irac.txt'

    cat = np.genfromtxt(final_file_name, dtype=None, names=['PearsID', 'Field', 'RA', 'DEC', 'zp_ms'], usecols=(0, 1, 2, 3, 36))

    pears_id = cat['PearsID']
    pears_field = cat['Field']
    pears_ra = cat['RA']
    pears_dec = cat['DEC']
    pears_ms = cat['zp_ms']

    pears_ms = np.log10(pears_ms)  # convert to log of stellar mass

    return pears_id, pears_field, pears_ra, pears_dec, pears_ms

def get_threed_stuff():

    selection = 'all_salp'
    pears_id, pears_field, pears_ra, pears_dec, pears_ms = compare_prep(selection)
    # While getting the 3D-HST stuff it doesn't matter what the selection
    # here is because the basic info (which is what you need) will not
    # change depending on the selection.

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

    return threed_ms

def compare_with_threedhst():

    # Select the datasets to get and compare
    selection = 'all_salp'
    # This has the following 6 options:
    # 1. 'all_salp': This will select the results which include all photometry and use the Salpeter IMF.
    # 2. 'all_salp_no_irac_ch3_ch4': This will select the results which exclude IRAC CH3 and CH4 and use Salpeter IMF.
    # 3. 'all_salp_no_irac': This will select the results which exclude all IRAC photometry and use Salpeter IMF.
    # 4. 'all_chab': This will select the results which include all photometry and use the Chabrier IMF.
    # 5. 'all_chab_no_irac_ch3_ch4': This will select the results which exclude IRAC CH3 and CH4 and use Chabrier IMF.
    # 6. 'all_chab_no_irac': This will select the results which exclude all IRAC photometry and use Chabrier IMF.

    pears_id, pears_field, pears_ra, pears_dec, pears_ms = compare_prep(selection)

    threed_ms = get_threed_stuff()

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

    # Show stellar mass histogram from this work as inset figure
    left, bottom, width, height = [0.21, 0.65, 0.3, 0.2]
    ax_ins = fig.add_axes([left, bottom, width, height])

    ax_ins.set_xlabel(r'$\rm log(M^{this\, work}_s)\ [M_\odot]$', fontsize=12)
    ax_ins.set_ylabel(r'$\rm \# objects$', fontsize=12)

    binsize = 0.5
    total_bins = int((13.0 - 4.0)/binsize)

    ax_ins.hist(pears_ms, total_bins, range=(4.0, 13.0), histtype='step', linewidth=1.2, color='k')

    ax_ins.set_xticks(np.arange(4.0, 13.5, 1.0))
    ax_ins.minorticks_on()

    # Other info on plot
    num = len(pears_ms)
    ax_ins.text(0.04, 0.92, r'$\rm N\, =\, $' + str(num), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax_ins.transAxes, color='k', size=12)

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

def get_zp_array():

    # Define empty list for redshifts
    zp = []

    # Loop over all results and store redshift values
    for fl in glob.glob(full_pears_results_dir + 'redshift_fitting_results_*.txt'):
        f = np.genfromtxt(fl, dtype=None, names=True, skip_header=1)
        zp.append(f['zp_minchi2'])

    # Convert to numpy array
    zp = np.asarray(zp)

    return zp

def make_z_hist():

    zp = get_zp_array()

    # Now make plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\rm z_{phot}$', fontsize=15)
    ax.set_ylabel(r'$\rm \# objects$', fontsize=15)

    binsize = 0.1
    total_bins = int((6.0 - 0.0)/binsize)

    ax.hist(zp, total_bins, range=(0.0, 6.0), histtype='step', linewidth=1.2, color='k')

    ax.minorticks_on()
    ax.set_xticks(np.arange(0.0, 6.1, 0.5))

    # Other info on plot
    num = len(zp)
    ax.text(0.75, 0.95, r'$\rm N\, =\, $' + str(num), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=15)
    
    fig.savefig(stacking_figures_dir + 'zphot_hist.pdf', \
        dpi=300, bbox_inches='tight')

    # --------------------------------
    # Decide redshift intervals
    # This is done by trial and error.
    # --------------------------------
    print "Number of galaxies within each redshift interval."
    print "0.0 <= z < 0.4", "    ", len(np.where((zp >= 0.0) & (zp < 0.4))[0])
    print "0.4 <= z < 0.7", "    ", len(np.where((zp >= 0.4) & (zp < 0.7))[0])
    print "0.7 <= z < 1.0", "    ", len(np.where((zp >= 0.7) & (zp < 1.0))[0])
    print "1.0 <= z < 2.0", "    ", len(np.where((zp >= 1.0) & (zp < 2.0))[0])
    print "2.0 <= z <= 6.0", "    ", len(np.where((zp >= 2.0) & (zp <= 6.0))[0])

    return None

def compare_all_salp():

    # 1st set
    selection = 'all_salp'
    pears_id, pears_field, pears_ra, pears_dec, pears_ms = compare_prep(selection)

    # 2nd set
    selection = 'all_salp_no_irac_ch3_ch4'
    pears_id_no_irac34, pears_field_no_irac34, pears_ra_no_irac34, pears_dec_no_irac34, pears_ms_no_irac34 = compare_prep(selection)

    # 3rd set
    selection = 'all_salp_no_irac'
    pears_id_no_irac, pears_field_no_irac, pears_ra_no_irac, pears_dec_no_irac, pears_ms_no_irac = compare_prep(selection)

    # Get 3D-HST stellar mass
    threed_ms = get_threed_stuff()

    # Now plot 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\rm log(M^{this\, work}_s)\ [M_\odot]$', fontsize=15)
    ax.set_ylabel(r'$\rm log(M^{3D-HST}_s)\ [M_\odot]$', fontsize=15)

    #ax.scatter(pears_ms, threed_ms, s=2.5, color='black', alpha=0.4)  # all_salp
    #ax.scatter(pears_ms_no_irac34, threed_ms, s=2.0, color='pink', alpha=0.4)  # all_salp_no_irac_ch3_ch4
    ax.scatter(pears_ms_no_irac, threed_ms, s=2.0, color='green', alpha=0.4)  # all_salp_no_irac
    ax.plot(np.arange(4.0, 13.1, 0.1), np.arange(4.0, 13.1, 0.1), '--', color='r')

    ax.set_xlim(4.0, 13.0)
    ax.set_ylim(4.0, 13.0)

    ax.minorticks_on()

    plt.show()

    return None

def compare_santini():

    return None


def main():

    #make_stellar_mass_hist()
    #compare_with_threedhst()
    #make_z_hist()
    #compare_all_salp()

    

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

