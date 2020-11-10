import numpy as np
from astropy.io import fits
import emcee
import corner
import scipy
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
astropy_cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

import os
import sys
from functools import reduce
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
# ---------- Define directories ---------- #
# ----- Data directories
pears_figs_dir = datadir = home + '/Documents/pears_figs_data/'
datadir = home + '/Documents/pears_figs_data/data_spectra_only/'
threedhst_datadir = home + '/Documents/3dhst_data/'

# ----- Directories for other useful codes
stacking_analysis_dir = home + '/Documents/GitHub/stacking-analysis-pears/'
stacking_utils = stacking_analysis_dir + 'util_codes/'
massive_galaxies_dir = home + '/Documents/GitHub/massive-galaxies/'
cluster_codedir = massive_galaxies_dir + 'cluster_codes/'
filter_curve_dir = massive_galaxies_dir + 'grismz_pipeline/'

sys.path.append(stacking_utils)
import proper_and_lum_dist as cosmo
from dust_utils import get_dust_atten_model
from bc03_utils import get_age_spec

sys.path.append(cluster_codedir)
import cluster_do_fitting as cf

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

# ------------------------------- Read in all photometry filters ------------------------------- #
# u band is given in transmission percentages
# all other filters are throughput fractions
uband_curve = np.genfromtxt(filter_curve_dir + 'kpno_mosaic_u.txt', dtype=None, \
    names=['wav', 'trans'], skip_header=14)
uband_curve['trans'] /= 100.0

f435w_filt_curve = np.genfromtxt(filter_curve_dir + 'f435w_filt_curve.txt', \
    dtype=None, names=['wav', 'trans'])
f606w_filt_curve = np.genfromtxt(filter_curve_dir + 'f606w_filt_curve.txt', \
    dtype=None, names=['wav', 'trans'])
f775w_filt_curve = np.genfromtxt(filter_curve_dir + 'f775w_filt_curve.txt', \
    dtype=None, names=['wav', 'trans'])
f850lp_filt_curve = np.genfromtxt(filter_curve_dir + 'f850lp_filt_curve.txt', \
    dtype=None, names=['wav', 'trans'])
f125w_filt_curve = np.genfromtxt(filter_curve_dir + 'f125w_filt_curve.txt', \
    dtype=None, names=['wav', 'trans'])
f140w_filt_curve = np.genfromtxt(filter_curve_dir + 'f140w_filt_curve.txt', \
    dtype=None, names=['wav', 'trans'])
f160w_filt_curve = np.genfromtxt(filter_curve_dir + 'f160w_filt_curve.txt', \
    dtype=None, names=['wav', 'trans'])
irac1_curve = np.genfromtxt(filter_curve_dir + 'irac1.txt', dtype=None, \
    names=['wav', 'trans'], skip_header=3)
irac2_curve = np.genfromtxt(filter_curve_dir + 'irac2.txt', dtype=None, \
    names=['wav', 'trans'], skip_header=3)
irac3_curve = np.genfromtxt(filter_curve_dir + 'irac3.txt', dtype=None, \
    names=['wav', 'trans'], skip_header=3)
irac4_curve = np.genfromtxt(filter_curve_dir + 'irac4.txt', dtype=None, \
    names=['wav', 'trans'], skip_header=3)

# IRAC wavelengths are in mixrons # convert to angstroms
irac1_curve['wav'] *= 1e4
irac2_curve['wav'] *= 1e4
irac3_curve['wav'] *= 1e4
irac4_curve['wav'] *= 1e4

all_filters = [uband_curve, f435w_filt_curve, f606w_filt_curve, f775w_filt_curve, f850lp_filt_curve, \
f125w_filt_curve, f140w_filt_curve, f160w_filt_curve, irac1_curve, irac2_curve, irac3_curve, irac4_curve]
all_filter_names = ['u', 'f435w', 'f606w', 'f775w', 'f850lp', \
'f125w', 'f140w', 'f160w', 'irac1', 'irac2', 'irac3', 'irac4']

# Read in all models and parameters
model_lam_grid = np.load(pears_figs_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')
model_grid = np.load(pears_figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')

log_age_arr = np.load(pears_figs_dir + 'log_age_arr_chab.npy', mmap_mode='r')
metal_arr = np.load(pears_figs_dir + 'metal_arr_chab.npy', mmap_mode='r')
tau_gyr_arr = np.load(pears_figs_dir + 'tau_gyr_arr_chab.npy', mmap_mode='r')
tauv_arr = np.load(pears_figs_dir + 'tauv_arr_chab.npy', mmap_mode='r')

"""
Array ranges are:
1. Age: 7.02 to 10.114 (this is log of the age in years)
2. Metals: 0.0001 to 0.05 (absolute fraction of metals. All CSP models although are fixed at solar = 0.02)
3. Tau: 0.01 to 63.095 (this is in Gyr. SSP models get -99.0)
4. TauV: 0.0 to 2.8 (Visual dust extinction in magnitudes. SSP models get -99.0)
"""


# This class came from stackoverflow
# SEE: https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'




def main():

    print("Starting at:", datetime.datetime.now())

    print(f"{bcolors.WARNING}")
    print("\n* * * *  [WARNING]: the downgraded model is offset by delta_lambda/2 where delta_lambda is the grism wavelength sampling.  * * * *")
    print(f"{bcolors.ENDC}")

    # Read in results for all of PEARS
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True, encoding='ascii')

    #print("Read in following catalog header.")
    #print(cat.dtype.names)

    # Now run MCMC fitting on all galaxies that are above say 10^8.5 M_sol 
    # Since we'd like to consider all galaxies above 10^9 M_sol, I'm giving
    # the stellar mass here some padding because it was derived from fitting
    # only the photometry. The MCMC fitting will derive the stellar mass again.
    # Trying 10.5 here for first test run
    ms_idx = np.where(np.log10(cat['zp_ms']) >= 10.5)[0]
    print("Will run MCMC fitting on", len(ms_idx), "galaxies.")

    all_ids_tofit = cat['PearsID'][ms_idx]
    all_fields_tofit = cat['Field'][ms_idx]

    # New catalog to save MCMC results
    fh = open(stacking_analysis_dir + 'massive_mcmc_fitting_results.txt', 'w')

    # Write header
    sh = str(cat.dtype.names)
    sh = sh.lstrip('(')
    sh = sh.rstrip(')')
    fh.write('# ' + sh.replace('\'', ''))
    fh.write('\n')

    # Loop over all galaxies
    for i in range(len(all_ids_tofit)):

        # Prep data and run emcee
        # this function will return a dict with the fitting results
        emcee_res = run_emcee_fitting(all_ids_tofit[i], all_fields_tofit[i])

        s = str(cat[ms_idx[i]])
        s = s.lstrip('(')
        s = s.rstrip(')')

        fh.write(s + ',')
        fh.write(emcee_res)
        fh.write('\n')

    fh.close()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)





