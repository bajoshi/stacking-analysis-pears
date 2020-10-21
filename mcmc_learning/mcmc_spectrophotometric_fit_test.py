"""
These MCMC codes are based on those from Mike Line's astrostatistics 
class. I'm using this to learn more about MCMC and fitting spectral data.

This particular code is used to see if I can fit a polynomial to one of
the grism spectra from the stacking project. 
"""

import numpy as np
from astropy.io import fits
import emcee
import corner
import scipy
from astropy.cosmology import Planck15
from scipy.interpolate import griddata
from scipy.integrate import simps

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
stacking_utils = home + '/Documents/GitHub/stacking-analysis-pears/util_codes/'
massive_galaxies_dir = home + '/Documents/GitHub/massive-galaxies/'
cluster_codedir = massive_galaxies_dir + 'cluster_codes/'
filter_curve_dir = massive_galaxies_dir + 'grismz_pipeline/'

# ----- Directory to save generated models
modeldir = home + '/Documents/bc03_output_dir/'

sys.path.append(stacking_utils)
import proper_and_lum_dist as cosmo
from dust_utils import get_dust_atten_model
from bc03_utils import get_bc03_spectrum

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

def get_template(age, tau, tauv, metallicity, \
    log_age_arr, metal_arr, tau_gyr_arr, tauv_arr, \
    model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap):

    """
    print("\nFinding closest model to --")
    print("Age [Gyr]:", 10**age / 1e9)
    print("Tau [Gyr]:", tau)
    print("Tau_v:", tauv)
    print("Metallicity [abs. frac.]:", metallicity)
    """

    # First find closest values and then indices corresponding to them
    # It has to be done this way because you typically wont find an exact match
    closest_age_idx = np.argmin(abs(log_age_arr - age))
    closest_tau_idx = np.argmin(abs(tau_gyr_arr - tau))
    closest_tauv_idx = np.argmin(abs(tauv_arr - tauv))

    # Now get indices
    age_idx = np.where(log_age_arr == log_age_arr[closest_age_idx])[0]
    tau_idx = np.where(tau_gyr_arr == tau_gyr_arr[closest_tau_idx])[0]
    tauv_idx = np.where(tauv_arr   ==    tauv_arr[closest_tauv_idx])[0]
    metal_idx = np.where(metal_arr == metallicity)[0]

    model_idx = int(reduce(np.intersect1d, (age_idx, tau_idx, tauv_idx, metal_idx)))

    model_llam = model_comp_spec_withlines_mmap[model_idx]

    chosen_age = 10**log_age_arr[model_idx] / 1e9
    chosen_tau = tau_gyr_arr[model_idx]
    chosen_av = 1.086 * tauv_arr[model_idx]
    chosen_metallicity = metal_arr[model_idx]

    """
    print("\nChosen model index:", model_idx)
    print("Chosen model parameters -- ")
    print("Age [Gyr]:", chosen_age)
    print("Tau [Gyr]:", chosen_tau)
    print("A_v:", chosen_av)
    print("Metallicity [abs. frac.]:", chosen_metallicity)
    """

    return model_llam

def loglike(theta, spec_x, spec_data, spec_err, phot_x, phot_data, phot_err):
    
    z, ms, age, tau, met, av, lsf_sigma = theta

    # ------- Get model
    model_mod, all_model_phot = model(spec_x, spec_data, spec_err, phot_x, phot_data, phot_err, z, ms, age, tau, met, av, lsf_sigma)

    # ------- Do combining of spectroscopy and photometry
    # For data
    comb_x, comb_data, comb_err = combine_all_data(spec_x, spec_data, spec_err, phot_x, phot_data, phot_err)

    # For model
    # Dummy arrays for model errors
    model_moderr = np.zeros(len(model_mod))
    all_model_photerr = np.zeros(len(phot_x))
    comb_model_wav, y, comb_model_ferr = combine_all_data(spec_x, model_mod, model_moderr, phot_x, all_model_phot, all_model_photerr)
    #print("Model func result:", y)

    # ------- Vertical scaling factor
    alpha = np.sum(comb_data * y / comb_err**2) / np.sum(y**2 / comb_err**2)
    #print("Alpha:", "{:.2e}".format(alpha))

    y = y * alpha

    lnLike = -0.5 * np.sum( (y - comb_data)**2/comb_err**2  +  np.log(2 * np.pi * comb_err**2))
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(spec_x, spec_data, color='k')
    ax.fill_between(spec_x, spec_data - spec_err, spec_data + spec_err, color='gray', alpha=0.5)
    ax.errorbar(phot_x, phot_data, yerr=phot_err, ms=5.0, fmt='o', \
        color='k', ecolor='k')

    ax.plot(spec_x, alpha * model_mod, color='firebrick')
    ax.scatter(phot_x, alpha * all_model_phot, s=15, color='firebrick')

    ax.set_xscale('log')
    plt.show()
    """
    
    return lnLike

def logprior(theta):

    z, ms, age, tau, met, av, lsf_sigma = theta
    
    # Make sure model is not older than the Universe
    # Allowing at least 100 Myr for the first galaxies to form after Big Bang
    age_at_z = Planck15.age(z).value  # in Gyr
    age_lim = age_at_z - 0.1  # in Gyr

    if ( 0.01 <= z <= 6.0 and \
         9.0 <= ms <= 12.0 and \
         0.01 <= age <= age_lim and  \
         0.01 <= tau <= 100.0 and  \
         0.0001 <= met <= 0.05 and \
         0.0 <= av <= 3.0 and \
         1.0 <= lsf_sigma <= 200.0 ):
        return 0.0
    
    return -np.inf

def logpost(theta, spec_x, spec_data, spec_err, phot_x, phot_data, phot_err):

    lp = logprior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    
    lnL = loglike(theta, spec_x, spec_data, spec_err, phot_x, phot_data, phot_err)
    
    return lp + lnL

def model(spec_x, spec_data, spec_err, phot_x, phot_data, phot_err, \
    z, ms, age, tau, met, av, lsf_sigma):
    """
    This function will return the closest BC03 template 
    from a large grid of pre-generated templates.

    Expects to get the following arguments
    x: observed wavelength grid
    z: redshift to apply to template
    age: age of SED in Gyr
    tau: exponential SFH timescale in Gyr
    av: visual dust extinction
    lsf_sigma: in angstroms
    """

    model_lam, model_llam = get_bc03_spectrum(age, tau, met, modeldir)

    # ------ Apply dust extinction
    model_dusty_llam = get_dust_atten_model(model_lam, model_llam, av)

    # ------ Multiply luminosity by stellar mass
    model_dusty_llam = model_dusty_llam * 10**ms

    # ------ Apply redshift
    model_lam_z, model_flam_z = cosmo.apply_redshift(model_lam, model_dusty_llam, z)

    # ------ Apply LSF
    model_lsfconv = scipy.ndimage.gaussian_filter1d(input=model_flam_z, sigma=lsf_sigma)

    # ------ Downgrade to grism resolution
    model_mod = np.zeros(len(spec_x))

    ### Zeroth element
    lam_step = spec_x[1] - spec_x[0]
    idx = np.where((model_lam_z >= spec_x[0] - lam_step) & (model_lam_z < spec_x[0] + lam_step))[0]
    model_mod[0] = np.mean(model_lsfconv[idx])

    ### all elements in between
    for j in range(1, len(spec_x) - 1):
        idx = np.where((model_lam_z >= spec_x[j-1]) & (model_lam_z < spec_x[j+1]))[0]
        model_mod[j] = np.mean(model_lsfconv[idx])
    
    ### Last element
    lam_step = spec_x[-1] - spec_x[-2]
    idx = np.where((model_lam_z >= spec_x[-1] - lam_step) & (model_lam_z < spec_x[-1] + lam_step))[0]
    model_mod[-1] = np.mean(model_lsfconv[idx])

    # ----------------- Now get the model photometry
    # and combine with the grism model
    # ------ THIS HAS TO BE THE SAME AS IN THE FUNC get_photometry_data()
    # ------ AND MUST BE in ascending order
    phot_lam = phot_x  # angstroms

    all_model_phot = np.zeros(len(phot_lam))

    # Loop over all filters
    for i in range(len(phot_lam)):

        # Interpolate the filter wavelengths to the model wavelength grid
        filt = all_filters[i]
        filt_interp = griddata(points=filt['wav'], values=filt['trans'], xi=model_lam_z, \
            method='linear')

        ## Set nan values in interpolated filter to 0.0
        filt_nan_idx = np.where(np.isnan(filt_interp))[0]
        filt_interp[filt_nan_idx] = 0.0

        """
        fig =  plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(filt['wav'], filt['trans'], lw=3.0, zorder=1, label=all_filter_names[i])
        ax.plot(model_lam_z, filt_interp, lw=1.0, zorder=2)
        ax.set_xlim(filt['wav'][0], filt['wav'][-1])
        ax.legend(loc=0)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        """

        dl = cosmo.luminosity_distance(z)
        dl = dl * 3.09e24  # convert to cm

        # multiply model spectrum to filter curve
        den = simps(y=filt_interp, x=model_lam_z)
        num = simps(y=model_flam_z * filt_interp, x=model_lam_z)
        filt_flam_model = num / den

        # now save to the list 
        all_model_phot[i] = filt_flam_model

    return model_mod, all_model_phot

def get_photometry_data(gal_id, field, survey, grism_lam_obs, grism_flam_obs, current_ra, current_dec):

    # Assign catalogs 
    if field == 'GOODS-N':
        phot_cat_3dhst = goodsn_phot_cat_3dhst
    elif field == 'GOODS-S':
        phot_cat_3dhst = goodss_phot_cat_3dhst

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
        print("Multiple matches found in photmetry catalog. Choosing the closest one.")

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
        print("Match not found in Photmetry catalog. Exiting.")
        sys.exit(0)

    # ------------- Need spectrum of Vega for correct coversion to AB mag from Vega mags
    # Read in Vega spectrum and get it in the appropriate forms
    vega = np.genfromtxt(pears_figs_dir + 'vega_reference.dat', dtype=None, \
        names=['wav', 'flam'], skip_header=7)

    speed_of_light = 299792458e10  # angstroms per second
    vega_lam = vega['wav']
    vega_spec_flam = vega['flam']
    vega_nu = speed_of_light / vega_lam
    vega_spec_fnu = vega_lam**2 * vega_spec_flam / speed_of_light

    # ------------------------------- Get photometric fluxes and their errors ------------------------------- #
    flam_f435w = cf.get_flam('F435W', phot_cat_3dhst['f_F435W'][threed_phot_idx])
    flam_f606w = cf.get_flam('F606W', phot_cat_3dhst['f_F606W'][threed_phot_idx])
    flam_f775w = cf.get_flam('F775W', phot_cat_3dhst['f_F775W'][threed_phot_idx])
    flam_f850lp = cf.get_flam('F850LP', phot_cat_3dhst['f_F850LP'][threed_phot_idx])
    flam_f125w = cf.get_flam('F125W', phot_cat_3dhst['f_F125W'][threed_phot_idx])
    flam_f140w = cf.get_flam('F140W', phot_cat_3dhst['f_F140W'][threed_phot_idx])
    flam_f160w = cf.get_flam('F160W', phot_cat_3dhst['f_F160W'][threed_phot_idx])

    flam_U = cf.get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['f_U'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac1 = cf.get_flam_nonhst('irac1', phot_cat_3dhst['f_IRAC1'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac2 = cf.get_flam_nonhst('irac2', phot_cat_3dhst['f_IRAC2'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac3 = cf.get_flam_nonhst('irac3', phot_cat_3dhst['f_IRAC3'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    flam_irac4 = cf.get_flam_nonhst('irac4', phot_cat_3dhst['f_IRAC4'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

    ferr_f435w = cf.get_flam('F435W', phot_cat_3dhst['e_F435W'][threed_phot_idx])
    ferr_f606w = cf.get_flam('F606W', phot_cat_3dhst['e_F606W'][threed_phot_idx])
    ferr_f775w = cf.get_flam('F775W', phot_cat_3dhst['e_F775W'][threed_phot_idx])
    ferr_f850lp = cf.get_flam('F850LP', phot_cat_3dhst['e_F850LP'][threed_phot_idx])
    ferr_f125w = cf.get_flam('F125W', phot_cat_3dhst['e_F125W'][threed_phot_idx])
    ferr_f140w = cf.get_flam('F140W', phot_cat_3dhst['e_F140W'][threed_phot_idx])
    ferr_f160w = cf.get_flam('F160W', phot_cat_3dhst['e_F160W'][threed_phot_idx])

    ferr_U = cf.get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['e_U'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac1 = cf.get_flam_nonhst('irac1', phot_cat_3dhst['e_IRAC1'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac2 = cf.get_flam_nonhst('irac2', phot_cat_3dhst['e_IRAC2'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac3 = cf.get_flam_nonhst('irac3', phot_cat_3dhst['e_IRAC3'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
    ferr_irac4 = cf.get_flam_nonhst('irac4', phot_cat_3dhst['e_IRAC4'][threed_phot_idx], \
        vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

    # ------------------------------- Apply aperture correction ------------------------------- #
    # First interpolate the given filter curve on to the wavelength frid of the grism data
    # You only need the F775W filter here since you're only using this filter to get the 
    # aperture correction factor.
    if survey == 'PEARS':
        f775w_filt_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/f775w_filt_curve.txt', \
            dtype=None, names=['wav', 'thru'])
        filt_trans_interp = griddata(points=f775w_filt_curve['wav'], values=f775w_filt_curve['thru'], \
            xi=grism_lam_obs, method='linear')

    # multiply grism spectrum to filter curve
    num = 0
    den = 0
    for w in range(len(grism_flam_obs)):
        num += grism_flam_obs[w] * filt_trans_interp[w]
        den += filt_trans_interp[w]

    avg_f775w_flam_grism = num / den
    aper_corr_factor = flam_f775w / avg_f775w_flam_grism
    print("Aperture correction factor:", "{:.3}".format(aper_corr_factor))

    grism_flam_obs *= aper_corr_factor  # applying factor

    # ------------------------------- Make unified photometry arrays ------------------------------- #
    phot_fluxes_arr = np.array([flam_U, flam_f435w, flam_f606w, flam_f775w, flam_f850lp, flam_f125w, flam_f140w, flam_f160w,
        flam_irac1, flam_irac2, flam_irac3, flam_irac4])
    phot_errors_arr = np.array([ferr_U, ferr_f435w, ferr_f606w, ferr_f775w, ferr_f850lp, ferr_f125w, ferr_f140w, ferr_f160w,
        ferr_irac1, ferr_irac2, ferr_irac3, ferr_irac4])

    # Pivot wavelengths
    # From here --
    # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/
    # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
    # KPNO/MOSAIC U-band: https://www.noao.edu/kpno/mosaic/filters/k1001.html
    # Spitzer IRAC channels: http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/6/#_Toc410728283
    phot_lam = np.array([3582.0, 4328.2, 5921.1, 7692.4, 9033.1, 12486.0, 13923.0, 15369.0, 
    35500.0, 44930.0, 57310.0, 78720.0])  # angstroms

    return phot_lam, phot_fluxes_arr, phot_errors_arr

def combine_all_data(wav, flam, ferr, phot_lam, phot_flam, phot_ferr):

    count = 0
        
    comb_wav = wav
    comb_flam = flam
    comb_ferr = ferr
        
    for l in phot_lam:

        # First find the index where to insert
        if l < comb_wav[0]:
            l_insert_idx = 0

        elif l > comb_wav[-1]:
            l_insert_idx = len(comb_wav)

        else:
            l_insert_idx = np.where(comb_wav > l)[0][0]

        # Now do the combining
        comb_wav = np.insert(comb_wav, l_insert_idx, l)
        comb_flam = np.insert(comb_flam, l_insert_idx, phot_flam[count])
        comb_ferr = np.insert(comb_ferr, l_insert_idx, phot_ferr[count])

        # Update counter
        count += 1 

    return comb_wav, comb_flam, comb_ferr

def main():

    print("Starting at:", datetime.datetime.now())

    print("\n* * * *   [WARNING]: the downgraded model is offset by delta_lambda/2 where delta_lambda is the grism wavelength sampling.   * * * *")
    print("\n* * * *   [WARNING]: using two different cosmologies for dl and Universe age at a redshift. Change to FlatLambdaCDM astropy.  * * * *")
    print("\n* * * *   [INFO]: check if you can use CANDELS photometry directly instead of 3D-HST   * * * *")

    # ---- Load in data
    pears_id = 126769
    pears_field = 'GOODS-S'

    print("\nWorking on:", pears_field, pears_id)

    fname = pears_field + '_' + str(pears_id) + '_' + 'PAcomb.fits'

    spec_hdu = fits.open(datadir + fname)

    # Get data
    wav = spec_hdu[1].data
    flam = spec_hdu[2].data[0]
    ferr = spec_hdu[2].data[1]

    # Chop wavelength grid to 6000A -- 9500A
    arg_low = np.argmin(abs(wav - 5900))
    arg_high = np.argmin(abs(wav - 9700))

    wav  = wav[arg_low:arg_high+1]
    flam = flam[arg_low:arg_high+1]
    ferr = ferr[arg_low:arg_high+1]

    # ferr /= 3.0

    # Get coordinates for matching with photometry
    gal_ra = spec_hdu[2].header['RA']
    gal_dec = spec_hdu[2].header['DEC']

    # ---- Get photometry as well
    phot_lam, phot_flam, phot_ferr = get_photometry_data(pears_id, pears_field, 'PEARS', \
        wav, flam, gal_ra, gal_dec)

    # ---- Plot data if you want to check what it looks like
    """
    #print("Photometry and errors:")
    #print(phot_flam)
    #print(phot_ferr)
    
    snr_arr = flam/ferr
    #print("Signal to noise array for this galaxy:", snr_arr)
    print("Mean of signal to noise array for this galaxy:", np.mean(snr_arr))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(wav, flam, color='k')
    ax.fill_between(wav, flam - ferr, flam + ferr, color='gray', alpha=0.5)

    ax.errorbar(phot_lam, phot_flam, yerr=phot_ferr, ms=5.0, fmt='o', \
        color='k', ecolor='k')

    ax.set_xscale('log')

    plt.show()
    sys.exit(0)
    """


    # ----------------------- Using numpy polyfitting ----------------------- #
    # -------- The best-fit will be used to set the initial position for the MCMC walkers.
    """
    pfit = np.ma.polyfit(wav, flam, deg=3)
    np_polynomial = np.poly1d(pfit)
    bestarr = pfit
    print("Numpy polynomial fit:")
    print(np_polynomial)
    """

    # Plotting fit and residuals
    """
    fig = plt.figure(figsize=(9,6))
    gs = gridspec.GridSpec(6,2)
    gs.update(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.00, hspace=0.7)

    ax1 = fig.add_subplot(gs[:4,:])
    ax2 = fig.add_subplot(gs[4:,:])

    ax1.plot(wav, flam, 'o-', markersize=3.0, color='turquoise', linewidth=1.5, label='PEARS obs data')
    ax1.plot(wav, np_polynomial(wav), color='crimson', zorder=1.0, label='NumPy polynomial fit')

    # Divide by polynomial fit
    flam = flam / np_polynomial(wav)
    ferr = ferr / np_polynomial(wav)

    # Plot "pure emission/absorption" spectrum
    ax2.axhline(y=1.0, ls='--', color='black', lw=1.5, zorder=1)
    ax2.plot(wav, flam, color='turquoise', linewidth=1.5, zorder=2)

    plt.show()
    """

    # ----------------------- Using explicit MCMC with Metropolis-Hastings ----------------------- #
    #*******Metropolis Hastings********************************
    mh_start = time.time()
    print("\nRunning explicit Metropolis-Hastings...")
    N = 1000   #number of "timesteps"

    # The parameter vector is (z, ms, age, tau, met, av, lsf_sigma)
    # age in gyr and tau in gyr
    # stellar mass is log(stellarmass/solarmass)
    # metallicity is absolute fraction
    # dust parameter is av not tauv
    # last param is LSF in Angstroms

    # Define initial guesses
    r = np.array([0.1, 10.0, 1.0, 1.0, 0.02, 0.1, 10.0])  # initial position
    print("Initial parameter vector:", r)

    # Array for metallicities
    metals_arr = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05])

    # Set jump sizes
    jump_size_z = 0.01  
    jump_size_ms = 0.02  # in log(ms)
    jump_size_age = 0.1  # in gyr
    jump_size_tau = 0.1  # in gyr
    jump_size_met = 0.001
    jump_size_av = 0.2  # magnitudes
    jump_size_lsf = 5.0  # angstroms

    label_list = [r'$z$', r'$log(Ms/M_\odot)$', r'$Age [Gyr]$', r'$\tau [Gyr]$', r'$Z$', r'$A_V [mag]$', r'$LSF [\AA]$']

    logp = logpost(r, wav, flam, ferr, phot_lam, phot_flam, phot_ferr)  # evaluating the probability at the initial guess
    
    print("Initial guess log(probability):", logp)

    samples = []  #creating array to hold parameter vector with time
    accept = 0.

    for i in range(N): #beginning the iteratitive loop

        print("MH Iteration", i, end='\r')

        rn0 = float(r[0] + jump_size_z * np.random.normal(size=1))  #creating proposal vecotr ("Y" or X_t+1)
        rn1 = float(r[1] + jump_size_ms * np.random.normal(size=1))
        rn2 = float(r[2] + jump_size_age * np.random.normal(size=1))
        rn3 = float(r[3] + jump_size_tau * np.random.normal(size=1))

        # Metallicity is very discrete in BC03 models 
        # so doing it this way
        # While the newer 2016 version has an additional metallicity
        # referred to as "m82", the documentation never specifies the 
        # actual metallicity associated with it. So I'm ignoring that one.
        rn4 = float(r[4] + jump_size_met * np.random.normal(size=1))
        metals_idx = np.argmin(abs(rn4 - metals_arr))
        rn4 = metals_arr[metals_idx]

        rn5 = float(r[5] + jump_size_av * np.random.normal(size=1))
        rn6 = float(r[6] + jump_size_lsf * np.random.normal(size=1))

        rn = np.array([rn0, rn1, rn2, rn3, rn4, rn5, rn6])

        #print("Proposed parameter vector", rn)
        
        logpn = logpost(rn, wav, flam, ferr, phot_lam, phot_flam, phot_ferr)  #evaluating probability of proposal vector
        #print("Proposed parameter vector log(probability):", logpn)
        dlogL = logpn - logp
        a = np.exp(dlogL)

        #print("Ratio of probabilities at proposed to current position:", a)

        if a >= 1:   #always keep it if probability got higher
            #print("Will accept point since probability increased.")
            logp = logpn
            r = rn
            accept+=1
        
        else:  #only keep it based on acceptance probability
            #print("Probability decreased. Will decide whether to keep point or not.")
            u = np.random.rand()  #random number between 0 and 1
            if u < a:  #only if proposal prob / previous prob is greater than u, then keep new proposed step
                logp = logpn
                r = rn
                accept+=1
                #print("Point kept.")

        samples.append(r)  #update

    print("Finished explicit Metropolis-Hastings.")

    mh_time = time.time() - mh_start
    mh_min, mh_sec = divmod(mh_time, 60.0)
    mh_hr, mh_min = divmod(mh_min, 60.0)
    print("Time taken for explicit Metropolis-Hastings:", \
        "{:.2f}".format(mh_hr), "hours", "{:.2f}".format(mh_min), "mins", "{:.2f}".format(mh_sec), "seconds.")

    # Plotting results from explicit MH
    samples = np.array(samples)

    # plot trace
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(samples[:,0], label='z')
    ax.plot(samples[:,1], label='Age [Gyr]')
    ax.plot(samples[:,2], label='Tau [Gyr]')
    ax.plot(samples[:,3], label=r'$A_v [mag]$')
    ax.plot(samples[:,4], label=r'$LSF [\AA]$')
    ax.legend(loc=0)

    # using corner
    corner.corner(samples, bins=30, labels=label_list, \
        show_titles='True', plot_contours='True')
    plt.show()

    print("Acceptance Rate:", accept/N)

    sys.exit(0)

    # ----------------------- Using emcee ----------------------- #
    print("\nRunning emcee...")
    ndim, nwalkers = 5, 100  # setting up emcee params--number of params and number of walkers

    # generating "intial" ball of walkers about best fit from min chi2
    pos = np.zeros(shape=(nwalkers, ndim))

    for i in range(nwalkers):

        rn0 = float(r[0] + jump_size_z * np.random.normal(size=1))
        rn1 = float(r[1] + jump_size_age * np.random.normal(size=1))
        rn2 = float(r[2] + jump_size_tau * np.random.normal(size=1))
        rn3 = float(r[3] + jump_size_av * np.random.normal(size=1))
        rn4 = float(r[4] + jump_size_lsf * np.random.normal(size=1))

        rn = np.array([rn0, rn1, rn2, rn3, rn4])

        pos[i] = rn

    from multiprocessing import Pool

    with Pool() as pool:
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=[comb_wav, comb_flam, comb_ferr], pool=pool)
        sampler.run_mcmc(pos, 1000, progress=True)

    print("Finished running emcee.")

    samples = sampler.get_chain()

    print("Samples shape:", samples.shape)

    # plot trace
    fig1, axes1 = plt.subplots(ndim, figsize=(10, 6), sharex=True)

    for i in range(ndim):
        ax1 = axes1[i]
        ax1.plot(samples[:, :, i], "k", alpha=0.1)
        ax1.set_xlim(0, len(samples))
        ax1.set_ylabel(label_list[i])
        ax1.yaxis.set_label_coords(-0.1, 0.5)

    axes1[-1].set_xlabel("Step number")

    # Get autocorrelation time
    try:
        tau = sampler.get_autocorr_time()
    except emcee.autocorr.AutocorrError as errmsg:
        print(errmsg)
        print("\n")
        print("Emcee AutocorrError occured.")
        print("The chain is shorter than 50 times the integrated autocorrelation time for 5 parameter(s).")
        print("Use this estimate with caution and run a longer chain!")
        print("\n")

        tau_list_str = str(errmsg).split('tau:')[-1]
        tau_list = tau_list_str.split()
        print("Tau list:", tau_list)

        tau = []
        for j in range(len(tau_list)):
            curr_elem = tau_list[j]
            if ('[' in curr_elem) and (len(curr_elem) > 1):
                tau.append(float(curr_elem.lstrip('[')))
            elif (']' in curr_elem) and (len(curr_elem) > 1):
                tau.append(float(curr_elem.rstrip(']')))
            elif len(curr_elem) > 1:
                tau.append(float(tau_list[j]))

    print("Autocorrelation time (i.e., steps that walkers take in each dimension before they forget where they started):", tau)

    # Discard burn-in. You do not want to consider the burn in the corner plots/estimation.
    burn_in = int(3 * tau[0])
    print("Burn-in:", burn_in)

    thinning_steps = int(0.5 * tau[0])
    print("Thinning steps:", thinning_steps)

    flat_samples = sampler.get_chain(discard=burn_in, thin=thinning_steps, flat=True)
    print("Flat samples shape:", flat_samples.shape)

    # plot corner plot
    fig2 = corner.corner(flat_samples, plot_contours='True', labels=label_list, \
        label_kwargs={"fontsize": 14}, show_titles='True', title_kwargs={"fontsize": 14})

    plt.show()

    # Plot 100 random models from the parameter space
    inds = np.random.randint(len(flat_samples), size=100)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    ax3.plot(wav, flam, color='k')
    ax3.fill_between(wav, flam - ferr, flam + ferr, color='gray', alpha=0.5, zorder=3)

    for ind in inds:
        sample = flat_samples[ind]
        m = model(wav, sample[0], sample[1], sample[2], sample[3], sample[4]) 
        a = np.sum(flam * m / ferr**2) / np.sum(m**2 / ferr**2)
        ax3.plot(wav, a * m, color='tab:red', alpha=0.2, zorder=2)

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)











