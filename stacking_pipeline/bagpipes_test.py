import numpy as np 
import bagpipes as pipes

from scipy.interpolate import griddata
from astropy.io import fits

import os
import sys

home = os.getenv('HOME')

datadir = home + '/Documents/pears_figs_data/data_spectra_only/'
massive_galaxies_dir = home + '/Documents/GitHub/massive-galaxies/'
threedhst_datadir = home + '/Documents/3dhst_data/'
pears_figs_dir = home + '/Documents/pears_figs_data/'

cluster_codedir = massive_galaxies_dir + 'cluster_codes/'

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

    #flam_U = cf.get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['f_U'][threed_phot_idx], \
    #    vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
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

    #ferr_U = cf.get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['e_U'][threed_phot_idx], \
    #    vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
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
    phot_fluxes_arr = np.array([flam_f435w, flam_f606w, flam_f775w, flam_f850lp, flam_f125w, flam_f140w, flam_f160w,
        flam_irac1, flam_irac2, flam_irac3, flam_irac4])
    phot_errors_arr = np.array([ferr_f435w, ferr_f606w, ferr_f775w, ferr_f850lp, ferr_f125w, ferr_f140w, ferr_f160w,
        ferr_irac1, ferr_irac2, ferr_irac3, ferr_irac4])

    # Pivot wavelengths
    # From here --
    # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/
    # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
    # KPNO/MOSAIC U-band: https://www.noao.edu/kpno/mosaic/filters/k1001.html
    # Spitzer IRAC channels: http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/6/#_Toc410728283
    phot_lam = np.array([4328.2, 5921.1, 7692.4, 9033.1, 12486.0, 13923.0, 15369.0, 
    35500.0, 44930.0, 57310.0, 78720.0])  # angstroms

    # From Adam Carnall's Github notebooks on bagpipes
    photometry = np.c_[phot_fluxes_arr, phot_errors_arr]

    # blow up the errors associated with any missing fluxes.
    for i in range(len(photometry)):
        if (photometry[i, 0] == 0.) or (photometry[i, 1] <= 0):
            photometry[i,:] = [0., 9.9*10**99.]
            
    # Enforce a maximum SNR of 20, or 10 in the IRAC channels.
    # we need this in our case too since we're using the same data
    for i in range(len(photometry)):
        if i < 7:  # in our case here, the first 7 channels are HST photometry
            max_snr = 20.
            
        else:
            max_snr = 10.
        
        if photometry[i, 0]/photometry[i, 1] > max_snr:
            photometry[i, 1] = photometry[i, 0]/max_snr

    return photometry

def load_data(pearsid_str):

    field = pearsid_str[:7]
    pears_id = int(pearsid_str[7:])

    print("PEARS ID and field:", pears_id, field)

    # ----------------- Get Spectroscopy
    fname = field + '_' + str(pears_id) + '_' + 'PAcomb.fits'
    spec_hdu = fits.open(datadir + fname)

    # Get data
    wav = spec_hdu[1].data
    flam = spec_hdu[2].data[0]
    ferr = spec_hdu[2].data[1]

    # Chop wavelength grid
    arg_low = np.argmin(abs(wav - 5900))
    arg_high = np.argmin(abs(wav - 9700))

    wav  = wav[arg_low:arg_high+1]
    flam = flam[arg_low:arg_high+1]
    ferr = ferr[arg_low:arg_high+1]

    spectrum = np.c_[wav, flam, ferr]

    # ----------------- Get Photometry
    # First you'll need the coordinates for matching
    current_ra = spec_hdu[2].header['RA']
    current_dec = spec_hdu[2].header['DEC']

    photometry = get_photometry_data(pears_id, field, 'PEARS', wav, flam, current_ra, current_dec)

    return spectrum, photometry

def main():

    pears_filter_list = [home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/f435w_filt_curve.txt', \
                         home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/f606w_filt_curve.txt', \
                         home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/f775w_filt_curve.txt', \
                         home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/f850lp_filt_curve.txt', \
                         home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/f125w_filt_curve.txt', \
                         home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/f140w_filt_curve.txt', \
                         home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/f160w_filt_curve.txt', \
                         home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/irac1_ang.txt', \
                         home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/irac2_ang.txt', \
                         home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/irac3_ang.txt', \
                         home + '/Documents/GitHub/massive-galaxies/grismz_pipeline/irac4_ang.txt']

    galaxy = pipes.galaxy('GOODS-S109151', load_data, filt_list=pears_filter_list)

    dblplaw = {}
    dblplaw["tau"] = (0., 15.)
    dblplaw["alpha"] = (0.01, 1000.)
    dblplaw["beta"] = (0.01, 1000.)
    dblplaw["alpha_prior"] = "log_10"
    dblplaw["beta_prior"] = "log_10"
    dblplaw["massformed"] = (1., 15.)
    dblplaw["metallicity"] = (0.1, 2.)
    dblplaw["metallicity_prior"] = "log_10"
    
    nebular = {}
    nebular["logU"] = -3.
    
    dust = {}
    dust["type"] = "CF00"
    dust["eta"] = 2.
    dust["Av"] = (0., 2.0)
    dust["n"] = (0.3, 2.5)
    dust["n_prior"] = "Gaussian"
    dust["n_prior_mu"] = 0.7
    dust["n_prior_sigma"] = 0.3
    
    fit_instructions = {}
    fit_instructions["redshift"] = (0.1, 1.2)
    
    fit_instructions["t_bc"] = 0.01
    #fit_instructions["redshift_prior"] = "Gaussian"
    #fit_instructions["redshift_prior_mu"] = 0.92
    #fit_instructions["redshift_prior_sigma"] = 0.05
    fit_instructions["dblplaw"] = dblplaw 
    fit_instructions["nebular"] = nebular
    fit_instructions["dust"] = dust

    fit_instructions["veldisp"] = (1., 1000.)   #km/s
    fit_instructions["veldisp_prior"] = "log_10"

    calib = {}
    calib["type"] = "polynomial_bayesian"
    
    calib["0"] = (0.5, 1.5) # Zero order is centred on 1, at which point there is no change to the spectrum.
    calib["0_prior"] = "Gaussian"
    calib["0_prior_mu"] = 1.0
    calib["0_prior_sigma"] = 0.25
    
    calib["1"] = (-0.5, 0.5) # Subsequent orders are centred on zero.
    calib["1_prior"] = "Gaussian"
    calib["1_prior_mu"] = 0.
    calib["1_prior_sigma"] = 0.25
    
    calib["2"] = (-0.5, 0.5)
    calib["2_prior"] = "Gaussian"
    calib["2_prior_mu"] = 0.
    calib["2_prior_sigma"] = 0.25
    
    fit_instructions["calib"] = calib

    noise = {}
    noise["type"] = "white_scaled"
    noise["scaling"] = (1., 10.)
    noise["scaling_prior"] = "log_10"
    fit_instructions["noise"] = noise

    fit = pipes.fit(galaxy, fit_instructions, run="spectroscopy")

    fit.fit(verbose=False)

    fig = fit.plot_spectrum_posterior(save=True, show=True)
    fig = fit.plot_calibration(save=True, show=True)
    fig = fit.plot_sfh_posterior(save=True, show=True)
    fig = fit.plot_corner(save=True, show=True)

    print("\a    \a    \a")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

