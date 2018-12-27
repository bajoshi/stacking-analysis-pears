from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt

home = os.getenv('HOME')
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figs_dir = home + "/Desktop/FIGS/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
threedhst_datadir = home + "/Desktop/3dhst_data/"
savedir = figs_dir + "stacking-analysis-figures/pears_all_photoz/"

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import photoz as phot

speed_of_light = 299792458e10  # angsroms per second

def main():

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Get directories
    figs_data_dir = "/Volumes/Bhavins_backup/bc03_models_npy_spectra/"
    threedhst_datadir = "/Volumes/Bhavins_backup/3dhst_data/"
    # This is if working on the laptop. 
    # Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(figs_data_dir):
        import pysynphot  # only import pysynphot on firstlight becasue that's the only place where I installed it.
        figs_data_dir = figs_dir  # this path only exists on firstlight
        threedhst_datadir = home + "/Desktop/3dhst_data/"  # this path only exists on firstlight
        if not os.path.isdir(figs_data_dir):
            print "Model files not found. Exiting..."
            sys.exit(0)

    # Flag to turn on-off emission lines in the fit
    use_emlines = True
    num_filters = 12

    # ------------------------------ Add emission lines to models ------------------------------ #
    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_data_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542

    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_lam_grid = model_lam_grid.astype(np.float64)

    total_emission_lines_to_add = 12  # Make sure that this changes if you decide to add more lines to the models
    model_comp_spec_withlines = np.zeros((total_models, len(model_lam_grid) + total_emission_lines_to_add), dtype=np.float64)

    # Read in models with emission lines adn put in numpy array
    bc03_all_spec_hdulist_withlines = fits.open(figs_data_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample_withlines.fits')
    model_lam_grid_withlines = np.load(figs_data_dir + 'model_lam_grid_withlines.npy')
    for q in range(total_models):
        model_comp_spec_withlines[q] = bc03_all_spec_hdulist_withlines[q+1].data

    bc03_all_spec_hdulist_withlines.close()
    del bc03_all_spec_hdulist_withlines

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

    # ---------------------------------- Read in look-up tables for model mags ------------------------------------- #
    # Using the look-up table now since it should be much faster
    # First get them all into an appropriate shape
    u = np.load(figs_data_dir + 'all_model_mags_par_u.npy')
    f435w = np.load(figs_data_dir + 'all_model_mags_par_f435w.npy')
    f606w = np.load(figs_data_dir + 'all_model_mags_par_f606w.npy')
    f775w = np.load(figs_data_dir + 'all_model_mags_par_f775w.npy')
    f850lp = np.load(figs_data_dir + 'all_model_mags_par_f850lp.npy')
    f125w = np.load(figs_data_dir + 'all_model_mags_par_f125w.npy')
    f140w = np.load(figs_data_dir + 'all_model_mags_par_f140w.npy')
    f160w = np.load(figs_data_dir + 'all_model_mags_par_f160w.npy')
    irac1 = np.load(figs_data_dir + 'all_model_mags_par_irac1.npy')
    irac2 = np.load(figs_data_dir + 'all_model_mags_par_irac2.npy')
    irac3 = np.load(figs_data_dir + 'all_model_mags_par_irac3.npy')
    irac4 = np.load(figs_data_dir + 'all_model_mags_par_irac4.npy')

    # put them in a list since I need to iterate over it
    all_model_flam = [u, f435w, f606w, f775w, f850lp, f125w, f140w, f160w, irac1, irac2, irac3, irac4]

    # cnovert to numpy array
    all_model_flam = np.asarray(all_model_flam)

    # ---------------------------------- Read in PEARS photometry ---------------------------------- #
    names_header = ['PearsID', 'field', 'ra', 'dec', 'U_flux', 'f435w_flux', 'f606w_flux', 'f775w_flux', 'f850lp_flux', \
        'f125w_flux', 'f140w_flux', 'f160w_flux', 'irac1_flux', 'irac2_flux', 'irac3_flux', 'irac4_flux', \
        'U_flux_err', 'f435w_flux_err', 'f606w_flux_err', 'f775w_flux_err', 'f850lp_flux_err', \
        'f125w_flux_err', 'f140w_flux_err', 'f160w_flux_err', 'irac1_flux_err', 'irac2_flux_err', 'irac3_flux_err', 'irac4_flux_err']
    pears_phot_cat = np.genfromtxt(stacking_analysis_dir + 'pears_all_photometry.txt', dtype=None, names=names_header)

    # ---------------------------------- Now loop over all galaxies ---------------------------------- #
    for i in range(len(pears_phot_cat)):

        current_id = pears_phot_cat['PearsID'][i]
        current_field = pears_phot_cat['field'][i]

        print "\n"
        print "Galaxies done so far:", i
        print "At ID", current_id, "in", current_field
        print "Total time taken:", time.time() - start, "seconds."

        # ------------------------------- Make unified photometry arrays ------------------------------- #
        flam_U = pears_phot_cat['U_flux'][i]
        flam_f435w = pears_phot_cat['f435w_flux'][i]
        flam_f606w = pears_phot_cat['f606w_flux'][i]
        flam_f775w = pears_phot_cat['f775w_flux'][i]
        flam_f850lp = pears_phot_cat['f850lp_flux'][i]
        flam_f125w = pears_phot_cat['f125w_flux'][i]
        flam_f140w = pears_phot_cat['f140w_flux'][i]
        flam_f160w = pears_phot_cat['f160w_flux'][i]
        flam_irac1 = pears_phot_cat['irac1_flux'][i]
        flam_irac2 = pears_phot_cat['irac2_flux'][i]
        flam_irac3 = pears_phot_cat['irac3_flux'][i]
        flam_irac4 = pears_phot_cat['irac4_flux'][i]

        ferr_U = pears_phot_cat['U_flux_err'][i]
        ferr_f435w = pears_phot_cat['f435w_flux_err'][i]
        ferr_f606w = pears_phot_cat['f606w_flux_err'][i]
        ferr_f775w = pears_phot_cat['f775w_flux_err'][i]
        ferr_f850lp = pears_phot_cat['f850lp_flux_err'][i]
        ferr_f125w = pears_phot_cat['f125w_flux_err'][i]
        ferr_f140w = pears_phot_cat['f140w_flux_err'][i]
        ferr_f160w = pears_phot_cat['f160w_flux_err'][i]
        ferr_irac1 = pears_phot_cat['irac1_flux_err'][i]
        ferr_irac2 = pears_phot_cat['irac2_flux_err'][i]
        ferr_irac3 = pears_phot_cat['irac3_flux_err'][i]
        ferr_irac4 = pears_phot_cat['irac4_flux_err'][i]

        # ------------------------------- Make unified photometry arrays ------------------------------- #
        phot_fluxes_arr = np.array([flam_U, flam_f435w, flam_f606w, flam_f775w, flam_f850lp, flam_f125w, flam_f140w, flam_f160w,
            flam_irac1, flam_irac2, flam_irac3, flam_irac4])
        phot_errors_arr = np.array([ferr_U, ferr_f435w, ferr_f606w, ferr_f775w, ferr_f850lp, ferr_f125w, ferr_f140w, ferr_f160w,
            ferr_irac1, ferr_irac2, ferr_irac3, ferr_irac4])

        # Pivot wavelengths
        # From here --
        # ACS: http://www.stsci.edu/hst/acs/analysis/bandwidths/#keywords
        # WFC3: http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir06.html#400352
        # KPNO/MOSAIC U-band: https://www.noao.edu/kpno/mosaic/filters/k1001.html
        # Spitzer IRAC channels: http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/6/#_Toc410728283
        phot_lam = np.array([3582.0, 4328.2, 5921.1, 7692.4, 9033.1, 12486.0, 13923.0, 15369.0, 
        35500.0, 44930.0, 57310.0, 78720.0])  # angstroms

        # ------------------------------ Now start fitting ------------------------------ #
        # --------- Force dtype for cython code --------- #
        # Apparently this (i.e. for flam_obs and ferr_obs) has  
        # to be done to avoid an obscure error from parallel in joblib --
        # AttributeError: 'numpy.ndarray' object has no attribute 'offset'
        phot_lam = phot_lam.astype(np.float64)
        phot_fluxes_arr = phot_fluxes_arr.astype(np.float64)
        phot_errors_arr = phot_errors_arr.astype(np.float64)

        # ------- Finite photometry values ------- # 
        # Make sure that the photometry arrays all have finite values
        # If any vlues are NaN then throw them out
        phot_fluxes_finite_idx = np.where(np.isfinite(phot_fluxes_arr))[0]
        phot_errors_finite_idx = np.where(np.isfinite(phot_errors_arr))[0]

        phot_fin_idx = reduce(np.intersect1d, (phot_fluxes_finite_idx, phot_errors_finite_idx))

        phot_fluxes_arr = phot_fluxes_arr[phot_fin_idx]
        phot_errors_arr = phot_errors_arr[phot_fin_idx]
        phot_lam = phot_lam[phot_fin_idx]

        # ------------- Call actual fitting function ------------- #
        zp_minchi2, zp, zerr_low, zerr_up, min_chi2, age, tau, av = \
        phot.do_photoz_fitting_lookup(phot_fluxes_arr, phot_errors_arr, phot_lam, \
            model_lam_grid_withlines, total_models, model_comp_spec_withlines, bc03_all_spec_hdulist, start,\
            current_id, current_field, all_model_flam, phot_fin_idx, savedir)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)