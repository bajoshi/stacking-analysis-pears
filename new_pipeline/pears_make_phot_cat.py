from __future__ import division

import numpy as np
from astropy.io import fits
import pysynphot

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figs_dir = home + "/Desktop/FIGS/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
threedhst_datadir = home + "/Desktop/3dhst_data/"

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
from pears_and_3dhst import read_3dhst_cats
import fullfitting_grism_broadband_emlines as ff

speed_of_light = 299792458e10  # angsroms per second

def get_all_filters():

    # ------------------------------- Read in filter curves ------------------------------- #
    f435w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f435w')
    f606w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f606w')
    f775w_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f775w')
    f850lp_filt_curve = pysynphot.ObsBandpass('acs,wfc1,f850lp')

    f125w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f125w')
    f140w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f140w')
    f160w_filt_curve = pysynphot.ObsBandpass('wfc3,ir,f160w')

    # non-HST filter curves
    # IRac wavelengths are in mixrons # convert to angstroms
    uband_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/kpno_mosaic_u.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=14)
    irac1_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac1.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac2_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac2.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac3_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac3.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)
    irac4_curve = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/irac4.txt', dtype=None, \
        names=['wav', 'trans'], skip_header=3)

    irac1_curve['wav'] *= 1e4
    irac2_curve['wav'] *= 1e4
    irac3_curve['wav'] *= 1e4
    irac4_curve['wav'] *= 1e4

    all_filters = [uband_curve, f435w_filt_curve, f606w_filt_curve, f775w_filt_curve, f850lp_filt_curve, \
    f125w_filt_curve, f140w_filt_curve, f160w_filt_curve, irac1_curve, irac2_curve, irac3_curve, irac4_curve]

    return all_filters

def main():
    
    # ------------------------------- Read in PEARS + 3DHST catalogs ------------------------------- #
    # Read 3dhst cats
    threed_ncat, threed_scat, threed_v41_phot = read_3dhst_cats()

    # Read PEARS cats
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['pearsid', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['pearsid', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['dec'] = pears_ncat['dec'] - dec_offset_goodsn_v19

    # ------------------------------- Read in photometry catalogs ------------------------------- #
    # GOODS-N from 3DHST
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
    'f_U', 'e_U', 'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
    'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), skip_header=3)

    # ------------------------------- Other preliminary stuff before looping ------------------------------- #
    all_filters = get_all_filters()

    # Lists to loop over
    all_master_cats = [pears_ncat, pears_scat]

    # Read in Vega spectrum and get it in the appropriate forms
    vega = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/' + 'vega_reference.dat', dtype=None, \
        names=['wav', 'flam'], skip_header=7)

    vega_lam = vega['wav']
    vega_spec_flam = vega['flam']
    vega_nu = speed_of_light / vega_lam
    vega_spec_fnu = vega_lam**2 * vega_spec_flam / speed_of_light

    # save lists for comparing after code is done
    id_list = []
    field_list = []
    ra_list = []
    dec_list = []

    u_flux_list = []
    f435w_flux_list = []
    f606w_flux_list = []
    f775w_flux_list = []
    f850lp_flux_list = []
    f125w_flux_list = []
    f140w_flux_list = []
    f160w_flux_list = []
    irac1_flux_list = []
    irac2_flux_list = []
    irac3_flux_list = []
    irac4_flux_list = []

    u_flux_err_list = []
    f435w_flux_err_list = []
    f606w_flux_err_list = []
    f775w_flux_err_list = []
    f850lp_flux_err_list = []
    f125w_flux_err_list = []
    f140w_flux_err_list = []
    f160w_flux_err_list = []
    irac1_flux_err_list = []
    irac2_flux_err_list = []
    irac3_flux_err_list = []
    irac4_flux_err_list = []

    fluxes_list_of_lists = [u_flux_list, f435w_flux_list, f606w_flux_list, f775w_flux_list, f850lp_flux_list, \
    f125w_flux_list, f140w_flux_list, f160w_flux_list, irac1_flux_list, irac2_flux_list, irac3_flux_list, irac4_flux_list]

    flux_errors_list_of_lists = [u_flux_err_list, f435w_flux_err_list, f606w_flux_err_list, f775w_flux_err_list, f850lp_flux_err_list, \
    f125w_flux_err_list, f140w_flux_err_list, f160w_flux_err_list, \
    irac1_flux_err_list, irac2_flux_err_list, irac3_flux_err_list, irac4_flux_err_list]

    # ------------------------------- Looping over all PEARS objects ------------------------------- #
    # Loop over all objects in the PEARS master catalogs
    # and get photo-z for each object as well as the photometry
    # start looping
    catcount = 0
    galaxy_count = 0
    for cat in all_master_cats:
        for i in xrange(len(cat)):
            # --------------------------------------------- GET OBS DATA ------------------------------------------- #
            current_id = cat['pearsid'][i]

            if catcount == 0:
                current_field = 'GOODS-N'
                phot_cat_3dhst = goodsn_phot_cat_3dhst
            elif catcount == 1: 
                current_field = 'GOODS-S'
                phot_cat_3dhst = goodss_phot_cat_3dhst
    
            threed_ra = phot_cat_3dhst['ra']
            threed_dec = phot_cat_3dhst['dec']

            print "At ID:", current_id, "in:", current_field

            # ------------------------------- Match and get photometry data ------------------------------- #
            # find obj ra,dec
            cat_idx = np.where(cat['pearsid'] == current_id)[0]
            if cat_idx.size:
                current_ra = float(cat['ra'][cat_idx])
                current_dec = float(cat['dec'][cat_idx])

            # Now match
            ra_lim = 0.5/3600  # arcseconds in degrees
            dec_lim = 0.5/3600
            threed_phot_idx = np.where((threed_ra >= current_ra - ra_lim) & (threed_ra <= current_ra + ra_lim) & \
                (threed_dec >= current_dec - dec_lim) & (threed_dec <= current_dec + dec_lim))[0]

            """
            If there are multiple matches with the photometry catalog 
            within 0.5 arseconds then choose the closest one.
            """
            if len(threed_phot_idx) > 1:

                ra_two = current_ra
                dec_two = current_dec

                dist_list = []
                for v in range(len(threed_phot_idx)):

                    ra_one = threed_ra[threed_phot_idx][v]
                    dec_one = threed_dec[threed_phot_idx][v]

                    dist = np.arccos(np.cos(dec_one*np.pi/180) * np.cos(dec_two*np.pi/180) * np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + 
                        np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
                    dist_list.append(dist)

                dist_list = np.asarray(dist_list)
                dist_idx = np.argmin(dist_list)
                threed_phot_idx = threed_phot_idx[dist_idx]

            elif len(threed_phot_idx) == 0:
                print "Match not found in Photmetry catalog. Skipping."
                continue

            # ------------------------------- Get photometric fluxes and their errors ------------------------------- #
            flam_f435w = ff.get_flam('F435W', phot_cat_3dhst['f_F435W'][threed_phot_idx])
            flam_f606w = ff.get_flam('F606W', phot_cat_3dhst['f_F606W'][threed_phot_idx])
            flam_f775w = ff.get_flam('F775W', phot_cat_3dhst['f_F775W'][threed_phot_idx])
            flam_f850lp = ff.get_flam('F850LP', phot_cat_3dhst['f_F850LP'][threed_phot_idx])
            flam_f125w = ff.get_flam('F125W', phot_cat_3dhst['f_F125W'][threed_phot_idx])
            flam_f140w = ff.get_flam('F140W', phot_cat_3dhst['f_F140W'][threed_phot_idx])
            flam_f160w = ff.get_flam('F160W', phot_cat_3dhst['f_F160W'][threed_phot_idx])

            flam_U = ff.get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['f_U'][threed_phot_idx], \
                vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
            flam_irac1 = ff.get_flam_nonhst('irac1', phot_cat_3dhst['f_IRAC1'][threed_phot_idx], \
                vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
            flam_irac2 = ff.get_flam_nonhst('irac2', phot_cat_3dhst['f_IRAC2'][threed_phot_idx], \
                vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
            flam_irac3 = ff.get_flam_nonhst('irac3', phot_cat_3dhst['f_IRAC3'][threed_phot_idx], \
                vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
            flam_irac4 = ff.get_flam_nonhst('irac4', phot_cat_3dhst['f_IRAC4'][threed_phot_idx], \
                vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

            ferr_f435w = ff.get_flam('F435W', phot_cat_3dhst['e_F435W'][threed_phot_idx])
            ferr_f606w = ff.get_flam('F606W', phot_cat_3dhst['e_F606W'][threed_phot_idx])
            ferr_f775w = ff.get_flam('F775W', phot_cat_3dhst['e_F775W'][threed_phot_idx])
            ferr_f850lp = ff.get_flam('F850LP', phot_cat_3dhst['e_F850LP'][threed_phot_idx])
            ferr_f125w = ff.get_flam('F125W', phot_cat_3dhst['e_F125W'][threed_phot_idx])
            ferr_f140w = ff.get_flam('F140W', phot_cat_3dhst['e_F140W'][threed_phot_idx])
            ferr_f160w = ff.get_flam('F160W', phot_cat_3dhst['e_F160W'][threed_phot_idx])

            ferr_U = ff.get_flam_nonhst('kpno_mosaic_u', phot_cat_3dhst['e_U'][threed_phot_idx], \
                vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
            ferr_irac1 = ff.get_flam_nonhst('irac1', phot_cat_3dhst['e_IRAC1'][threed_phot_idx], \
                vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
            ferr_irac2 = ff.get_flam_nonhst('irac2', phot_cat_3dhst['e_IRAC2'][threed_phot_idx], \
                vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
            ferr_irac3 = ff.get_flam_nonhst('irac3', phot_cat_3dhst['e_IRAC3'][threed_phot_idx], \
                vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)
            ferr_irac4 = ff.get_flam_nonhst('irac4', phot_cat_3dhst['e_IRAC4'][threed_phot_idx], \
                vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam)

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
            phot_lam = np.array([3582.0, 4328.2, 5921.1, 7692.4, 9033.1, 12486, 13923, 15369, 
            35500.0, 44930.0, 57310.0, 78720.0])  # angstroms

            # Now append to the lists            
            id_list.append(current_id)
            field_list.append(current_field)
            ra_list.append(current_ra)
            dec_list.append(current_dec)

            # ------- Finite photometry and error values ------- #
            for k in range(len(phot_fluxes_arr)):

                #  Check fluxes
                if not np.isfinite(phot_fluxes_arr[k]):
                    fluxes_list_of_lists[k].append(-99.0)
                else:
                    fluxes_list_of_lists[k].append(phot_fluxes_arr[k])

                # Same for errors
                # Check has to be done separately since they're two separate lists
                if not np.isfinite(phot_errors_arr[k]):
                    flux_errors_list_of_lists[k].append(-99.0)
                else:
                    flux_errors_list_of_lists[k].append(phot_errors_arr[k])

            galaxy_count += 1

        catcount += 1

    print "Total galaxies considered:", galaxy_count

    # ------------------------------- Save to plain text file ------------------------------- #
    # First convert ot numpy arrays
    id_list = np.asarray(id_list)
    field_list = np.asarray(field_list)
    ra_list = np.asarray(ra_list)
    dec_list = np.asarray(dec_list)

    u_flux_list = np.asarray(u_flux_list)
    f435w_flux_list = np.asarray(f435w_flux_list)
    f606w_flux_list = np.asarray(f606w_flux_list)
    f775w_flux_list = np.asarray(f775w_flux_list)
    f850lp_flux_list = np.asarray(f850lp_flux_list)
    f125w_flux_list = np.asarray(f125w_flux_list)
    f140w_flux_list = np.asarray(f140w_flux_list)
    f160w_flux_list = np.asarray(f160w_flux_list)
    irac1_flux_list = np.asarray(irac1_flux_list)
    irac2_flux_list = np.asarray(irac2_flux_list)
    irac3_flux_list = np.asarray(irac3_flux_list)
    irac4_flux_list = np.asarray(irac4_flux_list)

    u_flux_err_list = np.asarray(u_flux_err_list)
    f435w_flux_err_list = np.asarray(f435w_flux_err_list)
    f606w_flux_err_list = np.asarray(f606w_flux_err_list)
    f775w_flux_err_list = np.asarray(f775w_flux_err_list)
    f850lp_flux_err_list = np.asarray(f850lp_flux_err_list)
    f125w_flux_err_list = np.asarray(f125w_flux_err_list)
    f140w_flux_err_list = np.asarray(f140w_flux_err_list)
    f160w_flux_err_list = np.asarray(f160w_flux_err_list)
    irac1_flux_err_list = np.asarray(irac1_flux_err_list)
    irac2_flux_err_list = np.asarray(irac2_flux_err_list)
    irac3_flux_err_list = np.asarray(irac3_flux_err_list)
    irac4_flux_err_list = np.asarray(irac4_flux_err_list)

    # Now save as one large ascii file
    data = np.array(zip(id_list, field_list, ra_list, dec_list, \
        u_flux_list, f435w_flux_list, f606w_flux_list, f775w_flux_list, f850lp_flux_list, \
        f125w_flux_list, f140w_flux_list, f160w_flux_list, irac1_flux_list, irac2_flux_list, irac3_flux_list, irac4_flux_list, \
        u_flux_err_list, f435w_flux_err_list, f606w_flux_err_list, f775w_flux_err_list, f850lp_flux_err_list, f125w_flux_err_list, \
        f140w_flux_err_list, f160w_flux_err_list, \
        irac1_flux_err_list, irac2_flux_err_list, irac3_flux_err_list, irac4_flux_err_list), \
        dtype=[('id_list', int), ('field_list', '|S7'), ('ra_list', float), ('dec_list', float), \
        ('u_flux_list', float), ('f435w_flux_list', float), ('f606w_flux_list', float), ('f775w_flux_list', float), \
        ('f850lp_flux_list', float), ('f125w_flux_list', float), ('f140w_flux_list', float), ('f160w_flux_list', float), \
        ('irac1_flux_list', float), ('irac2_flux_list', float), ('irac3_flux_list', float), ('irac4_flux_list', float), \
        ('u_flux_err_list', float), ('f435w_flux_err_list', float), ('f606w_flux_err_list', float), ('f775w_flux_err_list', float), 
        ('f850lp_flux_err_list', float), ('f125w_flux_err_list', float), ('f140w_flux_err_list', float), ('f160w_flux_err_list', float), \
        ('irac1_flux_err_list', float), ('irac2_flux_err_list', float), ('irac3_flux_err_list', float), ('irac4_flux_err_list', float)])

    np.savetxt(stacking_analysis_dir + 'pears_all_photometry.txt', data, \
        fmt=['%d', '%s', '%.6f', '%.6f', \
        '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', \
        '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e', '%.3e'], delimiter=' ', \
        header='PearsID field ra dec U_flux f435w_flux f606w_flux f775w_flux f850lp_flux' + \
        ' f125w_flux f140w_flux f160w_flux irac1_flux irac2_flux irac3_flux irac4_flux' + \
        ' U_flux_err f435w_flux_err f606w_flux_err f775w_flux_err f850lp_flux_err' + \
        ' f125w_flux_err f140w_flux_err f160w_flux_err irac1_flux_err irac2_flux_err irac3_flux_err irac4_flux_err')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)