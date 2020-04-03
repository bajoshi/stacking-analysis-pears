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

sys.path.append(stacking_analysis_dir + 'stacking_pipeline')
from make_col_ms_plots import get_ur_color

# We need the model info to compute the u-r colors for each matched galaxy.
# This is done outside of any function so 
# that I don't have to call it everytime
# I call get_ur_color_wrapper.
print("\n", end='')
print("Imported/Called match_pears_figs_gencat. Be aware that this module reads model info outside of any function.")
print("\n", end='')
# Read model lambda grid # In agnstroms
model_lam_grid_withlines_mmap = np.load(figs_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')
# Now read the model spectra # In erg s^-1 A^-1
model_comp_spec_llam_withlines_mmap = np.load(figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')

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

def return_matched_data(figs_match_idx, figscat):

    chosen_idx = int(figs_match_idx)

    return figscat['ra'][chosen_idx], figscat['dec'][chosen_idx], figscat['id'][chosen_idx]

def pick_closest_match(current_pears_ra, current_pears_dec, figscat, matched_idx):

    assert len(matched_idx) > 1

    ra_two = current_pears_ra
    dec_two = current_pears_dec

    dist_list = []
    for v in range(len(matched_idx)):

        ra_one = figscat['ra'][matched_idx][v]
        dec_one = figscat['dec'][matched_idx][v]

        dist = np.arccos(np.cos(dec_one*np.pi/180) * np.cos(dec_two*np.pi/180) * \
            np.cos(ra_one*np.pi/180 - ra_two*np.pi/180) + \
            np.sin(dec_one*np.pi/180) * np.sin(dec_two*np.pi/180))
        dist_list.append(dist)

    dist_list = np.asarray(dist_list)
    dist_idx = np.argmin(dist_list)
    chosen_idx = matched_idx[dist_idx] 

    return figscat['ra'][chosen_idx], figscat['dec'][chosen_idx], figscat['id'][chosen_idx]

def get_ur_color_wrapper(index, pearscat):
    """
    Expects index from full results file and the file itself.
    """

    # First get teh full res model spectrum
    # then call get_ur_color
    best_model_idx = pearscat[index]['zp_model_idx']
    current_spec = model_comp_spec_llam_withlines_mmap[best_model_idx]
    ur_col = get_ur_color(current_spec, model_lam_grid_withlines_mmap)

    return ur_col

def main():

    # Read in FIGS catalogs
    gn1cat, gn2cat, gs1cat, gs2cat = get_figs_cats()

    # Read in PEARS results
    pearscat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True, encoding=None)

    # Create empty file for saving final sample
    fh = open(stacking_analysis_dir + 'pears_figs_combined_final_sample.txt', 'w')
    # Copy pasted header from full_pears_results_chabrier.txt and 
    # added the FIGS related quantities at the end along with the u-r color
    header = "#   PearsID  Field  RA  DEC  zspec  zp_minchi2  zspz_minchi2  zg_minchi2" + \
    "  zp  zspz  zg  zp_zerr_low  zp_zerr_up  zspz_zerr_low  zspz_zerr_up  zg_zerr_low  zg_zerr_up" + \
    "  zp_min_chi2  zspz_min_chi2  zg_min_chi2  zp_bestalpha  zspz_bestalpha  zg_bestalpha" + \
    "  zp_model_idx  zspz_model_idx  zg_model_idx  zp_age  zp_tau  zp_av" + \
    "  zspz_age  zspz_tau  zspz_av  zg_age  zg_tau  zg_av" + \
    "  zp_template_ms  zp_ms  zp_sfr  zp_uv  zp_vj  zspz_template_ms" + \
    "  zspz_ms  zspz_sfr  zspz_uv  zspz_vj  zg_template_ms  zg_ms  zg_sfr  zg_uv  zg_vj" + \
    "  figs_id  figs_field  figs_ra  figs_dec  ur_col"
    fh.write(header + "\n")

    # Define error message
    ErrorMessage = "This case should not have happened. Printing all info above for this object."

    # Matching radius # arcseconds in degrees
    ra_lim = 0.3/3600
    dec_lim = 0.3/3600

    # Mass and photoz limits
    mass_lim = 10.5
    zp_low_lim = 0.5
    zp_high_lim = 2.5

    # Now match!
    matches = 0  # This is just to print out how many galaxies are >= the mass limit solar masses within our z range
    for i in range(len(pearscat)):

        # find PEARS ra, dec
        # need stellar mass and zp just to check total galaxies in final sample
        current_pears_ra = float(pearscat['RA'][i])
        current_pears_dec = float(pearscat['DEC'][i])

        ms = np.log10(pearscat['zp_ms'][i])
        zp = pearscat['zp_minchi2'][i]

        # ------------------------- Find FIGS idx ------------------------- #
        if current_pears_dec > 0.0:
            gn1_idx = np.where((gn1cat['ra'] >= current_pears_ra - ra_lim) & (gn1cat['ra'] <= current_pears_ra + ra_lim) & \
                (gn1cat['dec'] >= current_pears_dec - dec_lim) & (gn1cat['dec'] <= current_pears_dec + dec_lim))[0]
            gn2_idx = np.where((gn2cat['ra'] >= current_pears_ra - ra_lim) & (gn2cat['ra'] <= current_pears_ra + ra_lim) & \
                (gn2cat['dec'] >= current_pears_dec - dec_lim) & (gn2cat['dec'] <= current_pears_dec + dec_lim))[0]

            if gn1_idx.size:
                if len(gn1_idx) == 1:
                    gn1_ra, gn1_dec, gn1id = return_matched_data(gn1_idx, gn1cat)
                elif len(gn1_idx) > 1:
                    print("PEARS RA DEC:", current_pears_ra, current_pears_dec, "Multiple matches GN1. Will pick the closest one.")
                    gn1_ra, gn1_dec, gn1id = pick_closest_match(current_pears_ra, current_pears_dec, gn1cat, gn1_idx)
                
                # Now get the u-r color for the match
                ur_col = get_ur_color_wrapper(i, pearscat)

                # Append stuff to the file handle for the file sample
                # This appends all matches regardless of whether they pass the mass or redshift cuts
                fh.write(str(pearscat[i]) + "  ")
                fh.write(str(gn1id) + "  ")
                fh.write("GN1" + "  ")
                fh.write("{:.7f}".format(gn1_ra) + "  ")
                fh.write("{:.6f}".format(gn1_dec) + "  ")
                fh.write("{:.3f}".format(ur_col))
                fh.write("\n")

                if (ms >= mass_lim) and (zp >= zp_low_lim) and (zp <= zp_high_lim):
                    matches += 1

            elif gn2_idx.size:
                if len(gn2_idx) == 1:
                    gn2_ra, gn2_dec, gn2id = return_matched_data(gn2_idx, gn2cat)
                elif len(gn2_idx) > 1:
                    print("PEARS RA DEC:", current_pears_ra, current_pears_dec, "Multiple matches GN2. Will pick the closest one.")
                    gn2_ra, gn2_dec, gn2id = pick_closest_match(current_pears_ra, current_pears_dec, gn2cat, gn2_idx)

                # Now get the u-r color for the match
                ur_col = get_ur_color_wrapper(i, pearscat)

                # Append stuff to the file handle for the file sample
                # This appends all matches regardless of whether they pass the mass or redshift cuts
                fh.write(str(pearscat[i]) + "  ")
                fh.write(str(gn2id) + "  ")
                fh.write("GN2" + "  ")
                fh.write("{:.7f}".format(gn2_ra) + "  ")
                fh.write("{:.6f}".format(gn2_dec) + "  ")
                fh.write("{:.3f}".format(ur_col))
                fh.write("\n")

                if (ms >= mass_lim) and (zp >= zp_low_lim) and (zp <= zp_high_lim):
                    matches += 1

            elif (not gn1_idx.size) and (not gn2_idx.size):
                continue

            else:
                print(current_pears_ra, current_pears_dec)
                print(gn1_idx, gn2_idx)
                raise ValueError(ErrorMessage)

        elif current_pears_dec < 0.0:
            gs1_idx = np.where((gs1cat['ra'] >= current_pears_ra - ra_lim) & (gs1cat['ra'] <= current_pears_ra + ra_lim) & \
                (gs1cat['dec'] >= current_pears_dec - dec_lim) & (gs1cat['dec'] <= current_pears_dec + dec_lim))[0]
            gs2_idx = np.where((gs2cat['ra'] >= current_pears_ra - ra_lim) & (gs2cat['ra'] <= current_pears_ra + ra_lim) & \
                (gs2cat['dec'] >= current_pears_dec - dec_lim) & (gs2cat['dec'] <= current_pears_dec + dec_lim))[0]

            if gs1_idx.size:
                if len(gs1_idx) == 1:
                    gs1_ra, gs1_dec, gs1id = return_matched_data(gs1_idx, gs1cat)
                elif len(gs1_idx) > 1:
                    print("PEARS RA DEC:", current_pears_ra, current_pears_dec, "Multiple matches GS1. Will pick the closest one.")
                    gs1_ra, gs1_dec, gs1id = pick_closest_match(current_pears_ra, current_pears_dec, gs1cat, gs1_idx)

                # Now get the u-r color for the match
                ur_col = get_ur_color_wrapper(i, pearscat)

                # Append stuff to the file handle for the file sample
                # This appends all matches regardless of whether they pass the mass or redshift cuts
                fh.write(str(pearscat[i]) + "  ")
                fh.write(str(gs1id) + "  ")
                fh.write("GS1" + "  ")
                fh.write("{:.7f}".format(gs1_ra) + "  ")
                fh.write("{:.6f}".format(gs1_dec) + "  ")
                fh.write("{:.3f}".format(ur_col))
                fh.write("\n")

                if (ms >= mass_lim) and (zp >= zp_low_lim) and (zp <= zp_high_lim):
                    matches += 1

            elif gs2_idx.size:
                if len(gs2_idx) == 1:
                    gs2_ra, gs2_dec, gs2id = return_matched_data(gs2_idx, gs2cat)
                elif len(gs2_idx) > 1:
                    print("PEARS RA DEC:", current_pears_ra, current_pears_dec, "Multiple matches GS2. Will pick the closest one.")
                    gs2_ra, gs2_dec, gs2id = pick_closest_match(current_pears_ra, current_pears_dec, gs2cat, gs2_idx)

                # Now get the u-r color for the match
                ur_col = get_ur_color_wrapper(i, pearscat)

                # Append stuff to the file handle for the file sample
                # This appends all matches regardless of whether they pass the mass or redshift cuts
                fh.write(str(pearscat[i]) + "  ")
                fh.write(str(gs2id) + "  ")
                fh.write("GS2" + "  ")
                fh.write("{:.7f}".format(gs2_ra) + "  ")
                fh.write("{:.6f}".format(gs2_dec) + "  ")
                fh.write("{:.3f}".format(ur_col))
                fh.write("\n")

                if (ms >= mass_lim) and (zp >= zp_low_lim) and (zp <= zp_high_lim):
                    matches += 1

            elif (not gs1_idx.size) and (not gs2_idx.size):
                continue

            else:
                print(current_pears_ra, current_pears_dec)
                print(gs1_idx, gs2_idx)
                raise ValueError(ErrorMessage)

    fh.close()
    print("Total galaxies with PEARS+FIGS data that are >= 10^%.1f solar masses within %.1f <= z <= %.1f: %d" \
        % (mass_lim, zp_low_lim, zp_high_lim, matches))

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

