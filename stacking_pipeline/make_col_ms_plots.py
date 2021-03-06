from __future__ import division

import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import simps
from scipy import stats

import os
import sys
import glob

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

home = os.getenv('HOME')
figs_dir = home + '/Desktop/FIGS/'

stacking_analysis_dir = figs_dir + 'stacking-analysis-pears/'
stacking_figures_dir = figs_dir + 'stacking-analysis-figures/'

# Get correct directory for 3D-HST data
if 'firstlight' in os.uname()[1]:
    threedhst_datadir = '/Users/baj/Desktop/3dhst_data/'
else:
    threedhst_datadir = '/Volumes/Bhavins_backup/3dhst_data/'

def compute_fnu(spec_llam, spec_wav, filt_wav, filt_thru):

    # Now convert to f_nu before you do anything
    speed_of_light_ang = 2.99792458e18  # angstroms per second

    filt_nu = speed_of_light_ang / filt_wav

    spec_nu = speed_of_light_ang / spec_wav
    spec_lnu = (spec_wav)**2 * spec_llam / speed_of_light_ang

    # Reverse arrays to get frequency in ascending order.
    # The griddata function on the filter doesn't work
    # if the frequencies are in decending order, which
    # they will be when converted from wavelength.
    filt_nu = filt_nu[::-1]
    filt_thru = filt_thru[::-1]

    spec_nu = spec_nu[::-1]
    spec_lnu = spec_lnu[::-1]

    # Now convert to fluxes.
    # i.e., put your model spectrum at a distance of 10 pc.
    # While this doesn't affect the color computation
    # this is to get some sensible number for the fnu
    # and the magnitude computation.
    # Therefore my magnitudes will now be absolute magnitudes.
    # Keep in mind that the magnitude numbers are for a model
    # spectrum which is normalized to 1 solar mass. 
    # So expect numbers in the ballpark of what you'd get if you
    # put the Sun 10 pc away.
    dl_10pc_cm = 3.086e19  # 10 parsecs in cm
    spec_fnu = spec_lnu / (4 * np.pi * dl_10pc_cm * dl_10pc_cm)

    # Now compute magnitudes
    # First, interpolate the filter curve to the model lambda/nu grid
    filt_interp = griddata(points=filt_nu, values=filt_thru, xi=spec_nu, method='linear')

    # Set nan values in interpolated filter to 0.0
    filt_nan_idx = np.where(np.isnan(filt_interp))[0]
    filt_interp[filt_nan_idx] = 0.0

    # Second, compute f_nu
    num = simps(y=spec_fnu * filt_interp / spec_nu, x=spec_nu)
    den = simps(y=filt_interp / spec_nu, x=spec_nu)

    fnu = num / den

    #print "{:.3e}".format(num), "{:.3e}".format(den), "{:.3e}".format(fnu)

    return fnu

def get_threed_match_idx(current_ra, current_dec, threed_ra, threed_dec):

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
        # Raise IndexError if match not found.
        # Because the code that generated the full pears sample
        # already matched with 3dhst and required a match to
        # be found to be included in the final full pears sample.
        print( "Match not found. This should not have happened. Exiting.")
        print("At ID and Field:", current_id, current_field)
        raise IndexError
        sys.exit(1)

    return threed_phot_idx

def get_threed_ur(current_ra, current_dec, threed_ra, threed_dec, phot_cat_3dhst):

    threed_phot_idx = get_threed_match_idx(current_ra, current_dec, threed_ra, threed_dec)

    # Now get 3D-HST u-r color
    threed_uflux = float(phot_cat_3dhst['f_U'][threed_phot_idx])
    threed_rflux = float(phot_cat_3dhst['f_R'][threed_phot_idx])

    threed_umag = 25.0 - 2.5 * np.log10(threed_uflux)
    threed_rmag = 25.0 - 2.5 * np.log10(threed_rflux)

    threed_ur = threed_umag - threed_rmag

    #print threed_phot_idx, "{:.3e}".format(threed_uflux), "{:.3e}".format(threed_rflux), "{:.3f}".format(threed_ur)

    return threed_ur

def get_threed_uv_vj(current_ra, current_dec, threed_ra, threed_dec, phot_cat_3dhst):

    threed_phot_idx = get_threed_match_idx(current_ra, current_dec, threed_ra, threed_dec)

    # Now get 3D-HST u-v and v-j colors
    threed_uflux = float(phot_cat_3dhst['f_U'][threed_phot_idx])
    threed_vflux = float(phot_cat_3dhst['f_V'][threed_phot_idx])
    threed_jflux = float(phot_cat_3dhst['f_J'][threed_phot_idx])

    threed_umag = 25.0 - 2.5 * np.log10(threed_uflux)
    threed_vmag = 25.0 - 2.5 * np.log10(threed_vflux)
    threed_jmag = 25.0 - 2.5 * np.log10(threed_jflux)

    threed_uv = threed_umag - threed_vmag
    threed_vj = threed_vmag - threed_jmag

    return threed_uv, threed_vj 

def add_contours(x, y, ax):

    # plot contours for point density
    counts, xbins, ybins = np.histogram2d(x, y, bins=25, normed=False)
    levels_to_plot = [1.5, 3.5, 6.5]

    #c = ax.contour(counts.transpose(), levels=levels_to_plot, \
    #    extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], \
    #    cmap=cm.viridis, linestyles='solid', zorder=10)

    #Perform a kernel density estimate on the data:
    xmin = 7.0  # x.min()
    xmax = 12.0  # x.max()
    ymin = -0.3  # y.min()
    ymax = 3.3  # y.max()
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, aspect='auto', extent=[xmin, xmax, ymin, ymax], zorder=1, alpha=0.95)

    # Add the ticks corresponding to the stacking grid
    ax.set_yticks(np.arange(0.0,3.5,0.5))
    ax.set_xticks(np.arange(8.0,13.0,1.0))

    # Plot grid 
    # I want this to have the lowest zorder
    grid_color = (0.9,0.9,0.9)
    ax.grid(True, color=grid_color, zorder=2)

    return None

def generate_all_ur_color():

    # Read in results for all of PEARS
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    # Define empty array for ur_color
    ur = np.zeros(len(cat))

    # Read model lambda grid # In agnstroms
    model_lam_grid_withlines_mmap = np.load(figs_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')
    # Now read the model spectra # In erg s^-1 A^-1
    model_comp_spec_llam_withlines_mmap = np.load(figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')

    # ------------------------------- Get required filter curves ------------------------------- # 
    uvj_filt_dir = stacking_analysis_dir + 'filter_curves/'

    # Wavelenght is in angstroms
    # Throughput is an absolute fraction
    u = np.genfromtxt(uvj_filt_dir + 'bessel_u.dat', dtype=None, names=['wav', 'filter_trans'])
    r = np.genfromtxt(uvj_filt_dir + 'bessel_r.dat', dtype=None, names=['wav', 'filter_trans'])

    # The Bessel filters are given in angstroms and fractions
    u_filt_wav = u['wav']
    u_filt_thru = u['filter_trans']

    r_filt_wav = r['wav']
    r_filt_thru = r['filter_trans']

    # ------------------------------- Now begin looping over all stellar mass selected galaxies ------------------------------- #
    for i in range(len(cat)):
        # First get teh full res model spectrum
        best_model_idx = cat[i]['zp_model_idx']
        current_spec = model_comp_spec_llam_withlines_mmap[best_model_idx]
            
        # Now get the u and r magnitudes
        ufnu = compute_fnu(current_spec, model_lam_grid_withlines_mmap, u_filt_wav, u_filt_thru)
        rfnu = compute_fnu(current_spec, model_lam_grid_withlines_mmap, r_filt_wav, r_filt_thru)
        umag = -2.5 * np.log10(ufnu) - 48.60
        rmag = -2.5 * np.log10(rfnu) - 48.60
        current_ur = umag - rmag

        ur[i] = current_ur

    # Now save it as a numpy array
    np.save(stacking_analysis_dir + 'ur_arr_all.npy', ur)

    return None

def get_ur_color(spec_llam, model_lam_grid):
    """
    Will compute the u-r color of a model spectrum.
    Expects a spectrum in L_lambda units along with
    wavelength in angstroms.
    """

    # ------------------------------- Get required filter curves ------------------------------- # 
    uvj_filt_dir = stacking_analysis_dir + 'filter_curves/'

    # Wavelenght is in angstroms
    # Throughput is an absolute fraction
    u = np.genfromtxt(uvj_filt_dir + 'bessel_u.dat', dtype=None, names=['wav', 'filter_trans'])
    r = np.genfromtxt(uvj_filt_dir + 'bessel_r.dat', dtype=None, names=['wav', 'filter_trans'])

    # The Bessel filters are given in angstroms and fractions
    u_filt_wav = u['wav']
    u_filt_thru = u['filter_trans']

    r_filt_wav = r['wav']
    r_filt_thru = r['filter_trans']

    # Now get the u and r magnitudes
    ufnu = compute_fnu(spec_llam, model_lam_grid, u_filt_wav, u_filt_thru)
    rfnu = compute_fnu(spec_llam, model_lam_grid, r_filt_wav, r_filt_thru)
    umag = -2.5 * np.log10(ufnu) - 48.60
    rmag = -2.5 * np.log10(rfnu) - 48.60
    ur = umag - rmag

    return ur

def ur_ms_plots():

    # Read in results for all of PEARS
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    # First get stellar mass and apply a cut to stellar mass
    ms = np.log10(cat['zp_ms'])
    low_mass_lim = 8.0
    ms_idx = np.where((ms >= low_mass_lim) & (ms <= 12.0))[0]  # Change the npy arrays to save and load accordingly
    if int(low_mass_lim) == 8:
        mass_str = '8_logM_12'
    elif int(low_mass_lim) == 9:
        mass_str = '9_logM_12'
    print("Galaxies from stellar mass cut:", len(ms_idx))

    # Now get indices based on redshift intervals
    # decided from the stellar_mass_diagnostic_plot.py
    # code. 
    # Look at the make_z_hist() function in there. 
    zp = cat['zp_minchi2'][ms_idx]
    ms = ms[ms_idx]

    # now loop over all galaxies within mass cut sample 
    # and get the u-r color for each galaxy
    if not os.path.isfile(stacking_analysis_dir + 'ur_arr_' + mass_str + '.npy'):

        ur = []
        threed_ur = []

        # Read model lambda grid # In agnstroms
        model_lam_grid_withlines_mmap = np.load(figs_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')
        # Now read the model spectra # In erg s^-1 A^-1
        model_comp_spec_llam_withlines_mmap = np.load(figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')

        # ------------------------------- Read in photometry catalogs ------------------------------- #
        # GOODS photometry catalogs from 3DHST
        # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
        # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
        photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
        'f_R', 'e_R', 'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
        'f_U', 'e_U', 'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
        'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
        goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cat', \
            dtype=None, names=photometry_names, \
            usecols=(0,3,4, 9,10, 15,16, 27,28, 30,31, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), \
            skip_header=3)
        goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cat', \
            dtype=None, names=photometry_names, \
            usecols=(0,3,4, 9,10, 18,19, 30,31, 33,34, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), \
            skip_header=3)

        # ------------------------------- Get required filter curves ------------------------------- # 
        uvj_filt_dir = stacking_analysis_dir + 'filter_curves/'

        # Wavelenght is in angstroms
        # Throughput is an absolute fraction
        u = np.genfromtxt(uvj_filt_dir + 'bessel_u.dat', dtype=None, names=['wav', 'filter_trans'])
        r = np.genfromtxt(uvj_filt_dir + 'bessel_r.dat', dtype=None, names=['wav', 'filter_trans'])

        # The Bessel U and R filters are given in angstroms and fractions
        u_filt_wav = u['wav']
        u_filt_thru = u['filter_trans']

        r_filt_wav = r['wav']
        r_filt_thru = r['filter_trans']

        # ------------------------------- Now begin looping over all stellar mass selected galaxies ------------------------------- #
        for idx in ms_idx:
            # First get teh full res model spectrum
            best_model_idx = cat[idx]['zp_model_idx']
            current_spec = model_comp_spec_llam_withlines_mmap[best_model_idx]
            
            # Now get the u and r magnitudes
            ufnu = compute_fnu(current_spec, model_lam_grid_withlines_mmap, u_filt_wav, u_filt_thru)
            rfnu = compute_fnu(current_spec, model_lam_grid_withlines_mmap, r_filt_wav, r_filt_thru)
            umag = -2.5 * np.log10(ufnu) - 48.60
            rmag = -2.5 * np.log10(rfnu) - 48.60
            current_ur = umag - rmag

            ur.append(current_ur)

            """
            # Check the 3DHST u-r color as well
            # They've used different filters so don't expect an exact match
            current_ra = cat[idx]['RA']
            current_dec = cat[idx]['DEC']
            current_field = cat[idx]['Field']

            # Assign catalogs 
            if current_field == 'GOODS-N':
                phot_cat_3dhst = goodsn_phot_cat_3dhst
            elif current_field == 'GOODS-S':
                phot_cat_3dhst = goodss_phot_cat_3dhst

            threed_ra = phot_cat_3dhst['ra']
            threed_dec = phot_cat_3dhst['dec']

            current_threed_ur = get_threed_ur(current_ra, current_dec, threed_ra, threed_dec, phot_cat_3dhst)
            threed_ur.append(current_threed_ur)
            """

            #print best_model_idx, "{:.3f}".format(umag), "{:.3f}".format(rmag), "{:.3f}".format(current_ur)

        # Convert to numpy array
        ur = np.asarray(ur)
        #threed_ur = np.asarray(threed_ur)
    
        np.save(stacking_analysis_dir + 'ur_arr_' + mass_str + '.npy', ur)

    else:
        ur = np.load(stacking_analysis_dir + 'ur_arr_' + mass_str + '.npy')

    print("Minimum and maximum in computed u-r color array:")
    print("Min:", min(ur), "             ", "Max:", max(ur))

    # Get z intervals and their indices
    z_interval1_idx = np.where((zp >= 0.0) & (zp < 0.4))[0]
    z_interval2_idx = np.where((zp >= 0.4) & (zp < 0.7))[0]
    z_interval3_idx = np.where((zp >= 0.7) & (zp < 1.0))[0]
    z_interval4_idx = np.where((zp >= 1.0) & (zp < 2.0))[0]
    z_interval5_idx = np.where((zp >= 2.0) & (zp <= 6.0))[0]

    print("Number of galaxies within each redshift interval.")
    print("0.0 <= z < 0.4", "    ", len(z_interval1_idx))
    print("0.4 <= z < 0.7", "    ", len(z_interval2_idx))
    print("0.7 <= z < 1.0", "    ", len(z_interval3_idx))
    print("1.0 <= z < 2.0", "    ", len(z_interval4_idx))
    print("2.0 <= z <= 6.0", "    ", len(z_interval5_idx))

    # Now make the plots
    # Define figure
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(10,12)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.25, hspace=0.25)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:5, :4])
    ax2 = fig.add_subplot(gs[:5, 4:8])
    ax3 = fig.add_subplot(gs[:5, 8:])
    ax4 = fig.add_subplot(gs[5:, :4])
    ax5 = fig.add_subplot(gs[5:, 4:8])
    ax6 = fig.add_subplot(gs[5:, 8:])

    # Labels
    ax5.set_xlabel(r'$\rm log(M_s)\ [M_\odot]$', fontsize=15)
    ax4.set_ylabel(r'$(u - r)_\mathrm{restframe}$', fontsize=15)
    ax4.yaxis.set_label_coords(-0.12, 1.05)

    # Actual plotting
    ax1.scatter(ms[z_interval1_idx], ur[z_interval1_idx], s=0.7, color='k', zorder=3)
    ax2.scatter(ms[z_interval2_idx], ur[z_interval2_idx], s=0.7, color='k', zorder=3)
    ax3.scatter(ms[z_interval3_idx], ur[z_interval3_idx], s=0.7, color='k', zorder=3)
    ax4.scatter(ms[z_interval4_idx], ur[z_interval4_idx], s=0.7, color='k', zorder=3)
    ax5.scatter(ms[z_interval5_idx], ur[z_interval5_idx], s=0.7, color='k', zorder=3)
    ax6.scatter(ms, ur, s=0.7, color='k', zorder=3)

    # Put contours on each plot
    add_contours(ms[z_interval1_idx], ur[z_interval1_idx], ax1)
    add_contours(ms[z_interval2_idx], ur[z_interval2_idx], ax2)
    add_contours(ms[z_interval3_idx], ur[z_interval3_idx], ax3)
    add_contours(ms[z_interval4_idx], ur[z_interval4_idx], ax4)
    add_contours(ms[z_interval5_idx], ur[z_interval5_idx], ax5)
    add_contours(ms, ur, ax6)

    # Add text 
    add_info_text_to_subplots(ax1, 0.0, 0.4, len(z_interval1_idx))
    add_info_text_to_subplots(ax2, 0.4, 0.7, len(z_interval2_idx))
    add_info_text_to_subplots(ax3, 0.7, 1.0, len(z_interval3_idx))
    add_info_text_to_subplots(ax4, 1.0, 2.0, len(z_interval4_idx))
    add_info_text_to_subplots(ax5, 2.0, 6.0, len(z_interval5_idx))
    add_info_text_to_subplots(ax6, 0.0, 6.0, len(zp))

    # Axes limits 
    ax1.set_xlim(7.0, 12.0)
    ax2.set_xlim(7.0, 12.0)
    ax3.set_xlim(7.0, 12.0)
    ax4.set_xlim(7.0, 12.0)
    ax5.set_xlim(7.0, 12.0)
    ax6.set_xlim(7.0, 12.0)

    ax1.set_ylim(-0.3, 3.3)
    ax2.set_ylim(-0.3, 3.3)
    ax3.set_ylim(-0.3, 3.3)
    ax4.set_ylim(-0.3, 3.3)
    ax5.set_ylim(-0.3, 3.3)
    ax6.set_ylim(-0.3, 3.3)

    # Don't show x tick labels on the upper row
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    # Don't show y tick labels on the two rightmost columns
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax5.set_yticklabels([])
    ax6.set_yticklabels([])

    fig.savefig(stacking_figures_dir + 'ur_ms_diagram_' + mass_str + '_KDE.pdf', dpi=300, bbox_inches='tight')

    return None

def uvj():

    # Read in results for all of PEARS
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    # First get stellar mass and apply a cut to stellar mass
    ms = np.log10(cat['zp_ms'])
    low_mass_lim = 8.0
    ms_idx = np.where((ms >= low_mass_lim) & (ms <= 12.0))[0]
    if int(low_mass_lim) == 8:
        mass_str = '8_logM_12'
    elif int(low_mass_lim) == 9:
        mass_str = '9_logM_12'
    print("Galaxies from stellar mass cut:", len(ms_idx))

    # Now get indices based on redshift intervals
    # decided from the stellar_mass_diagnostic_plot.py
    # code. 
    # Look at the make_z_hist() function in there. 
    zp = cat['zp_minchi2'][ms_idx]
    ms = ms[ms_idx]

    # now loop over all galaxies within mass cut sample 
    # and get the u-r color for each galaxy
    if not os.path.isfile(stacking_analysis_dir + 'uv_arr_' + mass_str + '.npy'):

        uv = []
        vj = []
        threed_uv = []
        threed_vj = []

        # Read model lambda grid # In agnstroms
        model_lam_grid_withlines_mmap = np.load(figs_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')
        # Now read the model spectra # In erg s^-1 A^-1
        model_comp_spec_llam_withlines_mmap = np.load(figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')

        # ------------------------------- Read in photometry catalogs ------------------------------- #
        # GOODS photometry catalogs from 3DHST
        # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
        # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
        photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
        'f_R', 'e_R', 'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
        'f_U', 'e_U', 'f_V', 'e_V', 'f_J', 'e_J', \
        'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
        'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
        goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cat', \
            dtype=None, names=photometry_names, \
            usecols=(0,3,4, 9,10, 15,16, 27,28, 30,31, 39,40, 45,46, 48,49, 54,55, 12,13, 24,25, 51,52, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), \
            skip_header=3)
        goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cat', \
            dtype=None, names=photometry_names, \
            usecols=(0,3,4, 9,10, 18,19, 30,31, 33,34, 39,40, 48,49, 54,55, 63,64, 15,16, 24,25, 57,58, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), \
            skip_header=3)

        # ------------------------------- Get required filter curves ------------------------------- # 
        uvj_filt_dir = stacking_analysis_dir + 'filter_curves/'

        # Wavelenght is in angstroms
        # Throughput is an absolute fraction
        u = np.genfromtxt(uvj_filt_dir + 'bessel_u.dat', dtype=None, names=['wav', 'filter_trans'])
        v = np.genfromtxt(uvj_filt_dir + 'bessel_v.dat', dtype=None, names=['wav', 'filter_trans'])
        j = np.genfromtxt(uvj_filt_dir + 'nsfcam_jmk_trans.dat', dtype=None, names=['wav', 'filter_trans'], skip_header=1)

        """
        When making the UVJ plot -- 
        it needs only the filter transmission without the 
        detector response included. See Williams et al. 2009
        prescription for making the UVJ plot. They use Bessel
        U and V filters and a Mauna Kea J filter. This is 
        also what I've done here.
        """
        # The Bessel U and V filters are given in angstroms and fractions
        u_filt_wav = u['wav']
        u_filt_thru = u['filter_trans']

        v_filt_wav = v['wav']
        v_filt_thru = v['filter_trans']

        # SOme extra stuff needed for J band
        j_filt_wav = j['wav'] * 1e4  # Convert microns to angstroms
        j_filt_thru = j['filter_trans'] / 100.0  # They give transmission percentages 

        # Now reverse because for some reason they provide the J band wavelengths in decending order
        j_filt_wav = j_filt_wav[::-1]
        j_filt_thru = j_filt_thru[::-1]

        # ------------------------------- Now begin looping over all stellar mass selected galaxies ------------------------------- #
        for idx in ms_idx:
            #print "\n", idx
            # First get teh full res model spectrum
            best_model_idx = cat[idx]['zp_model_idx']
            current_spec = model_comp_spec_llam_withlines_mmap[best_model_idx]
            
            # Now get the u and r magnitudes
            ufnu = compute_fnu(current_spec, model_lam_grid_withlines_mmap, u_filt_wav, u_filt_thru)
            vfnu = compute_fnu(current_spec, model_lam_grid_withlines_mmap, v_filt_wav, v_filt_thru)
            jfnu = compute_fnu(current_spec, model_lam_grid_withlines_mmap, j_filt_wav, j_filt_thru)
            umag = -2.5 * np.log10(ufnu) - 48.60
            vmag = -2.5 * np.log10(vfnu) - 48.60
            jmag = -2.5 * np.log10(jfnu) - 48.60
            current_uv = umag - vmag
            current_vj = vmag - jmag

            uv.append(current_uv)
            vj.append(current_vj)

            #print umag, vmag, jmag, current_uv, current_vj

            """
            # Check the 3DHST colors as well
            # They've used different filters so don't expect an exact match
            current_ra = cat[idx]['RA']
            current_dec = cat[idx]['DEC']
            current_field = cat[idx]['Field']

            # Assign catalogs 
            if current_field == 'GOODS-N':
                phot_cat_3dhst = goodsn_phot_cat_3dhst
            elif current_field == 'GOODS-S':
                phot_cat_3dhst = goodss_phot_cat_3dhst

            threed_ra = phot_cat_3dhst['ra']
            threed_dec = phot_cat_3dhst['dec']

            current_threed_uv, current_threed_vj = get_threed_uv_vj(current_ra, current_dec, threed_ra, threed_dec, phot_cat_3dhst)
            threed_uv.append(current_threed_uv)
            threed_vj.append(current_threed_vj)

            #print current_threed_uv, current_threed_vj
            """

        # Save colors
        uv = np.asarray(uv)
        vj = np.asarray(vj)
        np.save(stacking_analysis_dir + 'uv_arr_' + mass_str + '.npy', uv)
        np.save(stacking_analysis_dir + 'vj_arr_' + mass_str + '.npy', vj)

        # Save threed colors as well
        #threed_uv = np.asarray(threed_uv)
        #threed_vj = np.asarray(threed_vj)
        #np.save(stacking_analysis_dir + 'threed_uv_arr_' + mass_str + '.npy', threed_uv)
        #np.save(stacking_analysis_dir + 'threed_vj_arr_' + mass_str + '.npy', threed_vj)

    else:
        uv = np.load(stacking_analysis_dir + 'uv_arr_' + mass_str + '.npy')
        vj = np.load(stacking_analysis_dir + 'vj_arr_' + mass_str + '.npy')

    uv_plt = uv
    vj_plt = vj

    # Get z intervals and their indices
    z_interval1_idx = np.where((zp >= 0.0) & (zp < 0.4))[0]
    z_interval2_idx = np.where((zp >= 0.4) & (zp < 0.7))[0]
    z_interval3_idx = np.where((zp >= 0.7) & (zp < 1.0))[0]
    z_interval4_idx = np.where((zp >= 1.0) & (zp < 2.0))[0]
    z_interval5_idx = np.where((zp >= 2.0) & (zp <= 6.0))[0]

    print("Number of galaxies within each redshift interval.")
    print("0.0 <= z < 0.4", "    ", len(z_interval1_idx))
    print("0.4 <= z < 0.7", "    ", len(z_interval2_idx))
    print("0.7 <= z < 1.0", "    ", len(z_interval3_idx))
    print("1.0 <= z < 2.0", "    ", len(z_interval4_idx))
    print("2.0 <= z <= 6.0", "    ", len(z_interval5_idx))

    # Now make the plots
    # Define figure
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(10,12)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.25, hspace=0.3)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:5, :4])
    ax2 = fig.add_subplot(gs[:5, 4:8])
    ax3 = fig.add_subplot(gs[:5, 8:])
    ax4 = fig.add_subplot(gs[5:, :4])
    ax5 = fig.add_subplot(gs[5:, 4:8])
    ax6 = fig.add_subplot(gs[5:, 8:])

    # Labels
    ax5.set_xlabel(r'$V - J$', fontsize=15)
    ax4.set_ylabel(r'$U - V$', fontsize=15)
    ax4.yaxis.set_label_coords(-0.12, 1.05)

    # Actual plotting 
    ax1.scatter(vj_plt[z_interval1_idx], uv_plt[z_interval1_idx], s=0.7, color='k')
    ax2.scatter(vj_plt[z_interval2_idx], uv_plt[z_interval2_idx], s=0.7, color='k')
    ax3.scatter(vj_plt[z_interval3_idx], uv_plt[z_interval3_idx], s=0.7, color='k')
    ax4.scatter(vj_plt[z_interval4_idx], uv_plt[z_interval4_idx], s=0.7, color='k')
    ax5.scatter(vj_plt[z_interval5_idx], uv_plt[z_interval5_idx], s=0.7, color='k')
    ax6.scatter(vj_plt, uv_plt, s=0.7, color='k')

    # Plot UVJ selection on each subplot
    plot_uvj_selection(ax1, 0.0, 0.4)
    plot_uvj_selection(ax2, 0.4, 0.7)
    plot_uvj_selection(ax3, 0.7, 1.0)
    plot_uvj_selection(ax4, 1.0, 2.0)
    plot_uvj_selection(ax5, 2.0, 6.0)
    plot_uvj_selection(ax6, 0.0, 6.0)

    # Add text 
    add_info_text_to_subplots(ax1, 0.0, 0.4, len(z_interval1_idx))
    add_info_text_to_subplots(ax2, 0.4, 0.7, len(z_interval2_idx))
    add_info_text_to_subplots(ax3, 0.7, 1.0, len(z_interval3_idx))
    add_info_text_to_subplots(ax4, 1.0, 2.0, len(z_interval4_idx))
    add_info_text_to_subplots(ax5, 2.0, 6.0, len(z_interval5_idx))
    add_info_text_to_subplots(ax6, 0.0, 6.0, len(zp))

    # Axes limits
    ax1.set_xlim(-0.3, 2.5)
    ax2.set_xlim(-0.3, 2.5)
    ax3.set_xlim(-0.3, 2.5)
    ax4.set_xlim(-0.3, 2.5)
    ax5.set_xlim(-0.3, 2.5)
    ax6.set_xlim(-0.3, 2.5)

    ax1.set_ylim(0.0, 3.0)
    ax2.set_ylim(0.0, 3.0)
    ax3.set_ylim(0.0, 3.0)
    ax4.set_ylim(0.0, 3.0)
    ax5.set_ylim(0.0, 3.0)
    ax6.set_ylim(0.0, 3.0)

    # Don't show x tick labels on the upper row
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    # Don't show y tick labels on the two rightmost columns
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax5.set_yticklabels([])
    ax6.set_yticklabels([])

    fig.savefig(stacking_figures_dir + 'uvj_diagram_' + mass_str + '.pdf', dpi=300, bbox_inches='tight')

    return None

def plot_uvj_selection(ax, zlow, zhigh):

    # Testing block
    # Just to do a quick plot and check by eye
    # Seems like the y_qg_line = 0.88*x_qg_line + 0.69
    # works best for now in case of photo-z.
    """ 
    x_qg_line = np.arange(0.0, 3.0, 0.1)  # qg stands for 'Quiescent Galaxies'

    if zhigh < 0.5:
        y_qg_line = 0.88*x_qg_line + 0.69
    elif zlow >= 0.4 and zhigh <= 1.0:
        y_qg_line = 0.88*x_qg_line + 0.59
    else:  # for the bottom row
        y_qg_line = 0.88*x_qg_line + 0.59

    ax.plot(x_qg_line, y_qg_line)
    """

    # Depending on the given equation for the lines
    # I simply computed the min and max coordinates here by hand
    ax.vlines(x=1.6, ymin=2.1, ymax=3.0, color='r')
    ax.hlines(y=1.3, xmin=-0.3, xmax=0.69, color='r')
    ax.plot([0.69, 1.6], [1.3, 2.1], '-', color='r')

    return None

def add_info_text_to_subplots(ax, zlow, zhigh, num):

    # add number of galaxies in plot
    ax.text(0.04, 0.94, r'$\rm N\, =\, $' + str(num), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=15)

    # Add redshift range
    if zhigh == 6.0:
        zstr = str(zlow) + r'$\, \leq z \leq \,$' + str(zhigh)
    else:
        zstr = str(zlow) + r'$\, \leq z < \,$' + str(zhigh)

    ax.text(0.04, 0.84, zstr, \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=15)

    return None

def check_salp_chab_z():

    # Read in results for all of PEARS
    cat_salp = np.genfromtxt(stacking_analysis_dir + 'full_pears_results.txt', dtype=None, names=True)
    cat_chab = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    zp_salp = cat_salp['zp_minchi2']
    zp_chab = cat_chab['zp_minchi2']

    # Now vs each other
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$z^\mathrm{Salpeter}_\mathrm{phot}$', fontsize=14)
    ax.set_ylabel(r'$z^\mathrm{Chabrier}_\mathrm{phot}$', fontsize=14)

    ax.scatter(zp_salp, zp_chab, s=2.0, color='k')
    ax.plot(np.arange(-0.5, 6.5, 0.1), np.arange(-0.5, 6.5, 0.1), '--', color='r', linewidth=0.75)

    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-0.2, 6.2)

    ax.minorticks_on()

    fig.savefig(stacking_figures_dir + 'zp_salp_chab_check.pdf', dpi=300, bbox_inches='tight')

    return None

def ssfr_ms_plots():

    # Read in results for all of PEARS
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    # First get stellar mass and apply a cut to stellar mass
    ms = np.log10(cat['zp_ms'])
    low_mass_lim = 8.0
    ms_idx = np.where((ms >= low_mass_lim) & (ms <= 12.0))[0]
    if int(low_mass_lim) == 8:
        mass_str = '8_logM_12'
    elif int(low_mass_lim) == 9:
        mass_str = '9_logM_12'
    print("Galaxies from stellar mass cut:", len(ms_idx))

    # Now get indices based on redshift intervals
    # decided from the stellar_mass_diagnostic_plot.py
    # code. 
    # Look at the make_z_hist() function in there. 
    zp = cat['zp_minchi2'][ms_idx]
    ms = ms[ms_idx]
    sfr = cat['zp_sfr'][ms_idx]  # The SFR is also normalized to 1M_sol
    sfr *= ms

    # Some of the models have exactly 0 star-formation
    # Make sure these show up too
    #sfr_zero_idx = np.where(sfr == 0.0)[0]
    #sfr[sfr_zero_idx] = 0.1

    print(sfr[:1000])
    sys.exit(0)


    # Make sure the units of SFR and SSFR are correct
    template_ms = cat['zp_template_ms'][ms_idx]
    ssfr = np.log10(sfr / ms)  # I think, this should be divided by the template mass to get it right

    # Get z intervals and their indices
    z_interval1_idx = np.where((zp >= 0.0) & (zp < 0.4))[0]
    z_interval2_idx = np.where((zp >= 0.4) & (zp < 0.7))[0]
    z_interval3_idx = np.where((zp >= 0.7) & (zp < 1.0))[0]
    z_interval4_idx = np.where((zp >= 1.0) & (zp < 2.0))[0]
    z_interval5_idx = np.where((zp >= 2.0) & (zp <= 6.0))[0]

    print("Number of galaxies within each redshift interval.")
    print("0.0 <= z < 0.4", "    ", len(z_interval1_idx))
    print("0.4 <= z < 0.7", "    ", len(z_interval2_idx))
    print("0.7 <= z < 1.0", "    ", len(z_interval3_idx))
    print("1.0 <= z < 2.0", "    ", len(z_interval4_idx))
    print("2.0 <= z <= 6.0", "    ", len(z_interval5_idx))

    # Now make the plots
    # Define figure
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(10,12)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.25, hspace=0.3)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:5, :4])
    ax2 = fig.add_subplot(gs[:5, 4:8])
    ax3 = fig.add_subplot(gs[:5, 8:])
    ax4 = fig.add_subplot(gs[5:, :4])
    ax5 = fig.add_subplot(gs[5:, 4:8])
    ax6 = fig.add_subplot(gs[5:, 8:])

    # Labels
    ax5.set_xlabel(r'$\rm log(M_s)\ [M_\odot]$', fontsize=15)
    ax4.set_ylabel(r'$\rm log(SSFR)\ [yr^{-1}]$', fontsize=15)
    ax4.yaxis.set_label_coords(-0.12, 1.05)

    # Actual plotting
    ax1.scatter(ms[z_interval1_idx], ssfr[z_interval1_idx], s=0.7, color='k', zorder=3)
    ax2.scatter(ms[z_interval2_idx], ssfr[z_interval2_idx], s=0.7, color='k', zorder=3)
    ax3.scatter(ms[z_interval3_idx], ssfr[z_interval3_idx], s=0.7, color='k', zorder=3)
    ax4.scatter(ms[z_interval4_idx], ssfr[z_interval4_idx], s=0.7, color='k', zorder=3)
    ax5.scatter(ms[z_interval5_idx], ssfr[z_interval5_idx], s=0.7, color='k', zorder=3)
    ax6.scatter(ms, ssfr, s=0.7, color='k', zorder=3)

    plt.show()

    return None

def main():

    #check_salp_chab_z()
    #ur_ms_plots()
    #uvj()
    generate_all_ur_color()

    #ssfr_ms_plots()
    
    return None

if __name__ == '__main__':
    main()
    sys.exit(0)