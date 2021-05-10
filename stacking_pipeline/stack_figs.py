import numpy as np
from astropy.io import fits
from scipy.interpolate import splev, splrep

import os
import sys
import time
import datetime
import glob

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
figs_dir = home + "/Documents/pears_figs_data/"
figs_allspectra = figs_dir + '1D/'
stacking_utils = home + "/Documents/GitHub/stacking-analysis-pears/util_codes/"
stacking_figures_dir = home + "/Documents/stacking_figures/"

sys.path.append(stacking_utils)
from get_total_extensions import get_total_extensions
import grid_coadd as gd

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

def get_obj_coords(objid, field):

    field_specdir = figs_allspectra + field.upper() + '/'
    obj_filename_prefix = field.upper() + '_' + str(objid)

    fl = glob.glob(field_specdir + obj_filename_prefix + '*.txt')[0]

    with open(fl, 'r') as fh:
        lines = fh.readlines()
        lspt = lines[1].split()

        ra = float(lspt[2])
        dec = float(lspt[4])

    return ra, dec

def gen_figs_matches():

    # Read in catalog from Santini et al.
    names_header=['id', 'ra', 'dec', 'zbest', 'zphot', 'zphot_l68', 
    'zphot_u68', 'Mmed', 'smed', 'Mdeltau']
    santini_cat = np.genfromtxt(home + \
                  '/Documents/GitHub/massive-galaxies/santini_candels_cat.txt',
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

    # Define our redshift range
    # Considering G102 coverage to be 8500 to 11300 Angstroms
    zlow = 0.41  # defined as the redshift when the Fe features enter the G102 coverage
    zhigh = 2.14  # defined as the redshift when the 4000break leaves the G102 coverage

    # Now try and match every figs object to
    # the candels catalog and then stack.
    # Empty file for saving results
    fh = open(figs_dir + 'figs_candels_goodss_matches.txt', 'w')
    # Write header
    fh.write('#  Field  ObjID  RA  DEC  zbest  Mmed  smed' + '\n') 

    all_cats = [figs_gs1, figs_gs2]
    all_fields = ['GS1', 'GS2']

    fieldcount = 0
    for figs_cat in all_cats:

        field = all_fields[fieldcount]

        nobj = get_total_extensions(figs_cat)
        print("# extensions:", nobj)

        for i in range(1, nobj+1):

            extname = figs_cat[i].header['EXTNAME']

            # Now match
            objid = extname.split('_')[-1].split('A')[0]
            obj_ra, obj_dec = get_obj_coords(objid, field)

            print(extname, "      Object:", field, objid, "at ", obj_ra, obj_dec)

            # Now match our object to ones in Santini's cat
            match_idx = np.where( (np.abs(search_ra  - obj_ra)  <= ra_tol) & \
                                  (np.abs(search_dec - obj_dec) <= dec_tol) )[0]

            #print("Match index:", match_idx, len(match_idx))
            if len(match_idx) == 0:
                print(f'{bcolors.WARNING}', 'No matches found. Skipping', f'{bcolors.ENDC}')
                continue

            assert len(match_idx)==1

            match_idx = int(match_idx)
            zbest = santini_cat['zbest'][match_idx]
            
            #print("Redshift for object:", zbest)

            if (zbest >= zlow) and (zbest <= zhigh):
                ms = santini_cat['Mmed'][match_idx]
                ms_err = santini_cat['smed'][match_idx]

                write_str = field + "  " + objid + "  " + \
                            "{:.7f}".format(obj_ra) + "  " + "{:.7f}".format(obj_dec) + "  " + \
                            "{:.3f}".format(zbest) + "  " + \
                            "{:.3e}".format(ms) + "  " + "{:.3e}".format(ms_err) + "\n"

                fh.write(write_str)

                print(f'{bcolors.GREEN}', 'Added', f'{bcolors.ENDC}')

        fieldcount += 1

    fh.close()

    return None

def main():

    # check if the matching is done already
    figs_match_file = figs_dir + 'figs_candels_goodss_matches.txt'
    if not os.path.isfile(figs_match_file):
        gen_figs_matches()

    # Read in FIGS catalogs
    # Only GOODS-S for now because 
    # Santini et al only have GOODS-S masses
    figs_gs1 = fits.open(figs_dir + 'GS1_G102_2.combSPC.fits')
    figs_gs2 = fits.open(figs_dir + 'GS2_G102_2.combSPC.fits')

    # Read in FIGS and CANDELS matching file
    figs_samp = np.genfromtxt(figs_match_file, dtype=None, names=True, encoding='ascii')

    #ms_idx = np.where(np.log10(figs_samp['Mmed']) >= 10.5)[0]
    #print(ms_idx, len(ms_idx))
    #sys.exit(0)

    # ----------------- Stacking prelims
    # Change only the parameters here to change how the code runs
    # Ideally you shouldn't have to change anything else.
    lam_step = 25  # somewhat arbitrarily chosen by trial and error

    # Set the ends of the lambda grid
    # This is dependent on the redshift range being considered
    lam_grid_low = 3000
    lam_grid_high = 7600

    lam_grid = np.arange(lam_grid_low, lam_grid_high, lam_step)

    # Define our redshift range
    # Considering G102 coverage to be 8500 to 11300 Angstroms
    zlow = 0.41  # defined as the redshift when the Fe features enter the G102 coverage
    zhigh = 2.14  # defined as the redshift when the 4000break leaves the G102 coverage

    ms_lim_low = 10.5
    ms_lim_high = 12.0

    # Define empty arrays and lists for saving stacks
    old_llam = np.zeros(len(lam_grid))
    old_llamerr = np.zeros(len(lam_grid))
    old_llam = old_llam.tolist()
    old_llamerr = old_llamerr.tolist()

    num_points = np.zeros(len(lam_grid))
    num_galaxies = np.zeros(len(lam_grid))

    # Create figure
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    checkplot = True
    num_massive = 0
    for i in range(len(figs_samp)):

        # This step should only be done on the first iteration within a grid cell
        # This converts every element (which are all 0 to begin with) 
        # in the flux and flux error arrays to an empty list
        # This is done so that function add_spec() can now append to every element
        if i == 0:
            for x in range(len(lam_grid)):
                old_llam[x] = []
                old_llamerr[x] = []

        # Cut on stellar mass
        ms = np.log10(figs_samp['Mmed'][i])

        if ms < ms_lim_low:
            continue

        # If mass okay then get the spectrum
        objid = figs_samp['ObjID'][i]
        field = figs_samp['Field'][i]
        zbest = figs_samp['zbest'][i]

        if field == 'GS1':
            figs_spec = figs_gs1
        elif field == 'GS2':
            figs_spec = figs_gs2

        obj_exten = 'BEAM_' + str(objid) + 'A'

        wav = figs_spec[obj_exten].data['LAMBDA']
        avg_wflux = figs_spec[obj_exten].data['AVG_WFLUX']
        std_wflux = figs_spec[obj_exten].data['STD_WFLUX']

        # Clip the spectra at teh ends
        wav_idx = np.where((wav >= 8500) & (wav <= 11300))[0]
        wav = wav[wav_idx]
        avg_wflux = avg_wflux[wav_idx]
        std_wflux = std_wflux[wav_idx]

        # Now fit the continuum 
        # First normalize
        avg_wflux_norm = avg_wflux / np.mean(avg_wflux)
        std_wflux_norm = std_wflux / np.mean(std_wflux)

        # Scipy spline fit
        spl = splrep(x=wav, y=avg_wflux_norm, k=3, s=0.2)
        wav_plt = np.arange(wav[0], wav[-1], 1.0)
        spl_eval = splev(wav_plt, spl)

        # Divide the given flux by the smooth spline fit
        cont_div_flux_g102 = avg_wflux_norm / splev(wav, spl)
        cont_div_err_g102  = std_wflux_norm / splev(wav, spl)

        #if checkplot:
        #    fig = plt.figure()
        #    ax = fig.add_subplot(111)
        
        #    ax.plot(wav, avg_wflux_norm, color='k')
        #    ax.plot(wav_plt, spl_eval, color='tab:red')

        #    plt.show()

        # Deredshift wavelength
        wav_em = wav / (1 + zbest)

        # Shift it to force stack value ~1.0 at ~4600A
        shift_idx = np.where((wav_em >= 4600) & (wav_em <= 4700))[0]
        scaling_fac = np.mean(cont_div_flux_g102[shift_idx])
        cont_div_flux_g102 /= scaling_fac

        # add the continuum subtracted spectrum
        old_llam, old_llamerr, num_points, num_galaxies = \
        gd.add_spec(wav_em, cont_div_flux_g102, cont_div_err_g102, 
            old_llam, old_llamerr, num_points, num_galaxies, lam_grid, lam_step)

        ax.plot(wav_em, cont_div_flux_g102, ls='-', 
           color='bisque', linewidth=1.0)

        num_massive += 1

    old_llam, old_llamerr = gd.take_median(old_llam, old_llamerr, lam_grid)

    old_llam = np.asarray(old_llam)
    old_llamerr = np.asarray(old_llamerr)

    ax.plot(lam_grid, old_llam, '.-', color='darkorange', linewidth=1.5, \
        markeredgecolor='darkorange', markersize=1.0, zorder=5)
    ax.fill_between(lam_grid, old_llam - old_llamerr, 
                    old_llam + old_llamerr, \
                    color='gray', alpha=0.5, zorder=5)

    ax.set_xlim(lam_grid_low, lam_grid_high)
    ax.set_ylim(0.9, 1.1)  # if dividing by the continuum instead of subtracting
    ax.axhline(y=1.0, ls='--', color='k')
    ax.minorticks_on()

    gd.add_line_labels(ax)

    # Number of galaxies and redshift range on plot
    ax.text(0.66, 0.97, r'$\mathrm{N\,=\,}$' + str(num_massive), 
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=16)
    ax.text(0.66, 0.92, str(zlow) + r'$\,\leq z <\,$' + str(zhigh), \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=16)

    # Mass range
    ax.text(0.66, 0.86, str(ms_lim_low) + r'$\,\leq \mathrm{M\ [M_\odot]} <\,$' + 
        str(ms_lim_high), 
        verticalalignment='top', horizontalalignment='left', 
        transform=ax.transAxes, color='k', size=16)

    # Labels
    ax.set_xlabel(r'$\lambda\ [\mathrm{\AA}]$', fontsize=15)
    ax.set_ylabel(r'$L_{\lambda}\ [\mathrm{divided\ by\ continuum}]$', fontsize=15)

    figname = stacking_figures_dir + 'massive_stack_figs_' + \
            str(ms_lim_low).replace('.','p') + '_Ms_' + str(ms_lim_high).replace('.','p') + \
            '_' + str(zlow).replace('.','p') + '_z_' + str(zhigh).replace('.','p') + '.pdf'
    fig.savefig(figname, dpi=300, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)