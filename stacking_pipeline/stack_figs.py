import numpy as np
from astropy.io import fits

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

sys.path.append(stacking_utils)
from get_total_extensions import get_total_extensions

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
    zlow = 0.54  # defined as the redshift when the Fe features enter the G102 coverage
    zhigh = 1.9  # defined as the redshift when the 4000break leaves the G102 coverage

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

    # Select galaxies and stack
    # Define our redshift range
    # Considering G102 coverage to be 8500 to 11300 Angstroms
    zlow = 0.54  # defined as the redshift when the Fe features enter the G102 coverage
    zhigh = 1.9  # defined as the redshift when the 4000break leaves the G102 coverage

    mslim = 10.5 # all galaxies equal to and above this mass are selected

    # Read in FIGS catalogs
    # Only GOODS-S for now because 
    # Santini et al only have GOODS-S masses
    figs_gs1 = fits.open(figs_dir + 'GS1_G102_2.combSPC.fits')
    figs_gs2 = fits.open(figs_dir + 'GS2_G102_2.combSPC.fits')

    # Read in FIGS and CANDELS matching file
    figs_samp = np.genfromtxt(figs_match_file, dtype=None, names=True, encoding='ascii')

    checkplot = True
    for i in range(len(figs_samp)):

        # Cut on stellar mass
        ms = np.log10(figs_samp['Mmed'][i])

        if ms < 10.5:
            continue

        objid = figs_samp['ObjID'][i]
        field = figs_samp['Field'][i]

        if field == 'GS1':
            figs_spec = figs_gs1
        elif field == 'GS2':
            figs_spec = figs_gs2

        obj_exten = 'BEAM_' + str(objid) + 'A'

        wav = figs_spec[obj_exten].data['LAMBDA']
        avg_wflux = figs_spec[obj_exten].data['AVG_WFLUX']
        std_wflux = figs_spec[obj_exten].data['STD_WFLUX']

        #if checkplot:
        #    fig = plt.figure()
        #    ax = fig.add_subplot(111)
        #
        #    ax.plot(wav, avg_wflux, color='k')
        #    ax.fill_between(wav, avg_wflux - std_wflux, avg_wflux + std_wflux, color='gray')
        #    plt.show()

        # Clip the spectra at teh ends
        wav_idx = np.where((wav >= 8500) & (wav <= 11300))[0]
        wav = wav[wav_idx]
        avg_wflux = avg_wflux[wav_idx]
        std_wflux = std_wflux[wav_idx]

        # Now fit the continuum 
        # First normalize
        avg_wflux_norm = avg_wflux / np.mean(avg_wflux)

        # Scipy spline fit
        


        if i > 20: sys.exit(0)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)