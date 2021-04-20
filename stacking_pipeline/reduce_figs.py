import pylinear
from astropy.io import fits
import numpy as np

import os
import sys
import time
import datetime as dt
import glob
import shutil
import socket

import logging

home = os.getenv('HOME')

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

def gen_lst_files():

    # ---------------------- Preliminary stuff
    logger = logging.getLogger('Reducing FIGS primary data.')

    logging_format = '%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, filename='figs_reduction.log', filemode='w', format=logging_format)

    # Get starting time
    start = time.time()
    logger.info("Starting now.")

    # Assign directories
    if 'firstlight' in socket.gethostname():
        figs_raw_datadir = ''
        lst_dir = ''
    else:
        figs_raw_datadir = '/Volumes/Joshi_external_HDD/figs_raw_data/mastDownload/HST/'
        lst_dir = '/Volumes/Joshi_external_HDD/figs_raw_data/pylinear_lst_files/'

    # ---------------------- Create empty files for saving LST files for pylinear
    flt_filename = 'flt_figs_gs1.lst'
    obs_filename = 'obs_figs_gs1.lst'

    fh_flt_gs1 = open(lst_dir + flt_filename, 'w')

    fh_flt_gs1.write("# Path to each flt image" + "\n")
    fh_flt_gs1.write("# This has to be a simulated or observed dispersed image" + "\n")
    fh_flt_gs1.write("\n")

    # ------
    fh_obs_gs1 = open(lst_dir + obs_filename, 'w')

    fh_obs_gs1.write("# Image File name" + "\n")
    fh_obs_gs1.write("# Observing band" + "\n")
    fh_obs_gs1.write("\n")

    # ---------------------- Loop over all data
    for root, dirs, files in os.walk(figs_raw_datadir):

        for dataset in dirs:

            print("\nDataset:", dataset)

            for r, d, f in os.walk(root + dataset):

                flts = 0

                for fl in f:

                    # Now pick drc or flt file if it exists and if not then use the drz file
                    if '_flt' in fl:
                        flname_tosave = root + dataset + '/' + fl
                        flts += 1
                        break
                    elif '_drc' in fl:
                        if flts == 0:
                            flname_tosave = root + dataset + '/' + fl
                            break
                    else:
                        flname_tosave = root + dataset + '/' + fl

                print("Found files:", f)
                print("Choosing file:", flname_tosave)

                h = fits.open(flname_tosave)

                # This exception handler is to distinguish between 
                # ACS and WFC3 observations.
                try:
                    filt = h[0].header['FILTER']
                    print("Observed filters for this file:", filt)
                except KeyError:
                    filt1 = h[0].header['FILTER1']
                    filt2 = h[0].header['FILTER2']
                    filt = ''
                    print("Observed filters for this file:", filt, filt1, filt2)

                # Get the observed direct image that matches the appropriate field
                if 'f105w' in dataset:

                    direct_targname = h[0].header['TARGNAME']

                    # Now assign the correct FIGS field
                    if 'GS1' in direct_targname:
                        print(f"{bcolors.CYAN}", "Found GS1 direct obs:", fl, f"{bcolors.ENDC}")
                        fh_obs_gs1.write(flname_tosave + '  ' + 'hst_wfc3_f105w' + '\n')

                # Get the observed dispersed image that matches the appropriate field
                if 'G102' in filt:

                    disp_targname = h[0].header['TARGNAME']

                    # Now assign the correct FIGS field
                    if 'GS1' in disp_targname:
                        print(f"{bcolors.GREEN}", "Found GS1 dispersed obs:", fl, f"{bcolors.ENDC}")
                        fh_flt_gs1.write(flname_tosave + '\n')

                h.close()

    # Close lst files to save
    fh_flt_gs1.close()
    fh_obs_gs1.close()

    return None

def main():

    gen_lst_files()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)