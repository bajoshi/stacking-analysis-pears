import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')

stacking_util_codes = home + "/Documents/GitHub/stacking-analysis-pears/util_codes/"

sys.path.append(stacking_util_codes)
from bc03_utils import get_bc03_spectrum
from get_total_extensions import get_total_extensions

def perform_test1():

    # Run the bc03 csp_galaxev code for some set of parameters
    # First through the command line 
    # provide path here
    spec_fitsfile_from_cl = home + '/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m62_chab_ssp_testrun.fits'
    # and then through python here

    tau = 1.0  # gyr
    metals = 0.02  # abs frac

    age = 6.0  # gyr

    # Get the metallicity in the format that BC03 needs
    if metals == 0.0001:
        metallicity = 'm22'
    elif metals == 0.0004:
        metallicity = 'm32'
    elif metals == 0.004:
        metallicity = 'm42'
    elif metals == 0.008:
        metallicity = 'm52'
    elif metals == 0.02:
        metallicity = 'm62'
    elif metals == 0.05:
        metallicity = 'm72'

    outdir_ised = home + '/Documents/bc03_test_output_dir/'
    output = outdir_ised + "bc2003_hr_" + metallicity + "_csp_tau" + str(int(float(str(tau)[0:6])*10000)) + "_chab"

    # get_bc03_spectrum(age, tau, metals, outdir_ised)
    # this is commented out since I've already run it once

    # check that all spectra are identical
    h1 = fits.open(spec_fitsfile_from_cl)
    h2 = fits.open(output + '.fits')

    # first check htat they both have equal number of extensions
    # and then check that each extension is identical
    h1ext = get_total_extensions(h1)
    h2ext = get_total_extensions(h2)

    print("Extensions in h1:", h1ext)
    print("Extensions in h2:", h2ext)

    if h1ext == h2ext:
        for i in range(h1ext):
            dat1 = h1[i+1].data
            dat2 = h2[i+1].data

            if not np.array_equal(dat1, dat2):
                print("Did not match at extension:". i+1)
                print("Exiting...")
                sys.exit(0)

        print("Finished checking all extensions. All match.")

    else:
        print("Number of extensions dont match. Exiting...")
        sys.exit(0)

    return None

def perform_test2():

    return None

def main():

    print("-------------------")
    print("This code is intended to test the working of BC03.")
    print("We will perform two tests.\n")
    print("1: This code will test that calls to csp_galaxev")
    print("from within python provide identical results to calls made")
    print("through the command line. You will need to do the command line")
    print("part and provide the path to those spectra here.\n")
    print("2: A much more IMPORTANT test!")
    print("Given the spectra of a handful of nearby NGC/IC galaxies")
    print("(probably through NED or some other literature sources and preferably UV to FIR)")
    print("and their corresponding stellar population parameters, which were")
    print("hopefully derived independently NOT using BC03 SPS models,")
    print("this code will test that the spectra given by csp_galaxev")
    print("are \"reasonably\" similar to those from the literature.")
    print("The stellar population parameters required are -- stellar mass, ")
    print("age, SFH, Av (and dust law), metallicity, and redshift for the galaxy.")
    print("Applying dust attenuation and redshift is done outside of the BC03 code.")
    print("-------------------\n")

    perform_test1()

    perform_test2()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
