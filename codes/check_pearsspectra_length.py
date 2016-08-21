"""
    This code tests the PEARS spectra to see that lambda = 6500 and lambda = 9000 fall on the same index
    for every spectrum.
    This is true apart from a few exceptions.
"""

from __future__ import division
import numpy as np
import pyfits as pf
import glob, os

data_path = "/Users/bhavinjoshi/Documents/PEARS/data_spectra_only/"

for file in glob.glob(data_path + "*.fits"):
    try:
        fitsfile = pf.open(file)
        fitsdata = fitsfile[1].data
        diff = np.argmin(abs(fitsdata['LAMBDA'] - 9000)) - np.argmin(abs(fitsdata['LAMBDA'] - 6500))
        if diff != 62:
            n_ext = fitsfile[0].header['NEXTEND']
            difference = []
            for count in range(2,n_ext+1):
                difference.append(np.argmin(abs(fitsfile[count].data['LAMBDA'] - 9000)) - np.argmin(abs(fitsfile[count].data['LAMBDA'] - 6500)))
            if (62 not in difference) & (n_ext != 1): print diff, difference, file, n_ext
    except ValueError: # this error is raised when the first extension is empty
        print os.path.basename(file), "with", n_ext, "extensions, has an empty first extension"
        n_ext = fitsfile[0].header['NEXTEND']
        difference = []
        for count in range(2,n_ext+1):
            difference.append(np.argmin(abs(fitsfile[count].data['LAMBDA'] - 9000)) - np.argmin(abs(fitsfile[count].data['LAMBDA'] - 6500)))
        if (62 not in difference) & (n_ext != 1): print diff, difference, file, n_ext
        pass