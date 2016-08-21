"""
    This code checks for NaN values in any given array.
"""
import numpy as np
import sys
import pyfits as pf

def checknan(dat):

    #print dat

    for i in range(len(dat)):
        if np.isnan(dat[i]):
            print i, " index has a NaN value. Exiting."
            sys.exit()

    """
    ans = raw_input("Is the data okay?")
    if ans == 'y':
        return
    else:
        sys.exit()
    """

if __name__ == "__main__":

    h = pf.open('/Users/baj/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra.fits')

    i = 2
    while 1:
        try:
            flam = h[i].data[0]
            ferr = h[i].data[1]
            checknan(flam)
            checknan(ferr)
            i += 1
        except IndexError, e:
            print e
            print "Reached end of fits file. Exiting."
            break