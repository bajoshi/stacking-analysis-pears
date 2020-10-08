import numpy as np
from astropy.io import fits
import array

import subprocess
import os
import sys
import shutil

def read_current_filepos(filehandle, type='i', number=1):
    """
        This function reads the given number of binary data which is of the supplied type
        from the specified filehandle. The default number of elements to extract is one
        and the default type is an integer.
    """

    arr = array.array(type)
    arr.fromfile(filehandle, number)

    return np.asarray(arr)

def read_ised(modelfile, del_modelfile=False):

    if os.path.isfile(modelfile.replace('.ised','.fits')):
        print modelfile.replace('.ised','.fits')
        print "Fits file already exists. Skipping."
        return None
    
    fh = open(modelfile, 'rb')
    
    # For this file
    # int = 4 bytes
    # float = 4 bytes
    # char = 1 byte

    ignore = read_current_filepos(fh) # this is the number 1208 for SSPs and depends on Tcut for CSPs
    totalages = read_current_filepos(fh)[0] # this should be 221 for the SSP models
    
    """
        FOR CSPs ----
        The number of ages you get in the previous line depends on the CUT OFF TIME set in the csp run.
        The max age of any model that is given by BC03 is 20 Gyr. I found by trial and error that if the
        ---> Tcut is set to above 20 Gyr then it gives 221 ages which is the same as that given by a SSP.
        ---> Tcut is set to undet 20 Gyr but more than 13.8 Gyr then it gives a number (for totalages)
             that is between 221 and 245.
        ---> Tcut is set to 13.8 Gyr then it gives 245 ages.
        ---> Tcut is set to 5 Gyr then it gives 261 ages.
        Keep in mind however that the total age range still remains between 0 to 20 Gyr.
        Also I'm not sure if having a lower/higher number of totalages (than 245) will have any effect on my analysis.
        For now I have kept Tcut at 13.8 Gyr.
    """
    
    allages = read_current_filepos(fh, type='f', number=totalages) # in years
    
    # Going past some junk now
    if 'salp' in modelfile:
        fh.seek(328, 1)
        """
        Done by trial and error and looking at the ezgal code but mostly the ezgal code helped/
        This goes 328 bytes forward from the current position.
        If the second keyword is not provided then it assumes absolute positioning.
        I think (although I couldn't see all the characters properly) that these 328 
        bytes simply contain their copyright string (or something like that).
        Couldn't quite tell properly because they won't show up properly (some unicode issue).
        """
    elif 'chab' in modelfile:
        # This is exactly the same as 
        # the code in the EZGAL package.
        junk = read_current_filepos(fh, number=2)
        iseg = read_current_filepos(fh, number=1)
        if iseg > 0: 
           junk = read_current_filepos(fh, type='f', number=6*iseg)
        junk = read_current_filepos(fh, type='f', number=3)
        junk = read_current_filepos(fh)
        junk = read_current_filepos(fh, type='f')
        junk = read_current_filepos(fh, type='c', number=80)
        junk = read_current_filepos(fh, type='f', number=4)
        junk = read_current_filepos(fh, type='c', number=160)
        junk = read_current_filepos(fh)
        junk = read_current_filepos(fh, number=3)

    totalwavelengths = read_current_filepos(fh)[0]
    allwavelengths = read_current_filepos(fh, type='f', number=totalwavelengths)
    
    seds = np.zeros((totalages, totalwavelengths))
    
    hdu = fits.PrimaryHDU()
    
    hdulist = fits.HDUList(hdu)
    hdulist.append(fits.ImageHDU(allwavelengths))
    hdulist.append(fits.ImageHDU(allages))
    
    for i in range(totalages):
        ignore = read_current_filepos(fh, number=2)
        nlam = read_current_filepos(fh)
    
        seds[i] = read_current_filepos(fh, type='f', number=totalwavelengths)
        
        no = read_current_filepos(fh)
        ignore = read_current_filepos(fh, type='f', number=no)
    
        hdulist.append(fits.ImageHDU(seds[i]))
    
    hdulist.writeto(modelfile.replace('.ised','.fits'), overwrite=True)
    print "Writing ...", modelfile.replace('.ised','.fits')
    
    if del_modelfile:
        os.remove(modelfile)
        print "Deleted ...", modelfile

    fh.close()

    return None

