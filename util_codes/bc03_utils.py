import numpy as np
from astropy.io import fits
import array

import subprocess
import os
import sys
import shutil

home = os.getenv('HOME')

def read_current_filepos(filehandle, dtype='i', number=1):
    """
        This function reads the given number of binary data which is of the supplied type
        from the specified filehandle. The default number of elements to extract is one
        and the default type is an integer.
    """

    arr = array.array(dtype)
    arr.fromfile(filehandle, number)

    return np.asarray(arr)

def ised2fits(modelfile, del_modelfile=False):
    """
        This function will save all the spectra within the isedfile
        saved by csp_galaxev to a fits file. The spectrum for each 
        age will be saved in a different extension.
        The zeroth extension is empty.
        The first extension is the grid of wavelength for the spectra.
        Wavelengths are in angstroms.
        The second extension is the grid of ages.
        Ages are in years.
        The third and subsequent extensions provide the spectra at the 
        ages given in the second extension.
        The spectra are in units of L_sol/A.

        This code is largely based on the code for reading ised files 
        from the ezgal package. 
        See the ezgal github page
        and also refer to Mancone & Gonzalez 2012, PASP, 124, 606.
    """

    if os.path.isfile(modelfile.replace('.ised','.fits')):
        print("\nChecking for:", modelfile.replace('.ised','.fits'))
        print("Fits file already exists. Skipping.")
        return None

    fh = open(modelfile, 'rb')
    
    # dtype for this file
    # int = i = 4 bytes
    # float = f = 4 bytes
    # byte = b = 1 byte

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
    
    allages = read_current_filepos(fh, dtype='f', number=totalages) # in years
    
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
           junk = read_current_filepos(fh, dtype='f', number=6*iseg)
        junk = read_current_filepos(fh, dtype='f', number=3)
        junk = read_current_filepos(fh)
        junk = read_current_filepos(fh, dtype='f')
        junk = read_current_filepos(fh, dtype='b', number=80)
        junk = read_current_filepos(fh, dtype='f', number=4)
        junk = read_current_filepos(fh, dtype='b', number=160)
        junk = read_current_filepos(fh)
        junk = read_current_filepos(fh, number=3)

    totalwavelengths = read_current_filepos(fh)[0]
    allwavelengths = read_current_filepos(fh, dtype='f', number=totalwavelengths)
    
    seds = np.zeros((totalages, totalwavelengths), dtype=np.float32)
    
    # Open fits file
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)

    hdulist.append(fits.ImageHDU(allwavelengths))
    hdulist.append(fits.ImageHDU(allages))
    
    for i in range(totalages):
        ignore = read_current_filepos(fh, number=2)
        nlam = read_current_filepos(fh)
    
        seds[i] = read_current_filepos(fh, dtype='f', number=totalwavelengths)
        
        num = int(read_current_filepos(fh))
        ignore = read_current_filepos(fh, dtype='f', number=num)
    
        hdulist.append(fits.ImageHDU(seds[i]))
    
    hdulist.writeto(modelfile.replace('.ised','.fits'), overwrite=True)
    #print("\nWriting ...", modelfile.replace('.ised','.fits'))
    
    if del_modelfile:
        os.remove(modelfile)
        #print("Deleted ...", modelfile)

    fh.close()

    return None

def call_cspgalaxev(isedfile, tau, output, dust='N', z='0', sfh='1', recycle='N', tcut='20.0', verbose=False):
    """
    This function expects to get:

    isedfile: file containing SSP spectra which will be integrated
              according to the SFH specified to give the CSP spectra.

    SFH parameters: This will depend on the chosen SFH,
                    which is why they're given as a list
                    which can be of varying length.

                    e.g., 
                    1. Exponentially declining SFH
                    sfh code = 1
                    this only requires tau to be specified.

                    2. 

    How it works:
    Step 1: Open a process constructor to the csp_galaxev program.
    You need this, i.e., Popen, instead of subprocess.run() or subprocess.call().
    This one allows you to interact with the process.
    The interaction is achieved by communicating parameters below.
    This is different from passing arguments to the process on the command line.
    Passing command line args can be done through subprocess.run().
    However, the csp_galaxev code does not accept command line args.
    It must be given the args interactively after the csp_galaxev program is called.

    The first argument to Popen is the path to the the program to be called.
    After that we set up the PIPES. This is very important if we want to 
    communicate parameters to the open program.
    We're essentially opening a hook to the stdin stream, i.e., this allows 
    the parameters to be piped into the program by using communicate below.
    The encoding argument specifies that all the parameters being passed 
    are simple strings which is what csp_galaxev requires. If this isn't 
    specified then the communicated parameters below will be assumed to be 
    byte objects (binary?) instead (which will throw an error).
    An optional stdout argument may also be given. This can be a logfile if you want.
    If it isn't given then the output from the program that is called will
    be displayed on the terminal (which is the stdout stream by default).

    Step 2: Join parameters to be communicated, which have to be strings, 
    by a newline, as expected by csp_galaxev.

    Step 3: Call process and wait until it finishes.

    """

    # Open the csp_galaxev process
    start = subprocess.Popen([home + '/Documents/GALAXEV2016/bc03/src/csp_galaxev'], \
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='ascii')
    
    # Join the parameters to be communnicated to the csp_galaxev code
    # by using os.linesep.join()
    communicate_params = os.linesep.join([isedfile, dust, z, sfh, str(tau), recycle, tcut, output])
    # communicate params is now a long string of parameters with the
    # parameters separated by lines. This is what csp_galaxev expects.

    # Print info to the screen
    if verbose:
        print("\nCommunicating the following parameters to csp_galaxev:")
        print("isedfile for SSP spectra:", isedfile)
        print("Include dust?:", dust)
        print("Redshift for spectrum within csp_galaxev:", z)
        print("SFH code:", sfh)
        print("Parameters for SFH, Tau [Gyr]:", "{:.2f}".format(tau))
        print("Other default parameters required by csp_galaxev:")
        print("Recycle gas from stars:", recycle)
        print("Time after which SFR is forced to be zero [Gyr]:", tcut)
        print("Path to output file generated by csp_galaxev:", output)

    # Call the process
    params = start.communicate(input=communicate_params)

    # Not entirely sure what this does but I think it's important.
    # I *think* this means we're asking the kernel/shell(?) to 
    # wait until the program is finished.
    start.poll()

    return None

def get_bc03_spectrum(age, tau, metals, outdir_ised, save2ascii=True):

    # Get the isedfile and the metallicity in the format that BC03 needs
    if metals == 0.0001:
        metallicity = 'm22'
        isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m22_chab_ssp.ised"
    elif metals == 0.0004:
        metallicity = 'm32'
        isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m32_chab_ssp.ised"
    elif metals == 0.004:
        metallicity = 'm42'
        isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m42_chab_ssp.ised"
    elif metals == 0.008:
        metallicity = 'm52'
        isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m52_chab_ssp.ised"
    elif metals == 0.02:
        metallicity = 'm62'
        isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m62_chab_ssp.ised"
    elif metals == 0.05:
        metallicity = 'm72'
        isedfile = home + "/Documents/GALAXEV2016/bc03/src/bc2003_hr_xmiless_m72_chab_ssp.ised"

    # Check that the isedfile exists
    if not os.path.isfile(isedfile):
        print("ised file missing...")
        sys.exit(0)

    # Create the name for the output files
    tau_str = "{:.3f}".format(tau).replace('.', 'p')
    output = outdir_ised + "bc2003_hr_" + metallicity + "_csp_tau" + tau_str + "_chab"

    # Check if the output file already exists
    # Checking for the fits files because the ised files usually get deleted
    if not os.path.isfile(output + '.fits'):
        
        # Now call csp_galaxev
        call_cspgalaxev(isedfile, tau, output)

        # now read in the output generated by csp_galaxev and 
        # get the spectrum for the age required.
        # First convert to fits for easier handling
        ised2fits(output + '.ised', del_modelfile=False)
        # the del_modelfile keyword should be false I think
        # if we're using calling csp_galaxev from within
        # emcee during the fitting process. I think there
        # is an issue if more than one walker starts looking 
        # for an ised file that was deleted by a concurrent walker.
        # Same issue below with trying to remove unnecessary files.

        # Remove some files we dont need
        """
        os.remove(output + '.1color')
        os.remove(output + '.2color')
        os.remove(output + '.1ABmag')
        os.remove(output + '.5color')
        os.remove(output + '.6lsindx_ffn')
        os.remove(output + '.6lsindx_sed')
        os.remove(output + '.7lsindx_ffn')
        os.remove(output + '.7lsindx_sed')
        os.remove(output + '.8lsindx_sed_fluxes')
        os.remove(output + '.9color')
        os.remove(output + '.acs_wfc_ABmag')
        os.remove(output + '.w_age_rf')
        os.remove(output + '.wfc3_ABmag')
        os.remove(output + '.wfc3_uvis1_ABmag')
        os.remove(output + '.wfc3_uvis1_legus_ABmag')
        os.remove(output + '.wfpc2_johnson_color')
        """

    #else:
    #    print("\nOutput fits file exists. Moving on to age extraction.")

    # Now open the fits file and get the correct age required
    h = fits.open(output + '.fits')
    lam = h[1].data
    ages = h[2].data

    # the ages in the fits files are in years
    age *= 1e9

    age_idx = np.argmin(abs(ages - age))

    llam = h[3 + age_idx].data

    # Scale to correct luminosity units
    L_sun = 3.84e33  # in erg per sec
    llam *= L_sun  # this is now in ergs / s / A

    h.close()

    return lam, llam


