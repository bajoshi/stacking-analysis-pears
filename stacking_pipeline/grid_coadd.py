# coding: utf-8
from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits
#from scipy.stats import gaussian_kde
#from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figures_dir = home + "/Desktop/FIGS/stacking-analysis-figures/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
pears_data_path = home + "/Documents/PEARS/data_spectra_only/"

sys.path.append(massive_galaxies_dir + 'cluster_codes/')
import cluster_do_fitting as cf

def get_net_sig(*args):
    """
    This function simply needs either the fits extension that 
    contains the spectrum for which netsig is to be computed
    or 
    the counts and the errors on the counts as separate arrays.

    It should be able to figure out what you gave but if you
    give it the separate arrays then make sure that the counts 
    arrays comes before the counts error array.

    DO NOT give it any additional arguments or it will fail.
    """

    if len(args) == 1:
        fitsdata = args[0]
        count_arr = fitsdata['COUNT']
        error_arr = fitsdata['ERROR']
    elif len(args) == 2:
        count_arr = args[0]
        error_arr = args[1]

    # Make sure that the arrays are not empty to begin with
    if not count_arr.size:
        print "Returning -99.0 for NetSig due to empty signal",
        print "and/or noise array for this object (or some PA for this object)."
        return -99.0
    if not error_arr.size:
        print "Returning -99.0 for NetSig due to empty signal",
        print "and/or noise array for this object (or some PA for this object)."
        return -99.0

    # Also check that the error array does not have ALL zeros
    if np.all(error_arr == 0.0):
        #print "Returning -99.0 for NetSig due to noise array",
        #print "containing all 0 for this object (or some PA for this object)."
        return -99.0

    try:
        signal_sum = 0
        noise_sum = 0
        totalsum = 0
        cumsum = []
        
        sn = count_arr/error_arr
        # mask NaNs in this array to deal with the division by errors than are 0
        mask = ~np.isfinite(sn)
        sn = ma.array(sn, mask=mask)
        sn_sorted = np.sort(sn)
        sn_sorted_reversed = sn_sorted[::-1]
        reverse_mask = ma.getmask(sn_sorted_reversed)
        # I need the reverse mask for checking since I'm reversing the sn sorted array
        # and I need to only compute the netsig using unmasked elements.
        # This is because I need to check that the reverse sorted array will not have 
        # a blank element when I use the where function later causing the rest of the 
        # code block to mess up. Therefore, the mask I'm checking i.e. the reverse_mask
        # and the sn_sorted_reversed array need to have the same order.
        sort_arg = np.argsort(sn)
        sort_arg_rev = sort_arg[::-1]

        i = 0
        for _count_ in sort_arg_rev:
            # if it a masked element then don't do anything
            if reverse_mask[i]:
                i += 1
                continue
            else:
                signal_sum += count_arr[_count_]
                noise_sum += error_arr[_count_]**2
                totalsum = signal_sum/np.sqrt(noise_sum)
                #print reverse_mask[i], sn_sorted_reversed[i], 
                #print _count_, signal_sum, totalsum  
                # Above print line useful for debugging. Do not remove. Just uncomment.
                cumsum.append(totalsum)
                i += 1

        cumsum = np.asarray(cumsum)
        if not cumsum.size:
            print "Exiting due to empty cumsum array. More debugging needed."
            print "Cumulative sum array:", cumsum
            sys.exit(0)
        netsig = np.nanmax(cumsum)
        
        return netsig
            
    except ZeroDivisionError:
        logging.warning("Division by zero! The net sig here cannot be trusted. Setting Net Sig to -99.")
        print "Exiting. This error should not have come up anymore."
        sys.exit(0)
        return -99.0

def get_interp_spec(lam, flam, ferr):
    interp_spec = np.interp(lam_grid, lam, flam, left=0, right=0)
    interp_spec_err = np.interp(lam_grid, lam, ferr, left=0, right=0)
    return interp_spec, interp_spec_err

def add_spec(specname, lam_em, flam_em, ferr, old_flam, old_flamerr, num_points, num_galaxies, lam_grid, lam_step):
    
    for i in range(len(lam_grid)):
        
        # add fluxes
        new_ind = np.where((lam_em >= lam_grid[i] - lam_step/2) & (lam_em < lam_grid[i] + lam_step/2))[0]

        if new_ind.size:
            
            # Only count a galaxy in a particular bin if for that bin at least one point is nonzero
            if np.any(flam_em[new_ind] != 0):
                num_galaxies[i] += 1
            
            # Reject points with excess contamination
            # Rejecting points that are more than 20% contaminated
            # Reject points that have negative signal
            # Looping over every point in a delta lambda bin
            for j in range(len(new_ind)):
                sig = flam_em[new_ind][j]
                noise = ferr[new_ind][j]
                
                if sig > 0: # only append those points where the signal is positive
                    if noise/sig < 0.20: # only append those points that are less than 20% contaminated
                        old_flam[i].append(sig)
                        old_flamerr[i].append(noise**2) # adding errors in quadrature
                        num_points[i] += 1 # keep track of how many points were added to each bin in lam_grid
                else:
                    continue

        else:            
            continue

    return old_flam, old_flamerr, num_points, num_galaxies

def rescale(id_arr_cell, field_arr_cell, z_arr_cell, dl_tbl):
    """
    This function will provide teh median of all median
    f_lambda values (see below) at approx 4500 A. This 
    is for the purposes of rescaling all spectra within
    a cell to this median of medians.
    """

    # Define empty array to store the median 
    # value of f_lambda between 4400 to 4600 A
    # for each observed spectrum
    medarr = np.zeros(len(id_arr_cell))

    # Define redshift array used in lookup table
    z_arr = np.arange(0.005, 6.005, 0.005)
    
    for k in range(len(id_arr_cell)):

        # Get current ID and Field
        current_id = id_arr_cell[k]
        current_field = field_arr_cell[k]
        
        # Get observed data and deredshift the spectrum
        grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = \
        cf.get_data(current_id, current_field)

        # If the return code was 0, then exit,
        # i.e., the observed spectrum is unuseable.
        # This condition should never be triggered. 
        if return_code == 0:
            print current_id, current_field
            print "Return code should not have been 0. Exiting."
            sys.exit(0)

        # Now deredshift the observed data
        redshift = z_arr_cell[k]

        zidx = np.argmin(abs(z_arr - redshift))
        # Make sure that the z_arr here is the same array that was 
        # used to generate the dl lookup table.
        dl = dl_tbl['dl_cm'][zidx]  # has to be in cm

        lam_em = grism_lam_obs / (1 + redshift)
        llam_em = grism_flam_obs * (1 + redshift) * (4 * np.pi * dl * dl)
        
        # Store median of values from 4400A-4600A for each spectrum
        arg4400 = np.argmin(abs(lam_em - 4400))
        arg4600 = np.argmin(abs(lam_em - 4600))
        medarr[k] = np.median(llam_em[arg4400:arg4600+1])
    
    medval = np.median(medarr)
    
    # Return the median in array of median values
    return medarr, medval, np.std(medarr)

def create_stacks(cat, urcol, z_low, z_high, z_indices, start):

    # Read in catalog of all PEARS fitting results and assign arrays
    # For now we need the id+field, spz, stellar mass, and u-r color
    pears_id = cat['PearsID'][z_indices]
    pears_field = cat['Field'][z_indices]
    zp = cat['zp_minchi2'][z_indices]

    stellar_mass = np.log10(cat['zp_ms'][z_indices])  # because the code below is expected log(stellar mass)
    ur_color = urcol[z_indices]

    # Read in dl lookup table
    # Required for deredshifting
    dl_tbl = np.genfromtxt(massive_galaxies_dir + 'cluster_codes/dl_lookup_table.txt', dtype=None, names=True)

    # ----------------------------------------- Code config params ----------------------------------------- #
    # Change only the parameters here to change how the code runs
    # Ideally you shouldn't have to change anything else.
    lam_step = 100
    lam_grid = np.arange(2700, 6000, lam_step)
    # Lambda grid decided based on observed wavelength range i.e. 6000 to 9500
    # and the initially chosen redshift range 0.6 < z < 1.2
    # This redshift range was chosen so that the 4000A break would fall in the observed wavelength range    
    col_low = 0.0
    col_high = 3.0
    col_step = 0.5

    mstar_low = 7.0
    mstar_high = 12.0
    mstar_step = 1.0
    
    # ----------------------------------------- Other preliminaries ----------------------------------------- #

    # ----------------------------------------- Begin creating stacks ----------------------------------------- #
    added_gal = 0
    skipped_gal = 0

    # Create empty array to store num of galaxies in each cell
    gal_per_cell = np.zeros((len(np.arange(col_low, col_high, col_step)), \
        len(np.arange(mstar_low, mstar_high, mstar_step))))
    
    for col in np.arange(col_low, col_high, col_step):
        for ms in np.arange(mstar_low, mstar_high, mstar_step):
            
            gal_current_cell = 0
            
            print "\n", "For z range:", z_low, "<= z <", z_high
            print "Stacking in cell:"
            print "Color range:", col, col+col_step
            print "Stellar mass range:", ms, ms+mstar_step
            
            # Find the indices (corresponding to catalog entries)
            # that are within the current cell
            indices = np.where((ur_color >= col) & (ur_color < col + col_step) &\
                            (stellar_mass >= ms) & (stellar_mass < ms + mstar_step))[0]

            num_galaxies_cell = int(len(pears_id[indices]))
            print "Number of spectra to coadd in this grid cell --", num_galaxies_cell

            if num_galaxies_cell == 0:
                continue
            
            old_flam = np.zeros(len(lam_grid))
            old_flamerr = np.zeros(len(lam_grid))
            num_points = np.zeros(len(lam_grid))
            num_galaxies = np.zeros(len(lam_grid))
            
            old_flam = old_flam.tolist()
            old_flamerr = old_flamerr.tolist()

            # this is to get the blue cloud sample
            #if (i == 0.6) and (j == 8.5):
            #    print np.unique(pears_id[indices]), len(np.unique(pears_id[indices]))
            #    sys.exit()

            # rescale to 200A band centered on a wavelength of 4500A # 4400A-4600A
            # This function returns the median of the median values (in the given band) from all given spectra
            # All spectra to be coadded in a given grid cell need to be divided by this value
            medarr, medval, stdval = rescale(pears_id[indices], pears_field[indices], zp[indices], dl_tbl)
            print "This cell has median value:", "{:.3e}".format(medval), " [erg s^-1 A^-1]"
            print "as the normalization value and a maximum possible of", len(pears_id[indices]), "spectra."
    
            """
            # Loop over all spectra in a grid cell and coadd them
            for u in range(len(pears_id[indices])):
                
                # This step should only be done on the first iteration within a grid cell
                # This converts every element (which are all 0 to begin with) 
                # in the flux and flux error arrays to an empty list
                # This is done so that function add_spec() can now append to every element
                if u == 0:
                    for x in range(len(lam_grid)):
                        old_flam[x] = []
                        old_flamerr[x] = []
            
                # Get redshift from catalog
                current_redshift = spz[indices][u]

                current_id = pears_id[indices][u]
                current_field = pears_field[indices][u]
                
                # ----------------------------- Get data ----------------------------- #
                grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code \
                = cf.get_data(current_id, current_field)

                if return_code == 0:
                    print current_id, current_field
                    print "Return code should not have been 0. Skipping this galaxy."
                    continue
 
                # Match with photometry catalog and get photometry data


                # Divide by the grism sensitivity curve
                # Wouldn't aXe do this itself when it extracted the PEARS spectra??
                # yes aXe has done it, if you use the spectrum in cgs units
                # but if you use the spectrum in counts/s then you'll need to divide by the sensitivity curve
                # Therefore, not needed right now
                # flam_obs, ferr = divide_sensitivity(flam_obs, ferr, lam_obs)
                
                # Divide by median value at 4400A to 4600A to rescale. 
                # Multiplying by median value of the flux medians to get it back to physical units
                flam_em = (flam_em / medarr[u]) * medval
                ferr = (ferr / medarr[u]) * medval

                # These (skipspec) were looked at by eye and seemed crappy
                # I'm also excluding spectra with emission lines.
                if (specname in skipspec) or (specname in em_lines):
                    print "Skipped", specname
                    skipped_gal += 1
                    continue

                # Reject spectrum if overall contamination too high
                if np.sum(abs(ferr)) > 0.3 * np.sum(abs(flam_em)):
                    print "Skipped", specname
                    skipped_gal += 1
                    continue
                else:
                    # add the spectrum
                    added_gal += 1
                    gal_current_cell += 1
                    old_flam, old_flamerr, num_points, num_galaxies = \
                    add_spec(specname, lam_em, flam_em, ferr, old_flam, old_flamerr, \
                        num_points, num_galaxies, lam_grid, lam_step)
            
                # Now intepolate the spectrum on to the lambda grid
                #interp_spec, interp_spec_err = get_interp_spec(lam_em, flam_em, ferr)
    
                #counts += interp_spec
                #countserr += interp_spec_err**2

            # taking median
            # maybe I should also try doing a mean after 3sigma clipping and compare
            for y in range(len(lam_grid)):
                if old_flam[y]:
                    old_flamerr[y] = \
                    np.sqrt((1.253 * np.std(old_flam[y]) / \
                        np.sqrt(len(old_flam[y])))**2 + np.sum(old_flamerr[y]) / gal_current_cell)
                    old_flam[y] = np.median(old_flam[y])
                else:
                    old_flam[y] = 0.0
                    old_flamerr[y] = 0.0

            hdr = fits.Header()
            
            #hdr["XTENSION"] = "IMAGE              / Image extension "
            #hdr["BITPIX"]  = "                 -64 / array data type"
            #hdr["NAXIS"]   = "                   2 / number of array dimensions"
            #hdr["NAXIS1"]  = "                 161"
            #hdr["NAXIS2"]  = "                   2"
            #hdr["PCOUNT"]  = "                   0 / number of parameters"
            #hdr["GCOUNT "] = "                   1 / number of groups"
            hdr["ONGRID"]  = str(i) + "," + str(j)
            hdr["NUMSPEC"] = str(int(gal_current_cell))
            hdr["NORMVAL"] = str(medval)
                   
            dat = np.array((old_flam, old_flamerr)).reshape(2, len(lam_grid))
            hdulist.append(fits.ImageHDU(data = dat, header = hdr))

            row = int(i/col_step)
            column = int((j - mstar_low)/mstar_step)
            gal_per_cell[row,column] = gal_current_cell

            print "ONGRID", i, j, "added", gal_current_cell, "spectra."
            print '\n'
            """

    # Time taken
    print "Time taken for stacking --", "{:.2f}".format(time.time() - start), "seconds"
    
    print added_gal, skipped_gal
    print np.flipud(gal_per_cell)
    print np.sum(gal_per_cell, axis=None)

    return None
    
if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Coaddition started at --",
    print dt.now()
    
    # ----------------------------------------- READ IN CATALOGS ----------------------------------------- #
    # Read in results for all of PEARS
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)
    # Read in U-R color
    urcol = np.load(stacking_analysis_dir + 'ur_arr_all.npy')

    """
    # ------------------------------- Read in photometry and grism+photometry catalogs ------------------------------- #
    # GOODS-N from 3DHST
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
    'f_U', 'e_U', 'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
    'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), \
        skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), \
        skip_header=3)
    """

    # ----------------------------------------- Now create stacks ----------------------------------------- #
    # Get z intervals and their indices
    zp = cat['zp_minchi2']

    all_z_low = np.array([0.0, 0.4, 0.7, 1.0])
    all_z_high = np.array([0.4, 0.7, 1.0, 2.0])

    # Separate grid stack for each redshift interval
    # This function will create and save the stacks in a fits file
    for i in range(1,4):
        
        # Get z range and indices
        z_low = all_z_low[i]
        z_high = all_z_high[i]
        z_indices = np.where((zp >= z_low) & (zp < z_high))[0]
        #goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst,
        create_stacks(cat, urcol, z_low, z_high, z_indices, start)

        sys.exit(0)

    # Total time taken
    print "Total time taken for all stacks --", "{:.2f}".format((time.time() - start)/60.0), "minutes."

    sys.exit(0)
