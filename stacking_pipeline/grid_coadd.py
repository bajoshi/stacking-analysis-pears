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

sys.path.append(massive_galaxies_dir + 'codes/')
# import combine_pas as cb

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

def rescale(ids, zs):
    medarr = np.zeros(len(ids))
    
    for k in range(len(ids)):
        
        redshift = zs[k]
        lam_em, flam_em, ferr, specname = fileprep(ids[k], redshift)
        
        # Store median of values from 4400A-4600A for each spectrum
        arg4400 = np.argmin(abs(lam_em - 4400))
        arg4600 = np.argmin(abs(lam_em - 4600))
        medarr[k] = np.median(flam_em[arg4400:arg4600+1])
    
    medval = np.median(medarr)
    
    # Return the maximum in array of median values
    return medarr, medval, np.std(medarr)

def smoothspec(flam, width, kernel_type):
    """
    This function will "smooth" the supplied spectrum by the specified 
    width, which is the width of the kernel that it uses; in pixels.

    The default width is 1 pixels. i.e. this will not smooth at all.
    
    The default kernel type is gaussian but the user can also choose to
    use a box kernel.

    These defaults are defined in fileprep instead of here. This is because 
    all my codes use fileprep
    """

    if kernel_type == 'gauss':
        gauss_kernel = Gaussian1DKernel(width)
        smoothed_flam = convolve(flam, gauss_kernel)
    elif kernel_type == 'box':
        box_kernel = Box1DKernel(width)
        smoothed_flam = convolve(flam, box_kernel)

    return smoothed_flam

def fileprep(pears_index, redshift, field, apply_smoothing=False, width=1, kernel_type='gauss', use_single_pa=True):

    # read in spectrum file
    data_path = home + "/Documents/PEARS/data_spectra_only/"
    # Get the correct filename and the number of extensions
    if field == 'GOODS-N':
        filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    elif field == 'GOODS-S':
        filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

    fitsfile = fits.open(filename)
    n_ext = fitsfile[0].header['NEXTEND']

    specname = os.path.basename(filename)

    if use_single_pa:
        # Get highest netsig to find the spectrum to be added
        if n_ext > 1:
            netsiglist = []
            palist = []
            for count in range(n_ext):
                #print "At PA", fitsfile[count+1].header['POSANG']  
                # Above line useful for debugging. Do not remove. Just uncomment.
                fitsdata = fitsfile[count+1].data
                netsig = get_net_sig(fitsdata)
                netsiglist.append(netsig)
                palist.append(fitsfile[count+1].header['POSANG'])
                #print "At PA", fitsfile[count+1].header['POSANG'], "with NetSig", netsig  
                # Above line also useful for debugging. Do not remove. Just uncomment.
            netsiglist = np.array(netsiglist)
            maxnetsigarg = np.argmax(netsiglist)
            netsig_chosen = np.max(netsiglist)
            spec_toadd = fitsfile[maxnetsigarg+1].data
            pa_chosen = fitsfile[maxnetsigarg+1].header['POSANG']
        elif n_ext == 1:
            spec_toadd = fitsfile[1].data
            pa_chosen = fitsfile[1].header['POSANG']
            netsig_chosen = get_net_sig(fitsfile[1].data)
            
        # Now get the spectrum to be added
        lam_obs = spec_toadd['LAMBDA']
        flam_obs = spec_toadd['FLUX']
        ferr = spec_toadd['FERROR']
        contam = spec_toadd['CONTAM']

        # Must include more input checks and better error handling
        # check that input wavelength array is not empty
        if not lam_obs.size:
            print pears_index, " in ", field, " has an empty wav array. Returning empty array..."
            return lam_obs, flam_obs, ferr, specname, pa_chosen, netsig_chosen
            
        # Subtract Contamination
        flam_obs = flam_obs - contam

        # apply smoothing if necessary
        if apply_smoothing:
            #print "Will apply smoothing using Gaussian kernel of width", width, "to", pears_index, "in", field
            flam_obs = smoothspec(flam_obs, width, kernel_type)
            
        # Now chop off the ends and only look at the observed spectrum from 6000A to 9500A
        arg6000 = np.argmin(abs(lam_obs - 6000))
        arg9500 = np.argmin(abs(lam_obs - 9500))
            
        lam_obs = lam_obs[arg6000:arg9500]
        flam_obs = flam_obs[arg6000:arg9500]
        ferr = ferr[arg6000:arg9500]

        # Now deredshift the spectrum
        lam_em = lam_obs / (1 + redshift)
        flam_em = flam_obs * (1 + redshift)
        ferr_em = ferr * (1 + redshift)
        # check the relations for deredshifting

        return lam_em, flam_em, ferr_em, specname, pa_chosen, netsig_chosen

    else:
        # read recarray if you're using a combined spectrum
        recarray = np.load(massive_galaxies_dir + 'pears_pa_combination_info_' + field + '.npy')

        # find id in recarray        
        idarg = np.where(recarray['pearsid'] == pears_index)[0]
        #print idarg
        idarg = int(idarg)  # correct shape if necessary
        # i.e. if the next few lines dont get an integer then the shape is off and it'll barf when it tries the convolution

        lam_grid = recarray['lam_grid'][idarg]
        comb_flam = recarray['combined_flam'][idarg]
        comb_flamerr = recarray['combined_ferr'][idarg]
        rejected_pa = recarray['rejected_pa'][idarg]
        combined_pa = recarray['combined_pa'][idarg]
        # the combined flux in here should already be contamination subtracted
        print pears_index, combined_pa

        # the combined pa might be a list type so make sure that it is a string
        if type(combined_pa) is list:
            if len(combined_pa) == 1:
                combined_pa = combined_pa[0]
                if 'PA' not in combined_pa:
                    combined_pa = 'PA' + combined_pa

        # apply smoothing if necessary
        if apply_smoothing:
            comb_flam = smoothspec(comb_flam, width, kernel_type)
            
        # Now chop off the ends and only look at the observed spectrum from 6000A to 9500A
        arg6000 = np.argmin(abs(lam_grid - 6000))
        arg9500 = np.argmin(abs(lam_grid - 9500))
            
        lam_grid = lam_grid[arg6000:arg9500]
        comb_flam = comb_flam[arg6000:arg9500]
        comb_flamerr = comb_flamerr[arg6000:arg9500]

        # Now unredshift the spectrum
        lam_em = lam_grid / (1 + redshift)
        flam_em = comb_flam * (1 + redshift)
        ferr_em = comb_flamerr * (1 + redshift)
        # check the relations for deredshifting

        return lam_em, flam_em, ferr_em, specname, combined_pa

if __name__ == '__main__':
    
    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Coaddition started at --",
    print dt.now()
    
    # ----------------------------------------- READ IN CATALOGS ----------------------------------------- #
    # Read in catalog of all PEARS fitting results and assign arrays
    # For now we need the id+field, spz, stellar mass, and u-r color
    cat = np.genfromtxt(stacking_analysis_dir + 'color_stellarmass.txt', dtype=None, names=True)
    
    pears_id = cat['pearsid']
    pears_field = cat['field']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    spz = cat['spz']

    # ----------------------------------------- Code config params ----------------------------------------- #
    # Change only the parameters here to change how the code runs
    # Ideally you shouldn't have to change anything else.
    lam_step = 100
    lam_grid = np.arange(2700, 6000, lam_step)
    # Lambda grid decided based on observed wavelength range i.e. 6000 to 9500
    # and the initially chosen redshift range 0.6 < z < 1.2
    # This redshift range was chosen so that the 4000A break would fall in the observed wavelength range
    col_step = 0.3
    mstar_step = 0.5
    col_low = 0.0
    col_high = 3.0
    mstar_low = 7.0
    mstar_high = 12.0
    
    # ----------------------------------------- Other preliminaries ----------------------------------------- #

    # ----------------------------------------- Begin creating stacks ----------------------------------------- #
    added_gal = 0
    skipped_gal = 0
    gal_per_bin = np.zeros((len(np.arange(col_low, col_high, col_step)), \
        len(np.arange(mstar_low, mstar_high, mstar_step))))
    
    for i in np.arange(col_low, col_high, col_step):
        for j in np.arange(mstar_low, mstar_high, mstar_step):
            
            gal_current_bin = 0
            
            logging.info("\n ONGRID %.1f %.1f", i, j)
            print "ONGRID", i, j
            
            indices = np.where((ur_color >= i) & (ur_color < i + col_step) &\
                            (stellarmass >= j) & (stellarmass < j + mstar_step))[0]
            
            old_flam = np.zeros(len(lam_grid))
            old_flamerr = np.zeros(len(lam_grid))
            num_points = np.zeros(len(lam_grid))
            num_galaxies = np.zeros(len(lam_grid))
            
            old_flam = old_flam.tolist()
            old_flamerr = old_flamerr.tolist()
            
            logging.info("Number of spectra to coadd in this grid cell -- %d.", len(pears_id[indices]))

            # this is to get the blue cloud sample
            """
            if (i == 0.6) and (j == 8.5):
                print np.unique(pears_id[indices]), len(np.unique(pears_id[indices]))
                sys.exit()
            """

            # rescale to 200A band centered on a wavelength of 4500A # 4400A-4600A
            # This function returns the maximum of the median values (in the given band) from all given spectra
            # All spectra to be coadded in a given grid cell need to be divided by this value
            if indices.size:
                medarr, medval, stdval = rescale(pears_id[indices], photz[indices])
                print "ONGRID", i, j, "with", medval, 
                print "as the normalization value and a maximum possible spectra of", len(pears_id[indices]), "spectra."
            else:
                print "ONGRID", i, j, "has no spectra to coadd. Moving to next grid cell."
                continue
    
            # Loop over all spectra in a grid cell and coadd them
            for u in range(len(pears_id[indices])):
                
                # This step should only be done on the first iteration within a grid cell
                # This converts every element (which are all 0 to begin with) 
                # in the flux and flux error arrays to an empty list
                # This is done so that function add_spec() can now append to every element
                if u == 0.0:
                    for x in range(len(lam_grid)):
                        old_flam[x] = []
                        old_flamerr[x] = []
            
                # Get redshift from previously saved 3DHST photz catalog
                redshift = photz[indices][u]
                
                # Get rest frame values for all quantities
                lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][u], redshift)
                
                """
                # Divide by the grism sensitivity curve
                # Wouldn't aXe do this itself when it extracted the PEARS spectra??
                # yes aXe has done it, if you use the spectrum in cgs units
                # but if you use the spectrum in counts/s then you'll need to divide by the sensitivity curve
                # Therefore, not needed right now
                # flam_obs, ferr = divide_sensitivity(flam_obs, ferr, lam_obs)
                """
                
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
                    gal_current_bin += 1
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
                        np.sqrt(len(old_flam[y])))**2 + np.sum(old_flamerr[y]) / gal_current_bin)
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
            hdr["NUMSPEC"] = str(int(gal_current_bin))
            hdr["NORMVAL"] = str(medval)
                   
            dat = np.array((old_flam, old_flamerr)).reshape(2, len(lam_grid))
            hdulist.append(fits.ImageHDU(data = dat, header = hdr))

            row = int(i/col_step)
            column = int((j - mstar_low)/mstar_step)
            gal_per_bin[row,column] = gal_current_bin

            print "ONGRID", i, j, "added", gal_current_bin, "spectra."
            print '\n'
    
    print added_gal, skipped_gal
    print np.flipud(gal_per_bin)
    print np.sum(gal_per_bin, axis=None)
    
    # Total time taken
    print "Time taken for coaddition -- %.2f seconds", time.time() - start

    sys.exit(0)
