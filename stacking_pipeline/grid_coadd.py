# coding: utf-8
from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy.interpolate import splev, splrep

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')  # Does not have a trailing slash at the end
figs_dir = home + "/Documents/pears_figs_data/"
stacking_analysis_dir = home + "/Documents/GitHub/stacking-analysis-pears/"
stacking_figures_dir = home + "/Documents/stacking_figures/"
massive_galaxies_dir = home + "/Documents/GitHub/massive-galaxies/"
pears_spectra_dir = home + "/Documents/pears_figs_data/data_spectra_only/"

stacking_utils_dir = stacking_analysis_dir + "util_codes"
sys.path.append(stacking_utils_dir)
sys.path.append(stacking_analysis_dir + "stacking_pipeline/")
import make_col_ms_plots
from convert_to_sci_not import convert_to_sci_not

# Define grid params
# Outside of any functions to make sure these are the same everywhere
col_low = 0.0
col_high = 3.0
col_step = 0.5

mstar_low = 8.0
mstar_high = 12.0
mstar_step = 1.0

numcol = int((col_high - col_low)/col_step)
nummass = int((mstar_high - mstar_low)/mstar_step)

def gen_balmer_lines():

    # Check the latest Rydberg constant data here
    # https://physics.nist.gov/cgi-bin/cuu/Value?ryd|search_for=Rydberg
    # short list here for quick reference: https://physics.nist.gov/cuu/pdf/wall_2018.pdf
    rydberg_const = 10973731.568  # in m^-1

    balmer_line_wav_list = []

    for lvl in range(3, 15):

        energy_levels_term = (1/4) - (1/lvl**2)
        lam_vac = (1/rydberg_const) * (1/energy_levels_term)

        lam_vac_ang = lam_vac*1e10  # meters to angstroms # since the rydberg const above is in (1/m)

        #print("Transition:", lvl, "--> 2,       wavelength in vacuum [Angstroms]:", "{:.3f}".format(lam_vac_ang))

        balmer_line_wav_list.append(lam_vac_ang)

    return balmer_line_wav_list

def get_mask_indices(obs_wav, redshift):

    # Define rest-frame wavelengths in vacuum
    # Emission or absorption doesn't matter
    gband = 4300
    #hbeta = 4862.72
    oiii4959 = 4960.295
    oiii5007 = 5008.239
    mg2_mgb = 5175
    fe5270 = 5270
    fe5335 = 5335
    fe5406 = 5406
    nad = 5890
    #halpha = 6564.614

    all_balmer_lines = gen_balmer_lines()

    all_lines = [gband, oiii4959, oiii5007, mg2_mgb, fe5270, fe5335, fe5406, nad]
    all_lines = all_lines + all_balmer_lines

    # Set up empty array for masking indices
    mask_indices = []

    # Loop over all lines and get masking indices
    for line in all_lines:

        obs_line_wav = (1 + redshift) * line
        if (obs_line_wav >= obs_wav[0]) and (obs_line_wav <= obs_wav[-1]):
            closest_obs_wav_idx = np.argmin(abs(obs_wav - obs_line_wav))

            #print(line, "  ", redshift, "  ", obs_line_wav, "  ", closest_obs_wav_idx)

            mask_indices.append(np.arange(closest_obs_wav_idx-3, closest_obs_wav_idx+4))

    # Convert to numpy array
    mask_indices = np.asarray(mask_indices)
    mask_indices = mask_indices.ravel()

    # Make sure the returned indices are unique and sorted
    mask_indices = np.unique(mask_indices)

    return mask_indices

def add_spec(lam_em, llam_em, lerr, old_llam, old_llamerr, num_points, num_galaxies, lam_grid, lam_step):
    
    for i in range(len(lam_grid)):
        
        # add fluxes
        new_ind = np.where((lam_em >= lam_grid[i] - lam_step/2) & (lam_em < lam_grid[i] + lam_step/2))[0]

        if new_ind.size:
            
            # Not requiring any points to be positive anymore
            # because the continuum is being subtracted.

            # Only count a galaxy in a particular bin if for that bin at least one point is positive
            #if np.any(llam_em[new_ind] > 0):
            #    num_galaxies[i] += 1
            
            # Looping over every point in a delta lambda bin
            # Don't include a datapoint if it is the only one in a stack bin
            # Reject points based on a signal-to-noise cut
            # Reject points that have negative signal
            if len(new_ind) > 1:
                for ind in new_ind:
                    sig = llam_em[ind]
                    noise = lerr[ind]

                    if np.isfinite(sig):
                        old_llam[i].append(sig)
                        old_llamerr[i].append(noise**2) # adding errors in quadrature
                        num_points[i] += 1 # keep track of how many points were added to each bin in lam_grid

            elif len(new_ind) == 1:
                sig = llam_em[int(new_ind)]
                noise = lerr[int(new_ind)]

                if np.isfinite(sig):
                    old_llam[i].append(sig)
                    old_llamerr[i].append(noise**2) # adding errors in quadrature
                    num_points[i] += 1 # keep track of how many points were added to each bin in lam_grid

            else:
                print("Not sure what happened.")
                print(new_ind)
                sys.exit(0)


            #if np.isfinite(sig):
            #    # Because I put in NaN values where there wasn't anything to add while combining PAs
            #
            #    if sig > 0: # only append those points where the signal is positive
            #        if sig/noise > 2.0:  # signal to noise cut
            #            old_llam[i].append(sig)
            #            old_llamerr[i].append(noise**2) # adding errors in quadrature
            #            num_points[i] += 1 # keep track of how many points were added to each bin in lam_grid
            #    else:
            #        print("Signal read in:", sig, end="\n")
            #        errmsg = "This error occurs when the code encounters a negative or zero signal." + "\n" + \
            #        "This error should not have been triggered if you're using PA combined PEARS spectra or . " + \
            #        "the FIGS spectra read in with the get_figs_data(...) function in grid_coadd.py. "
            #        "These flux points should've been taken out by the code that combines spectra " + \
            #        "for each galaxy at different PAs and the function that returns FIGS data." + "\n" + \
            #        "Check/Run the PA combining code for this galaxy."
            #        raise ValueError(errmsg)

        else:            
            continue

    return old_llam, old_llamerr, num_points, num_galaxies

def get_figs_data(figs_id, field):

    # First read in the correct spectrum file
    if field == 'GN1':
        figs_spec = fits.open(figs_dir + 'GN1_G102_2.combSPC.fits')
    elif field == 'GN2':
        figs_spec = fits.open(figs_dir + 'GN2_G102_2.combSPC.fits')
    elif field == 'GS1':
        figs_spec = fits.open(figs_dir + 'GS1_G102_2.combSPC.fits')
    elif field == 'GS2':
        figs_spec = fits.open(figs_dir + 'GS2_G102_2.combSPC.fits')

    # Now get the spectrum and return
    extname = 'BEAM_' + str(figs_id) + 'A'

    try:
        lam_obs = figs_spec[extname].data["LAMBDA"] # Wavelength 
        flam_obs = figs_spec[extname].data["AVG_WFLUX"]  # Flux (erg/s/cm^2/A)
        ferr_obs = figs_spec[extname].data["STD_WFLUX"]  # Flux error (erg/s/cm^2/A)
        #contam = figs_spec[extname].data["CONTAM"] # Flux contamination (erg/s/cm^2/A)

        # Now chop the spectrum to be in between 8800 to 11000
        lam_idx = np.where((lam_obs >= 9000) & (lam_obs <= 11000))[0]
        lam_obs = lam_obs[lam_idx]
        flam_obs = flam_obs[lam_idx]
        ferr_obs = ferr_obs[lam_idx]

        # Convert the negative and zero values to NaN
        invalid_idx = np.where(flam_obs <= 0.0)[0]
        flam_obs[invalid_idx] = np.nan
        ferr_obs[invalid_idx] = np.nan

        return_code = 1

        figs_spec.close()

        return lam_obs, flam_obs, ferr_obs, return_code

    except KeyError:
        print("FIGS spectrum for object", figs_id, field, "not found.")
        return_code = 0
        lam_obs = np.zeros(100)
        flam_obs = np.zeros(100)
        ferr_obs = np.zeros(100)

        figs_spec.close()

        return lam_obs, flam_obs, ferr_obs, return_code

def get_pears_data(pears_id, field):

    filename = pears_spectra_dir + field + '_' + str(pears_id) + '_PAcomb.fits'
    spec_hdu = fits.open(filename)

    # Get data
    lam_obs = spec_hdu[1].data
    flam_obs = spec_hdu[2].data[0]
    ferr_obs = spec_hdu[2].data[1]

    # Only proceed if there are at least 20 valid points in the PA combined spectrum
    if len(np.where(np.isfinite(flam_obs))[0]) >= 20:

        # Chop wavelength grid to 6000A -- 9500A
        arg_low = np.argmin(abs(lam_obs - 6000))
        arg_high = np.argmin(abs(lam_obs - 9500))

        lam_obs  = lam_obs[arg_low:arg_high+1]
        flam_obs = flam_obs[arg_low:arg_high+1]
        ferr_obs = ferr_obs[arg_low:arg_high+1]

        return_code = 1

    else:
        return_code = 0
        lam_obs = np.zeros(100)
        flam_obs = np.zeros(100)
        ferr_obs = np.zeros(100)

    return lam_obs, flam_obs, ferr_obs, return_code

def rescale(id_arr_cell, field_arr_cell, z_arr_cell, dl_tbl):
    """
    New description:
    This function will provide teh median of all median
    f_lambda values (see below) at the center of the 
    FIGS observed wavelengths. 

    Old description:
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
        #current_figs_id = id_arr_cell[k]
        #current_figs_field = field_arr_cell[k]

        current_pears_id = id_arr_cell[k]
        current_pears_field = field_arr_cell[k]

        # Get observed data and deredshift the spectrum
        # FIGS data
        #lam_obs, flam_obs, ferr_obs, return_code = get_figs_data(current_figs_id, current_figs_field)
        lam_obs, flam_obs, ferr_obs, return_code = get_pears_data(current_pears_id, current_pears_field)

        # If the return code was 0, then exit,
        # i.e., the observed spectrum is unuseable.
        # This condition should never be triggered. 
        if return_code == 0:
            print(current_id, current_field)
            print("Return code should not have been 0. Exiting.")
            sys.exit(0)

        # Now deredshift the observed data
        redshift = z_arr_cell[k]

        zidx = np.argmin(abs(z_arr - redshift))
        # Make sure that the z_arr here is the same array that was 
        # used to generate the dl lookup table.
        dl = dl_tbl['dl_cm'][zidx]  # has to be in cm

        lam_em = lam_obs / (1 + redshift)
        llam_em = flam_obs * (1 + redshift) * (4 * np.pi * dl * dl)

        # Code block from previous version        
        # Store median of values from 4400A-4600A for each spectrum
        """
        arg4400 = np.argmin(abs(lam_em - 4400))
        arg4600 = np.argmin(abs(lam_em - 4600))
        medarr[k] = np.median(llam_em[arg4400:arg4600+1])
        """

        # ----------------- New code ----------------- #
        # Take the average of a 200 A bandpass approx centered 
        # in the FIGS coverage. I'm just finding the center of 
        # the observed coverage and taking 5 points on each side 
        # of it.
        lam_cen = (lam_em[-1] + lam_em[0])/2.0
        lam_cen_idx = np.argmin(abs(lam_em - lam_cen))

        lam_begin_idx = lam_cen_idx - 5
        lam_end_idx = lam_cen_idx + 5

        medarr[k] = np.nanmedian(llam_em[lam_begin_idx:lam_end_idx+1])
    
    medval = np.nanmedian(medarr)
    
    # Return the median in array of median values
    return medarr, medval, np.std(medarr)

def take_median(old_llam, old_llamerr, lam_grid):

    # taking median
    for y in range(len(lam_grid)):

        if old_llam[y]:

            #old_llam[y] = np.median(old_llam[y])
            #num_in_bin = len(old_llamerr[y])
            #old_llamerr[y] = np.sqrt(np.sum(old_llamerr[y])) / num_in_bin

            # Actual stack value after 3 sigma clipping
            # Only allowing 3 iterations right now
            masked_data = sigma_clip(data=old_llam[y], sigma=3, maxiters=3)
            old_llam[y] = np.ma.median(masked_data)

            #print(masked_data)
            #print(old_llam[y])
            #print(old_llamerr[y])

            # Get mask from the masked_data array
            mask = np.ma.getmask(masked_data)

            # Apply mask to error array
            masked_dataerr = np.ma.array(old_llamerr[y], mask=mask)

            # take the square root of all errors that were added in quadrature
            # couldn't do this earlier because it is a list of lists which 
            # has to be fully constructed before I can do this
            num_after_mask = len(masked_dataerr) - len(np.where(mask)[0])
            old_llamerr[y] = np.ma.sqrt(np.ma.sum(masked_dataerr)) / num_after_mask

            # Error on each point of the stack
            # This only uses the points that passed the 3 sigma clipping before
            #old_llamerr[y] = \
            #np.sqrt((1.253 * np.std(masked_data) / \
            #    np.sqrt(len(masked_data)))**2 + np.sum(masked_dataerr) / len(masked_data))

        else:
            old_llam[y] = np.nan
            old_llamerr[y] = np.nan

        #print("%d %.2e %.2e %.2f" % (lam_grid[y], old_llam[y], old_llamerr[y], (old_llam[y]/old_llamerr[y])))

    return old_llam, old_llamerr

def create_stacks(cat, urcol, z_low, z_high, z_indices, start):

    print("Working on stacks for redshift range:", z_low, "<= z <", z_high)

    # Read in catalog of all fitting results and assign arrays
    # For now we need the id+field, spz, stellar mass, and u-r color
    pears_id = cat['PearsID'][z_indices]
    pears_field = cat['Field'][z_indices]
    zp = cat['zp_minchi2'][z_indices]

    #figs_id = cat['figs_id'][z_indices]
    #figs_field = cat['figs_field'][z_indices]

    ur_color = urcol[z_indices]
    stellar_mass = np.log10(cat['zp_ms'][z_indices])  # because the code below expects log(stellar mass)

    # Read in dl lookup table
    # Required for deredshifting
    dl_tbl = np.genfromtxt(massive_galaxies_dir + 'cluster_codes/dl_lookup_table.txt', dtype=None, names=True)
    # Define redshift array used in lookup table
    z_arr = np.arange(0.005, 6.005, 0.005)

    # ----------------------------------------- Code config params ----------------------------------------- #
    # Change only the parameters here to change how the code runs
    # Ideally you shouldn't have to change anything else.
    lam_step = 80  # somewhat arbitrarily chosen # pretty much trial and error

    # Set the ends of the lambda grid
    # This is dependent on the redshift range being considered
    lam_grid_low = 1800
    lam_grid_high = 7400

    lam_grid = np.arange(lam_grid_low, lam_grid_high, lam_step)
    # Lambda grid decided based on observed wavelength range i.e. 6000 to 9500
    # and the initially chosen redshift range 0.6 < z < 1.2
    # This redshift range was chosen so that the 4000A break would fall in the observed wavelength range
    
    # ----------------------------------------- Other preliminaries ----------------------------------------- #
    # Create HDUList for writing final fits file with stacks
    hdu = fits.PrimaryHDU()
    hdr = fits.Header()
    hdulist = fits.HDUList(hdu)
    hdulist.append(fits.ImageHDU(lam_grid, header=hdr))

    # ----------------------------------------- Begin creating stacks ----------------------------------------- #
    added_gal = 0
    skipped_gal = 0

    # Create empty array to store num of galaxies in each cell
    gal_per_cell = np.zeros((len(np.arange(col_low, col_high, col_step)), \
        len(np.arange(mstar_low, mstar_high, mstar_step))))
    
    for col in np.arange(col_low, col_high, col_step):
        for ms in np.arange(mstar_low, mstar_high, mstar_step):
            
            # Counter for galaxies in each cell
            # Reset at the start of stacking within each cell
            gal_current_cell = 0
            
            print("\n" + "Stacking in cell:")
            print("Color range:", col, col+col_step)
            print("Stellar mass range:", ms, ms+mstar_step)
            
            # Find the indices (corresponding to catalog entries)
            # that are within the current cell
            indices = np.where((ur_color >= col) & (ur_color < col + col_step) &\
                            (stellar_mass >= ms) & (stellar_mass < ms + mstar_step))[0]

            num_galaxies_cell = int(len(pears_id[indices]))
            print("Number of spectra to coadd in this grid cell --", num_galaxies_cell)

            redshift_arr_cell = zp[indices]
            print("Redshifts within this cell are --", redshift_arr_cell)

            if num_galaxies_cell == 0:
                continue
            
            # Define empty arrays and lists for saving stacks
            pears_old_llam = np.zeros(len(lam_grid))
            pears_old_llamerr = np.zeros(len(lam_grid))
            pears_old_llam = pears_old_llam.tolist()
            pears_old_llamerr = pears_old_llamerr.tolist()

            #figs_old_llam = np.zeros(len(lam_grid))
            #figs_old_llamerr = np.zeros(len(lam_grid))
            #figs_old_llam = figs_old_llam.tolist()
            #figs_old_llamerr = figs_old_llamerr.tolist()

            pears_num_points = np.zeros(len(lam_grid))
            pears_num_galaxies = np.zeros(len(lam_grid))

            #figs_num_points = np.zeros(len(lam_grid))
            #figs_num_galaxies = np.zeros(len(lam_grid))

            # rescale to 200A band centered on the observed wavelengths
            # This function returns the median of the median values (in the given band) from all given spectra
            # All spectra to be coadded in a given grid cell need to be divided by this value
            medarr, medval, stdval = rescale(pears_id[indices], pears_field[indices], zp[indices], dl_tbl)
            print("This cell has a maximum possible of", len(pears_id[indices]), "spectra.")
            print("The spectra in this cell have a median value of:", end=' ')
            print("{:.3e}".format(medval), " [erg s^-1 A^-1]")

            # Loop over all spectra in a grid cell and coadd them
            for u in range(len(pears_id[indices])):
                
                # This step should only be done on the first iteration within a grid cell
                # This converts every element (which are all 0 to begin with) 
                # in the flux and flux error arrays to an empty list
                # This is done so that function add_spec() can now append to every element
                if u == 0:
                    for x in range(len(lam_grid)):
                        pears_old_llam[x] = []
                        pears_old_llamerr[x] = []
            
                        #figs_old_llam[x] = []
                        #figs_old_llamerr[x] = []

                # Get redshift from catalog
                current_redshift = zp[indices][u]

                current_pears_id = pears_id[indices][u]
                current_pears_field = pears_field[indices][u]

                #current_figs_id = figs_id[indices][u]
                #current_figs_field = figs_field[indices][u]

                # ----------------------------- Get data ----------------------------- #
                # PEARS PA combined data
                grism_lam_obs, grism_flam_obs, grism_ferr_obs, return_code = \
                get_pears_data(current_pears_id, current_pears_field)
                # FIGS data # This is PA combined already, from Nor
                #g102_lam_obs, g102_flam_obs, g102_ferr_obs, return_code = \
                #get_figs_data(current_figs_id, current_figs_field)

                # Deredshift the observed data 
                zidx = np.argmin(abs(z_arr - current_redshift))
                # Make sure that the z_arr here is the same array that was 
                # used to generate the dl lookup table.
                dl = dl_tbl['dl_cm'][zidx]  # has to be in cm

                pears_lam_em = grism_lam_obs / (1 + current_redshift)
                pears_llam_em = grism_flam_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
                pears_lerr = grism_ferr_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)

                #figs_lam_em = g102_lam_obs / (1 + current_redshift)
                #figs_llam_em = g102_flam_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
                #figs_lerr = g102_ferr_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)

                # Match with photometry catalog and get photometry data
                
                # Divide by median value at 4400A to 4600A to rescale. 
                # Multiplying by median value of the flux medians to get it back to physical units
                pears_llam_em = (pears_llam_em / medarr[u]) * medval
                pears_lerr = (pears_lerr / medarr[u]) * medval

                #figs_llam_em = (figs_llam_em / medarr[u]) * medval
                #figs_lerr = (figs_lerr / medarr[u]) * medval

                # add the spectrum
                added_gal += 1
                gal_current_cell += 1
                pears_old_llam, pears_old_llamerr, pears_num_points, pears_num_galaxies = \
                add_spec(pears_lam_em, pears_llam_em, pears_lerr, pears_old_llam, pears_old_llamerr, \
                    pears_num_points, pears_num_galaxies, lam_grid, lam_step)

                #figs_old_llam, figs_old_llamerr, figs_num_points, figs_num_galaxies = \
                #add_spec(figs_lam_em, figs_llam_em, figs_lerr, figs_old_llam, figs_old_llamerr, \
                #    figs_num_points, figs_num_galaxies, lam_grid, lam_step)

            # Now take the median of all flux points appended within the list of lists
            # This function also does the 3-sigma clipping
            pears_old_llam, pears_old_llamerr = take_median(pears_old_llam, pears_old_llamerr, lam_grid)
            #figs_old_llam, figs_old_llamerr = take_median(figs_old_llam, figs_old_llamerr, lam_grid)

            # ---------------- Check stack by making a preliminary plot ---------------- #
            # ---------------- DO NOT DELETE CODE BLOCK! Useful for checking ---------------- #
            # Uncomment if not needed
            """
            for u in range(len(pears_id[indices])):

                fig = plt.figure()
                ax = fig.add_subplot(111)

                # Get redshift from catalog
                current_redshift = zp[indices][u]

                current_pears_id = pears_id[indices][u]
                current_pears_field = pears_field[indices][u]

                current_figs_id = figs_id[indices][u]
                current_figs_field = figs_field[indices][u]

                # ----------------------------- Get data ----------------------------- #
                grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code \
                = cf.get_data(current_pears_id, current_pears_field)

                # FIGS data
                g102_lam_obs, g102_flam_obs, g102_ferr_obs, return_code = get_figs_data(current_figs_id, current_figs_field)

                # Deredshift the observed data 
                zidx = np.argmin(abs(z_arr - current_redshift))
                # Make sure that the z_arr here is the same array that was 
                # used to generate the dl lookup table.
                dl = dl_tbl['dl_cm'][zidx]  # has to be in cm

                pears_lam_em = grism_lam_obs / (1 + current_redshift)
                pears_llam_em = grism_flam_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
                pears_lerr = grism_ferr_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)

                figs_lam_em = g102_lam_obs / (1 + current_redshift)
                figs_llam_em = g102_flam_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
                figs_lerr = g102_ferr_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)

                # Divide by median value at 4400A to 4600A to rescale. 
                # Multiplying by median value of the flux medians to get it back to physical units
                pears_llam_em = (pears_llam_em / medarr[u]) * medval
                pears_lerr = (pears_lerr / medarr[u]) * medval

                figs_llam_em = (figs_llam_em / medarr[u]) * medval
                figs_lerr = (figs_lerr / medarr[u]) * medval

                ax.plot(pears_lam_em, pears_llam_em, color='paleturquoise')
                ax.plot(figs_lam_em, figs_llam_em, color='bisque')

                print(current_pears_id, current_pears_field)
                print(current_figs_id, current_figs_field)

                # Fit with a second degree polynomial
                # to catch the U shaped spectra. Print chi2 
                # value on plot
                p_init = models.Polynomial1D(degree=2)
                fit_p = fitting.LinearLSQFitter()

                p_pears = fit_p(p_init, pears_lam_em, pears_llam_em)
                p_figs  = fit_p(p_init, figs_lam_em, figs_llam_em)

                # plot fit
                ax.plot(pears_lam_em, p_pears(pears_lam_em), color='teal')
                ax.plot(figs_lam_em,  p_figs(figs_lam_em), color='brown')

                # Compute a chi2
                pears_chi2 = compute_chi2(pears_lam_em, pears_llam_em, pears_lerr, p_pears)
                figs_chi2 = compute_chi2(figs_lam_em, figs_llam_em, figs_lerr, p_figs)

                ax.text(x=0.05, y=0.1, s=r"$\chi^2_{PEARS} = $" + "{:.2f}".format(pears_chi2), \
                    verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='k', size=12)
                ax.text(x=0.05, y=0.05, s=r"$\chi^2_{FIGS} = $" + "{:.2f}".format(figs_chi2), \
                    verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='k', size=12)

                plt.show()
                plt.clf()
                plt.cla()
                plt.close()
            """

            # Separate header for each extension
            exthdr = fits.Header()
            
            #exthdr["XTENSION"] = "IMAGE              / Image extension "
            #exthdr["BITPIX"]  = "                 -64 / array data type"
            #exthdr["NAXIS"]   = "                   2 / number of array dimensions"
            #exthdr["NAXIS1"]  = "                 161"
            #exthdr["NAXIS2"]  = "                   2"
            #exthdr["PCOUNT"]  = "                   0 / number of parameters"
            #exthdr["GCOUNT "] = "                   1 / number of groups"
            exthdr["CRANGE"]  = str(col) + " to " + str(col+col_step)
            exthdr["MSRANGE"] = str(ms) + " to " + str(ms+mstar_step)
            exthdr["NUMSPEC"] = str(int(gal_current_cell))
            exthdr["NORMVAL"] = str(medval)

            # Also add the average color and stellar mass for the cell to each 
            avgcol = np.mean(ur_color[indices])
            avgmass = np.mean(stellar_mass[indices])

            exthdr["AVGCOL"] = str(avgcol)
            exthdr["AVGMASS"] = str(avgmass)

            # Now reshape the data and append
            """
            pears_dat = np.array((pears_old_llam, pears_old_llamerr)).reshape(2, len(lam_grid))
            figs_dat = np.array((figs_old_llam, figs_old_llamerr)).reshape(2, len(lam_grid))
            all_dat = np.vstack((pears_dat, figs_dat))
            hdulist.append(fits.ImageHDU(data=all_dat, header=exthdr))
            """
            pears_dat = np.array((pears_old_llam, pears_old_llamerr)).reshape(2, len(lam_grid))
            hdulist.append(fits.ImageHDU(data=pears_dat, header=exthdr))

            # Update the array containing the cellwise distribution of galaxies
            row = int(col/col_step)
            column = int((ms - mstar_low)/mstar_step)
            gal_per_cell[row,column] = gal_current_cell

            print("Stacked", gal_current_cell, "spectra.")

    # Also write the galaxy distribution per cell
    galdist_hdr = fits.Header()
    galdist_hdr["NAME"] = "Galaxy distribution per cell"
    hdulist.append(fits.ImageHDU(data=np.flipud(gal_per_cell), header=galdist_hdr))

    # Write stacks to fits file
    final_fits_filename = 'stacks_' + str(z_low).replace('.','p') + '_' + str(z_high).replace('.','p') + '.fits'
    hdulist.writeto(stacking_analysis_dir + final_fits_filename, overwrite=True)

    # Time taken
    print("Time taken for stacking --", "{:.2f}".format(time.time() - start), "seconds")
    
    print("Total galaxies stacked in all stacks for this redshift range:", added_gal)
    print("Total galaxies skipped in all stacks for this redshift range:", skipped_gal)
    print("\n", "Cellwise distribution of galaxies:")
    print(np.flipud(gal_per_cell))
    print(int(np.sum(gal_per_cell, axis=None)))

    return None
    
def compute_chi2(x, y, err, model_fit):

    chi2 = 0
    for i in range(len(x)):
        if err[i] == 0.0:
            continue
        else:
            chi2 += (y[i] - model_fit(x[i]))**2 / err[i]**2

    return chi2

def get_avg_col_mass_arrays(ur_color, stellar_mass, stack_hdu):

    # Find the averages of all grid cells in a particular row/column
    # While these are useful numbers to have, they are currently only used in the plotting routine.
    cellcount = 0

    avgcolarr = np.zeros(numcol)
    avgmassarr = np.zeros(nummass)
        
    avgmassarr = avgmassarr.tolist()
    for k in range(len(avgmassarr)):
        avgmassarr[k] = []

    for i in np.arange(col_low, col_high, col_step):
        colcount = 0
        for j in np.arange(mstar_low, mstar_high, mstar_step):
            
            # Find the indices (corresponding to catalog entries)
            # that are within the current cell
            indices = np.where((ur_color >= i) & (ur_color < i + col_step) &\
                       (stellar_mass >= j) & (stellar_mass < j + mstar_step))[0]

            row = int((i - col_low)/col_step)
            column = int((j - mstar_low)/mstar_step)

            if indices.size:
                avgcol = stack_hdu[cellcount+2].header['AVGCOL']
                avgmass = stack_hdu[cellcount+2].header['AVGMASS']
                avgcolarr[row] += float(avgcol)
                avgmassarr[column].append(float(avgmass))

                cellcount += 1
                colcount += 1
            else:
                continue

        avgcolarr[row] /= (colcount)

    for x in range(len(avgmassarr)):
        avgmassarr[x] = np.sum(avgmassarr[x]) / len(avgmassarr[x])

    return avgcolarr, avgmassarr

def plot_ur_ms_diagram(ax, ur_color, stellar_mass, z_low, z_high, z_indices):

    # Because the axes object is already defined you can just start plotting
    # Labels first
    ax.set_xlabel(r'$\rm log(M_s)\ [M_\odot]$', fontsize=15)
    ax.set_ylabel(r'$(u - r)_\mathrm{rest}$', fontsize=15)

    # Plot the points
    ax.scatter(stellar_mass, ur_color, s=1.5, color='k', zorder=3)  # The arrays here already have the z_indices applied

    # Use the utility functions from the make_col_ms_plots.py code
    make_col_ms_plots.add_contours(stellar_mass, ur_color, ax)
    make_col_ms_plots.add_info_text_to_subplots(ax, z_low, z_high, len(z_indices))

    # Limits
    ax.set_xlim(mstar_low, mstar_high)
    ax.set_ylim(col_low, col_high)

    return None

def remove_axes_spines_ticks(ax):

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    return None

def plot_stacks(cat, urcol, z_low, z_high, z_indices, start):

    print("Working on plotting stacks for redshift range:", z_low, "<= z <", z_high)

    # Assign arrays
    pears_id = cat['PearsID'][z_indices]
    pears_field = cat['Field'][z_indices]
    zp = cat['zp_minchi2'][z_indices]

    #figs_id = cat['figs_id'][z_indices]
    #figs_field = cat['figs_field'][z_indices]

    ur_color = urcol[z_indices]
    stellar_mass = np.log10(cat['zp_ms'][z_indices])  # because the code below expects log(stellar mass)

    # Read in dl lookup table
    # Required for deredshifting
    dl_tbl = np.genfromtxt(massive_galaxies_dir + 'cluster_codes/dl_lookup_table.txt', dtype=None, names=True)
    # Define redshift array used in lookup table
    z_arr = np.arange(0.005, 6.005, 0.005)

    # Read in fits file for stacks
    final_fits_filename = 'stacks_' + str(z_low).replace('.','p') + '_' + str(z_high).replace('.','p') + '.fits'
    stack_hdu = fits.open(stacking_analysis_dir + final_fits_filename)

    # Get teh average color and mass arrays for plotting
    avgcolarr, avgmassarr = get_avg_col_mass_arrays(ur_color, stellar_mass, stack_hdu)
    avgcolarr_to_print = avgcolarr[::-1]

    # Also replot the corresponding u-r vs color plot with the same grid overlaid
    # Define figure and grid
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(numcol, 2 * nummass + 1)
    # For the grid definition:
    # This gives rows equal to the number of color cells.
    # It gives columns equal to two times (because there are two plots side-by-side)
    # the number of stellar mass cells. The +1 is to have a gap between the two plots.
    gs.update(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.00, hspace=0.00)

    # Define axes
    # Firstly, the axes object for plotting the u-r vs M* diagram
    ax1 = fig.add_subplot(gs[:,:4])
    plot_ur_ms_diagram(ax1, ur_color, stellar_mass, z_low, z_high, z_indices)

    # Now all the other axes objects with each one corresponding 
    # to a grid cell will be defined dynamically.
    # Begin looping through cells to plot
    cellcount = 0

    # Get lambda grid
    lam = stack_hdu[1].data

    for i in np.arange(col_low, col_high, col_step):
        for j in np.arange(mstar_low, mstar_high, mstar_step):

            # Create subplots for different grid positions
            # First get teh row and column
            row = numcol - 1 - int((i - col_low)/col_step)
            # The numcol - 1 - ... is to make sure that the plotting is being done 
            # from the bottom left which is what the for loop is doing.
            column = int((j - mstar_low)/mstar_step) + nummass + 1
            # The columns have to be offset by nummass + 1 because 
            # of the plot to its left. 
            #print "\n", "i and j:", i, j 
            #print "Color and M* (bottom left of cell):", i, j
            #print "Row and column:", row, column
            #print "GridSpec row range:", row, "to", row+1
            #print "GridSpec column range:", column, "to", column+1
            ax = fig.add_subplot(gs[row:row+1, column:column+1])

            # Add the label for the avgmass and avgcolor
            # This appears before gettting indices so that these labels arent skipped
            if row == 0:
                ax.text(0.25, 1.2, "{:.2f}".format(float(avgmassarr[column - nummass - 1])), \
                verticalalignment='top', horizontalalignment='left', \
                transform=ax.transAxes, color='k', size=14)

            if column == (2 * nummass):
                ax.text(0.98, 0.65, "{:.2f}".format(float(avgcolarr_to_print[row])), \
                verticalalignment='top', horizontalalignment='left', \
                transform=ax.transAxes, color='k', size=14, rotation=270)

            if row == 0 and column == nummass + 1:
                masslabel = r'$\left<\mathrm{log}\left(\frac{M_*}{M_\odot}\right)\right>$'
                ax.text(-0.03, 0.75, masslabel, \
                verticalalignment='top', horizontalalignment='left', \
                transform=ax.transAxes, color='k', size=13, zorder=10)

            if row == numcol - 2 and column == 2*nummass:
                colorlabel = r'$\left<(u-r)_\mathrm{rest}\right>$'
                ax.text(0.5, 1.0, colorlabel, \
                verticalalignment='top', horizontalalignment='left', \
                transform=ax.transAxes, color='k', size=13, rotation=270, zorder=10)

            # Find the indices (corresponding to catalog entries)
            # that are within the current cell
            indices = np.where((ur_color >= i) & (ur_color < i + col_step) &\
                            (stellar_mass >= j) & (stellar_mass < j + mstar_step))[0]

            # Check that the cell isn't empty and then proceed
            if indices.size:
                print("Number of spectra in this grid cell --", len(indices))
                #medarr, medval, stdval = rescale(figs_id[indices], figs_field[indices], zp[indices], dl_tbl)
                medarr, medval, stdval = rescale(pears_id[indices], pears_field[indices], zp[indices], dl_tbl)
            else:
                # Delete axes spines and labels if skipping
                remove_axes_spines_ticks(ax)
                continue

            # Do not plot any cells with less than 5 spectra
            if (len(indices) < 5):
                print("At u-r color and M* (bottom left of cell):", i, j)
                print("Too few spectra in stack (i.e., less than 5). Continuing to the next grid cell...")
                cellcount += 1
                # The cellcount has to be increased here (and not in the case 
                # of the continue statement right after this) is because
                # this stack was actually created but we're just not plotting it.
                # Delete axes spines and labels if skipping
                remove_axes_spines_ticks(ax)
                continue

            # Add labels
            if (row == 5) and (column == nummass + 1):
                ax.set_xlabel('$\lambda\ [\mu m]$', fontsize=13)
            if (row == 3) and (column == nummass + 1):
                ax.set_ylabel('$L_{\lambda}\ [\mathrm{erg\, s^{-1}\, \AA^{-1}}]$', fontsize=13)

            # Loop over all spectra in a grid cell and plot them in a light grey color
            for u in range(len(pears_id[indices])):

                # Get redshift from catalog
                current_redshift = zp[indices][u]

                current_pears_id = pears_id[indices][u]
                current_pears_field = pears_field[indices][u]

                #current_figs_id = figs_id[indices][u]
                #current_figs_field = figs_field[indices][u]

                # ----------------------------- Get data ----------------------------- #
                # PEARS PA combined data
                grism_lam_obs, grism_flam_obs, grism_ferr_obs, return_code = \
                get_pears_data(current_pears_id, current_pears_field)
                # FIGS data
                #g102_lam_obs, g102_flam_obs, g102_ferr_obs, return_code = \
                #get_figs_data(current_figs_id, current_figs_field)

                # Deredshift the observed data 
                zidx = np.argmin(abs(z_arr - current_redshift))
                # Make sure that the z_arr here is the same array that was 
                # used to generate the dl lookup table.
                dl = dl_tbl['dl_cm'][zidx]  # has to be in cm

                pears_lam_em = grism_lam_obs / (1 + current_redshift)
                pears_llam_em = grism_flam_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
                pears_lerr = grism_ferr_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
                
                #figs_lam_em = g102_lam_obs / (1 + current_redshift)
                #figs_llam_em = g102_flam_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
                #figs_lerr = g102_ferr_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)

                # Divide by median value at 4400A to 4600A to rescale. 
                # Multiplying by median value of the flux medians to get it back to physical units
                pears_llam_em = (pears_llam_em / medarr[u]) * medval
                pears_lerr = (pears_lerr / medarr[u]) * medval

                #figs_llam_em = (figs_llam_em / medarr[u]) * medval
                #figs_lerr = (figs_lerr / medarr[u]) * medval

                # Plotting
                ax.plot(pears_lam_em, pears_llam_em, ls='-', color='paleturquoise', linewidth=0.5)
                #ax.plot(figs_lam_em, figs_llam_em, ls='-', color='bisque', linewidth=0.5)
                ax.get_yaxis().set_ticklabels([])
                #ax.get_xaxis().set_ticklabels([])
                ax.get_xaxis().set_ticklabels(['','0.4','0.6'])
                ax.set_xlim(3500, 6700)

                ax.minorticks_on()

            # Plot stack
            pears_llam = stack_hdu[cellcount+2].data[0]
            pears_llam_err = stack_hdu[cellcount+2].data[1]
            #figs_llam = stack_hdu[cellcount+2].data[2]
            #figs_llam_err = stack_hdu[cellcount+2].data[3]

            # Force zeros to NaNs so that they're not plotted
            pears_llam_zero_idx = np.where(pears_llam == 0.0)[0]
            pears_llam[pears_llam_zero_idx] = np.nan
            ax.errorbar(lam, pears_llam, yerr=pears_llam_err, fmt='.-', color='mediumblue', linewidth=0.5,\
                        elinewidth=0.2, ecolor='r', markeredgecolor='mediumblue', capsize=0, markersize=0.5, zorder=5)

            #figs_llam_zero_idx = np.where(figs_llam == 0.0)[0]
            #figs_llam[figs_llam_zero_idx] = np.nan
            #ax.errorbar(lam, figs_llam, yerr=figs_llam_err, fmt='.-', color='darkorange', linewidth=0.5,\
            #            elinewidth=0.2, ecolor='r', markeredgecolor='darkorange', capsize=0, markersize=0.5, zorder=5)

            # Y Limits 
            # Find min and max within the stack and add some padding
            # Using the nan functions here because some stacks have nan values
            #stack_min = np.min([np.nanmin(pears_llam), np.nanmin(figs_llam)])
            #stack_max = np.max([np.nanmax(pears_llam), np.nanmax(figs_llam)])
            #stack_mederr = np.median([np.nanmedian(pears_llam_err), np.nanmedian(figs_llam_err)])

            stack_min = np.nanmin(pears_llam)
            stack_max = np.nanmax(pears_llam)
            stack_mederr = np.nanmedian(pears_llam_err)
            # median of all errors on hte stack
            ax.set_ylim(stack_min - 3 * stack_mederr, stack_max + 3 * stack_mederr)

            # Add other info to plot
            numspec = int(stack_hdu[cellcount+2].header['NUMSPEC'])
            normval = float(stack_hdu[cellcount+2].header['NORMVAL'])
            normval = convert_to_sci_not(normval)  # Returns a properly formatted string

            # add number of galaxies in plot
            ax.text(0.8, 0.2, numspec, verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=9)

            # Normalization value
            ax.text(0.02, 0.2, normval, verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=9)

            cellcount += 1

    fig.savefig(stacking_figures_dir + final_fits_filename.replace('.fits','.pdf'), dpi=300, bbox_inches='tight')

    # Close fits file and return
    stack_hdu.close()

    return None

def stack_plot_massive(cat, urcol, z_low, z_high, z_indices, start):

    print("Working on massive galaxy stacks for redshift range:", z_low, "<= z <", z_high)

    # Assign arrays
    pears_id = cat['PearsID'][z_indices]
    pears_field = cat['Field'][z_indices]
    zp = cat['zp_minchi2'][z_indices]
    zs = cat['zspec'][z_indices]

    #figs_id = cat['figs_id'][z_indices]
    #figs_field = cat['figs_field'][z_indices]

    ur_color = urcol[z_indices]
    stellar_mass = np.log10(cat['zp_ms'][z_indices])  # because the code below expects log(stellar mass)

    # Read in dl lookup table
    # Required for deredshifting
    dl_tbl = np.genfromtxt(massive_galaxies_dir + 'cluster_codes/dl_lookup_table.txt', dtype=None, names=True)
    # Define redshift array used in lookup table
    z_arr = np.arange(0.005, 6.005, 0.005)

    # ----------------------------------------- Code config params ----------------------------------------- #
    # Change only the parameters here to change how the code runs
    # Ideally you shouldn't have to change anything else.
    lam_step = 25  # somewhat arbitrarily chosen # pretty much trial and error

    # Set the ends of the lambda grid
    # This is dependent on the redshift range being considered
    lam_grid_low = 3000
    lam_grid_high = 7600

    lam_grid = np.arange(lam_grid_low, lam_grid_high, lam_step)
    # Lambda grid decided based on observed wavelength range i.e. 6000 to 9500
    # and the initially chosen redshift range 0.16 < z < 0.96

    ms_lim_low = 10.5
    ms_lim_high = 12.0

    # Find the indices (corresponding to massive galaxies)
    indices = np.where((ur_color >= 1.8) & (ur_color < 3.0) &\
                    (stellar_mass >= ms_lim_low) & (stellar_mass < ms_lim_high))[0]
  
    num_massive = int(len(pears_id[indices]))
    print("Number of massive galaxies in this redshift range --", num_massive)

    # rescale to 200A band centered on the observed wavelengths
    # This function returns the median of the median values (in the given band) from all given spectra
    # All spectra to be coadded in a given grid cell need to be divided by this value
    #medarr, medval, stdval = rescale(figs_id[indices], figs_field[indices], zp[indices], dl_tbl)
    #print("The spectra in this cell have a median value of:", end=' ')
    #print("{:.3e}".format(medval), " [erg s^-1 A^-1]")

    # Define empty arrays and lists for saving stacks
    pears_old_llam = np.zeros(len(lam_grid))
    pears_old_llamerr = np.zeros(len(lam_grid))
    pears_old_llam = pears_old_llam.tolist()
    pears_old_llamerr = pears_old_llamerr.tolist()

    pears_num_points = np.zeros(len(lam_grid))
    pears_num_galaxies = np.zeros(len(lam_grid))

    #figs_old_llam = np.zeros(len(lam_grid))
    #figs_old_llamerr = np.zeros(len(lam_grid))
    #figs_old_llam = figs_old_llam.tolist()
    #figs_old_llamerr = figs_old_llamerr.tolist()

    #figs_num_points = np.zeros(len(lam_grid))
    #figs_num_galaxies = np.zeros(len(lam_grid))

    # Rejected galaxies list
    galaxies_to_reject = []# [(95761, 'GOODS-N'), (70857, 'GOODS-N'), (114677, 'GOODS-S'), \
    #(63307, 'GOODS-S'), (77558, 'GOODS-S'), (123736, 'GOODS-N'), (43381, 'GOODS-S'), \
    #(111206, 'GOODS-S'), (123255, 'GOODS-N'), (70407, 'GOODS-S'), (90998, 'GOODS-S'), \
    #(24439, 'GOODS-N'), (119621, 'GOODS-N'), (89237, 'GOODS-N'), (18862, 'GOODS-S'), \
    #(106130, 'GOODS-S'), (16496, 'GOODS-S')]

    to_reject_but_might_be_okay_with_masking = [(94867, 'GOODS-N')]

    fails_with_MKL_err = [(89031, 'GOODS-S'), (44756, 'GOODS-N'), (86216, 'GOODS-N')]

    #to_check = [(109151, 'GOODS-S'), (56663, 'GOODS-N'), (21592, 'GOODS-S'), (109794, 'GOODS-S'), (47814, 'GOODS-N')]
    # these are galaxies with strong absorption features
    # if the code works well on these then it should work 
    # well on the rest.

    # Create figure
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    # Loop over all spectra and coadd them
    for u in range(len(pears_id[indices])):
        
        # This step should only be done on the first iteration within a grid cell
        # This converts every element (which are all 0 to begin with) 
        # in the flux and flux error arrays to an empty list
        # This is done so that function add_spec() can now append to every element
        if u == 0:
            for x in range(len(lam_grid)):
                pears_old_llam[x] = []
                pears_old_llamerr[x] = []
            
                #figs_old_llam[x] = []
                #figs_old_llamerr[x] = []

        # Get redshift from catalog
        current_redshift = zp[indices][u]

        current_pears_id = pears_id[indices][u]
        current_pears_field = pears_field[indices][u]

        #if (current_pears_id, current_pears_field) not in to_check:
        #    continue

        print("PEARS object:", current_pears_id, current_pears_field)

        # Apply cut on redshift error
        redshift_err_tol = 0.05
        current_spec_redshift = zs[indices][u]
        zerr = abs(current_redshift - current_spec_redshift) / (1 + current_spec_redshift)
        print("Spec-z, photo-z, and redshift error for this galaxy:", current_spec_redshift, current_redshift, zerr)
        # This will let the ones with spec-z = -99.0, 
        # i.e., unknown spec-z, through, which is what we want.
        if zerr >= redshift_err_tol:
            print("Skipping due to large redshift error.", "\n")
            num_massive -= 1
            continue

        # Reject galaxy if it is in one of the rejection lists
        if (current_pears_id, current_pears_field) in galaxies_to_reject or \
        (current_pears_id, current_pears_field) in to_reject_but_might_be_okay_with_masking or \
        (current_pears_id, current_pears_field) in fails_with_MKL_err:
            print("Skipping:", current_pears_id, current_pears_field)
            num_massive -= 1
            continue

        #current_figs_id = figs_id[indices][u]
        #current_figs_field = figs_field[indices][u]

        # Subtract continuum by fitting a third degree polynomial
        # Continuum fitted with potential emission line areas masked
        """
        p_init = models.Polynomial1D(degree=3)
        fit_p = fitting.LinearLSQFitter()

        # mask emission lines 
        pears_llam_em_masked, pears_mask_ind = mask_em_lines(pears_lam_em, pears_llam_em)
        #figs_llam_em_masked, figs_mask_ind = mask_em_lines(figs_lam_em, figs_llam_em)

        p_pears = fit_p(p_init, pears_lam_em, pears_llam_em_masked)
        #p_figs  = fit_p(p_init, figs_lam_em, figs_llam_em_masked)

        # splinefit = interpolate.UnivariateSpline(pears_lam_em, pears_llam_em_masked)
        pfit = np.ma.polyfit(pears_lam_em, pears_llam_em_masked, deg=3)
        np_polynomial = np.poly1d(pfit)
        #splinefit = CubicSpline(pears_lam_em, pears_llam_em_masked)

        # Rebin data and refit the same low degree polynomial
        rebin_step = 250.0
        rebin_start = int(pears_lam_em[0] / rebin_step) * rebin_step
        rebin_end = int(pears_lam_em[-1] / rebin_step) * rebin_step
        rebin_grid = np.arange(rebin_start, rebin_end + rebin_step, rebin_step)
        pears_llam_em_rebinned = interpolate.griddata(points=pears_lam_em, values=pears_llam_em, \
            xi=rebin_grid, method='cubic')

        # find NaNs and replace them with the nearest value in rebinned data
        nan_idx = np.where(np.isnan(pears_llam_em_rebinned))[0]

        if nan_idx.size:
            for i in nan_idx:
                if i == (len(pears_llam_em_rebinned) -1):
                    pears_llam_em_rebinned[i] = pears_llam_em_rebinned[i-1]
                else:
                    pears_llam_em_rebinned[i] = pears_llam_em_rebinned[i+1]

        rebin_fit = np.ma.polyfit(rebin_grid, pears_llam_em_rebinned, deg=3)
        rebin_polynomial = np.poly1d(rebin_fit)
        """

        # plot data and fit
        # ----------------------------- Get data ----------------------------- #
        # PEARS PA combined data
        grism_lam_obs, grism_flam_obs, grism_ferr_obs, return_code = get_pears_data(current_pears_id, current_pears_field)
        # FIGS data
        #g102_lam_obs, g102_flam_obs, g102_ferr_obs, return_code = get_figs_data(current_figs_id, current_figs_field)

        # Normalize flux levels to approx 1.0
        grism_flam_norm = grism_flam_obs / np.mean(grism_flam_obs)
        grism_ferr_norm = grism_ferr_obs / np.mean(grism_flam_obs)

        # Mask lines
        mask_indices = get_mask_indices(grism_lam_obs, current_redshift)

        # Make sure masking indices are consistent with array to be masked
        remove_mask_idx = np.where(mask_indices >= len(grism_lam_obs))[0]
        mask_indices = np.delete(arr=mask_indices, obj=remove_mask_idx)

        weights = np.ones(len(grism_lam_obs))
        weights[mask_indices] = 0

        #print("\n Masking the following indices:", mask_indices)

        # SciPy smoothing spline fit
        spl = splrep(x=grism_lam_obs, y=grism_flam_norm, k=3, s=5.0)
        wav_plt = np.arange(grism_lam_obs[0], grism_lam_obs[-1], 1.0)
        spl_eval = splev(wav_plt, spl)

        # Divide the given flux by the smooth spline fit
        cont_div_flux = grism_flam_norm / splev(grism_lam_obs, spl)
        cont_div_err  = grism_ferr_norm / splev(grism_lam_obs, spl)

        # Plotting
        fig1 = plt.figure(figsize=(9,5))
        gs = gridspec.GridSpec(6,2)
        gs.update(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.00, hspace=0.7)

        ax1 = fig1.add_subplot(gs[:4,:])
        ax2 = fig1.add_subplot(gs[4:,:])

        ax1.plot(grism_lam_obs, grism_flam_norm, 'o-', markersize=3.0, color='k', linewidth=1.5, label='PEARS obs data')
        ax1.fill_between(grism_lam_obs, grism_flam_norm - grism_ferr_norm, grism_flam_norm + grism_ferr_norm, \
        color='gray', alpha=0.5, zorder=5)
        ax1.plot(wav_plt, spl_eval, color='crimson', lw=3.0, label='SciPy smooth spline fit')

        #ax1.plot(figs_lam_em, figs_llam_em, color='gold', linewidth=1.5, label='FIGS obs data')

        #ax1.plot(pears_lam_em, p_pears(pears_lam_em), color='teal', zorder=1.0, label='AstroPy polynomial fit')
        #ax1.plot(figs_lam_em,  p_figs(figs_lam_em), color='brown', label='FIGS obs data')
        #ax1.plot(pears_lam_em, np_polynomial(pears_lam_em), color='crimson', zorder=1.0, label='NumPy polynomial fit')
        #ax1.plot(pears_lam_em, rebin_polynomial(pears_lam_em), lw=3.0, zorder=2.0, color='tab:brown', label='NumPy rebin polynomial fit')

        # Show mask as shaded region
        """
        all_lines, all_line_labels = get_all_line_wav()
        all_lines = np.asarray(all_lines)
        all_lines = all_lines * (1 + current_redshift)
        for l in range(len(all_lines)):

            current_line_wav = all_lines[l]
            current_line_idx = np.argmin(abs(grism_lam_obs - current_line_wav))

            lam_em_formaskplot = grism_lam_obs
            # if current_line_wav < pears_lam_em[-1]:
            #     current_line_idx = np.argmin(abs(pears_lam_em - current_line_wav))
            #     lam_em_formaskplot = pears_lam_em
            # elif current_line_wav > figs_lam_em[0]:
            #     current_line_idx = np.argmin(abs(figs_lam_em - current_line_wav))
            #     lam_em_formaskplot = figs_lam_em

            ls_idx = current_line_idx-3
            le_idx = current_line_idx+3
            if ls_idx < 0:
                ls_idx = 0
            if le_idx >= len(lam_em_formaskplot):
                le_idx = len(lam_em_formaskplot) - 1 
            line_start = lam_em_formaskplot[ls_idx]
            line_end = lam_em_formaskplot[le_idx]

            ax1.axvspan(line_start, line_end, alpha=0.25, color='gray')
            #xlab_line = (lam_em_formaskplot[current_line_idx] - lam_grid_low) / (lam_grid_high - lam_grid_low)
            #ylab_line = 0.2 + np.power(-1,l)*0.1
            #ax1.text(x=xlab_line, y=ylab_line, s=all_line_labels[l], color='k', size=9, \
            #verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes)
        """

        # Compute a chi2
        # use the masked array and the fit because the fitting was done on the masked array
        #pears_chi2 = compute_chi2(grism_lam_obs, grism_flam_norm, grism_ferr_norm, p_pears)
        #figs_chi2 = compute_chi2(figs_lam_em, figs_llam_em, figs_lerr, p_figs)

        # Add galaxy info to plot
        ax1.text(x=0.05, y=0.7, s=str(current_pears_id) + "  " + current_pears_field, \
            verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='k', size=12)
        #ax1.text(x=0.05, y=0.9, s=r"$\chi^2_{PEARS} = $" + "{:.2e}".format(pears_chi2), \
        #    verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='k', size=12)
        #ax1.text(x=0.05, y=0.8, s=r"$\chi^2_{FIGS} = $" + "{:.2e}".format(figs_chi2), \
        #    verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='k', size=12)

        # Now divide continuum
        # Using astropy fits
        #pears_llam_em = pears_llam_em / p_pears(pears_lam_em)
        #figs_llam_em = figs_llam_em / p_figs(figs_lam_em)

        # Also divide errors 
        #pears_lerr = pears_lerr / p_pears(pears_lam_em)
        #figs_lerr = figs_lerr / p_figs(figs_lam_em)
        
        # Using Numpy fits
        #pears_llam_em = pears_llam_em / np_polynomial(pears_lam_em)
        #pears_lerr = pears_lerr / np_polynomial(pears_lam_em)

        # Using fits on rebinned data
        #pears_llam_em = pears_llam_em / rebin_polynomial(pears_lam_em)
        #pears_lerr = pears_lerr / rebin_polynomial(pears_lam_em)

        # Plot "pure emission/absorption" spectrum
        ax2.plot(grism_lam_obs, cont_div_flux, color='teal', lw=2.0, label='Continuum divided flux')
        ax2.axhline(y=1.0, ls='--', color='k', lw=1.8)

        # Limits 
        #ax1.set_xlim(lam_grid_low, lam_grid_high)
        #ax2.set_xlim(lam_grid_low, lam_grid_high)

        ax1.minorticks_on()
        ax2.minorticks_on()

        ax1.legend(loc=0)

        #plt.show()
        plt.cla()
        plt.clf()
        plt.close()

        # Deredshift the observed data 
        zidx = np.argmin(abs(z_arr - current_redshift))
        # Make sure that the z_arr here is the same array that was 
        # used to generate the dl lookup table.
        dl = dl_tbl['dl_cm'][zidx]  # has to be in cm

        pears_lam_em = grism_lam_obs / (1 + current_redshift)
        #pears_llam_em = grism_flam_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
        #pears_lerr = grism_ferr_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)

        #figs_lam_em = g102_lam_obs / (1 + current_redshift)
        #figs_llam_em = g102_flam_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
        #figs_lerr = g102_ferr_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)

        # Shift it to force stack value ~1.0 at ~4600A
        shift_idx = np.where((pears_lam_em >= 4600) & (pears_lam_em <= 4700))[0]
        scaling_fac = np.mean(cont_div_flux[shift_idx])
        cont_div_flux /= scaling_fac

        # add the continuum subtracted spectrum
        pears_old_llam, pears_old_llamerr, pears_num_points, pears_num_galaxies = \
        add_spec(pears_lam_em, cont_div_flux, cont_div_err, pears_old_llam, pears_old_llamerr, \
            pears_num_points, pears_num_galaxies, lam_grid, lam_step)

        #figs_old_llam, figs_old_llamerr, figs_num_points, figs_num_galaxies = \
        #add_spec(figs_lam_em, figs_llam_em, figs_lerr, figs_old_llam, figs_old_llamerr, \
        #    figs_num_points, figs_num_galaxies, lam_grid, lam_step)

        ax.plot(pears_lam_em, cont_div_flux, ls='-', color='turquoise', linewidth=0.5, alpha=0.4)
        #ax.plot(figs_lam_em, figs_llam_em, ls='-', color='bisque', linewidth=1.0)

    # Now take the median of all flux points appended within the list of lists
    # This function also does the 3-sigma clipping
    pears_old_llam, pears_old_llamerr = take_median(pears_old_llam, pears_old_llamerr, lam_grid)
    #figs_old_llam, figs_old_llamerr = take_median(figs_old_llam, figs_old_llamerr, lam_grid)

    pears_old_llam = np.asarray(pears_old_llam)
    pears_old_llamerr = np.asarray(pears_old_llamerr)
    #figs_old_llam = np.asarray(figs_old_llam)
    #figs_old_llamerr = np.asarray(figs_old_llamerr)

    # Plot stacks
    ax.plot(lam_grid, pears_old_llam, '.-', color='mediumblue', linewidth=1.5, \
        markeredgecolor='mediumblue', markersize=1.0, zorder=5)
    ax.fill_between(lam_grid, pears_old_llam - pears_old_llamerr, pears_old_llam + pears_old_llamerr, \
        color='gray', alpha=0.5, zorder=5)

    # Save stack as plain text file
    fh = open(stacking_analysis_dir + 'massive_stack_pears_' + str(z_low) + 'z' + str(z_high) + '.txt', 'w')
    fh.write("# lam flam flam_err")
    fh.write("\n")
    for q in range(len(lam_grid)):
        # These flux and error values should be representable by "{:.6f}"
        # because they're all around 1.0. I don't think the numbers need the sci notation.
        fh.write("{:.2f}".format(lam_grid[q]) + " " + "{:.6f}".format(pears_old_llam[q]) + " " + "{:.6f}".format(pears_old_llamerr[q]))
        fh.write("\n")
    fh.close()

    #figs_llam_zero_idx = np.where(figs_old_llam == 0.0)[0]
    #figs_old_llam[figs_llam_zero_idx] = np.nan
    #figs_old_llamerr[figs_llam_zero_idx] = np.nan
    #ax.errorbar(lam_grid, figs_old_llam, yerr=figs_old_llamerr, fmt='.-', color='darkorange', linewidth=2.5,\
    #            elinewidth=1.0, ecolor='r', markeredgecolor='darkorange', capsize=0, markersize=4.0, zorder=5)

    ax.set_xlim(lam_grid_low, lam_grid_high)
    ax.set_ylim(0.9, 1.1)  # if dividing by the continuum instead of subtracting
    ax.axhline(y=1.0, ls='--', color='k')
    ax.minorticks_on()

    add_line_labels(ax)

    # Number of galaxies and redshift range on plot
    ax.text(0.66, 0.97, r'$\mathrm{N\,=\,}$' + str(num_massive), verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=16)
    ax.text(0.66, 0.92, str(z_low) + r'$\,\leq z \leq\,$' + str(z_high), \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=16)

    # Mass range
    ax.text(0.66, 0.86, str(ms_lim_low) + r'$\,\leq \mathrm{M\ [M_\odot]} <\,$' + str(ms_lim_high), \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=16)

    # Labels
    ax.set_xlabel(r'$\lambda\ [\mathrm{\AA}]$', fontsize=15)
    ax.set_ylabel(r'$L_{\lambda}\ [\mathrm{divided\ by\ continuum}]$', fontsize=15)

    #ax.text(0.67, 0.26, 'PEARS ACS/G800L', verticalalignment='top', horizontalalignment='left', \
    #        transform=ax.transAxes, color='royalblue', size=20)
    #ax.text(0.67, 0.195, 'FIGS WFC3/G102', verticalalignment='top', horizontalalignment='left', \
    #        transform=ax.transAxes, color='darkorange', size=20)

    # Measure Mg/Fe
    #mg2fe = fit_gauss_mgfe_astropy(lam_grid, pears_old_llam, num_massive, z_low, z_high, ms_lim_low, ms_lim_high)

    figname = stacking_figures_dir + 'massive_stack_' + str(z_low).replace('.','p') \
    + '_' + str(z_high).replace('.','p') + '.pdf'
    fig.savefig(figname, dpi=300, bbox_inches='tight')

    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    return None

def add_line_labels(ax, label_flag='all'):
    """
    Mark some important features on the plot.
    """

    if label_flag == 'all':
        # [OII]
        ax.axvline(x=3727.0, ls='--', ymin=0.53, ymax=0.6, color='firebrick')
        ax.axvline(x=3729.0, ls='--', ymin=0.53, ymax=0.6, color='firebrick')
        ax.text(3695.0, 1.03, r'$\mathrm{[OII]}\lambda\lambda 3727,3729$', \
            verticalalignment='top', horizontalalignment='left', \
            transform=ax.transData, color='firebrick', size=12)

        # Ca H & K
        ax.text(3920.0, 0.935, r'$\mathrm{Ca}$' + '\n' + r'$\mathrm{H\ \&\ K}$', \
            verticalalignment='center', horizontalalignment='center', \
            transform=ax.transData, color='firebrick', size=12)

        # TiO
        ax.text(6230.0, 0.975, r'$\mathrm{TiO}$', \
            verticalalignment='center', horizontalalignment='center', \
            transform=ax.transData, color='firebrick', size=12)

    # These remaining features below will always be marked
    # Mgb
    ax.axvline(x=5175.0, ls='--', ymin=0.18, ymax=0.27, color='firebrick')
    ax.text(5165.0, 0.925, r'$\mathrm{Mg_2 + Mgb}$', \
        verticalalignment='top', horizontalalignment='right', \
        transform=ax.transData, color='firebrick', size=12)
    # FeII
    ax.axvline(x=5270.0, ls='--', ymin=0.2, ymax=0.29, color='firebrick')
    ax.axvline(x=5335.0, ls='--', ymin=0.2, ymax=0.29, color='firebrick')
    ax.axvline(x=5406.0, ls='--', ymin=0.2, ymax=0.29, color='firebrick')
    fe_str = r'$\mathrm{Fe}\lambda 5270$' + '+ \n' + r'$\mathrm{Fe}\lambda 5335$' + '+' + r'$\mathrm{Fe}\lambda 5406$'
    ax.text(5270, 0.935, fe_str, \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transData, color='firebrick', size=12)

    # Hbeta
    ax.axvline(x=4861.0, ls='--', ymin=0.3, ymax=0.38, color='firebrick')
    ax.text(4861.0, 0.955, r'$\mathrm{H}\beta$', \
        verticalalignment='top', horizontalalignment='right', \
        transform=ax.transData, color='firebrick', size=12)

    # [OIII]
    ax.axvline(x=4959.0, ls='--', ymin=0.53, ymax=0.6, color='firebrick')
    ax.axvline(x=5007.0, ls='--', ymin=0.53, ymax=0.6, color='firebrick')
    ax.text(4980.0, 1.03, r'$\mathrm{[OIII]}\lambda\lambda 4959,5007$', \
        verticalalignment='top', horizontalalignment='center', \
        transform=ax.transData, color='firebrick', size=12)

    # Gband
    ax.text(4300.0, 0.965, 'G-band', \
        verticalalignment='center', horizontalalignment='center', \
        transform=ax.transData, color='firebrick', size=12)

    # NaD + TiO
    ax.text(5890.0, 0.97, r'$\mathrm{NaD}$' + '+ \n' + r'$\mathrm{TiO}$', \
        verticalalignment='center', horizontalalignment='center', \
        transform=ax.transData, color='firebrick', size=12)

    return None

def get_all_line_wav():

    # Define rest-frame wavelengths in vacuum
    # Typically seen in emission
    oii3727 = 3727.0
    hbeta = 4862.7
    oiii4959 = 4960.3
    oiii5007 = 5008.2
    halpha = 6564.6

    # Typically seen in absorption
    # These are in air # Need to find vacuum wav
    gband = 4300
    mg2_mgb = 5175
    fe5270 = 5270
    fe5335 = 5335
    #fe5406 = 5406
    #nad = 5890

    # Now put all the line wavelengths in a list and return
    all_lines = [oii3727, hbeta, oiii4959, oiii5007, halpha, gband, mg2_mgb, fe5270, fe5335]#, fe5406, nad]
    all_line_labels = [r'$[OII]3727$', r'H$\beta$', r'$[OIII]4959$', r'$[OIII]5007$', \
    r'H$\alpha + [NII]$', 'G-band', r'$\mathrm{Mg_2 + Mgb}$', 'Fe5270', 'Fe5335']#, 'Fe5406', 'NaD+TiO']

    return all_lines, all_line_labels

def mask_em_lines(lam_em, llam_em):

    # create empty mask index array
    all_lines, all_line_labels = get_all_line_wav()
    mask_indices = np.zeros(len(lam_em))

    # Now loop over all lines and mask each
    # For now just masking two points on either side of central wavelength
    # For [NII] we do one additional point on each side
    for i in range(len(all_lines)):

        current_line_wav = all_lines[i]
        current_line_label = all_line_labels[i]
        current_line_idx = np.argmin(abs(lam_em - current_line_wav))

        if '[NII]' in current_line_label:
            mask_indices[current_line_idx-4:current_line_idx+5] = 1
        else:
            mask_indices[current_line_idx-3:current_line_idx+4] = 1

    # also mask additional points on either end
    mask_indices[:4] = 1
    mask_indices[-4:] = 1

    # Now create the masked array
    masked_llam_arr = ma.masked_array(llam_em, mask=mask_indices)

    return masked_llam_arr, mask_indices

def Gauss(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def GaussAbs(x, amp1, mu1, sigma1, amp2, mu2, sigma2, amp3, mu3, sigma3, \
    amp4, mu4, sigma4, amp5, mu5, sigma5):
    return amp1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + \
           amp2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)) + \
           amp3 * np.exp(-(x - mu3)**2 / (2 * sigma3**2)) + \
           amp4 * np.exp(-(x - mu4)**2 / (2 * sigma4**2)) + \
           amp5 * np.exp(-(x - mu5)**2 / (2 * sigma5**2))

def GaussAbs_central_wav_fixed(x, amp1, sigma1, amp2, sigma2, amp3, sigma3):
    return (amp1 * np.exp(-(x - 4861.0)**2 / (2 * sigma1**2))) + \
           (amp2 * np.exp(-(x - 5160.0)**2 / (2 * sigma2**2))) + \
           (amp3 * np.exp(-(x - 5335.0)**2 / (2 * sigma3**2)))

def fit_gauss_mgfe_astropy(stack_lam, stack_llam, num_massive, z_low, z_high, ms_lim_low, ms_lim_high):

    # First constrain the region to be fit
    fitreg_idx = np.where((stack_lam >= 4600) & (stack_lam <= 5650))[0]
    stack_lam_to_fit = stack_lam[fitreg_idx]
    stack_llam_to_fit = stack_llam[fitreg_idx]

    # Inititalize models
    const_init = models.Const1D(amplitude=1.0)
    hb_abs_init = models.Gaussian1D(amplitude=0.01, mean=4861.0, stddev=100.0)
    mg_abs_init = models.Gaussian1D(amplitude=0.05, mean=5150.0, stddev=100.0)
    fe_abs_init = models.Gaussian1D(amplitude=0.02, mean=5335.0, stddev=150.0)

    # Fitting
    # gauss_init = (const_init - hb_abs_init) + (const_init - mg_abs_init) + (const_init - fe_abs_init)
    gauss_init = const_init - hb_abs_init -  mg_abs_init - fe_abs_init

    # Freeze central wavelengths
    # and give bounds
    gauss_init.mean_1.fixed = True
    gauss_init.mean_2.fixed = True
    gauss_init.mean_3.fixed = True

    gauss_init.stddev_3.min, gauss_init.stddev_3.max = 5.0, 250.0
    # In Angstroms # These units are NOT km/s !!! 

    fit_gauss = fitting.LevMarLSQFitter()
    g = fit_gauss(gauss_init, stack_lam_to_fit, stack_llam_to_fit)

    #res = abs(g(stack_lam_to_fit) - stack_llam_to_fit)
    #print(np.std(res))
    
    print(g)
    print("\n")
    print(np.array_repr(g.parameters, precision=3))

    # Compute mg2fe ratio
    mg_amp = g.parameters[4]
    mg_stddev = g.parameters[6]
    fe_amp = g.parameters[7]
    fe_stddev = g.parameters[9]

    mg_area = mg_amp * mg_stddev
    fe_area = fe_amp * fe_stddev
    mg2fe = mg_area/fe_area

    print("Mg flux:", mg_area * np.sqrt(2*np.pi))
    print("Fe flux:", fe_area * np.sqrt(2*np.pi))

    print("Mg to Fe flux ratio: %.2f" % mg2fe)
    print("log(Mg/Fe) measured to be: %.2f" % np.log10(mg2fe))

    # plot fit
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Show data and combined fit
    ax.plot(stack_lam, stack_llam, '.-', color='royalblue', linewidth=1.5, \
        markeredgecolor='royalblue', markersize=1.0, zorder=1)
    ax.plot(stack_lam, g(stack_lam), ls='--', color='orange', lw=2, zorder=5)

    # Show individual gaussians
    ax.plot(stack_lam, g[0](stack_lam) - g[1](stack_lam), ls='--', color='dodgerblue', lw=2, zorder=4)
    ax.plot(stack_lam, g[0](stack_lam) - g[2](stack_lam), ls='--', color='springgreen', lw=2, zorder=4)
    ax.plot(stack_lam, g[0](stack_lam) - g[3](stack_lam), ls='--', color='crimson', lw=2, zorder=4)

    # Horizontal line
    ax.axhline(y=1.0, ls='--', color='k')

    # Labels
    ax.set_xlabel(r'$\lambda\ [\mathrm{\AA}]$', fontsize=15)
    ax.set_ylabel(r'$L_{\lambda}\ [\mathrm{continuum\ subtracted}]$', fontsize=15)

    ax.set_xlim(4000, 6200)
    ax.set_ylim(0.9, 1.1)
    ax.minorticks_on()

    add_line_labels(ax, None)

    # Number of galaxies and redshift range on plot
    ax.text(0.66, 0.97, r'$\mathrm{N\,=\,}$' + str(num_massive), \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=12)
    ax.text(0.66, 0.92, str(z_low) + r'$\,\leq z \leq\,$' + str(z_high), \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=12)

    # Mass range
    ax.text(0.66, 0.86, str(ms_lim_low) + r'$\,\leq \mathrm{M\ [M_\odot]} <\,$' + str(ms_lim_high), \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=12)

    # [Mg/Fe] on plot
    mg2fe_str = r'$\mathrm{[Mg/Fe] = }$' + "{:.2f}".format(np.log10(mg2fe)) + r'$\pm\, 0.11$'
    ax.text(0.66, 0.81, mg2fe_str, \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='k', size=12)

    # Add inset figure to show line broadening 
    left, bottom, width, height = [0.18, 0.66, 0.35, 0.2]
    axins = fig.add_axes([left, bottom, width, height])

    width_z_err = 0.029 * 5175  # in angstrom; assumes a mean 3% error on redshifts
    width_lsf_err = 3 * 50  # in angstrom; number of ACS pixels of a typical galaxy times spectral res at mgb
    quad_width = np.sqrt(width_z_err**2 + width_lsf_err**2)

    y_ins = 1.0 - Gauss(stack_lam, 1.0, 5175.0, quad_width)
    axins.plot(stack_lam, y_ins, color='k')
    axins.set_xlim(4000, 6200)
    axins.minorticks_on()

    # Save figure
    if ms_lim_low == 9.5:
        msstr = 'intermediate_mass_'
    elif ms_lim_low == 10.5:
        msstr = 'massive_'

    fig.savefig(stacking_figures_dir + 'Mg2Fe_fit_result_'+ msstr + \
        str(z_low).replace('.','p') + '_' + str(z_high).replace('.','p') +
        '.pdf', dpi=300, bbox_inches='tight')

    return None

def fit_gauss_mgfe_scipy(stack_lam, stack_llam, z_low, z_high):

    # Add/Subtract factor to have continuum subtracted value at 4500 A be exactly zero
    #lam4500 = np.argmin(abs(stack_lam - 4500))
    #stack_llam -= stack_llam[lam4500]

    # First constrain the region to be fit
    fitreg_idx = np.where((stack_lam >= 4600) & (stack_lam <= 5600))[0]
    stack_lam_to_fit = stack_lam[fitreg_idx]
    stack_llam_to_fit = stack_llam[fitreg_idx]

    # The fitting cannot handle NaNs 
    # Make sure any nan values are replaced by interpolated values
    """
    for i in range(len(stack_lam_to_fit)):

        current_lam = stack_lam_to_fit[i]
        current_llam = stack_llam_to_fit[i]
        if not np.isfinite(stack_llam_to_fit[i]):

            stack_lam_idx = np.argmin(abs(stack_lam - current_lam))
            # Referenced to the original stack lam array because 
            # just the fitting array might not have the values
            # we need for interpolation

            # I'm calling it interpolation but for now I'm just 
            # taking an average of the two points around it.
            stack_llam_to_fit[i] = (stack_llam[stack_lam_idx - 1] + stack_llam[stack_lam_idx + 1])/2
    """

    # Initial params
    # Hbeta
    amp_hb = 0.5
    mu_hb = 4861.0
    sigma_hb = 80.0

    # OIII 4959
    amp_oiii1 = 0.1
    mu_oiii1 = 4959.0
    sigma_oiii1 = 20.0

    # OIII 5007
    amp_oiii2 = 0.4
    mu_oiii2 = 5007.0
    sigma_oiii2 = 20.0

    # Mg2 + Mgb
    amp_mg = 0.8
    mu_mg = 5175.0
    sigma_mg = 100.0

    # Fe 5270
    amp_fe1 = 0.3
    mu_fe1 = 5270.0
    sigma_fe1 = 10.0

    # Fe 5335
    amp_fe2 = 0.3
    mu_fe2 = 5335.0
    sigma_fe2 = 80.0

    # Fe 5406
    amp_fe3 = 0.3
    mu_fe3 = 5406.0
    sigma_fe3 = 10.0

    # initial_guess = [amp_hb, mu_hb, sigma_hb, \
    # amp_mg, mu_mg, sigma_mg, \
    # amp_fe1, mu_fe1, sigma_fe1, \
    # amp_fe2, mu_fe2, sigma_fe2, \
    # amp_fe3, mu_fe3, sigma_fe3]

    # Set bounds. You need to pass to curve_fit,
    # a tuple which is a set of two lists. 
    # First list is lower bounds on all params 
    # and second list is upper bound on all params.
    """
    hb_amp_low, hb_amp_high = -0.8e39, -0.1e39
    hb_mu_low, hb_mu_high = 4855.0, 4865.0
    hb_sigma_low, hb_sigma_high = 5.0, 200.0

    mg_amp_low, mg_amp_high = -1.5e39, -0.1e39
    mg_mu_low, mg_mu_high = 5174.0, 5176.0
    mg_sigma_low, mg_sigma_high = 10.0, 250.0

    fe1_amp_low, fe1_amp_high = -0.7e39, -0.1e39
    fe1_mu_low, fe1_mu_high = 5265.0, 5275.0
    fe1_sigma_low, fe1_sigma_high = 5.0, 75.0

    fe2_amp_low, fe2_amp_high = -0.7e39, -0.1e39
    fe2_mu_low, fe2_mu_high = 5330.0, 5340.0
    fe2_sigma_low, fe2_sigma_high = 5.0, 75.0

    fe3_amp_low, fe3_amp_high = -0.7e39, -0.1e39
    fe3_mu_low, fe3_mu_high = 5400.0, 5410.0
    fe3_sigma_low, fe3_sigma_high = 5.0, 75.0

    
    hb_amp_low, hb_amp_high = -np.inf, np.inf
    hb_mu_low, hb_mu_high = 4820.0, 4940.0
    hb_sigma_low, hb_sigma_high = -np.inf, np.inf

    mg_amp_low, mg_amp_high = -np.inf, np.inf
    mg_mu_low, mg_mu_high = 5174.0, 5176.0
    mg_sigma_low, mg_sigma_high = -np.inf, np.inf

    fe1_amp_low, fe1_amp_high = -np.inf, np.inf
    fe1_mu_low, fe1_mu_high = 5265.0, 5275.0
    fe1_sigma_low, fe1_sigma_high = -np.inf, np.inf

    fe2_amp_low, fe2_amp_high = -np.inf, np.inf
    fe2_mu_low, fe2_mu_high = 5330.0, 5340.0
    fe2_sigma_low, fe2_sigma_high = -np.inf, np.inf

    fe3_amp_low, fe3_amp_high = -np.inf, np.inf
    fe3_mu_low, fe3_mu_high = 5400.0, 5410.0
    fe3_sigma_low, fe3_sigma_high = -np.inf, np.inf

    bounds = ([hb_amp_low, hb_mu_low, hb_sigma_low, mg_amp_low, mg_mu_low, mg_sigma_low, \
        fe1_amp_low, fe1_mu_low, fe1_sigma_low, fe2_amp_low, fe2_mu_low, fe2_sigma_low, \
        fe3_amp_low, fe3_mu_low, fe3_sigma_low], \
        [hb_amp_high, hb_mu_high, hb_sigma_high, mg_amp_high, mg_mu_high, mg_sigma_high, \
        fe1_amp_high, fe1_mu_high, fe1_sigma_high, fe2_amp_high, fe2_mu_high, fe2_sigma_high, \
        fe3_amp_high, fe3_mu_high, fe3_sigma_high])
    """

    initial_guess = [amp_hb, sigma_hb, \
    amp_mg, sigma_mg, \
    amp_fe2, sigma_fe2]

    popt, pcov = curve_fit(GaussAbs_central_wav_fixed, xdata=stack_lam_to_fit, ydata=stack_llam_to_fit)#, \
    #p0=initial_guess)#, bounds=bounds)

    print("Optimal param values:", np.array_repr(popt, precision=2))
    print("\n")
    perr = np.sqrt(np.diag(pcov))
    print("Errors:", np.array_repr(perr, precision=2))

    # plot fit
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Show data and combined fit
    ax.plot(stack_lam, stack_llam, '.-', color='royalblue', linewidth=1.5, \
        markeredgecolor='royalblue', markersize=1.0, zorder=1)
    ax.plot(stack_lam, GaussAbs_central_wav_fixed(stack_lam, *popt), ls='--', color='orange', lw=2, zorder=4)

    # Show individual gaussians
    # hbeta_params = popt[0:3]
    # oiii4959_params = popt[3:6]
    # oiii5007_params = popt[6:9]
    # mg_params = popt[9:12]
    # fe_params = popt[12:15]

    hb_amp_best, hb_sigma_best = popt[0], popt[1]
    mg_amp_best, mg_sigma_best = popt[2], popt[3]
    #fe1_amp_best, fe1_sigma_best = popt[4], popt[5]
    fe2_amp_best, fe2_sigma_best = popt[4], popt[5]
    #fe3_amp_best, fe3_sigma_best = popt[8], popt[9]

    hbeta_gaussian = Gauss(stack_lam, hb_amp_best, 4861.0, hb_sigma_best)
    mg_gaussian = Gauss(stack_lam, mg_amp_best, 5160.0, mg_sigma_best)
    #fe1_gaussian = Gauss(stack_lam, fe1_amp_best, 5270.0, fe1_sigma_best)
    fe2_gaussian = Gauss(stack_lam, fe2_amp_best, 5335.0, fe2_sigma_best)
    #fe3_gaussian = Gauss(stack_lam, fe3_amp_best, 5406.0, fe3_sigma_best)

    ax.plot(stack_lam, hbeta_gaussian, color='dodgerblue', zorder=2)
    ax.fill_between(stack_lam, hbeta_gaussian.max(), hbeta_gaussian, facecolor='dodgerblue', alpha=0.4, zorder=2)

    ax.plot(stack_lam, mg_gaussian, color='springgreen', zorder=2)
    ax.fill_between(stack_lam, mg_gaussian.max(), mg_gaussian, facecolor='springgreen', alpha=0.4, zorder=2)

    #ax.plot(stack_lam, fe1_gaussian, color='crimson', zorder=2)
    #ax.fill_between(stack_lam, fe1_gaussian.max(), fe1_gaussian, facecolor='crimson', alpha=0.4, zorder=2)
    ax.plot(stack_lam, fe2_gaussian, color='crimson', zorder=2)
    ax.fill_between(stack_lam, fe2_gaussian.max(), fe2_gaussian, facecolor='crimson', alpha=0.4, zorder=2)
    #ax.plot(stack_lam, fe3_gaussian, color='crimson', zorder=2)
    #ax.fill_between(stack_lam, fe3_gaussian.max(), fe3_gaussian, facecolor='crimson', alpha=0.4, zorder=2)

    # Horizontal line
    ax.axhline(y=0.0, ls='--', color='k')

    # Labels
    ax.set_xlabel(r'$\lambda\ [\mathrm{\AA}]$', fontsize=15)
    ax.set_ylabel(r'$L_{\lambda}\ [\mathrm{continuum\ subtracted}]$', fontsize=15)

    ax.minorticks_on()

    mg_area = mg_amp_best * abs(mg_sigma_best)
    fe_area = fe2_amp_best * abs(fe2_sigma_best)
    mg2fe = mg_area/fe_area
    print("Mg to Fe flux ratio: %.2f" % mg2fe)
    print("log(Mg/Fe) measured to be: %.2f" % np.log10(mg2fe))

    plt.show()
    sys.exit(0)

    fig.savefig(stacking_figures_dir + 'Mg2Fe_fit_result_'+ \
        str(z_low).replace('.','p') + '_' + str(z_high).replace('.','p') +
        '.pdf', dpi=300, bbox_inches='tight')

    return mg2fe

def main():

    # Start time
    start = time.time()
    dt = datetime.datetime
    print("Coaddition started at --")
    print(dt.now())
    
    # ----------------------------------------- READ IN CATALOGS ----------------------------------------- #
    # Read in results for all of PEARS
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True, encoding='ascii')
    # Read in U-R color  # This was generated by make_col_ms_plots.py
    urcol = np.load(stacking_analysis_dir + 'ur_arr_all.npy')

    print("Read in following catalog header.")
    print(cat.dtype.names)

    #cat = np.genfromtxt(stacking_analysis_dir + 'pears_figs_combined_final_sample.txt', \
    #    dtype=None, names=True, encoding='ascii')
    #urcol = cat['ur_col']

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

    all_z_low  = np.array([0.16]) #([0.16,0.16,0.60,0.16])
    all_z_high = np.array([0.96]) #([0.96,0.60,0.96,1.42])

    # Separate grid stack for each redshift interval
    # This function will create and save the stacks in a fits file
    for i in range(len(all_z_low)):
        
        # Get z range and indices
        z_low = all_z_low[i]
        z_high = all_z_high[i]
        z_indices = np.where((zp >= z_low) & (zp < z_high))[0]

        #create_stacks(cat, urcol, z_low, z_high, z_indices, start)
        #plot_stacks(cat, urcol, z_low, z_high, z_indices, start)
        stack_plot_massive(cat, urcol, z_low, z_high, z_indices, start)

    # Total time taken
    print("Total time taken for all stacks --", "{:.2f}".format((time.time() - start)/60.0), "minutes.")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
