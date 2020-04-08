# coding: utf-8
from __future__ import division

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')  # Does not have a trailing slash at the end
figs_dir = home + "/Desktop/FIGS/"
stacking_analysis_dir = figs_dir + "stacking-analysis-pears/"
stacking_figures_dir = figs_dir + "stacking-analysis-figures/"
massive_galaxies_dir = figs_dir + "massive-galaxies/"
pears_data_path = home + "/Documents/PEARS/data_spectra_only/"

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
sys.path.append(massive_galaxies_dir + 'cluster_codes/')
sys.path.append(stacking_analysis_dir + 'stacking_pipeline/')
import cluster_do_fitting as cf
import make_col_ms_plots

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

def add_spec(lam_em, llam_em, lerr, old_llam, old_llamerr, num_points, num_galaxies, lam_grid, lam_step):
    
    for i in range(len(lam_grid)):
        
        # add fluxes
        new_ind = np.where((lam_em >= lam_grid[i] - lam_step/2) & (lam_em < lam_grid[i] + lam_step/2))[0]

        if new_ind.size:
            
            # Only count a galaxy in a particular bin if for that bin at least one point is nonzero
            if np.any(llam_em[new_ind] != 0):
                num_galaxies[i] += 1
            
            # Reject points with excess contamination
            # Rejecting points that are more than 20% contaminated
            # Reject points that have negative signal
            # Looping over every point in a delta lambda bin
            for j in range(len(new_ind)):
                sig = llam_em[new_ind][j]
                noise = lerr[new_ind][j]
                
                if sig > 0: # only append those points where the signal is positive
                    if noise/sig < 0.20: # only append those points that are less than 20% contaminated
                        old_llam[i].append(sig)
                        old_llamerr[i].append(noise**2) # adding errors in quadrature
                        num_points[i] += 1 # keep track of how many points were added to each bin in lam_grid
                else:
                    continue

        else:            
            continue

    return old_llam, old_llamerr, num_points, num_galaxies

def add_spec2(pears_lam_em, pears_llam_em, pears_lerr, figs_lam_em, figs_llam_em, figs_lerr, 
    pears_old_llam, pears_old_llamerr, figs_old_llam, figs_old_llamerr, 
    pears_num_points, pears_num_galaxies, figs_num_points, figs_num_galaxies, lam_grid, lam_step)

    for i in range(len(lam_grid)):
        
        # add fluxes
        new_ind_pears = np.where((pears_lam_em >= lam_grid[i] - lam_step/2) & (pears_lam_em < lam_grid[i] + lam_step/2))[0]
        new_ind_figs  = np.where((figs_lam_em >= lam_grid[i] - lam_step/2)  & (figs_lam_em < lam_grid[i] + lam_step/2))[0]

        # Do the coadding for PEARS
        if new_ind_pears.size:

            # Only count a galaxy in a particular bin if for that bin at least one point is nonzero
            if np.any(pears_llam_em[new_ind_pears] != 0):
                pears_num_galaxies[i] += 1

            # Reject points with excess contamination
            # Rejecting points that are more than 33% contaminated
            # Reject points that have negative signal
            # Looping over every point in a delta lambda bin
            for j in range(len(new_ind_pears)):
                sig = pears_llam_em[new_ind_pears][j]
                noise = pears_lerr[new_ind_pears][j]
                
                if sig > 0: # only append those points where the signal is positive
                    if noise/sig < 0.33: # only append those points that are less than 20% contaminated
                        pears_old_llam[i].append(sig)
                        pears_old_llamerr[i].append(noise**2) # adding errors in quadrature
                        pears_num_points[i] += 1 # keep track of how many points were added to each bin in lam_grid
                else:
                    continue

        # Do the coadding for FIGS
        if new_ind_figs.size:

            # Only count a galaxy in a particular bin if for that bin at least one point is nonzero
            if np.any(figs_llam_em[new_ind] != 0):
                figs_num_galaxies[i] += 1


    return pears_old_llam, pears_old_llamerr, figs_old_llam, figs_old_llamerr, 
    pears_num_points, pears_num_galaxies, figs_num_points, figs_num_galaxies

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

    lam_obs = figs_spec[extname].data["LAMBDA"] # Wavelength 
    flam_obs = figs_spec[extname].data["AVG_WFLUX"]  # Flux (erg/s/cm^2/A)
    ferr_obs = figs_spec[extname].data["STD_WFLUX"]  # Flux error (erg/s/cm^2/A)
    #contam = figs_spec[extname].data["CONTAM"] # Flux contamination (erg/s/cm^2/A)

    return lam_obs, flam_obs, ferr_obs

def rescale(pears_id_arr_cell, pears_field_arr_cell, z_arr_cell, dl_tbl):
    """
    This function will provide teh median of all median
    f_lambda values (see below) at approx 4500 A. This 
    is for the purposes of rescaling all spectra within
    a cell to this median of medians.
    """

    # Define empty array to store the median 
    # value of f_lambda between 4400 to 4600 A
    # for each observed spectrum
    medarr = np.zeros(len(pears_id_arr_cell))

    # Define redshift array used in lookup table
    z_arr = np.arange(0.005, 6.005, 0.005)
    
    for k in range(len(pears_id_arr_cell)):

        # Get current ID and Field
        current_pears_id = pears_id_arr_cell[k]
        current_pears_field = pears_field_arr_cell[k]

        # Get observed data and deredshift the spectrum
        # PEARS data 
        grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = \
        cf.get_data(current_pears_id, current_pears_field)

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

    print("Working on stacks for redshift range:", z_low, "<= z <", z_high)

    # Read in catalog of all fitting results and assign arrays
    # For now we need the id+field, spz, stellar mass, and u-r color
    pears_id = cat['PearsID'][z_indices]
    pears_field = cat['Field'][z_indices]
    zp = cat['zp_minchi2'][z_indices]

    figs_id = cat['figs_id'][z_indices]
    figs_field = cat['figs_field'][z_indices]

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
    lam_grid_low = 2400
    lam_grid_high = 9200

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
            
            print("\n", "Stacking in cell:")
            print("Color range:", col, col+col_step)
            print("Stellar mass range:", ms, ms+mstar_step)
            
            # Find the indices (corresponding to catalog entries)
            # that are within the current cell
            indices = np.where((ur_color >= col) & (ur_color < col + col_step) &\
                            (stellar_mass >= ms) & (stellar_mass < ms + mstar_step))[0]

            num_galaxies_cell = int(len(pears_id[indices]))
            print("Number of spectra to coadd in this grid cell --", num_galaxies_cell)

            if num_galaxies_cell == 0:
                continue
            
            # Define empty arrays and lists for saving stacks
            pears_old_llam = np.zeros(len(lam_grid))
            pears_old_llamerr = np.zeros(len(lam_grid))
            pears_old_llam = pears_old_llam.tolist()
            pears_old_llamerr = pears_old_llamerr.tolist()

            figs_old_llam = np.zeros(len(lam_grid))
            figs_old_llamerr = np.zeros(len(lam_grid))
            figs_old_llam = figs_old_llam.tolist()
            figs_old_llamerr = figs_old_llamerr.tolist()

            pears_num_points = np.zeros(len(lam_grid))
            pears_num_galaxies = np.zeros(len(lam_grid))

            figs_num_points = np.zeros(len(lam_grid))
            figs_num_galaxies = np.zeros(len(lam_grid))

            # rescale to 200A band centered on a wavelength of 4500A # 4400A-4600A
            # This function returns the median of the median values (in the given band) from all given spectra
            # All spectra to be coadded in a given grid cell need to be divided by this value
            medarr, medval, stdval = rescale(pears_id[indices], pears_field[indices], zp[indices], dl_tbl)
            print("This cell has median value:", "{:.3e}".format(medval), " [erg s^-1 A^-1]")
            print("as the normalization value and a maximum possible of", len(pears_id[indices]), "spectra.")

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
            
                        figs_old_llam[x] = []
                        figs_old_llamerr[x] = []

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
                print(current_figs_id, current_figs_field)
                g102_lam_obs, g102_flam_obs, g102_ferr_obs = get_figs_data(current_figs_id, current_figs_field)

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

                # Match with photometry catalog and get photometry data
                
                # Divide by median value at 4400A to 4600A to rescale. 
                # Multiplying by median value of the flux medians to get it back to physical units
                pears_llam_em = (pears_llam_em / medarr[u]) * medval
                pears_lerr = (pears_lerr / medarr[u]) * medval

                figs_llam_em = (figs_llam_em / medarr[u]) * medval
                figs_lerr = (figs_lerr / medarr[u]) * medval

                # add the spectrum
                added_gal += 1
                gal_current_cell += 1
                pears_old_llam, pears_old_llamerr, figs_old_llam, figs_old_llamerr, \
                pears_num_points, pears_num_galaxies, figs_num_points, figs_num_galaxies = \
                add_spec2(pears_lam_em, pears_llam_em, pears_lerr, figs_lam_em, figs_llam_em, figs_lerr, 
                    pears_old_llam, pears_old_llamerr, figs_old_llam, figs_old_llamerr, 
                    pears_num_points, pears_num_galaxies, figs_num_points, figs_num_galaxies, lam_grid, lam_step)

                # Call to old function
                # add_spec(lam_em, llam_em, lerr, old_llam, old_llamerr, \
                #          num_points, num_galaxies, lam_grid, lam_step)

                sys.exit(0)

            # taking median
            for y in range(len(lam_grid)):
                if old_llam[y]:

                    # Actual stack value after 3 sigma clipping
                    # Only allowing 3 iterations right now
                    masked_data = sigma_clip(data=old_llam[y], sigma=3, iters=3)
                    old_llam[y] = np.median(masked_data)

                    # Get mask from the masked_data array
                    mask = np.ma.getmask(masked_data)

                    # Apply mask to error array
                    masked_dataerr = np.ma.array(old_llamerr[y], mask=mask)

                    # Error on each point of the stack
                    # This only uses the points that passed the 3 sigma clipping before
                    old_llamerr[y] = \
                    np.sqrt((1.253 * np.std(masked_data) / \
                        np.sqrt(len(masked_data)))**2 + np.sum(masked_dataerr) / gal_current_cell)

                else:
                    old_llam[y] = 0.0
                    old_llamerr[y] = 0.0

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
            dat = np.array((old_llam, old_llamerr)).reshape(2, len(lam_grid))
            hdulist.append(fits.ImageHDU(data=dat, header=exthdr))

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
    print(np.sum(gal_per_cell, axis=None))

    return None
    
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

    ur_color = urcol[z_indices]
    stellar_mass = np.log10(cat['zp_ms'][z_indices])  # because the code below is expected log(stellar mass)

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
                ax.text(0.93, 0.65, "{:.2f}".format(float(avgcolarr_to_print[row])), \
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

                current_id = pears_id[indices][u]
                current_field = pears_field[indices][u]

                # ----------------------------- Get data ----------------------------- #
                grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code \
                = cf.get_data(current_id, current_field)

                # Deredshift the observed data 
                zidx = np.argmin(abs(z_arr - current_redshift))
                # Make sure that the z_arr here is the same array that was 
                # used to generate the dl lookup table.
                dl = dl_tbl['dl_cm'][zidx]  # has to be in cm

                lam_em = grism_lam_obs / (1 + current_redshift)
                llam_em = grism_flam_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
                lerr = grism_ferr_obs * (1 + current_redshift) * (4 * np.pi * dl * dl)
                
                # Divide by median value at 4400A to 4600A to rescale. 
                # Multiplying by median value of the flux medians to get it back to physical units
                llam_em = (llam_em / medarr[u]) * medval
                lerr = (lerr / medarr[u]) * medval

                # Plotting
                ax.plot(lam_em, llam_em, ls='-', color='lightgray', linewidth=0.5)
                ax.get_yaxis().set_ticklabels([])
                ax.get_xaxis().set_ticklabels([])
                ax.set_xlim(2000, 8000)

            # Plot stack in blue
            llam = stack_hdu[cellcount+2].data[0]
            llam_err = stack_hdu[cellcount+2].data[1]

            # Force zeros to NaNs so that they're not plotted
            llam_zero_idx = np.where(llam == 0.0)[0]
            llam[llam_zero_idx] = np.nan
            ax.errorbar(lam, llam, yerr=llam_err, fmt='.-', color='b', linewidth=0.5,\
                        elinewidth=0.2, ecolor='r', markeredgecolor='b', capsize=0, markersize=0.5, zorder=5)

            # Y Limits 
            # Find min and max within the stack and add some padding
            # Using the nan functions here because some stacks have nan values
            stack_min = np.nanmin(llam)
            stack_max = np.nanmax(llam)
            stack_mederr = np.nanmedian(llam_err)  # median of all errors on hte stack
            ax.set_ylim(stack_min - 3 * stack_mederr, stack_max + 3 * stack_mederr)

            # Add other info to plot
            numspec = int(stack_hdu[cellcount+2].header['NUMSPEC'])
            normval = float(stack_hdu[cellcount+2].header['NORMVAL'])
            normval = cf.convert_to_sci_not(normval)  # Returns a properly formatted string

            # add number of galaxies in plot
            ax.text(0.8, 0.2, numspec, verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=10)

            # Normalization value
            ax.text(0.04, 0.94, normval, verticalalignment='top', horizontalalignment='left', \
            transform=ax.transAxes, color='k', size=10)

            cellcount += 1

    fig.savefig(stacking_figures_dir + final_fits_filename.replace('.fits','.pdf'), dpi=300, bbox_inches='tight')

    # Close fits file and return
    stack_hdu.close()

    return None

def main():

    # Start time
    start = time.time()
    dt = datetime.datetime
    print("Coaddition started at --")
    print(dt.now())
    
    # ----------------------------------------- READ IN CATALOGS ----------------------------------------- #
    # Read in results for all of PEARS
    #cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True, encoding='ascii')
    # Read in U-R color  # This was generated by make_col_ms_plots.py
    #urcol = np.load(stacking_analysis_dir + 'ur_arr_all.npy')

    cat = np.genfromtxt(stacking_analysis_dir + 'pears_figs_combined_final_sample.txt', dtype=None, names=True, encoding='ascii')
    urcol = cat['ur_col']

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

    all_z_low = np.array([0.5, 0.8, 1.0, 1.5, 2.0])
    all_z_high = np.array([0.8, 1.0, 1.5, 2.0, 2.5])

    # Separate grid stack for each redshift interval
    # This function will create and save the stacks in a fits file
    for i in range(len(all_z_low)):
        
        # Get z range and indices
        z_low = all_z_low[i]
        z_high = all_z_high[i]
        z_indices = np.where((zp >= z_low) & (zp < z_high))[0]

        create_stacks(cat, urcol, z_low, z_high, z_indices, start)
        plot_stacks(cat, urcol, z_low, z_high, z_indices, start)

    # Total time taken
    print("Total time taken for all stacks --", "{:.2f}".format((time.time() - start)/60.0), "minutes.")

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
