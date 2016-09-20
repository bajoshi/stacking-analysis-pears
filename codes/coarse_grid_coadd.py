from __future__ import division
import os
import sys
import time
import datetime
import logging

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

import grid_coadd as gd

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figures_dir = stacking_analysis_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"

if __name__ == '__main__':
    # Start time
    start = time.time()
    
    data_path = home + "/Documents/PEARS/data_spectra_only/"
    threedphot = home + "/Documents/3D-HST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat"
    threed = fits.open(home + '/Documents/3D-HST/3dhst.v4.1.5.master.fits')
    
    logging.basicConfig(filename='coadd_coarsegrid.log', format='%(levelname)s:%(message)s', filemode='w', level=logging.DEBUG)
    logging.info("\n This file is overwritten every time the program runs.\n You will have to change the filemode to log messages for every run.")
        
    dt = datetime.datetime
    logging.info("Coaddition started at --")
    logging.info(dt.now())

    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt', dtype=None, names=True, skip_header=2)
    
    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']
    
    lam_step = 100
    lam_grid = np.arange(2700, 6000, lam_step)
    # Lambda grid decided based on observed wavelength range i.e. 6000 to 9500
    # and the initially chosen redshift range 0.6 < z < 1.2
    # This redshift range was chosen so that the 4000A break would fall in the observed wavelength range

    # Change only the parameters here to change how the code runs
    # Ideally you shouldn't have to change anything else.
    col_step = 0.6
    mstar_step = 1.0
    final_fits_filename = 'coadded_PEARSgrismspectra_coarsegrid.fits'
    col_low = 0.0
    col_high = 3.0
    mstar_low = 7.0
    mstar_high = 12.0

    hdu = fits.PrimaryHDU()
    hdr = fits.Header()
    hdulist = fits.HDUList(hdu)
    hdulist.append(fits.ImageHDU(lam_grid, header=hdr))
    
    skipspec = np.loadtxt(home + '/Desktop/FIGS/stacking-analysis-pears/specskip.txt', dtype=np.str, delimiter=',')
    em_lines = np.loadtxt(home + '/Desktop/FIGS/stacking-analysis-pears/em_lines_readin.txt', dtype=np.str, delimiter=',')
    # This little for loop is to fix formatting issues with the skipspec and em_lines arrays that are read in with loadtxt.
    for i in range(len(skipspec)):
        skipspec[i] = skipspec[i].replace('\'', '')
    for i in range(len(em_lines)):
        em_lines[i] = em_lines[i].replace('\'', '')
        
    added_gal = 0
    skipped_gal = 0
    gal_per_bin = np.zeros((len(np.arange(col_low, col_high, col_step)), len(np.arange(mstar_low, mstar_high, mstar_step))))

    for i in np.arange(col_low, col_high, col_step):
        for j in np.arange(mstar_low, mstar_high, mstar_step):
            
            gal_current_bin = 0
            print "ONGRID", i, j
            logging.info("\n ONGRID %.1f %.1f", i, j)
            
            indices = np.where((ur_color >= i) & (ur_color < i + col_step) &\
                               (stellarmass >= j) & (stellarmass < j + mstar_step))[0]

            old_flam = np.zeros(len(lam_grid))
            old_flamerr = np.zeros(len(lam_grid))
            num_points = np.zeros(len(lam_grid))
            num_galaxies = np.zeros(len(lam_grid))
            
            old_flam = old_flam.tolist()
            old_flamerr = old_flamerr.tolist()

            curr_pearsids = pears_id[indices]
            curr_zs = photz[indices]
            
            logging.info("Number of spectra to coadd in this grid cell -- %d.", len(pears_id[indices]))
            
            # rescale to 200A band centered on a wavelength of 4500A # 4400A-4600A
            # This function returns the maximum of the median values (in the given band) from all given spectra
            # All spectra to be coadded in a given grid cell need to be divided by this value
            if indices.size:
                medarr, medval, stdval = gd.rescale(curr_pearsids, curr_zs)
                print "ONGRID", i, j, "with", medval, "as the normalization value and a maximum possible spectra of", len(pears_id[indices]), "spectra."
            else:
                print "ONGRID", i, j, "has no spectra to coadd. Moving to next grid cell."
                continue
            
            # Loop over all spectra in a grid cell and coadd them
            for u in range(len(pears_id[indices])):
                
                # This step should only be done on the first iteration within a grid cell
                # This converts every element (which are all 0 to begin with) in the flux and flux error arrays to an empty list
                # This is done so that function add_spec() can now append to every element
                if u == 0.0:
                    for x in range(len(lam_grid)):
                        old_flam[x] = []
                        old_flamerr[x] = []
            
                # Get redshift from previously saved 3DHST photz catalog
                redshift = photz[indices][u]
        
                # Get rest frame values for all quantities
                lam_em, flam_em, ferr, specname = gd.fileprep(pears_id[indices][u], redshift)

                # Divide by max value to rescale
                flam_em = (flam_em / medarr[u])
                ferr = (ferr / medarr[u])
        
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
                    old_flam, old_flamerr, num_points, num_galaxies =\
                        gd.add_spec(specname, lam_em, flam_em, ferr, old_flam, old_flamerr, num_points, num_galaxies, lam_grid, lam_step)

            # taking median
            # maybe I should also try doing a mean after 3sigma clipping and compare
            for y in range(len(lam_grid)):
                if old_flam[y]:
                    old_flamerr[y] = np.sqrt((1.253 * np.std(old_flam[y]) / np.sqrt(len(old_flam[y])))**2 + np.sum(old_flamerr[y]) / gal_current_bin)
                    old_flam[y] = np.median(old_flam[y])
                else:
                    old_flam[y] = 0.0
                    old_flamerr[y] = 0.0

            avgcol = np.mean(ur_color[indices])
            avgmass = np.mean(stellarmass[indices])

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
            hdr["AVGCOL"] = str(avgcol)
            hdr["AVGMASS"] = str(avgmass)
            
            dat = np.array((old_flam, old_flamerr)).reshape(2, len(lam_grid))
            hdulist.append(fits.ImageHDU(data = dat, header = hdr))
            
            row = int(i/col_step)
            column = int((j - mstar_low)/mstar_step)
            gal_per_bin[row,column] = gal_current_bin
    
    print added_gal, skipped_gal
    print np.flipud(gal_per_bin)
    print np.sum(gal_per_bin, axis=None)
    hdulist.writeto(savefits_dir + final_fits_filename, clobber=True)
    
    # Total time taken
    logging.info("Time taken for coaddition -- %.2f seconds", time.time() - start)
