from __future__ import division
import os, sys, time, datetime
import logging

import numpy as np
import pyfits as pf
from scipy.stats import gaussian_kde

import matplotlib as mpl
import matplotlib.pyplot as plt
pgf_preamble = {"pgf.texsystem": "pdflatex"}
mpl.rcParams.update(pgf_preamble)

def get_net_sig(fitsdata, filename):
    
    try:
        signal_sum = 0
        noise_sum = 0
        totalsum = 0
        cumsum = []
        
        if np.count_nonzero(fitsdata['ERROR']) != len(fitsdata['ERROR']):
            raise ZeroDivisionError
    
        sn = fitsdata['COUNT']/fitsdata['ERROR']
        sn_sorted = np.sort(sn)
        sn_sorted_reversed = sn_sorted[::-1]
        
        for _count_ in range(len(fitsdata)):
            idx = np.where(sn==sn_sorted_reversed[_count_])
            signal_sum += fitsdata['COUNT'][idx]
            noise_sum += fitsdata['ERROR'][idx]**2
            totalsum = signal_sum/np.sqrt(noise_sum)
            cumsum.append(totalsum)

        netsig = np.amax(cumsum)
    
        return netsig
    
    except ValueError as detail:
        logging.warning(filename)
        #logging.warning(detail.value)
        logging.warning("The above spectrum will be given net sig of -99. Not sure of this error yet.")
    except ZeroDivisionError:
        logging.warning(filename)
        logging.warning("Division by zero! The net sig here cannot be trusted. Setting Net Sig to -99.")
        return -99.0

def add_spec(specname, lam_em, flam_em, ferr, old_flam, old_flamerr, num_points, num_galaxies):
    
    for i in range(len(lam_grid)):
        
        # add fluxes
        new_ind = np.where((lam_em >= lam_grid[i] - lam_step/2) & (lam_em < lam_grid[i] + lam_step/2))[0]
        
        if new_ind.size:
            
            # Only count a galaxy in a particular bin if for that bin at least one point is nonzero
            if np.any(flam_em[new_ind] != 0):
                num_galaxies[i] += 1
            
            # Reject points with excess contamination
            # Rejecting points that are more than 30% contaminated
            for j in range(len(new_ind)):
                sig = flam_em[new_ind][j]
                noise = ferr[new_ind][j]
                
                if sig != 0: # only append those points where the signal is non-zero
                    if noise/sig < 0.20:
                        old_flam[i].append(sig)
                        old_flamerr[i].append(noise**2) # adding errors in quadrature
                        num_points[i] += 1 # keep track of how many points were added to each bin in lam_grid
                else:
                    continue
    
        else:
            # figure out if sth else is needed to be done here if it cannot find any points to coadd
            # logging.warning("In spectrum -- %s at %d.  No points found for coaddition! Continuing...", specname, lam_grid[i])
            # This warning line produces too many lines in the log and bloats up the log file.
            
            continue

    return old_flam, old_flamerr, num_points, num_galaxies


def rescale(indices):
    medarr = np.zeros(len(pears_id[indices]))
    
    for k in range(len(pears_id[indices])):
        
        redshift = photz[indices][k]
        
        lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][k], redshift)
        
        # Store median of values from 4400A-4600A for each spectrum
        arg4400 = np.argmin(abs(lam_em - 4400))
        arg4600 = np.argmin(abs(lam_em - 4600))
        medarr[k] = np.median(flam_em[arg4400:arg4600+1])
    
    medval = np.median(medarr)
    
    # Return the maximum in array of median values
    return medarr, medval, np.std(medarr)

def fileprep(pears_index, redshift):
    
    # Get the correct filename and the number of extensions
    filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    if not os.path.isfile(filename):
        filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

    fitsfile = pf.open(filename)
    n_ext = fitsfile[0].header['NEXTEND']

    specname = os.path.basename(filename)
    
    # Get highest netsig to find the spectrum to be added
    if n_ext > 1:
        netsiglist = []
        for count in range(n_ext):
            fitsdata = fitsfile[count+1].data
            netsig = get_net_sig(fitsdata, filename)
            netsiglist.append(netsig)
        netsiglist = np.array(netsiglist)
        maxnetsigarg = np.argmax(netsiglist)
        spec_toadd = fitsfile[maxnetsigarg+1].data
    elif n_ext == 1:
        spec_toadd = fitsfile[1].data
    
    # Now get the spectrum to be added
    lam_obs = spec_toadd['LAMBDA']
    flam_obs = spec_toadd['FLUX']
    ferr = spec_toadd['FERROR']
    contam = spec_toadd['CONTAM']
    
    # Subtract Contamination
    flam_obs = flam_obs - contam
    
    # First chop off the ends and only look at the observed spectrum from 6000A to 9500A
    arg6000 = np.argmin(abs(lam_obs - 6000))
    arg9500 = np.argmin(abs(lam_obs - 9500))
    
    lam_obs = lam_obs[arg6000:arg9500]
    flam_obs = flam_obs[arg6000:arg9500]
    ferr = ferr[arg6000:arg9500]
    
    flam_obs[flam_obs < 0] = 0
    
    # Now unredshift the spectrum
    lam_em = lam_obs / (1 + redshift)
    flam_em = flam_obs * (1 + redshift)
    # check the relations for unredshifting
    
    return lam_em, flam_em, ferr, specname

if __name__ == '__main__':
    # Start time
    start = time.time()
    
    data_path = "/Users/baj/Documents/PEARS/data_spectra_only/"
    threedphot = "/Users/baj/Documents/3D-HST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat"
    threed = pf.open('/Users/baj/Documents/3D-HST/3dhst.v4.1.5.master.fits')
    
    logging.basicConfig(filename='coadd_coarsegrid.log', format='%(levelname)s:%(message)s', filemode='w', level=logging.DEBUG)
    logging.info("\n This file is overwritten every time the program runs.\n You will have to change the filemode to log messages for every run.")
        
    dt = datetime.datetime
    logging.info("Coaddition started at --")
    logging.info(dt.now())

    cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/color_stellarmass.txt', dtype=None, names=True, skip_header=2)
    
    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']
    
    lam_step = 100
    lam_grid = np.arange(2700, 6000, lam_step)
    # Lambda grid decided based on observed wavelength range i.e. 6000 to 9500
    # and the initially chosen redshift range 0.6 < z < 1.2
    # This redshift range was chosen so that the 4000A break would fall in the observed wavelength range
    
    color_step = 0.6
    mstar_step = 1.0
    
    hdu = pf.PrimaryHDU()
    
    hdr = pf.Header()
    nextend = len(np.arange(0.0,3.0, color_step)) * len(np.arange(7.0, 11.5, mstar_step))
    hdr["NEXTEND"] = str(nextend)
    # This isn't correct. This will give the number of grid cells but
    # not all grid cells will have results that go into the fits file.
    
    hdulist = pf.HDUList(hdu)
    
    hdulist.append(pf.ImageHDU(lam_grid, header=hdr))
    
    skipspec = ['h_pears_n_id78220.fits',\
                'h_pears_n_id47644.fits', 'h_pears_s_id74234.fits', 'h_pears_s_id124266.fits',\
                'h_pears_n_id111054.fits',\
                'h_pears_s_id106446.fits',\
                'h_pears_n_id120710.fits',\
                'h_pears_s_id78417.fits',\
                'h_pears_s_id59792.fits','h_pears_s_id93218.fits','h_pears_n_id104213.fits','h_pears_s_id115422.fits','h_pears_s_id72113.fits','h_pears_s_id107858.fits','h_pears_s_id45223.fits','h_pears_s_id23920.fits',\
                'h_pears_s_id64963.fits','h_pears_s_id128268.fits',\
                'h_pears_s_id110795.fits','h_pears_s_id108561.fits','h_pears_n_id123162.fits',\
                'h_pears_s_id79154.fits','h_pears_s_id114978.fits',\
                'h_pears_s_id115024.fits','h_pears_n_id67343.fits',\
                'h_pears_s_id113895.fits',\
                'h_pears_n_id79581.fits',\
                'h_pears_s_id120441.fits','h_pears_n_id77819.fits',\
                'h_pears_s_id48182.fits',\
                'h_pears_n_id103918.fits',\
                'h_pears_s_id70340.fits','h_pears_n_id110223.fits','h_pears_n_id59873.fits',\
                'h_pears_s_id56575.fits']
        
    added_gal = 0
    skipped_gal = 0
    gal_per_bin = np.zeros((len(np.arange(0.0,3.0, color_step)), len(np.arange(7.0, 11.5, mstar_step))))

    for i in np.arange(0.0, 3.0, color_step):
        for j in np.arange(7.0, 11.5, mstar_step):
            
            gal_current_bin = 0
            print "ONGRID", i, j
            logging.info("\n ONGRID %.1f %.1f", i, j)
            
            indices = np.where((ur_color >= i) & (ur_color < i + color_step) &\
                               (stellarmass >= j) & (stellarmass < j + mstar_step))[0]
                

            old_flam = np.zeros(len(lam_grid))
            old_flamerr = np.zeros(len(lam_grid))
            num_points = np.zeros(len(lam_grid))
            num_galaxies = np.zeros(len(lam_grid))
            
            old_flam = old_flam.tolist()
            old_flamerr = old_flamerr.tolist()
            
            logging.info("Number of spectra to coadd in this grid cell -- %d.", len(pears_id[indices]))
            
            # rescale to 200A band centered on a wavelength of 4500A # 4400A-4600A
            # This function returns the maximum of the median values (in the given band) from all given spectra
            # All spectra to be coadded in a given grid cell need to be divided by this value
            if indices.size:
                medarr, medval, stdval = rescale(indices)
                #print "ONGRID", i, j, "with", maxval, "as the normalization value and", len(pears_id[indices]), "spectra."
            else:
                #print "ONGRID", i, j, "has no spectra to coadd. Moving to next grid cell."
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
                lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][u], redshift)

                # Divide by max value to rescale
                flam_em = (flam_em / medarr[u])
                ferr = (ferr / medarr[u])
        
                if specname in skipspec:
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
                        add_spec(specname, lam_em, flam_em, ferr, old_flam, old_flamerr, num_points, num_galaxies)

            for y in range(len(lam_grid)):
                old_flam[y] = np.median(old_flam[y])
                old_flamerr[y] = np.median(old_flamerr[y])

            avgcol = np.mean(ur_color[indices])
            avgmass = np.mean(stellarmass[indices])

            hdr = pf.Header()
    
            #hdr["XTENSION"] = "IMAGE              / Image extension "
            #hdr["BITPIX"]  = "                 -64 / array data type"
            #hdr["NAXIS"]   = "                   2 / number of array dimensions"
            #hdr["NAXIS1"]  = "                 161"
            #hdr["NAXIS2"]  = "                   2"
            #hdr["PCOUNT"]  = "                   0 / number of parameters"
            #hdr["GCOUNT "] = "                   1 / number of groups"
            hdr["ONGRID"]  = str(i) + "," + str(j)
            hdr["NUMSPEC"] = str(np.max(num_galaxies))
            hdr["NORMVAL"] = str(medval)
            hdr["AVGCOL"] = str(avgcol)
            hdr["AVGMASS"] = str(avgmass)
            
            dat = np.array((old_flam, old_flamerr)).reshape(2,len(lam_grid))
            hdulist.append(pf.ImageHDU(data = dat, header=hdr))
            
            row = int(i/color_step)
            column = int((j - 7.0)/mstar_step)
            gal_per_bin[row,column] = gal_current_bin
    
    print added_gal, skipped_gal
    print gal_per_bin
    hdulist.writeto('coadded_PEARSgrismspectra_coarsegrid.fits', clobber=True)
    
    # Total time taken
    logging.info("Time taken for coaddition -- %.2f seconds", time.time() - start)
