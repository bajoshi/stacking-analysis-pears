from __future__ import division
import os, sys, time, datetime
import logging

import numpy as np
from astropy.io import fits
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
        #logging.warning(filename)
        #logging.warning("Division by zero! The net sig here cannot be trusted. Setting Net Sig to -99.")
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
            # figure out if sth else is needed to be done here if it cannot find any points to coadd
            # logging.warning("In spectrum -- %s at %d.  No points found for coaddition! Continuing...", specname, lam_grid[i])
            # This warning line produces too many lines in the log and bloats up the log file.
            
            continue

    return old_flam, old_flamerr, num_points, num_galaxies

def divide_sensitivity(flam_obs, ferr, lam_obs):
    
    midval = np.mean(np.array(lam_obs[0], lam_obs[1]))
        
    sens_ind = np.where((senslam <= midval) & (senslam >= lam_obs[0]))[0]
    flam_obs[0] = flam_obs[0] / np.mean(sens[sens_ind])
    
    for i in range(1, len(lam_obs) - 1):
        
        midval_up = np.mean([lam_obs[i], lam_obs[i+1]])
        midval_down = np.mean([lam_obs[i], lam_obs[i-1]])
        
        sens_ind = np.where((senslam >= midval_down) & (senslam < midval_up))[0]
        flam_obs[i] = flam_obs[i] / np.mean(sens[sens_ind])

    i += 1
    midval = np.mean(np.array(lam_obs[i], lam_obs[i - 1]))
    
    sens_ind = np.where((senslam >= midval) & (senslam <= lam_obs[i]))[0]
    flam_obs[i] = flam_obs[i] / np.mean(sens[sens_ind])

    flam_obs[np.isnan(flam_obs)] = 0.0

    return flam_obs, ferr

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

def fileprep(pears_index, redshift):

    data_path = "/Users/baj/Documents/PEARS/data_spectra_only/"
    # Get the correct filename and the number of extensions
    filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    if not os.path.isfile(filename):
        filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

    fitsfile = fits.open(filename)
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
    threed = fits.open('/Users/baj/Documents/3D-HST/3dhst.v4.1.5.master.fits')

    """
    # This block sets up the division by the sensitivity curve
    sens_curve = np.genfromtxt('ACS.WFC.1st.sens.7.dat', dtype=None, names=True, skip_header=4)
    # sensitivity is for the 1st order spectrum
    # sensitivity given in (e s-1)/(erg s-1 cm-2 A-1)
    senslam = sens_curve['Lambda']
    sens = sens_curve['sensitivity']
    senserr = sens_curve['error']
    # rescale
    sensmax = max(sens)
    sens = sens / sensmax
    senserr = senserr / sensmax
    """
    
    logging.basicConfig(filename='coadd.log', format='%(levelname)s:%(message)s', filemode='w', level=logging.DEBUG)
    logging.info("\n This file is overwritten every time the program runs.\n You will have to change the filemode to log messages for every run.")
    
    dt = datetime.datetime
    logging.info("Coaddition started at --")
    logging.info(dt.now())
    
    cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/color_stellarmass.txt',
                       dtype=None, names=True, skip_header=2)
    
    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    #print len(pears_id) # 2032
    #print len(np.unique(pears_id)) # 1783
    
    lam_step = 100
    lam_grid = np.arange(2700, 6000, lam_step)
    # Lambda grid decided based on observed wavelength range i.e. 6000 to 9500
    # and the initially chosen redshift range 0.6 < z < 1.2
    # This redshift range was chosen so that the 4000A break would fall in the observed wavelength range
    
    # Change only the parameters here to change how the code runs
    # Ideally you shouldn't have to change anything else.
    col_step = 0.3
    mstar_step = 0.5
    final_fits_filename = 'coadded_PEARSgrismspectra.fits'
    col_low = 0.0
    col_high = 3.0
    mstar_low = 7.0
    mstar_high = 12.0
    
    hdu = fits.PrimaryHDU()
    hdr = fits.Header()
    hdulist = fits.HDUList(hdu)
    hdulist.append(fits.ImageHDU(lam_grid, header=hdr))
    
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

    em_lines = ['h_pears_n_id90194.fits', 'h_pears_s_id17321.fits', 'h_pears_s_id106641.fits', 'h_pears_s_id106993.fits',\
    'h_pears_s_id110400.fits', 'h_pears_s_id21196.fits', 'h_pears_s_id94632.fits', 'h_pears_n_id42058.fits', 'h_pears_n_id105902.fits',\
    'h_pears_n_id118534.fits', 'h_pears_s_id26909.fits', 'h_pears_n_id113595.fits', 'h_pears_s_id16788.fits', 'h_pears_s_id64769.fits',\
    'h_pears_s_id121506.fits', 'h_pears_s_id122913.fits', 'h_pears_n_id44207.fits', 'h_pears_n_id50210.fits', 'h_pears_n_id63410.fits',\
    'h_pears_n_id72081.fits', 'h_pears_n_id84467.fits', 'h_pears_n_id85556.fits', 'h_pears_n_id108010.fits', 'h_pears_n_id107999.fits',\
    'h_pears_n_id108239.fits', 'h_pears_s_id17587.fits', 'h_pears_s_id22486.fits', 'h_pears_s_id40117.fits', 'h_pears_s_id59595.fits',\
    'h_pears_s_id63612.fits', 'h_pears_s_id64393.fits', 'h_pears_s_id65179.fits', 'h_pears_s_id68852.fits', 'h_pears_s_id94873.fits',\
    'h_pears_s_id98046.fits', 'h_pears_s_id98242.fits', 'h_pears_s_id107055.fits', 'h_pears_s_id109596.fits', 'h_pears_s_id121911.fits',\
    'h_pears_s_id109435.fits', 'h_pears_s_id119341.fits', 'h_pears_s_id122735.fits', 'h_pears_s_id129968.fits', 'h_pears_s_id17587.fits',\
    'h_pears_s_id22486.fits', 'h_pears_s_id68852.fits', 'h_pears_s_id82307.fits', 'h_pears_n_id82356.fits', 'h_pears_s_id90246.fits',\
    'h_pears_s_id94873.fits', 'h_pears_n_id35090.fits', 'h_pears_n_id42526.fits', 'h_pears_n_id66045.fits', 'h_pears_n_id82693.fits',\
    'h_pears_n_id93025.fits', 'h_pears_s_id18882.fits', 'h_pears_s_id21363.fits', 'h_pears_s_id25474.fits', 'h_pears_s_id27652.fits',\
    'h_pears_s_id40163.fits', 'h_pears_s_id41078.fits', 'h_pears_s_id43170.fits', 'h_pears_s_id46284.fits', 'h_pears_s_id50298.fits',\
    'h_pears_s_id51976.fits', 'h_pears_s_id65708.fits', 'h_pears_s_id69168.fits', 'h_pears_s_id69234.fits', 'h_pears_s_id104478.fits',\
    'h_pears_s_id104514.fits', 'h_pears_s_id105015.fits', 'h_pears_s_id107052.fits', 'h_pears_s_id113279.fits', 'h_pears_s_id117686.fits',\
    'h_pears_s_id121974.fits', 'h_pears_s_id21614.fits', 'h_pears_s_id79520.fits', 'h_pears_s_id81609.fits', 'h_pears_s_id83804.fits',\
    'h_pears_s_id90198.fits', 'h_pears_s_id91382.fits', 'h_pears_n_id39842.fits', 'h_pears_n_id47252.fits', 'h_pears_n_id61065.fits',\
    'h_pears_n_id90145.fits', 'h_pears_n_id84077.fits', 'h_pears_n_id112486.fits', 'h_pears_n_id110727.fits', 'h_pears_n_id121428.fits',\
    'h_pears_s_id31086.fits', 'h_pears_s_id32905.fits', 'h_pears_s_id118526.fits', 'h_pears_n_id75917.fits', 'h_pears_n_id54384.fits',\
    'h_pears_n_id109561.fits', 'h_pears_n_id124893.fits', 'h_pears_s_id51293.fits', 'h_pears_s_id98951.fits', 'h_pears_s_id100652.fits',\
    'h_pears_s_id124410.fits', 'h_pears_s_id125699.fits', 'h_pears_s_id133306.fits', 'h_pears_s_id78045.fits', 'h_pears_s_id87735.fits',\
    'h_pears_n_id33944.fits', 'h_pears_n_id41064.fits', 'h_pears_n_id77596.fits', 'h_pears_n_id81988.fits', 'h_pears_n_id81946.fits',\
    'h_pears_n_id84715.fits', 'h_pears_s_id52445.fits', 'h_pears_s_id104446.fits', 'h_pears_s_id107036.fits', 'h_pears_s_id118673.fits',\
    'h_pears_s_id71864.fits', 'h_pears_s_id73385.fits', 'h_pears_s_id76154.fits', 'h_pears_s_id92860.fits', 'h_pears_n_id91947.fits',\
    'h_pears_n_id101185.fits', 'h_pears_n_id119723.fits', 'h_pears_n_id125948.fits', 'h_pears_s_id35878.fits', 'h_pears_s_id38417.fits',\
    'h_pears_s_id103422.fits', 'h_pears_s_id115331.fits', 'h_pears_s_id120803.fits', 'h_pears_s_id121733.fits', 'h_pears_s_id126769.fits',\
    'h_pears_s_id21647.fits', 'h_pears_s_id80500.fits', 'h_pears_s_id83686.fits', 'h_pears_n_id42014.fits', 'h_pears_s_id109992.fits',\
    'h_pears_s_id124964.fits', 'h_pears_s_id78077.fits', 'h_pears_s_id69713.fits', 'h_pears_s_id87679.fits', 'h_pears_n_id38344.fits',\
    'h_pears_s_id71524.fits', 'h_pears_n_id123476.fits', 'h_pears_s_id119489.fits', 'h_pears_s_id109167.fits', 'h_pears_s_id110397.fits',\
    'h_pears_s_id14215.fits', 'h_pears_s_id117429.fits', 'h_pears_n_id53563.fits', 'h_pears_s_id62528.fits', 'h_pears_n_id114628.fits',\
    'h_pears_s_id114108.fits', 'h_pears_s_id124313.fits', 'h_pears_s_id119193.fits']
                
    added_gal = 0
    skipped_gal = 0
    gal_per_bin = np.zeros((len(np.arange(col_low, col_high, col_step)), len(np.arange(mstar_low, mstar_high, mstar_step))))
    
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
                lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][u], redshift)
                
                """
                # Divide by the grism sensitivity curve
                # Wouldn't aXe do this itself when it extracted the PEARS spectra??
                # yes aXe has done it, if you use the spectrum in cgs units
                # but if you use the spectrum in counts/s then you'll need to divide by the sensitivity curve
                # Therefore, not needed right now
                # flam_obs, ferr = divide_sensitivity(flam_obs, ferr, lam_obs)
                """
                
                # Divide by median value at 4400A to 4600A to rescale. Multiplying by median value of the flux medians to get it back to physical units
                flam_em = (flam_em / medarr[u]) * medval
                ferr = (ferr / medarr[u]) * medval
                
                """
                # Set any negative points in spectrum to 0.
                # If more than 30% of the points in a spectrum are negative then skip it.
                if np.any(flam_em < 0.0):
                    if len(np.where(flam_em < 0.0)[0]) > 0.3 * len(flam_em):
                        continue
                    flam_em_neg_ind = np.where(flam_em < 0.0)[0]
                    flam_em[flam_em_neg_ind] = 0.0
                    ferr[flam_em_neg_ind] = 0.0
                """

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
                    old_flam, old_flamerr, num_points, num_galaxies = add_spec(specname, lam_em, flam_em, ferr, old_flam, old_flamerr, num_points, num_galaxies, lam_grid, lam_step)
            
                # Now intepolate the spectrum on to the lambda grid
                #interp_spec, interp_spec_err = get_interp_spec(lam_em, flam_em, ferr)
    
                #counts += interp_spec
                #countserr += interp_spec_err**2

            # taking median
            # maybe I should also try doing a mean after 3sigma clipping and compare
            for y in range(len(lam_grid)):
                if old_flam[y]:
                    old_flamerr[y] = 1.253 * np.std(old_flam[y]) / np.sqrt(len(old_flam[y]))
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
                   
            dat = np.array((old_flam, old_flamerr)).reshape(2,len(lam_grid))
            hdulist.append(fits.ImageHDU(data = dat, header=hdr))

            row = int(i/col_step)
            column = int((j - mstar_low)/mstar_step)
            gal_per_bin[row,column] = gal_current_bin

            print "ONGRID", i, j, "added", gal_current_bin, "spectra."
            print '\n'
    
    print added_gal, skipped_gal
    print np.flipud(gal_per_bin)
    print np.sum(gal_per_bin, axis=None)
    hdulist.writeto(final_fits_filename, clobber=True)
    
    # Total time taken
    logging.info("Time taken for coaddition -- %.2f seconds", time.time() - start)
    print "Total time taken", time.time() - start, "seconds"
