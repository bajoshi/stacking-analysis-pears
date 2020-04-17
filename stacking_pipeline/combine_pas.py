from __future__ import division

import numpy as np
from astropy.io import fits
from scipy import stats

import glob
import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
figs_dir = home + "/Desktop/FIGS/"

pears_spectra_dir = home + "/Documents/PEARS/data_spectra_only/"
stacking_analysis_dir = figs_dir + "stacking-analysis-pears/"
stacking_figures_dir = figs_dir + "stacking-analysis-figures/"

stacking_utils_dir = stacking_analysis_dir + "util_codes"
sys.path.append(stacking_utils_dir)
from get_total_extensions import get_total_extensions

def add_position_angle(lam, flux, contam, fluxerr, old_flux, old_fluxerr, lam_grid, lam_step):

    for i in range(len(lam)):

        # find wavelength indices within lam grid bin
        new_ind = np.where((lam >= lam_grid[i] - lam_step/2) & (lam < lam_grid[i] + lam_step/2))[0]

        if new_ind.size:

            for ind in new_ind:

                signal = flux[ind]
                noise = fluxerr[ind]
                contamination = contam[ind]

                """
                Subtract the contamination for each PA. 
                Must be done before stacking spectra for different galaxies.

                This has to be done at this stage because the contamination 
                estimate is different for each PA, and each wavelength point 
                within each PA as well. Therefore, you can't combine spectra 
                at different PAs and then subtract "some" contamination level,
                which would correspond to some specific PA, unless you somehow 
                averaged the contaminations as well. This would not be advisable 
                because you'd be introducing additional (unrequired) assumptions 
                and pathologies into the process by combining contaminations. 
                For example, say you have two PAs, one contaminated and the other 
                pristine; by combining the contaminations, you'll have made a 
                mess of both of them. Much easier to just subtract contamination
                from each individual PA.
                """
        
                if signal > 0: # only append those points where the signal is positive
                    if (contamination < 0.33 * signal):  # contamination cut
                        signal -= contamination
                        
                        old_flux[i].append(signal)
                        old_fluxerr[i].append(noise**2) # adding errors in quadrature
                else:
                    continue
        else:
            continue

    return old_flux, old_fluxerr

def combine_all_position_angles_pears(pears_index, field):

    # Get the correct filename and the number of extensions
    if field == 'GOODS-N':
        filename = pears_spectra_dir + 'h_pears_n_id' + str(pears_index) + '.fits'
    elif field == 'GOODS-S':
        filename = pears_spectra_dir + 'h_pears_s_id' + str(pears_index) + '.fits'

    spec_hdu = fits.open(filename, memmap=False)
    spec_extens = get_total_extensions(spec_hdu)

    print("Working on PEARS object:", pears_index, field, end='. ')
    print("Has", spec_extens, "PA(s).")

    # Loop over all extensions and combine them.
    old_flam = np.zeros(len(lam_grid))
    old_flamerr = np.zeros(len(lam_grid))
    
    old_flam = old_flam.tolist()
    old_flamerr = old_flamerr.tolist()

    for count in range(spec_extens):
        
        flam = spec_hdu[count + 1].data['FLUX']
        ferr = spec_hdu[count + 1].data['FERROR']
        contam = spec_hdu[count + 1].data['CONTAM']
        lam_obs = spec_hdu[count + 1].data['LAMBDA']
        exten_pa = spec_hdu[count + 1].header['EXTNAME'].split('PA')[1]

        if count == 0.0:
            for x in range(len(lam_grid)):
                old_flam[x] = []
                old_flamerr[x] = []

        # Now add the spectra from this PA to a "running list"
        old_flam, old_flamerr = add_position_angle(lam_obs, flam, contam, ferr, old_flam, old_flamerr, lam_grid, lam_step)

    for y in range(len(lam_grid)):
        if old_flam[y]:
            # you could do a 100 bootstrap samples
            old_flamerr[y] = np.sqrt(np.sum(old_flamerr[y]))
            # old formula for error on median -- 1.253 * np.std(old_flam[y]) / np.sqrt(len(old_flam[y]))
            old_flam[y] = np.mean(old_flam[y])
        else:
            # this shoudl not be set to 0
            # you DID NOT measure an exactly zero signal
            old_flam[y] = np.nan
            old_flamerr[y] = np.nan

    comb_flam = np.asarray(old_flam)
    comb_flamerr = np.asarray(old_flamerr)

    # close opened fits file
    spec_hdu.close()

    return lam_grid, comb_flam, comb_flamerr

def plot_all_pa_and_combinedspec(allpearsids, allpearsfields):
    """
    This function will plot, for all the pears galaxies supplied, 
    the spectra from individual PAs and the combined spectrum 
    overlaid on top.

    Make sure that there is a one-to-one correspondence between the 
    id array and the field array supplied.
    """

    totalgalaxies = len(allpearsids)

    # Plot all PAs and plot the median
    for u in range(totalgalaxies):

        pears_index = allpearsids[u]
        field = allpearsfields[u]

        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        # Get the correct filename and the number of extensions
        if field == 'GOODS-N':
            filename = pears_spectra_dir + 'h_pears_n_id' + str(pears_index) + '.fits'
        elif field == 'GOODS-S':
            filename = pears_spectra_dir + 'h_pears_s_id' + str(pears_index) + '.fits'
    
        fitsfile = fits.open(filename)
        fits_extens = get_total_extensions(fitsfile)

        for j in range(fits_extens):
            ax.plot(fitsfile[j+1].data['LAMBDA'], fitsfile[j+1].data['FLUX'])

        lam_obs, combined_spec, combined_spec_err = combine_all_position_angles_pears(pears_index, field)
        ax.plot(lam_obs, combined_spec, color='k', linewidth=2)
        ax.fill_between(lam_obs, combined_spec + combined_spec_err, combined_spec - combined_spec_err, color='lightgray')

        ax.axhline(y=0.0, lw=1.5, ls='--', color='r')

        ax.minorticks_on()
        ax.tick_params('both', width=1, length=3, which='minor')
        ax.tick_params('both', width=1, length=4.7, which='major')

        #print("\n")
        #for k in range(len(lam_obs)):
        #    print(lam_obs[k], combined_spec[k], combined_spec_err[k])

        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    return None

if __name__ == '__main__':

    # ------------------------------- Read in PEARS catalogs ------------------------------- #
    pears_ncat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['pearsid', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['pearsid', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['dec'] = pears_ncat['dec'] - dec_offset_goodsn_v19

    allcats = [pears_ncat, pears_scat]

    # First set up a wavelength grid.
    # This grid is extended on both ends on purpose to
    # account for data at the ends of the grism coverage.
    # The ends will of course be chopped later before
    # doing any analysis but I don't want to do it here.
    lam_step = 40
    lam_grid = np.arange(5500, 10540, lam_step)

    catcount = 0
    for cat in allcats:

        if catcount == 0:
            fieldname = 'GOODS-N'
        elif catcount == 1:
            fieldname = 'GOODS-S'

        # Loop over all spectra 
        for i in range(len(cat)):

            current_pears_index = cat['pearsid'][i]

            lam_grid, comb_flam, comb_flamerr = combine_all_position_angles_pears(current_pears_index, fieldname)
            #plot_all_pa_and_combinedspec([current_pears_index], [fieldname])

            # Write the combined PA spectrum to a fits file
            hdu = fits.PrimaryHDU()
            hdr = fits.Header()
            hdulist = fits.HDUList(hdu)
            hdr['EXTNAME'] = 'LAM_GRID'
            hdulist.append(fits.ImageHDU(lam_grid, header=hdr))

            # Add info to header
            hdr['EXTNAME'] = 'COMB_SPEC'
            hdr['PEARSID'] = current_pears_index
            hdr['FIELD'] = fieldname
            hdr['RA'] = cat['ra'][i]
            hdr['DEC'] = cat['dec'][i]

            # Add data
            dat = np.array((comb_flam, comb_flamerr)).reshape(2, len(lam_grid))
            hdulist.append(fits.ImageHDU(dat, header=hdr))

            # Write file
            final_fits_filename = pears_spectra_dir + fieldname + '_' + str(current_pears_index) + '_PAcomb.fits'
            hdulist.writeto(final_fits_filename, overwrite=True)

        catcount += 1

    sys.exit(0)
