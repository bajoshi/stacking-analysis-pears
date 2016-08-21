from __future__ import division
import numpy as np
import numpy.random as npr
from astropy.io import fits

import matplotlib.pyplot as plt

import grid_coadd as gd
import sys

def bootstrap(total_spectra):
    """
       This is the first version of bootstrap that I wrote.
       As the total number of samples drawn increases the probability 
       of getting any integer the same number of times as the total samples drawn 
       approaches 1.
       i.e. There are very few repetitions within any given sample drawn from here.
       So it seems like this version of bootstrap is not what I want because I'd 
       like more repetition to get reliable error bars. If there isn't much repetition
       then its just giving me my original data sample over and over again with the 
       indices shuffled which will give the same best fit parameters and no new
       knowledge.

       The code used to test the above written statements is ---
           fig = plt.figure()
           ax = fig.add_subplot(111)

           num_samp_to_draw = # define
           total_spectra = # define
       
           arr = np.empty((num_samp_to_draw, total_spectra))
           for i in range(num_samp_to_draw):
               arr[i] = bootstrap(total_spectra)
       
           ax.hist(arr.flatten(), total_spectra, facecolor='b', alpha=0.5)
       
           plt.show()
    """

    idx = npr.randint(0, total_spectra, total_spectra)
    # returns random integers
    # for the above line -- first arg is min val, second is max val, and 
    # third is total number of random values required.
    return idx

def bootstrap1(idx_orig):
    """
       As it is written here, I do not see a difference between this one and the previous
       bootstrap func.
       This seems right because here it is assigning a probability of 1 to each element
       in the sample. Although it explicitly incorporates replacement it is just the same 
       as drawing each element randomly.
    """

    idx = npr.choice(idx_orig, idx_orig, replace=True, p=None)
    return idx

if __name__ == '__main__':

    cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/color_stellarmass.txt',
                   dtype=None, names=True, skip_header=2)
    
    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    col_step = 0.3
    mstar_step = 0.5
    lam_step = 100
    lam_grid = np.arange(2700, 6000, lam_step)

    num_samp_to_draw = 1000
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

    for i in np.arange(2.1, 2.4, col_step):
        for j in np.arange(10.5, 11.0, mstar_step):
            
            indices = np.where((ur_color >= i) & (ur_color < i + 0.3) &\
                            (stellarmass >= j) & (stellarmass < j + 0.5))[0]
            total_spectra = len(pears_id[indices])
            print "ONGRID", i, j
            if total_spectra < 5: 
                # if the total possible combinations of spectra i.e. total_spectra**total_spectra < num_samp_to_draw
                # then don't do the bootstrap sampling on these
                # do these separately by going through all possible combinations of the samplings 
                continue

            hdu = fits.PrimaryHDU()
            hdulist = fits.HDUList(hdu)
            hdulist.append(fits.ImageHDU(lam_grid))

            for runcount in range(num_samp_to_draw):
                print "Bootstrap Run --", runcount+1
                new_indx = bootstrap1(total_spectra)
                curr_pearsid = pears_id[indices][new_indx]
                curr_z = photz[indices][new_indx]

                old_flam = np.zeros(len(lam_grid))
                old_flamerr = np.zeros(len(lam_grid))
                num_points = np.zeros(len(lam_grid))
                num_galaxies = np.zeros(len(lam_grid))
                
                old_flam = old_flam.tolist()
                old_flamerr = old_flamerr.tolist()

                medarr, medval, stdval = gd.rescale(curr_pearsid, curr_z)

                for u in range(len(curr_pearsid)):
    
                    if u == 0.0:
                        for x in range(len(lam_grid)):
                            old_flam[x] = []
                            old_flamerr[x] = []
    
                    redshift = curr_z[u]
                    lam_em, flam_em, ferr, specname = gd.fileprep(curr_pearsid[u], redshift)
    
                    flam_em = (flam_em / medarr[u]) * medval
                    ferr = (ferr / medarr[u]) * medval
    
                    if specname in skipspec:
                        #print "Skipped", specname
                        continue

                    # Reject spectrum if overall contamination too high
                    if np.sum(abs(ferr)) > 0.3 * np.sum(abs(flam_em)):
                        #print "Skipped", specname
                        continue
                    else:
                        old_flam, old_flamerr, num_points, num_galaxies =\
                            gd.add_spec(specname, lam_em, flam_em, ferr, old_flam, old_flamerr, num_points, num_galaxies, lam_grid, lam_step)
    
                for y in range(len(lam_grid)):
                    if old_flam[y]:
                        old_flamerr[y] = 1.253 * np.std(old_flam[y]) / np.sqrt(len(old_flam[y]))
                        old_flam[y] = np.median(old_flam[y])
                    else:
                        old_flam[y] = 0.0
                        old_flamerr[y] = 0.0

                hdr = fits.Header()
                hdr["ONGRID"]  = str(i) + "," + str(j)
                hdr["NUMSPEC"] = str(len(curr_pearsid))
                hdr["NORMVAL"] = str(medval)
                dat = np.array((old_flam, old_flamerr)).reshape(2,len(lam_grid))
                hdulist.append(fits.ImageHDU(data = dat, header=hdr))

            ongrid_w = str(i).replace('.','p') + '_' + str(j).replace('.','p')
            hdulist.writeto('spectra_newbootstrap_' + ongrid_w + '.fits', clobber=True)

    """
        you will need to run grid_coadd's  methods t ostack the spectra chosen in any bootstrap run.
        Save these new stacks to new fits files.
        i.e. Say you have 100 bootstrap runs then save the 100 resampled stacks to a single
        multi-extension fits file.
        Now access each  one of these separately to get best fit parametr for each indiv stack
        and a distribution of best fit params for a given fits file.
    """

