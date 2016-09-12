from __future__ import division
import numpy as np
import numpy.random as npr
from astropy.io import fits

import grid_coadd as gd

import sys
import glob
import os

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figures_dir = stacking_analysis_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"

def bootstrap(idx_orig):
    """It is assigning a probability of 1 to each element
       in the sample. Although it explicitly incorporates replacement it is just the same 
       as drawing each element randomly.
    """

    idx = npr.choice(idx_orig, idx_orig, replace=True, p=None)
    return idx

def main():

    home = os.getenv('HOME')

    cat = np.genfromtxt(home + '/Desktop/FIGS/new_codes/color_stellarmass.txt',
                   dtype=None, names=True, skip_header=2)
    
    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    # Change only the parameters here to change how the code runs
    # Ideally you shouldn't have to change anything else.
    col_step = 0.6
    mstar_step = 1.0
    col_low = 0.0
    col_high = 3.0
    mstar_low = 7.0
    mstar_high = 12.0

    # Set up lambda grid
    lam_step = 100
    lam_grid = np.arange(2700, 6000, lam_step)

    num_samp_to_draw = 100

    # Exclude spectra in the skipspec file and also exclude emission line spectra
    skipspec = np.loadtxt(home + '/Desktop/FIGS/stacking-analysis-pears/specskip.txt', dtype=np.str, delimiter=',')
    em_lines = np.loadtxt(home + '/Desktop/FIGS/stacking-analysis-pears/em_lines_readin.txt', dtype=np.str, delimiter=',')
    # This little for loop is to fix formatting issues with the skipspec and em_lines arrays that are read in with loadtxt.
    for i in range(len(skipspec)):
        skipspec[i] = skipspec[i].replace('\'', '')
        em_lines[i] = em_lines[i].replace('\'', '')

    for i in np.arange(col_low, col_high, col_step):
        for j in np.arange(mstar_low, mstar_high, mstar_step):
            
            indices = np.where((ur_color >= i) & (ur_color < i + col_step) &\
                               (stellarmass >= j) & (stellarmass < j + mstar_step))[0]
            total_spectra = len(pears_id[indices])
            print "ONGRID", i, j
            if total_spectra < 5: 
                print "Too few spectra in stack. Continuing to the next grid cell..."
                # if the total possible combinations of spectra i.e. total_spectra**total_spectra < num_samp_to_draw
                # then don't do the bootstrap sampling on these
                # do these separately by going through all possible combinations of the samplings 
                continue

            hdu = fits.PrimaryHDU()
            hdulist = fits.HDUList(hdu)
            hdulist.append(fits.ImageHDU(lam_grid))

            for runcount in range(num_samp_to_draw):
                print "Bootstrap Run --", runcount+1
                new_indx = bootstrap(total_spectra)
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
            hdulist.writeto(savefits_dir + 'bootstrap-err-stacks/' + 'spectra_bootstrap_' + ongrid_w + '.fits', clobber=True)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)