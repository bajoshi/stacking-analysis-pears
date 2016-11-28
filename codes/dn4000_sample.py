from __future__ import division

import numpy as np
from astropy.io import fits

import sys
import os

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
massive_figures_dir = massive_galaxies_dir + "figures/"
newcodes_dir = home + "/Desktop/FIGS/new_codes/"

if __name__ == '__main__':
    
    # read in dn4000 catalogs 
    pears_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/pears_4000break_catalog.txt',\
     dtype=None, names=True, skip_header=1)
    gn1_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/figs_gn1_4000break_catalog.txt',\
     dtype=None, names=True, skip_header=1)
    gn2_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/figs_gn2_4000break_catalog.txt',\
     dtype=None, names=True, skip_header=1)
    gs1_cat = np.genfromtxt(home + '/Desktop/FIGS/stacking-analysis-pears/figs_gs1_4000break_catalog.txt',\
     dtype=None, names=True, skip_header=1)

    pears_redshift_indices = np.where((pears_cat['redshift'] >= 0.558) & (pears_cat['redshift'] <= 1.317))[0]

    # galaxies in the possible redshift range
    #print len(pears_redshift_indices)  # 2318

    # galaxies that are outside the redshift range
    # not sure how these originally got into the pears and 3dhst matched sample....need to check again
    # these were originally selected to be within the above written redshift range
    #print np.setdiff1d(np.arange(len(pears_cat)), pears_redshift_indices)  # [1136 2032 2265]

    # galaxies with significant breaks
    sig_4000break_indices_pears = np.where(((pears_cat['dn4000'] / pears_cat['dn4000_err']) >= 5.0) &\
        ((pears_cat['dn4000'] / pears_cat['dn4000_err']) <= 20.0))[0]

    # Galaxies with believable breaks; im calling them proper breaks
    prop_4000break_indices_pears = \
    np.where((pears_cat['dn4000'][sig_4000break_indices_pears] >= 1.2) & \
        (pears_cat['dn4000'][sig_4000break_indices_pears] <= 2.5))[0]

    #print len(prop_4000break_indices_pears)  # 477

    #### FIGS ####

    # galaxies with significant breaks
    sig_4000break_indices_gn1 = np.where(((gn1_cat['dn4000'] / gn1_cat['dn4000_err']) >= 5.0) &\
        ((gn1_cat['dn4000'] / gn1_cat['dn4000_err']) <= 20.0))[0]
    sig_4000break_indices_gn2 = np.where(((gn2_cat['dn4000'] / gn2_cat['dn4000_err']) >= 5.0) &\
        ((gn2_cat['dn4000'] / gn2_cat['dn4000_err']) <= 20.0))[0]
    sig_4000break_indices_gs1 = np.where(((gs1_cat['dn4000'] / gs1_cat['dn4000_err']) >= 5.0) &\
        ((gs1_cat['dn4000'] / gs1_cat['dn4000_err']) <= 20.0))[0]

    # Galaxies with believable breaks
    prop_4000break_indices_gn1 = \
    np.where((gn1_cat['dn4000'][sig_4000break_indices_gn1] >= 1.2) & \
        (gn1_cat['dn4000'][sig_4000break_indices_gn1] <= 2.5))[0]
    prop_4000break_indices_gn2 = \
    np.where((gn2_cat['dn4000'][sig_4000break_indices_gn2] >= 1.2) & \
        (gn2_cat['dn4000'][sig_4000break_indices_gn2] <= 2.5))[0]
    prop_4000break_indices_gs1 = \
    np.where((gs1_cat['dn4000'][sig_4000break_indices_gs1] >= 1.2) & \
        (gs1_cat['dn4000'][sig_4000break_indices_gs1] <= 2.5))[0]

    #print len(prop_4000break_indices_gn1)  # 6
    #print len(prop_4000break_indices_gn2)  # 7
    #print len(prop_4000break_indices_gs1)  # 5

    #print len(sig_4000break_indices_pears)  # 1226
    #print len(sig_4000break_indices_gn1)  # 33
    #print len(sig_4000break_indices_gn2)  # 37
    #print len(sig_4000break_indices_gs1)  # 28

    #### SDSS ####
    # these are from SDSS DR8
    galspecindx = fits.open(newcodes_dir + 'galSpecIndx-dr8.fits')
    galspecinfo = fits.open(newcodes_dir + 'galSpecInfo-dr8.fits')

    # get dn4000 and redshift arrays
    dn4000_sdss = galspecindx[1].data['D4000_N_SUB']
    dn4000_err_sdss = galspecindx[1].data['D4000_N_SUB_ERR']
    redshift_sdss = galspecinfo[1].data['Z']

    # apply basic cuts
    sdss_use_indx = np.where((galspecinfo[1].data['Z_WARNING'] == 0) &\
     (galspecinfo[1].data['TARGETTYPE'] == 'GALAXY') & (galspecinfo[1].data['SPECTROTYPE'] == 'GALAXY') &\
     (galspecinfo[1].data['RELIABLE'] == 1))[0]

    print len(sdss_use_indx)  # 869037

    # apply more cuts i.e. significance and break range
    sig_4000break_indices_sdss = np.where(((dn4000_sdss[sdss_use_indx] / dn4000_err_sdss[sdss_use_indx]) >= 5.0) &\
     ((dn4000_sdss[sdss_use_indx] / dn4000_err_sdss[sdss_use_indx]) <= 20.0))[0]
    dn4000_sdss_sig = dn4000_sdss[sdss_use_indx][sig_4000break_indices_sdss]
    dn4000_sdss_range_indx = np.where((dn4000_sdss_sig >= 0) & (dn4000_sdss_sig <= 3))[0]
    dn4000_sdss_plot = dn4000_sdss[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx]

    redshift_sdss_plot = redshift_sdss[sdss_use_indx][sig_4000break_indices_sdss][dn4000_sdss_range_indx]

    print len(dn4000_sdss_plot)  # 866883

    # Make sure that all the galaxies in the sample are unique
    # there is the possibility of duplicate entries only in hte redshift
    # range where PEARS and FIGS overlap i.e. 1.2 <= z <= 1.32
    # the catalog will somehow have to be matched with itself?
    # it can't just be np.unique. 
    print len(pears_cat)
    print len(np.unique(pears_cat['ra']))

    # could you somehow demonstrate that the dn4000 that you get is the same that 
    # other studies got.
    # there are some sdss galaxies with the same redshifts as PEARS. 
    # IDK if there is any overlap in the fields?
    # You could also download a convincing number of SDSS spectra and run your code on them
    # and see if you get the same number for dn4000 as they do.
    

    # ------------------------- PLOTS ------------------------- #

    # make histograms of dn4000 distribution for PEARS and FIGS 
    # also showing (shaded or with vertical lines) the values of 
    # dn4000 that are selected in the final sample
    # also show the numbers on top of the bars

    # PEARS dn4000 histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # create proper array for plotting
    dn4000_pears = pears_cat['dn4000'][sig_4000break_indices_pears]
    dn4000_pears_plot = dn4000_pears[np.where(dn4000_pears <= 3)[0]]
    #print len(np.where(dn4000_pears > 3)[0])  # 10
    print len(dn4000_pears_plot)
    # There are only 10 out of 1226 values that are greater than 3 so 
    # I won't plot them just for the sake of making my plot look a little better.

    # get total bins and plot histogram
    iqr = np.std(dn4000_pears_plot, dtype=np.float64)
    binsize = 2*iqr*np.power(len(dn4000_pears_plot),-1/3)
    totalbins = np.floor((max(dn4000_pears_plot) - min(dn4000_pears_plot))/binsize)

    ncount, edges, patches = ax.hist(dn4000_pears_plot, totalbins, color='lightgray', align='mid')
    ax.grid(True)

    # shade the selection region
    edges_plot = np.where((edges >= 1.2) & (edges <= 2.5))[0]
    patches_plot = [patches[edge_ind] for edge_ind in edges_plot]
    col = np.full(len(patches_plot), 'lightblue', dtype='|S9')  
    # make sure the length of the string given in the array initialization is the same as the color name
    for c, p in zip(col, patches_plot):
        plt.setp(p, 'facecolor', c)

    # save figure
    fig.savefig(massive_figures_dir + 'pears_dn4000_hist.eps', dpi=300)

    # FIGS dn4000 histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # create proper array for plotting
    dn4000_gn1 = gn1_cat['dn4000'][sig_4000break_indices_gn1]
    dn4000_gn2 = gn2_cat['dn4000'][sig_4000break_indices_gn2]
    dn4000_gs1 = gs1_cat['dn4000'][sig_4000break_indices_gs1]

    dn4000_figs = np.concatenate((dn4000_gn1, dn4000_gn2, dn4000_gs1))
    dn4000_figs_plot = dn4000_figs[np.where(dn4000_figs <= 3)[0]]
    #print len(np.where(dn4000_figs > 3)[0])  # 12
    print len(dn4000_figs_plot)
    # Same thing done as PEARS. I did not plot the 12 out of 98 objects 
    # that have dn4000 values greater than 3 so that my plot can look better

    # get total bins and plot histogram
    #iqr = np.std(dn4000_figs_plot, dtype=np.float64)
    #binsize = 2*iqr*np.power(len(dn4000_figs_plot),-1/3)
    #totalbins = np.floor((max(dn4000_figs_plot) - min(dn4000_figs_plot))/binsize)
    # the freedman-diaconis rule here does not seem to give me a proper number of bins.
    # which it generally does not for arrays that do not have a large number of elements.
    totalbins = 3

    ncount, edges, patches = ax.hist(dn4000_figs_plot, totalbins, color='lightgray', align='mid')
    ax.grid(True)    
   
    ## shade the selection region
    #edges_plot = np.where((edges >= 1.2) & (edges <= 2.5))[0]
    #patches_plot = [patches[edge_ind] for edge_ind in edges_plot]
    #col = np.full(len(patches_plot), 'lightblue', dtype='|S9')  
    ## make sure the length of the string given in the array initialization is the same as the color name
    #for c, p in zip(col, patches_plot):
    #    plt.setp(p, 'facecolor', c)

    # save figure
    fig.savefig(massive_figures_dir + 'figs_dn4000_hist.eps', dpi=300)    

    # dn4000 vs redshift 
    # This should include the big clud of SDSS points and 
    # any other points that you can find which give dn4000 for the whole redshift range

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get proper redshift and dn4000_err arrays for PEARS and FIGS for plotting
    redshift_pears = pears_cat['redshift'][sig_4000break_indices_pears]
    redshift_pears_plot = redshift_pears[np.where(dn4000_pears <= 3)[0]]

    dn4000_err_pears = pears_cat['dn4000_err'][sig_4000break_indices_pears]
    dn4000_err_pears_plot = dn4000_err_pears[np.where(dn4000_pears <= 3)[0]]

    redshift_gn1 = gn1_cat['redshift'][sig_4000break_indices_gn1]
    redshift_gn2 = gn2_cat['redshift'][sig_4000break_indices_gn2]
    redshift_gs1 = gs1_cat['redshift'][sig_4000break_indices_gs1]
    redshift_figs = np.concatenate((redshift_gn1, redshift_gn2, redshift_gs1))
    redshift_figs_plot = redshift_figs[np.where(dn4000_figs <= 3)[0]]

    dn4000_err_gn1 = gn1_cat['dn4000_err'][sig_4000break_indices_gn1]
    dn4000_err_gn2 = gn2_cat['dn4000_err'][sig_4000break_indices_gn2]
    dn4000_err_gs1 = gs1_cat['dn4000_err'][sig_4000break_indices_gs1]
    dn4000_err_figs = np.concatenate((dn4000_err_gn1, dn4000_err_gn2, dn4000_err_gs1))
    dn4000_err_figs_plot = dn4000_err_figs[np.where(dn4000_figs <= 3)[0]] 

    #ax.plot(redshift_pears_plot, dn4000_pears_plot, 'o', markersize=2, color='k', markeredgecolor='k')
    #ax.plot(redshift_figs_plot, dn4000_figs_plot, 'o', markersize=2, color='b', markeredgecolor='b')

    ax.errorbar(redshift_pears_plot, dn4000_pears_plot, yerr=dn4000_err_pears_plot,\
     fmt='.', color='k', markeredgecolor='k', capsize=0, markersize=7)
    ax.errorbar(redshift_figs_plot, dn4000_figs_plot, yerr=dn4000_err_figs_plot,\
     fmt='.', color='b', markeredgecolor='b', capsize=0, markersize=7)

    ax.plot(redshift_sdss_plot, dn4000_sdss_plot, '.', markersize=1, color='lightgray')

    ax.axhline(y=1, linewidth=1, linestyle='--', color='g')

    ax.set_xlim(0,1.85)

    # save the figure
    fig.savefig(massive_figures_dir + 'dn4000_redshift.png')

    sys.exit(0)
