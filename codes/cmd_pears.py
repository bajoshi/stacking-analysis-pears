from __future__ import division

import numpy as np
from astropy.io import fits as pf

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
pgf_preamble = {"pgf.texsystem": "pdflatex"}
mpl.rcParams.update(pgf_preamble)

home = os.getenv('HOME')

def get_color_stellarmass(matchedfile, fieldname, ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master):

    count = 0

    for i in range(len(matchedfile)):
        cat = matchedfile
        current_id = cat['pearsid'][i]
        threedid = cat['threed_hst_id'][i]
        
        threedra = cat['threedra'][i]
        threeddec = cat['threeddec'][i]
        threed_zphot = cat['threed_zphot'][i]
        
        threedindex = np.where((threed[1].data['phot_id'] == threedid) & (abs(threed[1].data['ra'] - threedra) < 1e-3) & (abs(threed[1].data['dec'] - threeddec) < 1e-3))[0][0]

        mstar = threed[1].data[threedindex]['lmass']
        urcol = -2.5*np.log10(threed[1].data[threedindex]['L156']/threed[1].data[threedindex]['L158'])
        
        ur_color.append(urcol)
        stellarmass.append(mstar)

        if (mstar >= 10.5) and (threed_zphot >= 0.5) and (threed_zphot <= 2.0) and (fieldname in all_fields_northnames):
          mastercat_idx = np.where(pears_north_master['pears_id'] == current_id)
          current_imag = '{:.2f}'.format(pears_north_master['imag'][mastercat_idx][0])
          if current_imag <= 20.0:
              print fieldname, current_id, threedra, threeddec, threed_zphot, mstar, '{:.2f}'.format(urcol), current_imag
              imag_arr.append(current_imag)
              zphot_arr.append('{:.2f}'.format(threed_zphot))
              count += 1

    print "total galaxies in chosen range", count, "for field", fieldname

    return ur_color, stellarmass, imag_arr, zphot_arr

if __name__ == '__main__':

    # read in all matched files
    cdfn1 = np.genfromtxt(home + '/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn1.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfn2 = np.genfromtxt(home + '/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn2.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfn3 = np.genfromtxt(home + '/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn3.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfn4 = np.genfromtxt(home + '/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn4.txt',\
                          dtype=None, skip_header=3, names=True)

    cdfs1 = np.genfromtxt(home + '/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs1.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfs2 = np.genfromtxt(home + '/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs2.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfs3 = np.genfromtxt(home + '/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs3.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfs4 = np.genfromtxt(home + '/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs4.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfs_new = np.genfromtxt(home + '/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs_new.txt',\
                             dtype=None, skip_header=3, names=True)
    cdfs_udf = np.genfromtxt(home + '/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs_udf.txt',\
                             dtype=None, skip_header=3, names=True)

    # read in master catalogs to get magnitudes
    pears_master_cat_names = ['pears_id', 'ra', 'dec', 'imag', 'imagerr', 'netsig_raw', 'netsig_corr']
    pears_north_master = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_north_master.cat', \
      dtype=None, names=pears_master_cat_names, skip_header=7)
    pears_south_master = np.genfromtxt(home + '/Documents/PEARS/master_catalogs/h_pears_south_master.cat', \
      dtype=None, names=pears_master_cat_names, skip_header=7)

    all_fields_north = [cdfn1, cdfn2, cdfn3, cdfn4]
    all_fields_south = [cdfs1, cdfs2, cdfs3, cdfs4, cdfs_new, cdfs_udf]
    # 2032 total objects

    all_fields_northnames = ['cdfn1', 'cdfn2', 'cdfn3', 'cdfn4']
    all_fields_southnames = ['cdfs1', 'cdfs2', 'cdfs3', 'cdfs4', 'cdfs_new', 'cdfs_udf']

    data_path = home + '/Documents/PEARS/data_spectra_only/'
    threedphot = home + '/Documents/3D-HST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat'
    threed = pf.open(home + '/Documents/3D-HST/3dhst.v4.1.5.master.fits')

    ur_color = []
    stellarmass = []
    imag_arr = []
    zphot_arr = []

    ur_color, stellarmass, imag_arr, zphot_arr = get_color_stellarmass(cdfn1, 'cdfn1', ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master)
    ur_color, stellarmass, imag_arr, zphot_arr = get_color_stellarmass(cdfn2, 'cdfn2', ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master)
    ur_color, stellarmass, imag_arr, zphot_arr = get_color_stellarmass(cdfn3, 'cdfn3', ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master)
    ur_color, stellarmass, imag_arr, zphot_arr = get_color_stellarmass(cdfn4, 'cdfn4', ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master)
    #ur_color, stellarmass, imag_arr, zphot_arr = get_color_stellarmass(cdfs1, 'cdfs1', ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master)
    #ur_color, stellarmass, imag_arr, zphot_arr = get_color_stellarmass(cdfs2, 'cdfs2', ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master)
    #ur_color, stellarmass, imag_arr, zphot_arr = get_color_stellarmass(cdfs3, 'cdfs3', ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master)
    #ur_color, stellarmass, imag_arr, zphot_arr = get_color_stellarmass(cdfs4, 'cdfs4', ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master)
    #ur_color, stellarmass, imag_arr, zphot_arr = get_color_stellarmass(cdfs_new, 'cdfs_new', ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master)
    #ur_color, stellarmass, imag_arr, zphot_arr = get_color_stellarmass(cdfs_udf, 'cdfs_udf', ur_color, stellarmass, imag_arr, zphot_arr, pears_north_master, pears_south_master)

    # convert all lists to numpy arrays
    ur_color = np.asarray(ur_color, dtype=np.float)
    stellarmass = np.asarray(stellarmass, dtype=np.float)
    imag_arr = np.asarray(imag_arr, dtype=np.float)
    zphot_arr = np.asarray(zphot_arr, dtype=np.float)

    # make histogram of mag and zphot
    valid_mag = np.where(imag_arr <= 23)[0]
    print len(valid_mag), "galaxies in chosen mag range"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(imag_arr[valid_mag], 8)

    plt.clf()
    plt.cla()
    plt.close()

    sys.exit(0)

    mstar_min = np.argmin(stellarmass)
    log_ml = -0.55 + 0.45*ur_color[mstar_min]
    Mr_min = 4.62 - ((min(stellarmass) - log_ml) * 2.5)
    redseq_colorlim_low = 2.06 - 0.244 * np.tanh((Mr_min + 20.07)/1.09)

    mstar_max = np.argmax(stellarmass)
    log_ml = -0.55 + 0.45*ur_color[mstar_max]
    Mr_max = 4.62 - ((max(stellarmass) - log_ml) * 2.5)
    redseq_colorlim_up = 2.06 - 0.244 * np.tanh((Mr_max + 20.07)/1.09)

    mstar_line = np.array([min(stellarmass), max(stellarmass)])
    redseqlim = np.array([redseq_colorlim_low, redseq_colorlim_up])

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\mathrm{log}\left(\frac{M_*}{M_\odot}\right)$')
    ax.set_ylabel('$\mathrm{(U-R)_{rest}}$')
    ax.plot(stellarmass, ur_color, 'o', markersize=2, color='k', markeredgecolor='none')

    #ax.plot(mstar_line, redseqlim, '--', color='r')

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')

    ax.set_xlim(7,11.5)
    ax.set_ylim(0,3)
    ax.grid('on')

    ax.set_aspect(abs((11.5-7)/(3-0)))
    #fig.savefig('pears_colormag',dpi=300,bbox_inches='tight')

    plt.show()
    sys.exit(0)
