# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pyfits as pf

import matplotlib.pyplot as plt
import os, sys, glob

def plot_delta_radec(closest_ra, closest_dec):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$\Delta \alpha$')
    ax.set_ylabel(r'$\Delta \delta$')
    ax.plot(closest_ra, closest_dec, 'o', markeredgecolor='b')
    
    #ax.set_xlim(-0.1/3600, 0.1/3600)
    #ax.set_ylim(-0.1/3600, 0.1/3600)
    
    ax.axhline(y=0, linestyle='--', color='k')
    ax.axvline(x=0, linestyle='--', color='k')
    
    ax.minorticks_on()
    ax.tick_params('both', width=0.8, length=3, which='minor')
    ax.tick_params('both', width=0.8, length=4.7, which='major')
    plt.show()

def catmatch(f_ra, f_dec, threed_ra, threed_dec, f_id):
    
    f_ra_mat = []
    f_dec_mat = []
    threed_ra_mat = []
    threed_dec_mat = []
    closest_ra = []
    closest_dec = []
    f_id_mat = []

    for i in range(len(ferreras_cat)):
        angdist = np.sqrt((f_ra[i] - threed_ra)**2 + (f_dec[i] - threed_dec)**2)
        match = np.argmin(angdist)
        if angdist[match] <= 0.5/3600:
            f_ra_mat.append(f_ra[i])
            f_dec_mat.append(f_dec[i])
            threed_ra_mat.append(threed_ra[match])
            threed_dec_mat.append(threed_dec[match])
            closest_ra.append(f_ra[i] - threed_ra[match])
            closest_dec.append(f_dec[i] - threed_dec[match])
            f_id_mat.append(f_id[i])

    return f_ra_mat, f_dec_mat, threed_ra_mat, threed_dec_mat, closest_ra, closest_dec, f_id_mat

if __name__ == '__main__':

    # Read 3D-HST cat
    threed_hst_cat = pf.open('/Users/baj/Documents/3D-HST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat.FITS')

    # Read Ferreras 2009 cat
    ferreras_cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/ferreras_2009_ETG_cat.txt', dtype=None,\
                                 names=['id', 'ra', 'dec', 'z'], usecols=(0,1,2,5), skip_header=23)

    f_ra = ferreras_cat['ra']
    f_dec = ferreras_cat['dec']
    f_id = ferreras_cat['id']
    threed_ra = threed_hst_cat[1].data['ra']
    threed_dec = threed_hst_cat[1].data['dec']

    f_ra_mat, f_dec_mat, threed_ra_mat, threed_dec_mat, closest_ra, closest_dec, f_id_mat = catmatch(f_ra, f_dec, threed_ra, threed_dec, f_id)

    plot_delta_radec(closest_ra, closest_dec)



































