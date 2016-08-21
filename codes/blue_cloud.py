from __future__ import division
import numpy as np
import numpy.ma as ma
from astropy.io import fits

import sys, os, time, glob, datetime
import warnings

import matplotlib.pyplot as plt

def makefig(xlab, ylab):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    return fig, ax

def make_masked_mag(mags):

    if np.any(mags == 99.0):
        indices_to_be_masked = np.where(mags == 99.0)[0]
        mask = np.zeros(len(mags)) # by default create a masked array where all values in the original array are assumed to be valid
        mask[indices_to_be_masked] = 1 # now set the indices to be masked as True
        mags = ma.masked_array(mags, mask = mask)

    return mags

if __name__ == '__main__':

    warnings.filterwarnings('error')

    blue_cl_ids = [10465,12457,15513,15793,16039,17413,18065,18213,18484,18882,20265,20877,21023,\
    21363,21614,22316,22784,23034,25474,27652,27862,29726,31227,33949,34050,34245,34789,35090,35692,\
    36001,37908,38345,40063,40163,40243,40875,40899,41078,41932,42526,43079,43170,44367,44835,45117,\
    45314,46284,46498,46626,46694,46952,47029,47997,48250,49165,50298,50542,51220,51533,51976,52497,\
    53198,53811,54005,54809,54815,55063,55951,56442,58615,58650,59121,61413,61418,62543,62562,64178,\
    64394,64963,65170,65572,65708,66045,67460,67558,67599,67654,68854,68918,69117,69168,69170,69234,\
    71305,72168,73362,74125,74166,74385,74900,74950,76612,78343,78417,78761,79520,79756,81609,82523,\
    82693,83789,83804,84143,85861,85879,85918,87611,89877,90198,90321,91382,91517,91695,91724,92376,\
    93025,93208,93242,93923,94073,94364,94425,94854,94858,96942,97327,100679,100950,101093,101176,\
    102160,102575,104478,104498,104514,104981,105015,105016,105328,105570,107052,107654,107754,107997,\
    108366,108620,108871,109019,109438,109511,109770,109948,110065,110145,110235,110493,110595,110664,\
    110733,111122,111151,112301,112745,112989,113279,113474,113558,114221,115319,115544,115684,116154,\
    116370,116527,117002,117222,117333,117686,118438,119893,120445,120859,121225,121350,121678,121974,\
    122039,122206,122303,122600,122961,125733,125809,126281,126934,127541,128198,128268,128352]
    # there are 209 blue cloud galaxies in the region 0.6 < U-R < 0.9 and 8.5 < m_star < 9.0 
    print len(blue_cl_ids)

    # Read PEARS cats
    pears_ncat = np.genfromtxt('/Users/baj/Documents/PEARS/master_catalogs/h_pears_north_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))
    pears_scat = np.genfromtxt('/Users/baj/Documents/PEARS/master_catalogs/h_pears_south_master.cat', dtype=None,\
                               names=['id', 'ra', 'dec', 'imag'], usecols=(0,1,2,3))

    dec_offset_goodsn_v19 = 0.32/3600 # from GOODS ACS v2.0 readme
    pears_ncat['dec'] = pears_ncat['dec'] - dec_offset_goodsn_v19

    # Read PEARS broadband cat
    names_header = ['PID', 'MAG_AUTO_b', 'MAG_AUTO_v', 'MAG_AUTO_i', 'MAG_AUTO_z', 'MAGERR_AUTO_b', 'MAGERR_AUTO_v', 'MAGERR_AUTO_i', 'MAGERR_AUTO_z', 'ALPHA_J2000', 'DELTA_J2000']
    # I can't tell if I should use ID or PID
    
    north_pears_cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/n_biz_bviz_all.pid.txt', dtype=None, names=names_header, skip_header=365, usecols=(362, 158, 164, 162, 166, 93, 91, 92, 89, 358, 248), delimiter=' ')
    south_pears_cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/s_biz_bviz_all.pid.txt', dtype=None, names=names_header, skip_header=365, usecols=(362, 158, 164, 162, 166, 93, 91, 92, 89, 358, 248), delimiter=' ')

    north_pears_cat['DELTA_J2000'] = north_pears_cat['DELTA_J2000'] - dec_offset_goodsn_v19
    # I'm assuming there is the same offset in this broadband photometry catalog as well. Applying the correction for now.

    countn = 0
    counts = 0
    tolerance = 0.5/3600
    data_path = "/Users/baj/Documents/PEARS/data_spectra_only/"

    mag_b = []
    mag_v = []
    mag_i = []
    mag_z = []

    for ind in blue_cl_ids:
        #if np.where(north_pears_cat['PID'] == ind)[0].size : print ind, "in GOODS-N"
        #if np.where(south_pears_cat['PID'] == ind)[0].size : print ind, "in GOODS-S"

        #if np.where(pears_ncat['id'] == ind)[0].size :
        #    #print ind, "in GOODS-N"
        #    countn += 1
        #if np.where(pears_scat['id'] == ind)[0].size : 
        #    #print ind, "in GOODS-S"
        #    counts += 1

        #if (np.where(pears_ncat['id'] == ind)[0].size) and (np.where(pears_scat['id'] == ind)[0].size):
        #    print ind, "is repeated"

        # Get the correct filename and the number of extensions
        pears_index = ind
        filename_n = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
        #if not os.path.isfile(filename):
        filename_s = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

        if os.path.isfile(filename_n) and os.path.isfile(filename_s): 
            continue
            # skipping these for now

        # you could just as well look at the dec and see if it is positive or negative and infer if hte object is in the northern or southern sky.
        # the dec wouldn't work probably because pears_ncat and pears_scat have some common ids
        if os.path.isfile(filename_n):
            idx = np.where(pears_ncat['id'] == ind)[0]
            curr_ra = pears_ncat['ra'][idx]
            curr_dec = pears_ncat['dec'][idx]

            ra_idx = np.where((north_pears_cat['ALPHA_J2000'] <= curr_ra + tolerance) & (north_pears_cat['ALPHA_J2000'] >= curr_ra - tolerance))[0]
            dec_idx = np.where((north_pears_cat['DELTA_J2000'] <= curr_dec + tolerance) & (north_pears_cat['DELTA_J2000'] >= curr_dec - tolerance))[0]

            match_id = np.intersect1d(ra_idx, dec_idx)

            if len(match_id) > 1:
                ang_dist = np.sqrt((curr_ra - north_pears_cat['ALPHA_J2000'][match_id])**2 + (curr_dec - north_pears_cat['DELTA_J2000'][match_id])**2)
                min_idx = np.argmin(ang_dist)
                #print ind, curr_ra, curr_dec, north_pears_cat['ALPHA_J2000'][match_id][min_idx], north_pears_cat['DELTA_J2000'][match_id][min_idx], ang_dist*3600

                mag_b.append(north_pears_cat['MAG_AUTO_b'][match_id][min_idx])
                mag_v.append(north_pears_cat['MAG_AUTO_v'][match_id][min_idx])
                mag_i.append(north_pears_cat['MAG_AUTO_i'][match_id][min_idx])
                mag_z.append(north_pears_cat['MAG_AUTO_z'][match_id][min_idx])

            elif len(match_id) == 1:
                min_idx = match_id
                #print ind, curr_ra, curr_dec, north_pears_cat['ALPHA_J2000'][min_idx], north_pears_cat['DELTA_J2000'][min_idx]

                mag_b.append(north_pears_cat['MAG_AUTO_b'][min_idx])
                mag_v.append(north_pears_cat['MAG_AUTO_v'][min_idx])
                mag_i.append(north_pears_cat['MAG_AUTO_i'][min_idx])
                mag_z.append(north_pears_cat['MAG_AUTO_z'][min_idx])

        if os.path.isfile(filename_s):
            idx = np.where(pears_scat['id'] == ind)[0]
            curr_ra = pears_scat['ra'][idx]
            curr_dec = pears_scat['dec'][idx]

            ra_idx = np.where((south_pears_cat['ALPHA_J2000'] <= curr_ra + tolerance) & (south_pears_cat['ALPHA_J2000'] >= curr_ra - tolerance))[0]
            dec_idx = np.where((south_pears_cat['DELTA_J2000'] <= curr_dec + tolerance) & (south_pears_cat['DELTA_J2000'] >= curr_dec - tolerance))[0]

            match_id = np.intersect1d(ra_idx, dec_idx)

            if len(match_id) > 1:
                ang_dist = np.sqrt((curr_ra - south_pears_cat['ALPHA_J2000'][match_id])**2 + (curr_dec - south_pears_cat['DELTA_J2000'][match_id])**2)
                min_idx = np.argmin(ang_dist)
                #print ind, curr_ra, curr_dec, south_pears_cat['ALPHA_J2000'][match_id][min_idx], south_pears_cat['DELTA_J2000'][match_id][min_idx], ang_dist*3600

                mag_b.append(south_pears_cat['MAG_AUTO_b'][match_id][min_idx])
                mag_v.append(south_pears_cat['MAG_AUTO_v'][match_id][min_idx])
                mag_i.append(south_pears_cat['MAG_AUTO_i'][match_id][min_idx])
                mag_z.append(south_pears_cat['MAG_AUTO_z'][match_id][min_idx])

            elif len(match_id) == 1:
                min_idx = match_id
                #print ind, curr_ra, curr_dec, north_pears_cat['ALPHA_J2000'][min_idx], north_pears_cat['DELTA_J2000'][min_idx]

                mag_b.append(south_pears_cat['MAG_AUTO_b'][min_idx])
                mag_v.append(south_pears_cat['MAG_AUTO_v'][min_idx])
                mag_i.append(south_pears_cat['MAG_AUTO_i'][min_idx])
                mag_z.append(south_pears_cat['MAG_AUTO_z'][min_idx])

    mag_b = np.asarray(mag_b)
    mag_v = np.asarray(mag_v)
    mag_i = np.asarray(mag_i)
    mag_z = np.asarray(mag_z)

    # mask the indices where the magnitudes are 99.0
    mag_b = make_masked_mag(mag_b)
    mag_v = make_masked_mag(mag_v)
    mag_i = make_masked_mag(mag_i)
    mag_z = make_masked_mag(mag_z)

    #fig, ax = makefig('V - I', 'B - V')
    #ax.plot(mag_v - mag_i, mag_b - mag_v, 'o', markersize=4, color='k', markeredgecolor='k')
#
    #fig, ax = makefig('V - Z', 'B - V')
    #ax.plot(mag_v - mag_z, mag_b - mag_v, 'o', markersize=4, color='k', markeredgecolor='k')    
#
    #fig, ax = makefig('I - Z', 'B - I')
    #ax.plot(mag_i - mag_z, mag_b - mag_i, 'o', markersize=4, color='k', markeredgecolor='k')
#
    #fig, ax = makefig('I - Z', 'V - I')
    #ax.plot(mag_i - mag_z, mag_v - mag_i, 'o', markersize=4, color='k', markeredgecolor='k')

    fig, ax = makefig('b', 'B-V')
    ax.plot(mag_b, mag_b - mag_v, 'o', markersize=4, color='k', markeredgecolor='k')

    fig, ax = makefig('v', 'B-V')
    ax.plot(mag_v, mag_b - mag_v, 'o', markersize=4, color='k', markeredgecolor='k')

    fig, ax = makefig('i', 'B-V')
    ax.plot(mag_i, mag_b - mag_v, 'o', markersize=4, color='k', markeredgecolor='k')

    fig, ax = makefig('z', 'B-V')
    ax.plot(mag_z, mag_b - mag_v, 'o', markersize=4, color='k', markeredgecolor='k')

    #############
    fig, ax = makefig('b', 'V-I')
    ax.plot(mag_b, mag_v - mag_i, 'o', markersize=4, color='k', markeredgecolor='k')

    fig, ax = makefig('v', 'V-I')
    ax.plot(mag_v, mag_v - mag_i, 'o', markersize=4, color='k', markeredgecolor='k')

    fig, ax = makefig('i', 'V-I')
    ax.plot(mag_i, mag_v - mag_i, 'o', markersize=4, color='k', markeredgecolor='k')

    fig, ax = makefig('z', 'V-I')
    ax.plot(mag_z, mag_v - mag_i, 'o', markersize=4, color='k', markeredgecolor='k')

    #############
    fig, ax = makefig('b', 'I-Z')
    ax.plot(mag_b, mag_i - mag_z, 'o', markersize=4, color='k', markeredgecolor='k')

    fig, ax = makefig('v', 'I-Z')
    ax.plot(mag_v, mag_i - mag_z, 'o', markersize=4, color='k', markeredgecolor='k')

    fig, ax = makefig('i', 'I-Z')
    ax.plot(mag_i, mag_i - mag_z, 'o', markersize=4, color='k', markeredgecolor='k')

    fig, ax = makefig('z', 'I-Z')
    ax.plot(mag_z, mag_i - mag_z, 'o', markersize=4, color='k', markeredgecolor='k')

    plt.show()


