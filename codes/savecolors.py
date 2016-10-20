from __future__ import division

import numpy as np
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')  # Does not have a trailing slash at the end

def get_colors(matchedfile, threedn, threeds):

    pears_id = []
    fieldname = []
    threed_id = []
    pearsra = []
    pearsdec = []
    ur_color = []
    stellarmass = []
    threed_z = []

    cat = matchedfile

    for i in range(len(cat)):

        pears_id.append(cat['pearsid'][i])
        fieldname.append(cat['field'][i])
        threed_id.append(cat['threed_id'][i])
        pearsra.append(cat['pearsra'][i])
        pearsdec.append(cat['pearsdec'][i])
        threed_z.append(cat['threed_zphot'][i])
        
        if cat['field'][i] == 'GOODS-N':
            threedindex = np.where((threedn[1].data['id'] == cat['threed_id'][i]))[0]
            urcol = -2.5*np.log10(threedn[1].data[threedindex]['l156']/threedn[1].data[threedindex]['l158'])
        elif cat['field'][i] == 'GOODS-S':
            threedindex = np.where((threeds[1].data['id'] == cat['threed_id'][i]))[0]
            urcol = -2.5*np.log10(threeds[1].data[threedindex]['l156']/threeds[1].data[threedindex]['l158'])
        
        ur_color.append(urcol)
        stellarmass.append(cat['threed_mstellar'][i])
    
    pears_id = np.array(pears_id)
    fieldname = np.array(fieldname)
    pearsra = np.array(pearsra)
    pearsdec = np.array(pearsdec)
    stellarmass = np.array(stellarmass)
    ur_color = np.array(ur_color)
    threed_id = np.array(threed_id)
    threed_z = np.array(threed_z)

    return pears_id, fieldname, pearsra, pearsdec, ur_color, stellarmass, threed_id, threed_z

def concat(pears_id, fieldname, pearsra, pearsdec, ur_color, stellarmass, threed_id, threed_z, pid, fname, pra, pdec, ur, ms, tid, tz):
    pears_id = np.concatenate((pears_id,pid))
    fieldname = np.concatenate((fieldname,fname))
    pearsra = np.concatenate((pearsra, pra))
    pearsdec = np.concatenate((pearsdec, pdec))
    ur_color = np.concatenate((ur_color,ur))
    stellarmass = np.concatenate((stellarmass,ms))
    threed_id = np.concatenate((threed_id,tid))
    threed_z = np.concatenate((threed_z,tz))
    
    return pears_id, fieldname, pearsra, pearsdec, ur_color, stellarmass, threed_id, threed_z

if __name__ == '__main__':

    # read in all matched files
    cdfn1 = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/matches_cdfn1.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfn2 = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/matches_cdfn2.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfn3 = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/matches_cdfn3.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfn4 = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/matches_cdfn4.txt',\
                          dtype=None, skip_header=3, names=True)
    
    cdfs1 = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/matches_cdfs1.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfs2 = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/matches_cdfs2.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfs3 = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/matches_cdfs3.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfs4 = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/matches_cdfs4.txt',\
                          dtype=None, skip_header=3, names=True)
    cdfs_new = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/matches_cdfs_new.txt',\
                             dtype=None, skip_header=3, names=True)
    cdfs_udf = np.genfromtxt(home + '/Desktop/FIGS/new_codes/pears_match_3dhst_v4.1/matches_cdfs_udf.txt',\
                             dtype=None, skip_header=3, names=True)
    
    data_path = home + "/Documents/PEARS/data_spectra_only/"
    threedn = fits.open(home + '/Desktop/FIGS/new_codes/goodsn_3dhst.v4.1.cats/RF_colors/goodsn_3dhst.v4.1.master.RF.FITS')
    threeds = fits.open(home + '/Desktop/FIGS/new_codes/goodss_3dhst.v4.1.cats/RF_colors/goodss_3dhst.v4.1.master.RF.FITS')

    all_fields = [cdfn1, cdfn2, cdfn3, cdfn4, cdfs1, cdfs2, cdfs3, cdfs4, cdfs_new, cdfs_udf]
    # 2321 objects

    pears_id, fieldname, pearsra, pearsdec, ur_color, stellarmass, threed_id, threed_z = get_colors(cdfn1, threedn, threeds)
    # cdfn1 has to be done separately to get the concatenation to work

    for field in all_fields[1:]:
        pid, fname, pra, pdec, ur, ms, tid, tz = get_colors(field, threedn, threeds)
        pears_id, fieldname, pearsra, pearsdec, ur_color, stellarmass, threed_id, threed_z =\
         concat(pears_id, fieldname, pearsra, pearsdec, ur_color, stellarmass, threed_id, threed_z, pid, fname, pra, pdec, ur, ms, tid, tz)

    outdir = home + '/Desktop/FIGS/stacking-analysis-pears/'
    data = np.array(zip(pears_id, fieldname, pearsra, pearsdec, ur_color, stellarmass, threed_id, threed_z),\
                    dtype=[('pears_id', int), ('fieldname', '|S7'), ('pearsra', float), ('pearsdec', float), ('ur_color', float), ('stellarmass', float), ('threed_id', int), ('threed_z', float)])

    np.savetxt(outdir + 'color_stellarmass.txt', data, fmt=['%d', '%s', '%.4f', '%.4f', '%.2f', '%.2f', '%d', '%.2f'], delimiter=' ',\
               header='The PEARS id is given here as only an integer.' + '\n' +\
               'The user must figure out if it is part of northern or southern fields.' + '\n' +\
               'pearsid field pearsra pearsdec urcol mstar threedid threedzphot')

    sys.exit(0)

