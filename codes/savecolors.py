from __future__ import division
import numpy as np
import pyfits as pf

# read in all matched files
cdfn1 = np.genfromtxt('/Users/baj/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn1.txt',\
                      dtype=None, skip_header=3, names=True)
cdfn2 = np.genfromtxt('/Users/baj/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn2.txt',\
                      dtype=None, skip_header=3, names=True)
cdfn3 = np.genfromtxt('/Users/baj/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn3.txt',\
                      dtype=None, skip_header=3, names=True)
cdfn4 = np.genfromtxt('/Users/baj/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfn4.txt',\
                      dtype=None, skip_header=3, names=True)

cdfs1 = np.genfromtxt('/Users/baj/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs1.txt',\
                      dtype=None, skip_header=3, names=True)
cdfs2 = np.genfromtxt('/Users/baj/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs2.txt',\
                      dtype=None, skip_header=3, names=True)
cdfs3 = np.genfromtxt('/Users/baj/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs3.txt',\
                      dtype=None, skip_header=3, names=True)
cdfs4 = np.genfromtxt('/Users/baj/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs4.txt',\
                      dtype=None, skip_header=3, names=True)
cdfs_new = np.genfromtxt('/Users/baj/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs_new.txt',\
                         dtype=None, skip_header=3, names=True)
cdfs_udf = np.genfromtxt('/Users/baj/Documents/4000break_check/pears_with_3dhstv4.1.5/matches_cdfs_udf.txt',\
                         dtype=None, skip_header=3, names=True)

data_path = "/Users/baj/Documents/PEARS/data_spectra_only/"
threedphot = "/Users/baj/Documents/3D-HST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat"
threed = pf.open('/Users/baj/Documents/3D-HST/3dhst.v4.1.5.master.fits')

def get_colors(matchedfile):
    ur_color = []
    stellarmass = []
    pears_id = []
    threed_id = []
    threed_z = []
    
    for i in range(len(matchedfile)):
        cat = matchedfile
        current_id = cat['pearsid'][i]
        pears_id.append(current_id)
        threedid = cat['threed_hst_id'][i]
        threed_id.append(threedid)
        
        threedra = cat['threedra'][i]
        threeddec = cat['threeddec'][i]
        threed_zphot = cat['threed_zphot'][i]
        threed_z.append(threed_zphot)
        
        threedindex = np.where((threed[1].data['phot_id'] == threedid) & (abs(threed[1].data['ra'] - threedra) < 1e-3) & (abs(threed[1].data['dec'] - threeddec) < 1e-3))[0][0]
        
        mstar = threed[1].data[threedindex]['lmass']
        urcol = -2.512*np.log10(threed[1].data[threedindex]['L156']/threed[1].data[threedindex]['L158'])
        
        ur_color.append(urcol)
        stellarmass.append(mstar)
    
    pears_id = np.array(pears_id)
    stellarmass = np.array(stellarmass)
    ur_color = np.array(ur_color)
    threed_id = np.array(threed_id)
    threed_z = np.array(threed_z)

    return pears_id, ur_color, stellarmass, threed_id, threed_z

def concat(pears_id, ur_color, stellarmass, threed_id, threed_z, pid, ur, ms, tid, tz):
    pears_id = np.concatenate((pears_id,pid))
    ur_color = np.concatenate((ur_color,ur))
    stellarmass = np.concatenate((stellarmass,ms))
    threed_id = np.concatenate((threed_id,tid))
    threed_z = np.concatenate((threed_z,tz))
    
    return pears_id, ur_color, stellarmass, threed_id, threed_z

all_fields = [cdfn1, cdfn2, cdfn3, cdfn4, cdfs1, cdfs2, cdfs3, cdfs4, cdfs_new, cdfs_udf]
# 2032 total objects

pears_id, ur_color, stellarmass, threed_id, threed_z = get_colors(cdfn1)

for field in all_fields[1:]:
    pid, ur, ms, tid, tz = get_colors(field)
    pears_id, ur_color, stellarmass, threed_id, threed_z = concat(pears_id, ur_color, stellarmass, threed_id, threed_z, pid, ur, ms, tid, tz)

outdir = '/Users/baj/Desktop/FIGS/new_codes/'
data = np.array(zip(pears_id, ur_color, stellarmass, threed_id, threed_z),\
                dtype=[('pears_id', int), ('ur_color', float), ('stellarmass', float), ('threed_id', int), ('threed_z', float)])
np.savetxt(outdir + 'color_stellarmass.txt', data, fmt=['%d', '%.2f', '%.2f', '%d', '%.2f'], delimiter=' ',\
           header='The PEARS id is given here as only an integer.' + '\n' +\
           'The user must figure out if it is part of northern or southern fields.' + '\n' +\
           'pearsid urcol mstar threedid threedzphot')