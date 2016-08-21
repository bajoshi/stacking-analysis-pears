from __future__ import division
import numpy as np
from astropy.io import fits
from scipy.stats import gaussian_kde

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
pgf_preamble = {"pgf.texsystem": "pdflatex"}
mpl.rcParams.update(pgf_preamble)

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

all_fields_north = [cdfn1, cdfn2, cdfn3, cdfn4]
all_fields_south = [cdfs1, cdfs2, cdfs3, cdfs4, cdfs_new, cdfs_udf]
# 2032 total objects

data_path = "/Users/baj/Documents/PEARS/data_spectra_only/"
threedphot = "/Users/baj/Documents/3D-HST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat"
threed = fits.open('/Users/baj/Documents/3D-HST/3dhst.v4.1.5.master.fits')

ur_color = []
stellarmass = []

def get_colors(matchedfile):
    
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


get_colors(cdfn1)
get_colors(cdfn2)
get_colors(cdfn3)
get_colors(cdfn4)
get_colors(cdfs1)
get_colors(cdfs2)
get_colors(cdfs3)
get_colors(cdfs4)
get_colors(cdfs_new)
get_colors(cdfs_udf)


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_xlabel(r'$\mathrm{log}\left(\frac{M_*}{M_\odot}\right)$', fontsize=16)
ax.set_ylabel('$\mathrm{(U-R)_{rest}}$', fontsize=16)

ax.minorticks_on()
ax.tick_params('both', width=1, length=3, which='minor')
ax.tick_params('both', width=1, length=4.7, which='major')

"""
stellarmass = np.array(stellarmass)
ur_color = np.array(ur_color)

smmin = min(stellarmass)
smmax = max(stellarmass)
urmin = min(ur_color)
urmax = max(ur_color)

X, Y = np.mgrid[smmin:smmax:100j, urmin:urmax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([stellarmass, ur_color])
kernel = gaussian_kde(values)

density = np.reshape(kernel(positions).T, X.shape)

ax.imshow(np.rot90(density), cmap=plt.cm.gist_earth_r, extent=[7, 12.0, 0, 3])
"""
ax.plot(stellarmass, ur_color, 'o', markersize=2, color='k', markeredgecolor='none')

color_intervals = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
color_loc = MultipleLocator(0.6)
ax.yaxis.set_major_locator(color_loc)
#ax.grid(b=True, which='major')
ax.set_xlim(7.0, 12.0)

ax.set_aspect(abs((12-7)/(3-0)))

fig.savefig('pears_colmstar_nogrid.png',dpi=300,bbox_inches='tight')

#plt.show()
