from __future__ import division
import numpy as np
import pyfits as pf

import matplotlib as mpl
import matplotlib.pyplot as plt
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
threed = pf.open('/Users/baj/Documents/3D-HST/3dhst.v4.1.5.master.fits')

ur_color = []
stellarmass = []

def measure_north(matchedfile):
    
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


measure_north(cdfn1)
measure_north(cdfn2)
measure_north(cdfn3)
measure_north(cdfn4)
measure_north(cdfs1)
measure_north(cdfs2)
measure_north(cdfs3)
measure_north(cdfs4)
measure_north(cdfs_new)
measure_north(cdfs_udf)

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
fig.savefig('pears_colormag',dpi=300,bbox_inches='tight')

#plt.show()
