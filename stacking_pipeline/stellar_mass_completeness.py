from __future__ import division

import numpy as np

import os
import sys

import matplotlib.pyplot as plt

from matplotlib import rc
rc("text", usetex=False)

home = os.getenv('HOME')
figs_dir = home + '/Desktop/FIGS/'

stacking_analysis_dir = figs_dir + 'stacking-analysis-pears/'
stacking_figures_dir = figs_dir + 'stacking-analysis-figures/'

# Get correct directory for 3D-HST data
if 'firstlight' in os.uname()[1]:
    threedhst_datadir = '/Users/baj/Desktop/3dhst_data/'
else:
    threedhst_datadir = '/Volumes/Bhavins_backup/3dhst_data/'

def get_threed_ms_z():

    # ------------------------------- Read in stellar mass catalogs ------------------------------- #
    # These are output from the FAST code (Kriek et al.)
    goodsn_ms_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.fout', dtype=None, names=True, skip_header=17)
    goodss_ms_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.fout', dtype=None, names=True, skip_header=17)

    # Get 3D-HST stellar mass and redshift
    # Separately for GOODS North and South and then combine
    threed_lmass_n = goodsn_ms_cat_3dhst['lmass']
    threed_lmass_s = goodss_ms_cat_3dhst['lmass']

    threed_z_n = goodsn_ms_cat_3dhst['z']
    threed_z_s = goodss_ms_cat_3dhst['z']

    threed_lmass = np.concatenate((threed_lmass_n, threed_lmass_s))
    threed_z = np.concatenate((threed_z_n, threed_z_s))

    return threed_lmass, threed_z

def main():

    # ------------------------------- Read in PEARS results ------------------------------- #
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    ms = np.log10(cat['zp_ms'])
    zp = cat['zp_minchi2']

    # ------------------------------- Read in 3D-HST results ------------------------------- #
    threed_lmass, threed_z = get_threed_ms_z()

    # ------------------------------- Plot ------------------------------- #
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    #ax.set_xlabel(r'$\mathrm{Redshift}$')
    #ax.set_ylabel(r'$\mathrm{log(M_s/M_\odot)}$')

    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel('log(Ms/Msolar)', fontsize=14)

    ax.scatter(threed_z, threed_lmass, color='k', s=1.0, linewidth=0.5, edgecolor='k')
    ax.scatter(zp, ms, color='green', s=5.0, linewidth=1.0, facecolors='None')

    ax.set_xlim(-0.2, 6.0)
    ax.set_ylim(-0.2, 12.0)

    ax.axhline(y=9.0, ls='--', color='red', lw=2.0)
    ax.axvline(x=3.0, ls='--', color='red', lw=2.0)
    #ax.axvline(x=0.5, ls='--', color='red', lw=2.0)

    fig.savefig(stacking_figures_dir + 'stellar_mass_completeness.pdf', dpi=150, bbox_inches='tight')
    #plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)