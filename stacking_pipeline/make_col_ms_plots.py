from __future__ import division

import numpy as np

import os
import sys
import glob

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
figs_dir = home + '/Desktop/FIGS/'
stacking_analysis_dir = figs_dir + 'stacking-analysis-pears/'

def main():
    # Read in results for all of PEARS
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results.txt', dtype=None, names=True)

    # Now get indices based on redshift intervals
    # decided from the stellar_mass_diagnostic_plot.py
    # code. 
    # Look at the make_z_hist() function in there. 
    zp = cat['zp_minchi2']
    ms = np.log10(cat['zp_ms'])

    print cat['zp_uv']
    print min(cat['zp_uv']), max(cat['zp_uv'])
    sys.exit(0)

    z_interval1_idx = np.where((zp >= 0.0) & (zp < 0.4))[0]
    z_interval2_idx = np.where((zp >= 0.4) & (zp < 0.7))[0]
    z_interval3_idx = np.where((zp >= 0.7) & (zp < 1.0))[0]
    z_interval4_idx = np.where((zp >= 1.0) & (zp < 2.0))[0]
    z_interval5_idx = np.where((zp >= 2.0) & (zp <= 6.0))[0]

    print "Number of galaxies within each redshift interval."
    print "0.0 <= z < 0.4", "    ", len(z_interval1_idx)
    print "0.4 <= z < 0.7", "    ", len(z_interval2_idx)
    print "0.7 <= z < 1.0", "    ", len(z_interval3_idx)
    print "1.0 <= z < 2.0", "    ", len(z_interval4_idx)
    print "2.0 <= z <= 6.0", "    ", len(z_interval5_idx)

    # Now make the plots
    # Define figure
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(10,12)
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.25)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:5, :4])
    ax2 = fig.add_subplot(gs[:5, 4:8])
    ax3 = fig.add_subplot(gs[:5, 8:])
    ax4 = fig.add_subplot(gs[5:, :4])
    ax5 = fig.add_subplot(gs[5:, 4:8])
    ax6 = fig.add_subplot(gs[5:, 8:])

    ax1.scatter(ms[z_interval1_idx], cat['zp_uv'][z_interval1_idx], s=1.5, color='k')
    ax2.scatter(ms[z_interval2_idx], cat['zp_uv'][z_interval2_idx], s=1.5, color='k')
    ax3.scatter(ms[z_interval3_idx], cat['zp_uv'][z_interval3_idx], s=1.5, color='k')
    ax4.scatter(ms[z_interval4_idx], cat['zp_uv'][z_interval4_idx], s=1.5, color='k')
    ax5.scatter(ms[z_interval5_idx], cat['zp_uv'][z_interval5_idx], s=1.5, color='k')
    ax6.scatter(ms, cat['zp_uv'], s=1.5, color='k')

    plt.show()


    return None

if __name__ == '__main__':
    main()
    sys.exit(0)