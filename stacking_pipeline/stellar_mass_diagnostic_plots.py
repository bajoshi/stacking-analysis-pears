from __future__ import division

import numpy as np

import os
import sys
import glob

import matplotlib.pyplot as plt

home = os.getenv('HOME')
figs_dir = home + '/Desktop/FIGS/'

massive_figures_dir = figs_dir + 'massive-galaxies-figures/'
full_pears_results_dir = massive_figures_dir + 'full_pears_results/'
stacking_figures_dir = figs_dir + 'stacking-analysis-figures/'

def make_stellar_mass_hist():

    # Define empty list for stellar mass
    stellar_mass = []

    # Loop over all results and store stellar mass values
    for fl in glob.glob(full_pears_results_dir + 'redshift_fitting_results_*.txt'):
        f = np.genfromtxt(fl, dtype=None, names=True, skip_header=1)
        stellar_mass.append(f['zp_ms'])

    # Convert to numpy array to make histogram
    stellar_mass = np.asarray(stellar_mass)
    ms = np.log10(stellar_mass)  # plot log of stellar mass to make it easier to see

    # Now make plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\rm log(M_s)\ [M_\odot]$', fontsize=15)
    ax.set_ylabel(r'$\rm \# objects$', fontsize=15)

    ax.hist(ms, 20, histtype='step', linewidth=1.2, color='k')

    ax.minorticks_on()
    
    fig.savefig(stacking_figures_dir + 'stellar_mass_hist.pdf', \
        dpi=300, bbox_inches='tight')

    return None

def main():

    make_stellar_mass_hist()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)