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

    binsize = 0.5
    total_bins = int((13.0 - 4.0)/binsize)

    ax.hist(ms, total_bins, range=(4.0, 13.0), histtype='step', linewidth=1.2, color='k')

    ax.minorticks_on()
    ax.set_xticks(np.arange(4.0, 13.5, 1.0))

    # Other info on plot
    num = len(stellar_mass)
    ax.text(0.04, 0.95, r'$\rm N\, =\, $' + str(num), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=15)
    
    fig.savefig(stacking_figures_dir + 'stellar_mass_hist.pdf', \
        dpi=300, bbox_inches='tight')

    return None

def compare_with_threedhst():

    # Define empty lists for id, field, ra, dec, 
    # and stellar mass from my code and from 3dhst
    pears_id = []
    pears_field = []
    pears_ra = []
    pears_dec = []
    pears_ms = []
    threed_ms = []

    # Loop over all results and store values from my pipeline
    fl_count = 0
    for fl in glob.glob(full_pears_results_dir + 'redshift_fitting_results_*.txt'):

        f = np.genfromtxt(fl, dtype=None, names=True, skip_header=1)

        current_id = f['PearsID'][fl_count]
        current_field = f['Field'][fl_count]
        current_ra = f['RA'][fl_count]
        current_dec = f['DEC'][fl_count]
        current_ms = f['zp_ms'][fl_count]

        pears_id.append(current_id)
        pears_field.append(current_field)
        pears_ra.append(current_ra)
        pears_dec.append(current_dec)
        pears_ms.append(current_ms)

        fl_count += 1

    # Convert to numpy arrays
    pears_id = np.asarray(pears_id)
    pears_field = np.asarray(pears_field)
    pears_ra = np.asarray(pears_ra)
    pears_dec = np.asarray(pears_dec)
    pears_ms = np.asarray(pears_ms)

    # Now match with 3D-HST and get their stellar mass values
    


    return None

def main():

    make_stellar_mass_hist()
    compare_with_threedhst()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

