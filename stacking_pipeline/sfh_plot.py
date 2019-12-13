from __future__ import division

import numpy as np

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
figs_dir = home + '/Desktop/FIGS/'

stacking_analysis_dir = figs_dir + 'stacking-analysis-pears/'
stacking_figures_dir = figs_dir + 'stacking-analysis-figures/'

def get_initial_sfr(tau, age, ms):
    """
    It expects to get tau and age in years and
    stellar mass in units of solar masses.
    Will return initial_sfr in units of solar masses per year.

    This assumes an exponential SFH.
    Won't do any checking in here, so give it sensible inputs.
    """

    initial_sfr = ms / (tau * (1 - np.exp(-1 * age/tau)))

    return initial_sfr

def main():
    """
    Based on the fitting results, this code will plot the SFHs for all galaxies.

    The galaxies are also classified as red-sequence, green-valley, and blue-cloud
    based on their location in the color vs stellar-mass diagram.
    """

    # --------------------- Read in results for all of PEARS --------------------- #
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    # --------------------- Get u-r color and stellar masses --------------------- #
    # First get stellar mass and apply a cut to stellar mass
    ms = np.log10(cat['zp_ms'])
    low_mass_lim = 8.0
    ms_idx = np.where((ms >= low_mass_lim) & (ms <= 12.0))[0]  # Change the npy arrays to save and load accordingly
    if int(low_mass_lim) == 8:
        mass_str = '8_logM_12'
    elif int(low_mass_lim) == 9:
        mass_str = '9_logM_12'
    print("Galaxies from stellar mass cut:", len(ms_idx))

    # Now get indices based on redshift intervals
    # decided from the stellar_mass_diagnostic_plot.py
    # code. 
    # Look at the make_z_hist() function in there. 
    zp = cat['zp_minchi2'][ms_idx]
    ms = ms[ms_idx]

    # This assumes that this color array has been saved to disk
    # See the make_col_ms_plots.py code
    ur = np.load(stacking_analysis_dir + 'ur_arr_' + mass_str + '.npy')

    print(ur.shape)
    print(ms.shape)
    print(zp.shape)
    
    # --------------------- Now classify --------------------- #
    plot_color_arr = np.zeros(len(ur))


    return None

if __name__ == '__main__':
    main()
    sys.exit(0)