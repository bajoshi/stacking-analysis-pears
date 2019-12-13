from __future__ import division

import numpy as np
import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck15

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

    verbose = False

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
    
    # --------------------- Classify and get initial SFRs --------------------- #
    # --------------------- and then plot --------------------- #
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    formation_redshift_arr = np.zeros(len(ms))

    formation_redshift_redseq = []
    formation_redshift_greenval = []
    formation_redshift_bluecloud = []

    tau_redseq = []
    tau_greenval = []
    tau_bluecloud = []

    ages_arr = np.zeros(len(ms))

    for i in range(len(ms)):

        # Get initial SFR
        current_tau = cat['zp_tau'][ms_idx][i] * 1e9
        current_age = 10**(cat['zp_age'][ms_idx][i]) 
        if cat['zp_tau'][ms_idx][i] == -99.0:
            continue

        current_initial_sfr = get_initial_sfr(current_tau, current_age, 10**ms[i])

        # Now get formation redshift
        current_z = zp[i]
        univ_age_at_z = Planck15.age(current_z) # in Gyr
        univ_age_at_formation = univ_age_at_z - (current_age/1e9)*u.Gyr

        if univ_age_at_z < (current_age/1e9)*u.Gyr:
            print("This galaxy is supposedly older than the Universe.")
            print("How did this galaxy get through the fitting pipeline?")
            print("i index:", i)
            continue

        formation_redshift = z_at_value(Planck15.age, univ_age_at_formation)

        formation_redshift_arr[i] = formation_redshift
        ages_arr[i] = current_age

        # Do classification 
        if ur[i] <= 1.50:
            current_color = 'b'
            formation_redshift_bluecloud.append(formation_redshift)
            tau_bluecloud.append(current_tau/1e9)
        elif (ur[i] > 1.50) and (ur[i] <= 2.0):
            current_color = 'g'
            formation_redshift_greenval.append(formation_redshift)
            tau_greenval.append(current_tau/1e9)
        elif ur[i] > 2.0:
            current_color = 'r'
            formation_redshift_redseq.append(formation_redshift)
            tau_redseq.append(current_tau/1e9)

        # If you want some info to be shown in the terminal
        if verbose:
            print('\n', "Plotting color:", current_color)
            print("Tau [Gyr]:", current_tau/1e9)
            print("Age [Gyr]:", current_age/1e9)
            print("log(Stellar mass [M_sol]):", ms[i])

            print("Initial SFR [M_sol/yr]:", current_initial_sfr)
            print("(u-r) color:", ur[i])
            print("Galaxy redshift:", current_z)

            print("Age of the Universe at galaxy observed redshift:", univ_age_at_z)
            print("Age of the Universe at galaxy formation redshift:", univ_age_at_formation)
            print("Formation redshift:", formation_redshift)

        if current_color != 'b':
            continue
        
        # Actual plotting
        redshifts = np.linspace(formation_redshift, current_z, 50)  # redshift steps to plot at
        xarr = np.linspace(0, current_age, 50)  # time steps
        yarr = current_initial_sfr * np.exp(-1 * xarr/current_tau)
        ax1.plot(redshifts, yarr, color=current_color, alpha=0.7)

    ax1.set_ylim(0.01, 1000)
    ax1.set_yscale('log')

    ax1.set_xlim(0, 50)

    ax1.set_xlabel(r'$\mathrm{Redshift}$', fontsize=15)
    ax1.set_ylabel(r'$\mathrm{Star-formation\ rate\ [M_\odot\, yr^{-1}]}$', fontsize=15)

    print("\n")
    print("Max for formation_redshift:", max(formation_redshift_arr))
    print("Max Age [Gyr]:", max(ages_arr/1e9))
    max_age_idx = np.argmax(ages_arr)

    print("Redshift of oldest galaxy:", zp[max_age_idx])
    print("Formation redshift of oldest galaxy:", formation_redshift_arr[max_age_idx])

    fig.savefig(stacking_figures_dir + 'blue_cloud_sfhs.pdf', dpi=300, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()

    """
    # parallel x axis for age of the Universe
    # This solution came from 
    # http://www.astropy.org/astropy-tutorials/edshift_plot.html
    ax2 = ax1.twiny()

    ages = np.linspace(1.0,13.5,5)*u.Gyr
    ageticks = [z_at_value(Planck15.age, age) for age in ages]
    ages_ticklabels = ['{:g}'.format(age) for age in ages.value]

    ax2.set_xticks(ageticks)
    ax2.set_xticklabels(ages_ticklabels)
    """

    # Formation redshift histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(formation_redshift_redseq, 60, range=(0,30), color='r', histtype='step', density=True)
    ax.hist(formation_redshift_greenval, 60, range=(0,30), color='g', histtype='step', density=True)
    ax.hist(formation_redshift_bluecloud, 60, range=(0,30), color='b', histtype='step', density=True)

    # Labels
    ax.set_xlabel(r'$\mathrm{Formation\ Redshift}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{\#\ objects\ [normalized]}$', fontsize=15)

    # Make tick labels larger
    ax.set_yticks(np.arange(0.0, 0.5, 0.1))
    ax.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4'], size=10)
    ax.set_xticklabels(ax.get_xticks().tolist(), size=10)
    #ax.set_yticklabels(ax.get_yticks().tolist(), size=10)

    fig.savefig(stacking_figures_dir + 'formation_redshifts.pdf', dpi=300, bbox_inches='tight')

    # Tau histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(np.log10(tau_redseq), 40, range=(-2,2), color='r', histtype='step', density=True)
    ax.hist(np.log10(tau_greenval), 40, range=(-2,2), color='g', histtype='step', density=True)
    ax.hist(np.log10(tau_bluecloud), 40, range=(-2,2), color='b', histtype='step', density=True)

    # Labels
    ax.set_xlabel(r'$\mathrm{SFH\ timescale\ log(\tau)\ [Gyr]}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{\#\ objects\ [normalized]}$', fontsize=15)

    # Make tick labels larger
    ax.set_xticklabels(ax.get_xticks().tolist(), size=10)
    ax.set_yticklabels(ax.get_yticks().tolist(), size=10)

    fig.savefig(stacking_figures_dir + 'taus.pdf', dpi=300, bbox_inches='tight')

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)