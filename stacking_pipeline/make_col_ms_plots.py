from __future__ import division

import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import simps
import pysynphot

import os
import sys
import glob

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
figs_dir = home + '/Desktop/FIGS/'

stacking_analysis_dir = figs_dir + 'stacking-analysis-pears/'
stacking_figures_dir = figs_dir + 'stacking-analysis-figures/'

def compute_flam(filtername, spec, spec_wav):

    # Select filter curve based on filter name
    # These are all from pysysnphot
    # Wavelenght is in angstroms
    # Throughput is an absolute fraction
    if filtername == 'u':
        band = pysynphot.ObsBandpass('sdss,u')
    elif filtername == 'g':
        band = pysynphot.ObsBandpass('sdss,g')
    elif filtername == 'r':
        band = pysynphot.ObsBandpass('sdss,r')
    #elif filtername == 'j':

    filt_wav = band.wave
    filt_thru = band.throughput

    # Now compute magnitudes
    # First, interpolate the transmission curve to the model lam grid
    filt_interp = griddata(points=filt_wav, values=filt_thru, xi=spec_wav, method='linear')

    # Set nan values in interpolated filter to 0.0
    filt_nan_idx = np.where(np.isnan(filt_interp))[0]
    filt_interp[filt_nan_idx] = 0.0

    # Second, compute f_lambda
    den = simps(y=filt_interp, x=spec_wav)
    num = simps(y=spec * filt_interp, x=spec_wav)
    flam = num / den

    return flam

def uv_ms_plots():

    # Read in results for all of PEARS
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    # First get stellar mass and apply a cut to stellar mass
    ms = np.log10(cat['zp_ms'])
    ms_idx = np.where((ms >= 9.0) & (ms <= 12.0))[0]
    print "Galaxies from stellar mass cut:", len(ms_idx)

    # Now get indices based on redshift intervals
    # decided from the stellar_mass_diagnostic_plot.py
    # code. 
    # Look at the make_z_hist() function in there. 
    zp = cat['zp_minchi2'][ms_idx]
    ms = ms[ms_idx]
    #uv = cat['zp_uv'][ms_idx]
    ur = []

    # now loop over all galaxies within mass cut sample 
    # and get the u-r color for each galaxy

    # Read model lambda grid # In agnstroms
    model_lam_grid_withlines_mmap = np.load(figs_dir + 'model_lam_grid_withlines_chabrier.npy', mmap_mode='r')
    # Now read the model spectra # In erg s^-1 A^-1
    model_comp_spec_llam_withlines_mmap = np.load(figs_dir + 'model_comp_spec_llam_withlines_chabrier.npy', mmap_mode='r')

    for idx in ms_idx:
        # First get teh full res model spectrum
        best_model_idx = cat[idx]['zp_model_idx']
        current_spec = model_comp_spec_llam_withlines_mmap[best_model_idx]
        
        # Now get the u and r magnitudes
        uflam = compute_flam('u', current_spec, model_lam_grid_withlines_mmap)
        rflam = compute_flam('r', current_spec, model_lam_grid_withlines_mmap)
        current_ur = -2.5 * np.log10(uflam / rflam)  # Because this is a color the zeropoint doesn't matter
        ur.append(current_ur)

    # Convert to numpy array
    ur = np.asarray(ur)

    # Get z intervals and their indices
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

    # Labels
    ax5.set_xlabel(r'$\rm log(M_s)\ [M_\odot]$', fontsize=15)
    ax4.set_ylabel(r'$(u - r)_\mathrm{restframe}$', fontsize=15)
    ax4.yaxis.set_label_coords(-0.12, 1.05)

    # Actual plotting
    ax1.scatter(ms[z_interval1_idx], ur[z_interval1_idx], s=1.5, color='k')
    ax2.scatter(ms[z_interval2_idx], ur[z_interval2_idx], s=1.5, color='k')
    ax3.scatter(ms[z_interval3_idx], ur[z_interval3_idx], s=1.5, color='k')
    ax4.scatter(ms[z_interval4_idx], ur[z_interval4_idx], s=1.5, color='k')
    ax5.scatter(ms[z_interval5_idx], ur[z_interval5_idx], s=1.5, color='k')
    ax6.scatter(ms, ur, s=1.5, color='k')

    # Add text 
    add_info_text_to_subplots(ax1, 0.0, 0.4, len(z_interval1_idx))
    add_info_text_to_subplots(ax2, 0.4, 0.7, len(z_interval2_idx))
    add_info_text_to_subplots(ax3, 0.7, 1.0, len(z_interval3_idx))
    add_info_text_to_subplots(ax4, 1.0, 2.0, len(z_interval4_idx))
    add_info_text_to_subplots(ax5, 2.0, 6.0, len(z_interval5_idx))
    add_info_text_to_subplots(ax6, 0.0, 6.0, len(zp))

    # Axes limits 
    ax1.set_xlim(7.0, 13.0)
    ax2.set_xlim(7.0, 13.0)
    ax3.set_xlim(7.0, 13.0)
    ax4.set_xlim(7.0, 13.0)
    ax5.set_xlim(7.0, 13.0)
    ax6.set_xlim(7.0, 13.0)

    ax1.set_ylim(-1.0, 2.5)
    ax2.set_ylim(-1.0, 2.5)
    ax3.set_ylim(-1.0, 2.5)
    ax4.set_ylim(-1.0, 2.5)
    ax5.set_ylim(-1.0, 2.5)
    ax6.set_ylim(-1.0, 2.5)

    # Don't show x tick labels on the upper row
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    # Don't show y tick labels on the two rightmost columns
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax5.set_yticklabels([])
    ax6.set_yticklabels([])

    fig.savefig(stacking_figures_dir + 'ur_ms_diagram.pdf', dpi=300, bbox_inches='tight')

    return None

def uvj():

    # Read in results for all of PEARS
    cat = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    # First get stellar mass and apply a cut to stellar mass
    ms = np.log10(cat['zp_ms'])
    ms_idx = np.where((ms >= 9.0) & (ms <= 12.0))[0]
    print "Galaxies from stellar mass cut:", len(ms_idx)

    # Now get indices based on redshift intervals
    # decided from the stellar_mass_diagnostic_plot.py
    # code. 
    # Look at the make_z_hist() function in there. 
    zp = cat['zp_minchi2'][ms_idx]
    ms = ms[ms_idx]
    uv = cat['zp_uv'][ms_idx]
    vj = cat['zp_vj'][ms_idx]

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
    gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3)

    # Put axes on grid
    ax1 = fig.add_subplot(gs[:5, :4])
    ax2 = fig.add_subplot(gs[:5, 4:8])
    ax3 = fig.add_subplot(gs[:5, 8:])
    ax4 = fig.add_subplot(gs[5:, :4])
    ax5 = fig.add_subplot(gs[5:, 4:8])
    ax6 = fig.add_subplot(gs[5:, 8:])

    # Labels
    ax5.set_xlabel(r'$V - J$', fontsize=15)
    ax4.set_ylabel(r'$U - V$', fontsize=15)
    ax4.yaxis.set_label_coords(-0.12, 1.05)

    # Actual plotting 
    ax1.scatter(vj[z_interval1_idx], uv[z_interval1_idx], s=1.5, color='k')
    ax2.scatter(vj[z_interval2_idx], uv[z_interval2_idx], s=1.5, color='k')
    ax3.scatter(vj[z_interval3_idx], uv[z_interval3_idx], s=1.5, color='k')
    ax4.scatter(vj[z_interval4_idx], uv[z_interval4_idx], s=1.5, color='k')
    ax5.scatter(vj[z_interval5_idx], uv[z_interval5_idx], s=1.5, color='k')
    ax6.scatter(vj, uv, s=1.5, color='k')

    # Plot UVJ selection on each subplot
    plot_uvj_selection(ax1)
    plot_uvj_selection(ax2)
    plot_uvj_selection(ax3)
    plot_uvj_selection(ax4)
    plot_uvj_selection(ax5)
    plot_uvj_selection(ax6)

    # Add text 
    add_info_text_to_subplots(ax1, 0.0, 0.4, len(z_interval1_idx))
    add_info_text_to_subplots(ax2, 0.4, 0.7, len(z_interval2_idx))
    add_info_text_to_subplots(ax3, 0.7, 1.0, len(z_interval3_idx))
    add_info_text_to_subplots(ax4, 1.0, 2.0, len(z_interval4_idx))
    add_info_text_to_subplots(ax5, 2.0, 6.0, len(z_interval5_idx))
    add_info_text_to_subplots(ax6, 0.0, 6.0, len(zp))

    # Axes limits 
    ax1.set_xlim(0.0, 3.0)
    ax2.set_xlim(0.0, 3.0)
    ax3.set_xlim(0.0, 3.0)
    ax4.set_xlim(0.0, 3.0)
    ax5.set_xlim(0.0, 3.0)
    ax6.set_xlim(0.0, 3.0)

    ax1.set_ylim(-1.0, 2.5)
    ax2.set_ylim(-1.0, 2.5)
    ax3.set_ylim(-1.0, 2.5)
    ax4.set_ylim(-1.0, 2.5)
    ax5.set_ylim(-1.0, 2.5)
    ax6.set_ylim(-1.0, 2.5)

    # Don't show x tick labels on the upper row
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    # Don't show y tick labels on the two rightmost columns
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax5.set_yticklabels([])
    ax6.set_yticklabels([])

    fig.savefig(stacking_figures_dir + 'uvj_diagram.pdf', dpi=300, bbox_inches='tight')

    return None

def plot_uvj_selection(ax):

    x_qg_line = np.arange(0.0, 3.0, 0.1)  # qg stands for 'Quiescent Galaxies'
    y_qg_line = 0.8*x_qg_line + 0.7

    # I simply computed the min and max coordinates here by hand
    ax.vlines(x=1.5, ymin=1.9, ymax=3.0, color='r')
    ax.hlines(y=1.3, xmin=0.0, xmax=0.75, color='r')
    ax.plot([0.75, 1.5], [1.3, 1.9], '-', color='r')

    return None

def add_info_text_to_subplots(ax, zlow, zhigh, num):

    # add number of galaxies in plot
    ax.text(0.04, 0.2, r'$\rm N\, =\, $' + str(num), \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=15)

    # Add redshift range
    if zhigh == 6.0:
        zstr = str(zlow) + r'$\, \leq z \leq \,$' + str(zhigh)
    else:
        zstr = str(zlow) + r'$\, \leq z < \,$' + str(zhigh)

    ax.text(0.04, 0.1, zstr, \
    verticalalignment='top', horizontalalignment='left', \
    transform=ax.transAxes, color='k', size=15)

    return None

def check_salp_chab_z():

    # Read in results for all of PEARS
    cat_salp = np.genfromtxt(stacking_analysis_dir + 'full_pears_results.txt', dtype=None, names=True)
    cat_chab = np.genfromtxt(stacking_analysis_dir + 'full_pears_results_chabrier.txt', dtype=None, names=True)

    zp_salp = cat_salp['zp_minchi2']
    zp_chab = cat_chab['zp_minchi2']

    # Now vs each other
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$z^\mathrm{Salpeter}_\mathrm{phot}$', fontsize=14)
    ax.set_ylabel(r'$z^\mathrm{Chabrier}_\mathrm{phot}$', fontsize=14)

    ax.scatter(zp_salp, zp_chab, s=2.0, color='k')
    ax.plot(np.arange(-0.5, 6.5, 0.1), np.arange(-0.5, 6.5, 0.1), '--', color='r', linewidth=0.75)

    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-0.2, 6.2)

    ax.minorticks_on()

    fig.savefig(stacking_figures_dir + 'zp_salp_chab_check.pdf', dpi=300, bbox_inches='tight')

    return None

def main():

    #check_salp_chab_z()
    uv_ms_plots()
    #uvj()
    
    return None

if __name__ == '__main__':
    main()
    sys.exit(0)