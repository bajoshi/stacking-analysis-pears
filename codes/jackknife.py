from __future__ import division
import numpy as np
import numpy.random as npr
from astropy.io import fits

import matplotlib.pyplot as plt

import grid_coadd as gd
import sys

def makefig():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\lambda\ [\AA]$')
    ax.set_ylabel('$F_{\lambda}\ [\mathrm{arbitrary\ units}]$')
    ax.axhline(y=0,linestyle='--')

    return fig, ax

def plot_spectrum_indiv(ax, flam_em_indiv, lam_em_indiv, label=False, labelmax=False):
    
    # matplotlib will not plot nan values so I'm setting 0's to nan's here.
    # This is only to make the plot look better.
    flam_em_indiv[flam_em_indiv == 0.0] = np.nan
    
    if label:
        # with label
        ax.plot(lam_em_indiv, flam_em_indiv, ls='-', color='gray')
        if labelmax:
            max_flam_arg = np.argmax(flam_em_indiv)
            max_flam = flam_em_indiv[max_flam_arg]
            max_flam_lam = lam_em_indiv[max_flam_arg]
            #print max_flam, max_flam_lam
            
            ax.annotate(specname_indiv, xy=(max_flam_lam,max_flam), xytext=(max_flam_lam,max_flam),\
                        arrowprops=dict(arrowstyle="->"))

        else:
            min_flam_arg = np.argmin(flam_em_indiv)
            min_flam = flam_em_indiv[min_flam_arg]
            min_flam_lam = lam_em_indiv[min_flam_arg]
            #print min_flam, min_flam_lam
            
            ax.annotate(specname_indiv, xy=(min_flam_lam,min_flam), xytext=(min_flam_lam,min_flam),\
                        arrowprops=dict(arrowstyle="->"))
    else:
        # default execution
        # without label
        ax.plot(lam_em_indiv, flam_em_indiv, color='gray', ls='-')
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

def plot_spectrum_median(flux, flux_err, lam_em):
    
    # matplotlib will not plot nan values so I'm setting 0's to nan's here.
    # This is only to make the plot look better.
    flux[flux == 0.0] = np.nan
    flux_err[flux_err == 0.0] = np.nan
    
    ax.errorbar(lam_em, flux, yerr=flux_err, fmt='o-', color='k', zorder=10, linewidth=2, ecolor='r', markeredgecolor='k', capsize=0, markersize=3)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

if __name__ == '__main__':

    stacks = fits.open('/Users/baj/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra.fits')

    lam_step = 100
    lam_lowfit = 3000
    lam_highfit = 6000
    lam_grid = np.arange(lam_lowfit, lam_highfit, lam_step)

    num_samp_to_draw = 1e5

    for stackcount in range(47,48,1):

        flam = stacks[stackcount+2].data[0]
        ferr = stacks[stackcount+2].data[1]

        #ferr = ferr + 0.05 * flam # putting in an additional error bar

        # Chop off the ends of the stacked spectrum
        orig_lam_grid = np.arange(2700, 6000, lam_step)
        arg_lamlow = np.argmin(abs(orig_lam_grid - lam_lowfit))
        arg_lamhigh = np.argmin(abs(orig_lam_grid - lam_highfit+100))
        flam = flam[arg_lamlow:arg_lamhigh+1]
        ferr = ferr[arg_lamlow:arg_lamhigh+1]     

        resampled_spec = np.empty((len(flam), num_samp_to_draw))

        for i in range(len(flam)):
            resampled_spec[i] = np.random.normal(flam[i], ferr[i], num_samp_to_draw)

        resampled_spec = resampled_spec.T

        # block to plot all randomly sampled spectra with the original one
        """
        fig, ax = makefig()

        plot_spectrum_median(flam, ferr, lam_grid)

        for i in range(100): # don't have to plot all of the to see if the random sampling was done right.
            plot_spectrum_indiv(ax, resampled_spec[i], lam_grid)

        plt.show()
        """

        f = open('spectra_jackknife_' + '2p1_10p5' +'.txt', 'wa')
        for i in range(int(num_samp_to_draw)):

            f.write(str(resampled_spec[i]) + '\n')

        f.close()








