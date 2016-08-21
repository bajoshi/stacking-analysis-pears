from __future__ import division
import numpy as np
import pyfits as pf

import datetime, sys, os
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_spectrum_data(lam, flux, flux_err):
    
    ax.errorbar(lam, flux, yerr=flux_err, fmt='o-', color='k', linewidth=2, ecolor='r', markeredgecolor='k', capsize=0, markersize=4)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)
    #ax.set_ylim(0,2)

def plot_spectrum_bc03(lam, flux, bestparams, legendstyle):
    
    ax.plot(lam, flux, 'o-', color='r', linewidth=3, label=bestparams)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)
    ax.legend(loc=0, numpoints=1, prop={'size':12, 'weight':legendstyle})

def shade_em_lines():

    ymin, ymax = ax.get_ylim()
    ax.fill_between(x=np.arange(4800,5200,100), y1=ymax, y2=ymin, facecolor='gray', alpha=0.5)

def resample(lam, spec):
    
    lam_em = lam
    resampled_flam = np.zeros(len(lam_grid_tofit))
    for i in range(len(lam_grid_tofit)):
        
        new_ind = np.where((lam_em >= lam_grid_tofit[i] - lam_step/2) & (lam_em < lam_grid_tofit[i] + lam_step/2))[0]    
        resampled_flam[i] = np.median(spec[new_ind])

    return resampled_flam

def rescale(lam, spec):
    
    lam_em = lam
    
    arg4400 = np.argmin(abs(lam_em - 4400))
    arg4600 = np.argmin(abs(lam_em - 4600))
    medval = np.median(spec[arg4400:arg4600+1])
    
    return medval

if __name__ == '__main__':

    #minchi2 = 161.23
    #bestfitmetal = 'm62'
    #bestfitage = 10.0
    #bestfittau = 3.981
    #bestfittauV = 2.3
    #bestalpha = 1.70119548583e-13
    #bestfitfile = '/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/m62/bc2003_hr_m62_tauV23_csp_tau39810_salp.fits'

    fitparams = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/fitparams.txt', dtype=None, names=True, delimiter='  ')
    pdfname = 'overplot_fits.pdf'
    pdf = PdfPages(pdfname)

    modelpath = '/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/'

    count = 0
    for bestfitfile in fitparams['filename']:
        ongrid = fitparams['ongrid'][count]
        minchi2 = fitparams['chi2'][count]
        bestfitmetal = fitparams['metals'][count]
        bestfitage = fitparams['age'][count]
        bestfittau = fitparams['tau'][count]
        bestfittauV = fitparams['tauV'][count]
        bestalpha = fitparams['alpha'][count]
        stackcount = int(fitparams['stackcount'][count])

        bestfitspec = pf.open(modelpath + bestfitmetal + '/' + bestfitfile)
        bestparams = ongrid + '\n' + '$\chi^2 = $' + str(minchi2) + '\n' + bestfitmetal + '\n' + str(bestfitage)[:6] + 'Gyr' + '\n' + 'tau' + str(bestfittau) + 'Gyr'+ '\n' + 'tauV' + str(bestfittauV)
        if (bestfittau == 0.01) or (bestfittau == 79.432) or (bestfittauV == 0.0) or (bestfittauV == 2.9) or (bestfitage == 20.0) or (bestfitage >= 8.0):
            legendstyle = 'bold'
        else:
            legendstyle = 'normal'
        # this really should be an anchored box not a legend to be displayed better
    
        # Read in best spec from model file
        lam = bestfitspec[1].data
        ages = bestfitspec[2].data
        bestage = bestfitage * 1e9
        minageindex = np.argmin(abs(ages - bestage))
        spec = bestfitspec[minageindex + 3].data
        
        # read in median stacked spectra
        stacks = pf.open('/Users/baj/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra.fits')
        lam_step = 100
        lam_lowfit = 2700
        lam_highfit = 6000
        lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)
        flam = stacks[stackcount].data[0]
        ferr = stacks[stackcount].data[1]
        urcol = float(ongrid.split(',')[0])
        stmass = float(ongrid.split(',')[1])
    
        ## Chop off the ends of the stacked spectrum
        #orig_lam_grid = np.arange(2700, 6000, lam_step)
        #arg_lamlow = np.argmin(abs(orig_lam_grid - lam_lowfit))
        #arg_lamhigh = np.argmin(abs(orig_lam_grid - lam_highfit-100))
        #flam = flam[arg_lamlow:arg_lamhigh+1]
        #ferr = ferr[arg_lamlow:arg_lamhigh+1]
    
        # make plots
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('$\lambda\ [\AA]$')
        ax.set_ylabel('$F_{\lambda}\ [\mathrm{erg/s/cm^2/\AA}] $')
        ax.axhline(y=0,linestyle='--')
        
        # Only consider the part of BC03 spectrum between 2700 to 6000
        arg2650 = np.argmin(abs(lam - 2650))
        arg5950 = np.argmin(abs(lam - 5950))
        lam = lam[arg2650:arg5950+1]
        spec = spec[arg2650:arg5950+1]
    
        spec = resample(lam, spec)
        lam = lam_grid_tofit
    
        print ongrid, "AT", stacks[stackcount].header['ONGRID']
        plot_spectrum_data(lam_grid_tofit, flam, ferr)
        plot_spectrum_bc03(lam, spec*bestalpha, bestparams, legendstyle)
        if urcol <= 1.2:
            shade_em_lines()
        pdf.savefig(bbox_inches='tight')
        #plt.show()
        count += 1

    pdf.close()
