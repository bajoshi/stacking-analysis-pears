from __future__ import division

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

import sys
import glob
import os

home = os.getenv('HOME')  # Does not have a trailing slash at the end
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
figures_dir = stacking_analysis_dir + "figures/"
savefits_dir = home + "/Desktop/FIGS/new_codes/"

def resample(lam, spec, lam_grid_tofit, lam_step, total_ages):
    
    lam_em = lam
    resampled_flam = np.zeros((total_ages, len(lam_grid_tofit)))

    for i in range(len(lam_grid_tofit)):
        new_ind = np.where((lam_em >= lam_grid_tofit[i] - lam_step/2) & (lam_em < lam_grid_tofit[i] + lam_step/2))[0]
        resampled_flam[:,i] = np.median(spec[:,[new_ind]], axis=2).reshape(total_ages)

    return resampled_flam

def normalize(spec):

    return np.mean(spec, axis=1)

def makefig():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\lambda\ [\AA]$')
    ax.set_ylabel('$F_{\lambda}\ [\mathrm{arbitrary\ units}]$')
    ax.axhline(y=0,linestyle='--')

    return fig, ax

def plot_spectrum_indiv(ax, flam_em_indiv, lam_em_indiv, currage, label=False, labelmax=False):
    
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
        ax.plot(lam_em_indiv, flam_em_indiv, ls='-', label=str(currage)[:5])
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)
    ax.legend(loc=0, ncol=5, prop={'size':8})

def updaterot(event):
    """Event to update the rotation of the labels"""
    xs = ax.transData.transform(zip(x[-2::],y[-2::]))
    rot = np.rad2deg(np.arctan2(*np.abs(np.gradient(xs)[0][0][::-1])))
    ltex.set_rotation(rot)

if __name__ == '__main__':

    cspout = '/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/'
    metals = ['m22','m32','m42','m52','m62','m72']
    #metals = ['m72']

    lam_step = 100
    lam_lowfit = 3600
    lam_highfit = 6500
    lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)

    example = fits.open('/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/m62/bc2003_hr_m62_tauV0_csp_tau100_salp.fits')
    ages = example[2].data[1:]
    age_ind = np.where((ages/1e9 < 8) & (ages/1e9 > 0.1))[0]
    total_ages = int(len(age_ind))

    # FITS file where the reduced number of spectra will be saved
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)

    # read in BC03 spectra
    # I've restricted tauV, tau, lambda, and ages in distinguishing spectra
    tauVarr = np.arange(0.0, 2.0, 0.1)
    logtauarr = np.arange(-2, 2, 0.2)
    tauarr = np.empty(len(logtauarr)).astype(np.str)

    for i in range(len(logtauarr)):
        tauarr[i] = str(int(float(str(10**logtauarr[i])[0:6])*10000))

    for metallicity in metals:
        metalfolder = metallicity + '/'
        for tauVarrval in tauVarr:
            for tauval in tauarr:
                filename = cspout + metalfolder + 'bc2003_hr_' + metallicity + '_tauV' + str(int(tauVarrval*10)) + '_csp_tau' + tauval + '_salp.fits'
                #print filename
                h = fits.open(filename, memmap=False)
                currentlam = h[1].data

                currentspec = np.zeros([total_ages, len(currentlam)], dtype=np.float64)
                for i in range(total_ages):
                    currentspec[i] = h[age_ind[i]+3].data

                # Only consider the part of the model spectrum between low and high lambda limits defined above
                #arg_lowfit = np.argmin(abs(currentlam - lam_lowfit))
                #arg_highfit = np.argmin(abs(currentlam - lam_highfit))
                #currentlam = currentlam[arg_lowfit:arg_highfit+1]  # chopping off unrequired lambda
                #currentspec = currentspec[:,arg_lowfit:arg_highfit+1]  # chopping off unrequired spectrum

                currentspec = resample(currentlam, currentspec, lam_grid_tofit, lam_step, total_ages)
                currentlam = lam_grid_tofit

                #meanvals = normalize(currentspec)
                #meanvals = meanvals.reshape(total_ages,1)
                #currentspec = np.divide(currentspec, meanvals)

                """
                fig, ax = makefig()
                # This block of code, to make inline labels, including the function updaterot came from a stackoverflow user.
                for i in range(total_ages):
                    line_string = str(ages[age_ind[i]]/1e9)[:5] # This is the string that should show somewhere over the plotted line.
                    l, = ax.plot(lam_grid_tofit, currentspec[i], label=line_string)

                    # transform data points to screen space
                    x = lam_grid_tofit
                    y = currentspec[i]
                    pos = [(x[2]+x[1])/2., (y[2]+y[1])/2.]
                    xscreen = ax.transData.transform(zip(x[-2::],y[-2::]))
                    rot = np.rad2deg(np.arctan2(*np.abs(np.gradient(xscreen)[0][0][::-1])))
                    ltex = plt.text(pos[0], pos[1], line_string, size=9, rotation=rot, color = l.get_color(), ha="center", va="center",bbox = dict(ec='1',fc='1'))

                fig.canvas.mpl_connect('button_release_event', updaterot)

                plt.show()
                sys.exit()
                """

                for i in range(total_ages):
                    hdr = fits.Header()
                    hdr['LOG_AGE'] = str(np.log10(ages[age_ind[i]]))
    
                    if metallicity == 'm22':
                        metal_val = 0.0001
                    elif metallicity == 'm32':
                        metal_val = 0.0004
                    elif metallicity == 'm42':
                        metal_val = 0.004
                    elif metallicity == 'm52':
                        metal_val = 0.008
                    elif metallicity == 'm62':
                        metal_val = 0.02
                    elif metallicity == 'm72':
                        metal_val = 0.05

                    hdr['METAL'] = str(metal_val)
                    hdr['TAU_GYR'] = str(float(tauval)/1e4)
                    hdr['TAUV'] = str(float(tauVarrval)/10)
                    hdulist.append(fits.ImageHDU(data = currentspec[i], header=hdr))

    hdulist.writeto(savefits_dir + 'all_comp_spectra_bc03.fits', clobber=True)

