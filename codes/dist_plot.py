from __future__ import division
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

import sys, glob, os, datetime
import collections

def makefig_hist(qty):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(qty)
    ax.set_ylabel('N')

    return fig, ax

if __name__ == '__main__':

    dt = datetime.datetime

    dist_list = np.genfromtxt('dist_spec.txt', dtype=None, names=['dist_spec_ind', 'sim_ind'], delimiter=',', skip_header=2)
    print len(dist_list), "distinguishable spectra."

    dist_spec = np.empty(len(dist_list), dtype=np.int)
    sim = []
    for i in range(len(dist_list)):
        dist_spec[i] = int(dist_list[i][0])
        if dist_list[i][1] != '':
            sim.append([int(j) for j in dist_list[i][1].split(' ')])
        else:
            sim.append(dist_list[i][0])

    filename = '/Users/baj/Desktop/FIGS/new_codes/all_comp_spectra.fits'
    #filename = '/Users/baj/Desktop/FIGS/new_codes/all_spectra_dist.fits'
    h = fits.open(filename, memmap=False)
    print "Comparison Spectra read in.", dt.now() # takes about 3 min

    # Block to average out parameters of similar spectra and save the distinguishable spectra.
    ages = []
    tau = []
    tauv = []
    metals = []
    for i in range(len(dist_spec)):
        ages_temp = []
        metals_temp = []
        tau_temp = []
        tauv_temp = []

        ages_temp.append(float(h[int(dist_spec[i])+1].header['LOG_AGE']))
        metals_temp.append(float(h[int(dist_spec[i])+1].header['METAL']))
        tau_temp.append(float(h[int(dist_spec[i])+1].header['TAU_GYR']))
        tauv_temp.append(float(h[int(dist_spec[i])+1].header['TAUV']))
        for j in range(len(sim[i])):
            ages_temp.append(float(h[sim[i][j]+1].header['LOG_AGE']))
            metals_temp.append(float(h[sim[i][j]+1].header['METAL']))
            tau_temp.append(float(h[int(dist_spec[i])+1].header['TAU_GYR']))
            tauv_temp.append(float(h[int(dist_spec[i])+1].header['TAUV']))

        ages.append(np.mean(ages_temp))
        metals.append(np.mean(metals_temp))
        tau.append(np.mean(tau_temp))
        tauv.append(np.mean(tauv_temp))
        #print ages_temp
        #print np.mean(ages_temp)
        #sys.exit()

    # FITS file where distinguishalbe spectra will be saved
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)

    count = 0
    for i in dist_spec:
        hdr = fits.Header()
        hdr['LOG_AGE'] = ages[count]
        hdr['METAL'] = metals[count]
        hdr['TAU_GYR'] = tau[count]
        hdr['TAUV'] = tauv[count]
        hdulist.append(fits.ImageHDU(data = h[int(i)+1].data, header=hdr))
        count += 1

    hdulist.writeto('all_spectra_dist.fits', clobber=True)

    # Block to print out what the parameters of the distinguishable spectra are and make plots
    # When running the for loop below here to make plots, make sure that you've read in the fits file 
    # containing distinguishable spectra. 
    """
    ages = []
    tau = []
    tauv = []
    metals = []    
    for i in range(len(dist_list)):
        ages.append(float(h[i+1].header['LOG_AGE']))
        metals.append(float(h[i+1].header['METAL']))
        tau.append(float(h[i+1].header['TAU']))
        tauv.append(float(h[i+1].header['TAUV']))

    ages = np.asarray(ages, dtype=np.float64)
    fig, ax = makefig_hist('Age [Gyr]')
    iqr = 1.349 * np.std(ages, dtype=np.float64)
    binsize = 2*iqr*np.power(len(ages),-1/3) # Freedman-Diaconis Rule
    totalbins = np.floor((13.85 - 0.1)/binsize)
    ax.hist(ages, totalbins, histtype='bar', align='mid', alpha=0.5)
    plt.show()

    # this block of code is useful if the metals are in their usual BC03 notation
    #ctr = collections.Counter()
    #for i in metals:
    #    ctr[i] += 1
    #fig, ax = makefig_hist('Metals')
    #ax.bar(np.arange(len(ctr.keys())), ctr.values(), alpha=0.5)
    #ax.set_xticklabels(ctr.keys())
    #ax.set_xticks(np.arange(0.4, len(ctr.keys()) + 0.4, 1))
    #plt.show()

    fig, ax = makefig_hist('Metals [Z]')
    iqr = 1.349 * np.std(metals, dtype=np.float64)
    binsize = 2*iqr*np.power(len(metals),-1/3) # Freedman-Diaconis Rule
    totalbins = np.floor((max(metals) - min(metals))/binsize)
    ax.hist(metals, totalbins, histtype='bar', align='mid', alpha=0.5)
    plt.show()

    tau = np.asarray(tau, dtype=np.float64)
    logtau = np.log10(tau)
    fig, ax = makefig_hist(r'$\mathrm{log}\tau$')
    iqr = 1.349 * np.std(logtau, dtype=np.float64)
    binsize = 2*iqr*np.power(len(logtau),-1/3) # Freedman-Diaconis Rule
    totalbins = np.floor((2 - (-2))/binsize)
    ax.hist(logtau, totalbins, histtype='bar', align='mid', alpha=0.5)
    plt.show()

    tauv = np.asarray(tauv, dtype=np.float64)
    fig, ax = makefig_hist(r'$\tau_V$')
    totalbins = np.floor((1.9 - 0.1)/0.1)
    ax.hist(tauv, totalbins, histtype='bar', align='mid', alpha=0.5)
    plt.show()
    """