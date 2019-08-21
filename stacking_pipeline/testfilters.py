from astropy.io import fits

import pysynphot

import os
import sys

import matplotlib.pyplot as plt

def main():

    # Read in all filters
    uband = pysynphot.ObsBandpass('sdss,u')
    gband = pysynphot.ObsBandpass('sdss,g')
    #vband = pysynphot.ObsBandpass('sdss,u')
    rband = pysynphot.ObsBandpass('sdss,r')
    #jband = pysynphot.ObsBandpass('sdss,u')

    # Now plot them to check
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel(r'$\mathrm{Wavelength\ [\AA]}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{Throughput}$', fontsize=15)

    ax.plot(uband.wave, uband.throughput, color='violet', label='SDSS u')
    ax.plot(gband.wave, gband.throughput, color='green', label='SDSS g')
    #ax.plot(vband.wave, vband.throughput, color='cyan', label='')
    ax.plot(rband.wave, rband.throughput, color='red', label='SDSS r')
    #ax.plot(jband.wave, jband.throughput, color='magenta', label='')

    ax.legend()

    ax.minorticks_on()

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)