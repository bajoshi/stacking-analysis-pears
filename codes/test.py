from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_spectrum_indiv(flam_em, ferr, lam_em, specname):
    
    ax.plot(lam_em, flam_em, ls='-', label=specname)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

    ax.set_yscale('log')
    ax.legend(loc=0)

def plot_spectrum_median(flux, flux_err, lam, ongrid, numspec):
    
    ax.errorbar(lam, flux, yerr=flux_err, fmt='o-', color='k', linewidth=1, label=ongrid+','+numspec,\
                ecolor='r', markeredgecolor='k', capsize=0, markersize=4)
        
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)
                
    ax.legend(loc=0)

def normalize(numspec):
    maxarr = np.zeros(numspec)
    
    lam_em = lam_grid
    
    for k in range(numspec):
        
        #redshift = photz[indices][k]
        #lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][k], redshift)
        
        flam_em = allspecarr[k]
        
        # Store median of values from 4400A-4600A for each spectrum
        arg4400 = np.argmin(abs(lam_em - 4400))
        arg4600 = np.argmin(abs(lam_em - 4600))
        maxarr[k] = np.median(flam_em[arg4400:arg4600+1])
    
    # Return the maximum in array of median values
    return max(maxarr)

lam_grid = np.arange(2500, 6000, 100)

a = np.linspace(1e-15, 8e-15, len(lam_grid))
b = np.linspace(2e-18, 5e-18, len(lam_grid))
c = np.linspace(1e-18, 8e-19, len(lam_grid))
d = np.linspace(1e-18, 6e-18, len(lam_grid))
e = np.linspace(1e-19, 4e-19, len(lam_grid))

d[24] = 3e-17
d[25] = 1e-17

allspecarr = [a,b,c,d,e]

ferr = np.ones(len(lam_grid)) * 1e-19

pdfname = 'coadded_spectra_test.pdf'
pdf = PdfPages(pdfname)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('$\lambda\ [\AA]$')
ax.set_ylabel('$F_{\lambda}\ [\mathrm{norm.\ at\ 4400[\AA]<\lambda<4600[\AA]}]$')
ax.axhline(y=0,linestyle='--')

for i in range(5):
    plot_spectrum_indiv(allspecarr[i], ferr, lam_grid, str(i))

fig.subplots_adjust(top=0.92)
pdf.savefig(bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('$\lambda\ [\AA]$')
ax.set_ylabel('$F_{\lambda}\ [\mathrm{norm.\ at\ 4400[\AA]<\lambda<4600[\AA]}]$')
ax.axhline(y=0,linestyle='--')

maxval = normalize(5)
print maxval, "is the normalization value."

for j in range(5):

    flam_em = allspecarr[j]
    flam_em /= maxval

    plot_spectrum_indiv(flam_em, ferr, lam_grid, str(j))

fig.subplots_adjust(top=0.92)
pdf.savefig(bbox_inches='tight')

pdf.close()


















