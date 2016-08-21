from __future__ import division
import numpy as np
import pyfits as pf

import sys, os
import logging

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea

def plot_spectrum_data(lam, flux, flux_err):
    
    ax.errorbar(lam, flux, yerr=flux_err, fmt='o-', color='k', linewidth=2, ecolor='r', markeredgecolor='k', capsize=0, markersize=4)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)
    #ax.set_ylim(0,2)

def plot_spectrum_bc03(lam, flux):
    
    ax.plot(lam, flux, 'o-', color='r', linewidth=3)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

def resample(lam, spec):
    
    lam_em = lam
    resampled_flam = np.zeros(len(lam_grid_tofit))
    for i in range(len(lam_grid_tofit)):
        
        new_ind = np.where((lam_em >= lam_grid_tofit[i] - lam_step/2) & (lam_em < lam_grid_tofit[i] + lam_step/2))[0]    
        resampled_flam[i] = np.median(spec[new_ind])

    return resampled_flam

def get_net_sig(fitsdata, filename):

    try:
        signal_sum = 0
        noise_sum = 0
        totalsum = 0
        cumsum = []
 
        if np.count_nonzero(fitsdata['ERROR']) != len(fitsdata['ERROR']):
            raise ZeroDivisionError

        sn = fitsdata['COUNT']/fitsdata['ERROR']
        sn_sorted = np.sort(sn)
        sn_sorted_reversed = sn_sorted[::-1]

        for _count_ in range(len(fitsdata)):
            idx = np.where(sn==sn_sorted_reversed[_count_])
            signal_sum += fitsdata['COUNT'][idx]
            noise_sum += fitsdata['ERROR'][idx]**2
            totalsum = signal_sum/np.sqrt(noise_sum)
            cumsum.append(totalsum)

        netsig = np.amax(cumsum)

        return netsig

    except ValueError as detail:
        logging.warning(filename)
        #logging.warning(detail.value)
        logging.warning("The above spectrum will be given net sig of -99. Not sure of this error yet.")
    except ZeroDivisionError:
        logging.warning(filename)
        logging.warning("Division by zero! The net sig here cannot be trusted. Setting Net Sig to -99.")
        return -99.0

def fileprep(pears_index, redshift):
    
    # Get the correct filename and the number of extensions
    filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    if not os.path.isfile(filename):
        filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

    fitsfile = pf.open(filename)
    n_ext = fitsfile[0].header['NEXTEND']

    specname = os.path.basename(filename)

    # Get highest netsig to find the spectrum to be added
    if n_ext > 1:
        netsiglist = []
        for count in range(n_ext):
            fitsdata = fitsfile[count+1].data
            netsig = get_net_sig(fitsdata, filename)
            netsiglist.append(netsig)
        netsiglist = np.array(netsiglist)
        maxnetsigarg = np.argmax(netsiglist)
        spec_toadd = fitsfile[maxnetsigarg+1].data
    elif n_ext == 1:
        spec_toadd = fitsfile[1].data

    # Now get the spectrum to be added
    lam_obs = spec_toadd['LAMBDA']
    flam_obs = spec_toadd['FLUX']
    ferr = spec_toadd['FERROR']
    contam = spec_toadd['CONTAM']

    # Subtract Contamination
    flam_obs = flam_obs - contam
            
    # First chop off the ends and only look at the observed spectrum from 6000A to 9500A
    arg6000 = np.argmin(abs(lam_obs - 6000))
    arg9500 = np.argmin(abs(lam_obs - 9500))
                    
    lam_obs = lam_obs[arg6000:arg9500]
    flam_obs = flam_obs[arg6000:arg9500]
    ferr = ferr[arg6000:arg9500]
                                    
    # Now unredshift the spectrum
    lam_em = lam_obs / (1 + redshift)
    flam_em = flam_obs * (1 + redshift)
    # check the relations for unredshifting

    return lam_em, flam_em, ferr, specname

def rescale(indices):
    medarr = np.zeros(len(pears_id[indices]))
        
    for k in range(len(pears_id[indices])):
                
        redshift = photz[indices][k]
        lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][k], redshift)
        
        # Store median of values from 4400A-4600A for each spectrum
        arg4400 = np.argmin(abs(lam_em - 4400))
        arg4600 = np.argmin(abs(lam_em - 4600))
        medarr[k] = np.median(flam_em[arg4400:arg4600+1])
    
    medval = np.median(medarr)
        
    # Return the maximum in array of median values
    return medarr, medval, np.std(medarr)

def plot_spectrum_indiv(flam_em_indiv, ferr_indiv, lam_em_indiv, specname_indiv, i, j, label=False, labelmax=True):
    
    # matplotlib will not plot nan values so I'm setting 0's to nan's here.
    # This is only to make the plot look better.
    flam_em_indiv[flam_em_indiv == 0.0] = np.nan
    ferr_indiv[ferr == 0.0] = np.nan
    
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
        ax.plot(lam_em_indiv, flam_em_indiv, ls='-', color='gray')
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)


if __name__ == '__main__':

    minchi2 = 1091.7
    bestfitmetal = 'm62'
    bestfitage = 1.8
    bestfittau = 0.1584
    bestfittauV = 0.4
    bestalpha = 8.52200460963e-14
    bestfitfile = '/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/m62/bc2003_hr_m62_tauV4_csp_tau1584_salp.fits'

    if bestfitmetal == 'm22':
        bestmetallicity = '0.0001'
    elif bestfitmetal == 'm32':
        bestmetallicity = '0.0004'
    elif bestfitmetal == 'm42':
        bestmetallicity = '0.004'
    elif bestfitmetal == 'm52':
        bestmetallicity = '0.008'
    elif bestfitmetal == 'm62':
        bestmetallicity = '0.02'
    elif bestfitmetal == 'm72':
        bestmetallicity = '0.05'

    bestfitspec = pf.open(bestfitfile)
    # Read in best spec from model file
    lam = bestfitspec[1].data
    ages = bestfitspec[2].data
    bestage = bestfitage * 1e9
    minageindex = np.argmin(abs(ages - bestage))
    spec = bestfitspec[minageindex + 3].data

    lam_step = 100
    lam_lowfit = 2700
    lam_highfit = 6000
    lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)

    # Only consider the part of BC03 spectrum between 2700 to 6000
    arg2650 = np.argmin(abs(lam - 2650))
    arg5950 = np.argmin(abs(lam - 5950))
    lam = lam[arg2650:arg5950+1]
    spec = spec[arg2650:arg5950+1]
    spec = resample(lam, spec)
    lam = lam_grid_tofit

    # make plots
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\lambda\ [\AA]$')
    ax.set_ylabel('$F_{\lambda}\ [\mathrm{erg/s/cm^2/\AA}] $')
    ax.axhline(y=0,linestyle='--')

    # first plot individual galaxy spectra
    data_path = "/Users/baj/Documents/PEARS/data_spectra_only/"
    cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/color_stellarmass.txt',\
                        dtype=None, names=True, skip_header=2)
    
    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    skipspec = ['h_pears_n_id78220.fits',\
                'h_pears_n_id47644.fits', 'h_pears_s_id74234.fits', 'h_pears_s_id124266.fits',\
                'h_pears_n_id111054.fits',\
                'h_pears_s_id106446.fits',\
                'h_pears_n_id120710.fits',\
                'h_pears_s_id78417.fits',\
                'h_pears_s_id59792.fits','h_pears_s_id93218.fits','h_pears_n_id104213.fits','h_pears_s_id115422.fits','h_pears_s_id72113.fits','h_pears_s_id107858.fits','h_pears_s_id45223.fits','h_pears_s_id23920.fits',\
                'h_pears_s_id64963.fits','h_pears_s_id128268.fits',\
                'h_pears_s_id110795.fits','h_pears_s_id108561.fits','h_pears_n_id123162.fits',\
                'h_pears_s_id79154.fits','h_pears_s_id114978.fits',\
                'h_pears_s_id115024.fits','h_pears_n_id67343.fits',\
                'h_pears_s_id113895.fits',\
                'h_pears_n_id79581.fits',\
                'h_pears_s_id120441.fits','h_pears_n_id77819.fits',\
                'h_pears_s_id48182.fits',\
                'h_pears_n_id103918.fits',\
                'h_pears_s_id70340.fits','h_pears_n_id110223.fits','h_pears_n_id59873.fits',\
                'h_pears_s_id56575.fits',\
                'h_pears_s_id116349.fits','h_pears_n_id103918.fits',\
                'h_pears_n_id42132.fits','h_pears_n_id56602.fits', 'h_pears_n_id122827.fits']

    color_step = 0.3
    mstar_step = 0.5
    for i in np.arange(2.1, 2.4, color_step):
        for j in np.arange(10.5, 11.0, mstar_step):
            
            indices = np.where((ur_color >= i) & (ur_color < i + color_step) &\
                            (stellarmass >= j) & (stellarmass < j + mstar_step))[0]

            if indices.size:
                medarr, medval, stdval = rescale(indices)
            else:
                continue
                           
            for u in range(len(pears_id[indices])):
                # Get redshift from previously saved 3DHST photz catalog
                redshift = photz[indices][u]
                
                # Get rest frame values for all quantities
                lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][u], redshift)
                    
                # Divide by max value to rescale
                flam_em = (flam_em / medarr[u]) * medval
                ferr = (ferr / medarr[u]) * medval

                # Reject spectrum if overall contamination too high
                if np.sum(abs(ferr)) > 0.3 * np.sum(abs(flam_em)):
                    #print "Skipped", specname
                    continue

                # These were looked at by eye and seemed crappy
                if specname in skipspec:
                    #print "Skipped", specname
                    continue
                else:
                    #print 'plotting ', specname
                    plot_spectrum_indiv(flam_em, ferr, lam_em, specname, i, j, label=False, labelmax=False) # plot individual spectrum

    boxtext = 'Best Fit Parameters' + '\n' + 'Age = ' + str(bestfitage) + ' Gyr' + '\n' + 'Z = ' + bestmetallicity + '\n' + r'$\tau$ = ' + str(bestfittau) + ' Gyr' + '\n' + r'$\mathrm{A_V}$ = ' + str(bestfittauV)
    box = TextArea(boxtext, textprops=dict(color="k"))
    anchored_box = AnchoredOffsetbox(loc=3, child=box, pad=0.3, frameon=True,\
                         bbox_to_anchor=(0.7, 0.12),\
                         bbox_transform=ax.transAxes, borderpad=0.1)
    ax.add_artist(anchored_box)

    box = TextArea('2.1 < (U-R) < 2.4' + '\n' + r'$10.5 <\ \mathrm{log\left(\frac{M_s}{M_\odot}\right)} < 11.0$', textprops=dict(color="k"))
    anchored_box = AnchoredOffsetbox(loc=3, child=box, pad=0.3, frameon=True,\
                         bbox_to_anchor=(0.7, 0.37),\
                         bbox_transform=ax.transAxes, borderpad=0.1)
    ax.add_artist(anchored_box)

    # read in median stacked spectra
    stacks = pf.open('/Users/baj/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra.fits')
    #stacks = pf.open('/Users/baj/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra_coarsegrid.fits')
    flam = stacks[49].data[0]
    ferr = stacks[49].data[1]
    ongrid = stacks[49].header['ONGRID']
    urcol = float(ongrid.split(',')[0])
    stmass = float(ongrid.split(',')[1])

    # Chop off the ends of the stacked spectrum
    orig_lam_grid = np.arange(2700, 6000, lam_step)
    arg_lamlow = np.argmin(abs(orig_lam_grid - lam_lowfit))
    arg_lamhigh = np.argmin(abs(orig_lam_grid - lam_highfit-100))
    flam = flam[arg_lamlow:arg_lamhigh+1]
    ferr = ferr[arg_lamlow:arg_lamhigh+1]

    plot_spectrum_bc03(lam, spec*bestalpha)
    plot_spectrum_data(lam_grid_tofit, flam, ferr)
    fig.savefig('redseq_fit.png', dpi=300, bbox_inches='tight')
    #plt.show()


