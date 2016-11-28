from __future__ import division
import numpy as np
from astropy.io import fits

import datetime
import sys
import os
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea

import grid_coadd as gd

def cycle_through(flam_em, ferr, lam_em, specname):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\lambda\ [\AA]$')
    ax.set_ylabel('$F_{\lambda}\ [\mathrm{arbitrary\ units}]$')
    ax.axhline(y=0,linestyle='--')

    # with label
    ax.plot(lam_em, flam_em, ls='-', color='gray', label=specname)
    ax.legend(loc=0)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

    plt.show()

    return None

def plot_spectrum_indiv(flam_em_indiv, ferr_indiv, lam_em_indiv, specname_indiv, label=False, labelmax=True):
    
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

    return None

def plot_spectrum_median(flux, flux_err, lam_em, ongrid, numspec, ymax_lim):
    
    # matplotlib will not plot nan values so I'm setting 0's to nan's here.
    # This is only to make the plot look better.
    flux[flux == 0.0] = np.nan
    flux_err[flux_err == 0.0] = np.nan
    
    ax.errorbar(lam_em, flux, yerr=flux_err, fmt='o-', color='k', linewidth=2, label=ongrid+','+numspec,\
                ecolor='r', markeredgecolor='k', capsize=0, markersize=4)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

    #ymin, ymax = ax.get_ylim()
    ax.set_ylim(0,ymax_lim)
    #ax.set_yscale('log')
    ax.legend(loc=0)

    return None

def get_ylim_stack(flux, flux_err, lam_em):

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)    
    ax1.errorbar(lam_em, flux, yerr=flux_err)
    ymin, ymax = ax1.get_ylim()

    del fig1, ax1
    return ymin, ymax

def plot_spectrum_bc03(lam, flux, bestparams, legendstyle):
    
    ax.plot(lam, flux, 'o-', color='r', linewidth=2)
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

    box = TextArea(bestparams, textprops=dict(size=10,color="k",weight=legendstyle))
    
    anchored_box = AnchoredOffsetbox(loc=3, child=box, pad=0.0, frameon=False,\
                                     bbox_to_anchor=(0.85, 0.05),\
                                     bbox_transform=ax.transAxes, borderpad=0.0)
    ax.add_artist(anchored_box)

    return None

def shade_em_lines():

    ymin, ymax = ax.get_ylim()
    ax.fill_between(x=np.arange(4800,5200,100), y1=ymax, y2=ymin, facecolor='gray', alpha=0.5)

    return None

def resample_bc03(lam_model, spec):
    
    lam_em = lam_model
    resampled_flam = np.zeros(len(lam_grid_tofit))
    for i in range(len(lam_grid_tofit)):
        
        new_ind = np.where((lam_em >= lam_grid_tofit[i] - lam_step/2) & (lam_em < lam_grid_tofit[i] + lam_step/2))[0]    
        resampled_flam[i] = np.median(spec[new_ind])

    return resampled_flam

if __name__ == '__main__':
    
    # Logging start info
    logging.basicConfig(filename='coadd.log', format='%(levelname)s:%(message)s', filemode='a', level=logging.DEBUG)
    dt = datetime.datetime
    logging.info("\n Plots started at --")
    logging.info(dt.now())
    
    # Prep for normalizing and plotting

    # Change only the parameters here to change how the code runs
    # Ideally you shouldn't have to change anything else.
    col_step = 0.3
    mstar_step = 0.5
    final_fits_filename = 'coadded_PEARSgrismspectra.fits'
    pdfname = 'coadded_spectra.pdf'
    col_low = 0.0
    col_high = 3.0
    mstar_low = 7.0
    mstar_high = 12.0

    h = fits.open('/Users/baj/Desktop/FIGS/new_codes/' + final_fits_filename)
    lam = h[1].data

    pdf = PdfPages(pdfname)

    data_path = "/Users/baj/Documents/PEARS/data_spectra_only/"
    threedphot = "/Users/baj/Documents/3D-HST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat"
    threed = fits.open('/Users/baj/Documents/3D-HST/3dhst.v4.1.5.master.fits')
    cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/color_stellarmass.txt',\
                        dtype=None, names=True, skip_header=2)
    
    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']

    modelpath = '/Users/baj/Documents/GALAXEV_BC03/bc03/src/cspout_new/'
    fitparams = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/fitparams.txt', dtype=None, names=True, delimiter='  ')
    ongrid_arr = fitparams['ongrid']
    
    skipspec = np.loadtxt(home + '/Desktop/FIGS/stacking-analysis-pears/specskip.txt', dtype=np.str, delimiter=',')
    em_lines = np.loadtxt(home + '/Desktop/FIGS/stacking-analysis-pears/em_lines_readin.txt', dtype=np.str, delimiter=',')
    # This little for loop is to fix formatting issues with the skipspec and em_lines arrays that are read in with loadtxt.
    for i in range(len(skipspec)):
        skipspec[i] = skipspec[i].replace('\'', '')
    for i in range(len(em_lines)):
        em_lines[i] = em_lines[i].replace('\'', '')
    
    # Find indices to be plotted
    cellcount = 0
    indiv_count = 0
    totalgalaxies = 0
    totalgalaxiesstacks = 0
    for i in np.arange(col_low, col_high, col_step):
        for j in np.arange(mstar_low, mstar_high, mstar_step):
            
            indices = np.where((ur_color >= i) & (ur_color < i + col_step) &\
                            (stellarmass >= j) & (stellarmass < j + mstar_step))[0]
                            
            print "ONGRID", i, j
            
            if len(indices) < 5:
                print "Too few spectra in stack. Continuing to the next grid cell..."
                cellcount += 1
                continue

            if indices.size:
                medarr, medval, stdval = rescale(pears_id[indices], photz[indices])
            else:
                continue

            # Running this once actually to get limits but it will also plot the spectrum once and delete the figure
            flam = h[cellcount+2].data[0]
            ferr = h[cellcount+2].data[1]
            ymin, ymax = get_ylim_stack(flam, ferr, lam)

            # Create plot for each grid cell
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('$\lambda\ [\AA]$')
            ax.set_ylabel('$F_{\lambda}\ [\mathrm{erg/s/cm^{-2}/\AA}]$')
            ax.axhline(y=0,linestyle='--')
            
            """
            ##### Start Test Block #####
            # Run this block if you want to cycle through individual spectra or plot all indiv spectra at once
            for x in range(len(pears_id[indices])):
                # Get redshift from previously saved 3DHST photz catalog
                redshift = photz[indices][x]
                
                # Get rest frame values for all quantities
                lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][x], redshift)
                
                # Divide by median value at 4400A to 4600A to rescale. Multiplying by median value of the flux medians to get it back to physical units
                flam_em = (flam_em / medarr[x]) * medval
                ferr = (ferr / medarr[x]) * medval
                
                # Reject spectrum if overall contamination too high
                if np.sum(abs(ferr)) > 0.3 * np.sum(abs(flam_em)):
                    #print "Skipped", specname
                    continue

                # These were looked at by eye and seemed crappy
                if specname in skipspec:
                    #print "Skipped", specname
                    continue
                else:
                    cycle_through(flam_em, ferr, lam_em, specname)
                    #plot_spectrum_indiv(flam_em, ferr, lam_em, specname, i, j, label=True, labelmax=True)

            plt.show()        
            sys.exit()
            ##### End Test Block #####
            """

            #box = TextArea('$\ \mathrm{erg/s/cm^{-2}/\AA}$', textprops=dict(color="k"))
            #anchored_box = AnchoredOffsetbox(loc=3, child=box, pad=0.0, frameon=False,\
            #                     bbox_to_anchor=(0.1, 1.004),\
            #                     bbox_transform=ax.transAxes, borderpad=0.0)
            #ax.add_artist(anchored_box)

            # Loop over all spectra in a grid cell and plot them
            for u in range(len(pears_id[indices])):
                """
                # Create plot for each grid cell
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_xlabel('$\lambda\ [\AA]$')
                ax.set_ylabel('$F_{\lambda}\ [\mathrm{norm.\ at\ 4400[\AA]<\lambda<4600[\AA]}]$')
                ax.axhline(y=0,linestyle='--')
            
                box = TextArea("{:.2e}".format(maxval) + '$\ \mathrm{erg/s/cm^{-2}/\AA}$', textprops=dict(color="k"))
            
                anchored_box = AnchoredOffsetbox(loc=3, child=box, pad=0.0, frameon=False,\
                                                 bbox_to_anchor=(0.0, 1.02),\
                                                 bbox_transform=ax.transAxes, borderpad=0.0)
                ax.add_artist(anchored_box)
                """
                # Get redshift from previously saved 3DHST photz catalog
                redshift = photz[indices][u]
                
                # Get rest frame values for all quantities
                lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][u], redshift)
                    
                # Divide by max value to rescale
                flam_em = (flam_em / medarr[u]) * medval
                ferr = (ferr / medarr[u]) * medval

                # Plotting
                """
                # Set any negative points in spectrum to 0.
                # If more than 30% of the points in a spectrum are negative then skip it.
                if np.any(flam_em < 0.0):
                    if len(np.where(flam_em < 0.0)[0]) > 0.3 * len(flam_em):
                        continue
                    flam_em_neg_ind = np.where(flam_em < 0.0)[0]
                    flam_em[flam_em_neg_ind] = 0.0
                    ferr[flam_em_neg_ind] = 0.0
                """
                # Reject spectrum if overall contamination too high
                if np.sum(abs(ferr)) > 0.3 * np.sum(abs(flam_em)):
                    #print "Skipped", specname
                    continue

                # These (skipspec) were looked at by eye and seemed crappy
                # I'm also excluding spectra with emission lines.
                if (specname in skipspec) or (specname in em_lines):
                    #print "Skipped", specname
                    continue
                else:
                    indiv_count += 1
                    plot_spectrum_indiv(flam_em, ferr, lam_em, specname, label=False, labelmax=False) # plot individual spectrum

            # plot stacked spectrum
            # Running this again to get it on top
            if h[cellcount+2].header['NUMSPEC'] != str(0):
                ongrid = h[cellcount+2].header['ONGRID']
                numspec = h[cellcount+2].header['NUMSPEC']
                normval = h[cellcount+2].header['NORMVAL']
                flam = h[cellcount+2].data[0]
                ferr = h[cellcount+2].data[1]
                
                print ongrid
                totalgalaxiesstacks += int(numspec)
                
                plot_spectrum_median(flam, ferr, lam, ongrid, numspec, ymax)
                
                cellcount += 1
            else:
                logging.warning("Skipping extension number %d", cellcount+2)
                cellcount += 1
                continue    
            
            #    fig.subplots_adjust(top=0.92)
            #    pdf.savefig(bbox_inches='tight')
            
            totalgalaxies += len(np.unique(pears_id[indices]))

            # Block to plot the best fit model spectrum if one exists.
            """
            fitarg = np.where(ongrid_arr == str(i) + ',' + str(j))[0]
            if fitarg.size:
                fitongrid = fitparams['ongrid'][fitarg][0]
                minchi2 = fitparams['chi2'][fitarg][0]
                bestfitmetal = fitparams['metals'][fitarg][0]
                bestfitage = fitparams['age'][fitarg][0]
                bestfittau = fitparams['tau'][fitarg][0]
                bestfittauV = fitparams['tauV'][fitarg][0]
                bestalpha = fitparams['alpha'][fitarg][0]
                stackcount = int(fitparams['stackcount'][fitarg][0])
                urcol = float(fitongrid.split(',')[0])
                stmass = float(fitongrid.split(',')[1])
                bestfitfile = fitparams['filename'][fitarg][0]
    
                bestfitspec = fits.open(modelpath + bestfitmetal + '/' + bestfitfile)
                bestparams = '$\chi^2 = $' + str(minchi2)[:6] + '\n' + bestfitmetal + '\n' + str(bestfitage)[:6] + 'Gyr' + '\n' + 'tau' + str(bestfittau) + 'Gyr'+ '\n' + 'tauV' + str(bestfittauV)
                if (bestfittau == 0.01) or (bestfittau == 79.432) or (bestfittauV == 0.0) or (bestfittauV == 2.9) or (bestfitage == 20.0) or (bestfitage >= 8.0):
                    legendstyle = 'bold'
                else:
                    legendstyle = 'normal'

                fitlam = bestfitspec[1].data
                ages = bestfitspec[2].data
                bestage = bestfitage * 1e9
                minageindex = np.argmin(abs(ages - bestage))
                spec = bestfitspec[minageindex + 3].data
                lam_step = 100
                lam_lowfit = 2700
                lam_highfit = 6000
                lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)

                # Only consider the part of BC03 spectrum between 2700 to 6000
                arg2650 = np.argmin(abs(fitlam - 2650))
                arg5950 = np.argmin(abs(fitlam - 5950))
                fitlam = fitlam[arg2650:arg5950+1]
                spec = spec[arg2650:arg5950+1]
                spec = resample_bc03(fitlam, spec)
                fitlam = lam_grid_tofit

                plot_spectrum_bc03(fitlam, spec*bestalpha, bestparams, legendstyle)
                if urcol <= 1.2:
                    shade_em_lines()
            """

            fig.subplots_adjust(top=0.92)
            pdf.savefig(bbox_inches='tight')

    print indiv_count
    print totalgalaxies
    print totalgalaxiesstacks
    print "Reached end of file. Exiting."
    
    """
        # To plot it to the screen at the same position every time
        # plt.get_current_fig_manager().window.wm_geometry("600x600+20+40")
        # the arguments here are width x height + xposition + yposition
    """
    
    # Save file metadata
    d = pdf.infodict()
    d['Title'] = 'Median stacked ACS G800L spectra from PEARS survey'
    d['Author'] = u'Bhavin Joshi'
    d['CreationDate'] = datetime.datetime(2016, 3, 9)
    d['ModDate'] = datetime.datetime.today()
    
    pdf.close()
    
    logging.info("Written pdf file -- %s", pdfname)
    print "Written pdf file --", pdfname
    
    dt = datetime.datetime
    logging.info("\n Finished at --")
    logging.info(dt.now())
