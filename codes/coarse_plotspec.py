from __future__ import division
import numpy as np
import pyfits as pf

import datetime, sys, os
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea

import matplotlib.gridspec as gridspec

#from grid_coadd import fileprep, rescale, get_net_sig

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
    #specname = specname.split('.')[0].split('_')[2:]

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

    flam_obs = flam_obs - contam
            
    # First chop off the ends and only look at the observed spectrum from 6000A to 9500A
    arg6000 = np.argmin(abs(lam_obs - 6000))
    arg9500 = np.argmin(abs(lam_obs - 9500))
			        
    lam_obs = lam_obs[arg6000:arg9500]
    flam_obs = flam_obs[arg6000:arg9500]
    ferr = ferr[arg6000:arg9500]
					            
    flam_obs[flam_obs < 0] = 0
							        
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

def plot_spectrum_indiv(flam_em, ferr, lam_em, specname):

    # without label
    ax.plot(lam_em, flam_em, ls='-', color='gray')
    
    """
    # with label
    ax.plot(lam_em, flam_em, ls='-')
    max_flam_arg = np.argmax(flam_em)
    max_flam = flam_em[max_flam_arg]
    max_flam_lam = lam_em[max_flam_arg]
    
    print max_flam, max_flam_lam
    
    ax.annotate(specname, xy=(max_flam_lam,max_flam), xytext=(max_flam_lam,max_flam),\
                arrowprops=dict(arrowstyle="->"))
    
    #ax.legend(loc=0, prop={'size':2}, ncol=3)
    """
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(3000, 5500)

def plot_spectrum_median(flux, flux_err, lam, ongrid, numspec):
    
    ax.errorbar(lam, flux, yerr=flux_err, fmt='o-', color='k', linewidth=2, label=ongrid+','+numspec,\
                ecolor='r', markeredgecolor='k', capsize=0, markersize=4)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(3000, 5500)
    
    ax.axvline(x=5200, color='maroon')
    #ax.set_yscale('log')
    ax.legend(loc=0)

if __name__ == '__main__':
    
    # Logging start info
    logging.basicConfig(filename="coadd_coarsegrid.log", format='%(levelname)s:%(message)s', filemode='a', level=logging.DEBUG)
    dt = datetime.datetime
    logging.info("\n Plots started at --")
    logging.info(dt.now())
    
    # Prep for normalizing and plotting
    coadd_dir = '/Users/baj/Desktop/FIGS/new_codes/'
    h = pf.open('/Users/baj/Desktop/FIGS/new_codes/coadded_PEARSgrismspectra_coarsegrid.fits')
    
    lam = h[1].data
    #n_ext = h[0].header['NUMEXT'] # you will have to edit the header and put this keyword in by hand
    #num_spectra = n_ext - 2
    
    savename = 'coadded_spectra_coarsegrid.png'

    data_path = "/Users/baj/Documents/PEARS/data_spectra_only/"
    threedphot = "/Users/baj/Documents/3D-HST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat"
    threed = pf.open('/Users/baj/Documents/3D-HST/3dhst.v4.1.5.master.fits')
    cat = np.genfromtxt('/Users/baj/Desktop/FIGS/new_codes/color_stellarmass.txt',\
                        dtype=None, names=True, skip_header=2)
    
    pears_id = cat['pearsid']
    ur_color = cat['urcol']
    stellarmass = cat['mstar']
    photz = cat['threedzphot']
    
    color_step = 0.6
    mstar_step = 1.0
    
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
                'h_pears_s_id56575.fits']
                
    gs = gridspec.GridSpec(15,15)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.00, hspace=0.00)
    
    
    # Find the averages of all grid cells in a particular row/column
    # While these are useful numbers to have, they are currently only used in the plotting routine.
    cellcount = 0

    avgcolarr = np.zeros(5)
    avgmassarr = np.zeros(5)
        
    avgmassarr = avgmassarr.tolist()
    for k in range(len(avgmassarr)):
        avgmassarr[k] = []

    for i in np.arange(0.0, 3.0, color_step):
        colcount = 0
        for j in np.arange(7.0, 11.5, mstar_step):
            
            indices = np.where((ur_color >= i) & (ur_color < i + color_step) &\
	                   (stellarmass >= j) & (stellarmass < j + mstar_step))[0]
            
            row = int(i/color_step)
            column = int((j - 7.0)/mstar_step)

            if indices.size:
                avgcol = h[cellcount+2].header['AVGCOL']
                avgmass = h[cellcount+2].header['AVGMASS']
                avgcolarr[row] += float(avgcol)
                avgmassarr[column].append(float(avgmass))

                cellcount += 1
                colcount += 1
            else:
	            continue

        avgcolarr[row] /= (colcount)

    for x in range(len(avgmassarr)):
        avgmassarr[x] = np.sum(avgmassarr[x]) / len(avgmassarr[x])

    # Begin looping through cells to plot
    cellcount = 0
    for i in np.arange(0.0, 3.0, color_step):
        for j in np.arange(7.0, 11.5, mstar_step):
            
            indices = np.where((ur_color >= i) & (ur_color < i + color_step) &\
                            (stellarmass >= j) & (stellarmass < j + mstar_step))[0]
                            
            print "ONGRID", i, j
            
            if indices.size:
                medarr, medval, stdval = rescale(indices)
            else:
                continue
            
            # Run this block if you want to cycle through individual spectra
            """
            for x in range(len(pears_id[indices])):
                # Get redshift from previously saved 3DHST photz catalog
                redshift = photz[indices][x]
                
                # Get rest frame values for all quantities
                lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][x], redshift)
               
                # Divide by max value to rescale
                flam_em /= maxval
                ferr /= maxval
                
                cycle_through(flam_em, ferr, lam_em, specname)
                              
            sys.exit()
            """

            # Create subplots for different grid positions
            row = 4 - int(i/color_step)
            column = int((j - 7.0)/mstar_step)
            ax = plt.subplot(gs[row*3:row*3+3, column*3:column*3+3])
            ax.set_ylim(0,2)
            
            if (row == 4) and (column == 2):
                ax.set_xlabel('$\lambda\ [\mu m]$', fontsize=13)
            if (row == 2) and (column == 0):
                ax.set_ylabel('$F_{\lambda}\ [\mathrm{arbitrary\ units}]$', fontsize=13)
            
            # If you want the twin axis labels above and to the right
            # then run the following block
            """
            if (row == 2) and (column == 3):
                ax1 = ax.twinx()
                ax1.set_ylabel(r'$\left<\mathrm{(U-R)_{rest}}\right>$', fontsize=13)
                ax1.get_yaxis().set_ticklabels([])
                ax1.yaxis.labelpad = 105
            if (row == 0) and (column == 2):
                ax2 = ax.twiny()
                ax2.set_xlabel(r'$\left<\mathrm{log}\left(\frac{M_*}{M_\odot}\right)\right>$', fontsize=13)
                ax2.get_xaxis().set_ticklabels([])
                ax2.xaxis.labelpad = 20
            """
            
            # This block is also to get the twin axis labels
            # They will be placed below and to the left of the numbers now
            if (row == 2) and (column == 0):
                masslabelbox = TextArea("{:s}".format(r'$\left<\mathrm{log}\left(\frac{M_*}{M_\odot}\right)\right>$'),\
                                        textprops=dict(color='k', size=13))
                anchored_masslabelbox = AnchoredOffsetbox(loc=3, child=masslabelbox, pad=0.0, frameon=False,\
                                                          bbox_to_anchor=(0.15, 2.68),\
                                                          bbox_transform=ax.transAxes, borderpad=0.0)
                ax.add_artist(anchored_masslabelbox)
                                                           
            if (row == 4) and (column == 2):
                colorlabelbox = TextArea("{:s}".format(r'$\left<\mathrm{(U-R)_{rest}}\right>$'),\
                                         textprops=dict(color='k', size=13, rotation=270))
                anchored_colorlabelbox = AnchoredOffsetbox(loc=3, child=colorlabelbox, pad=0.0, frameon=False,\
                                                           bbox_to_anchor=(2.85, 0.7),\
                                                           bbox_transform=ax.transAxes, borderpad=0.0)
                ax.add_artist(anchored_colorlabelbox)
            
            
            # Loop over all spectra in a grid cell and plot them
            for u in range(len(pears_id[indices])):
                # Get redshift from previously saved 3DHST photz catalog
                redshift = photz[indices][u]
                
                # Get rest frame values for all quantities
                lam_em, flam_em, ferr, specname = fileprep(pears_id[indices][u], redshift)
                    
                # Divide by max value to rescale
                flam_em = (flam_em / medarr[u])
                ferr = (ferr / medarr[u])

                # Plotting
                # Reject spectrum if overall contamination too high
                if np.sum(abs(ferr)) > 0.3 * np.sum(abs(flam_em)):
                    print "Skipped", specname
                    continue
                if specname in skipspec:
                    print "Skipped", specname
                    continue
                else:
                    ax.plot(lam_em, flam_em, ls='-', color='lightgray')
                    ax.get_yaxis().set_ticklabels([])
                    ax.get_xaxis().set_ticklabels([])
                    ax.set_xlim(3000, 5500)
        

            # plot stacked spectrum
            if h[cellcount+2].header['NUMSPEC'] != str(0):
                ongrid = h[cellcount+2].header['ONGRID']
                numspec = h[cellcount+2].header['NUMSPEC']
                normval = h[cellcount+2].header['NORMVAL']
                flux = h[cellcount+2].data[0]
                flux_err = h[cellcount+2].data[1]
                
                print ongrid
                
                # Box in fig for rescaling flux value
                fluxbox = TextArea("{:.2e}".format(medval), textprops=dict(color='k', size=8))
            
                anchored_fluxbox = AnchoredOffsetbox(loc=3, child=fluxbox, pad=0.0, frameon=False,\
                                                     bbox_to_anchor=(0.03, 0.82),\
                                                     bbox_transform=ax.transAxes, borderpad=0.0)
                ax.add_artist(anchored_fluxbox)
                
                # Box in fig for rescaling flux value
                numbox = TextArea("{:d}".format(int(numspec)), textprops=dict(color='darkgreen'))
                
                anchored_numbox = AnchoredOffsetbox(loc=4, child=numbox, pad=0.0, frameon=False,\
                                                    bbox_to_anchor=(0.9, 0.07),\
                                                    bbox_transform=ax.transAxes, borderpad=0.0)
                ax.add_artist(anchored_numbox)
                
                #plot_spectrum_median(flux, flux_err, lam, ongrid, numspec)
                ax.errorbar(lam, flux, yerr=flux_err, fmt='.-', color='b', linewidth=1,\
                            ecolor='r', markeredgecolor='b', capsize=0, markersize=3)
                ax.get_yaxis().set_ticklabels([])
                ax.get_xaxis().set_ticklabels([])
                
                ### For all of the next if statements, keep in mind that it starts counting
                ### the row and column from the top left.
                if (row < 2):
                    ax.axvline(x=5173, color='purple', linestyle='--')
                #if column >= 1:
                #    ax.axvline(x=4000, color='red')
                if (column < 3) and (row >= 2):
                    ax.axvline(x=4942, color='darkolivegreen', linestyle='--')

                if (row == 4)and (column == 0):
                    ax.get_xaxis().set_ticklabels(['0.3', '0.35', '0.4', '0.45', '0.5', '0.55'], fontsize=7, rotation=45)
                
                if (row == 4) and (column == 1):
                    ax.get_xaxis().set_ticklabels(['', '0.35', '0.4', '0.45', '0.5', '0.55'], fontsize=7, rotation=45)
                
                if (row == 4) and (column == 2):
                    ax.get_xaxis().set_ticklabels(['', '0.35', '0.4', '0.45', '0.5', '0.55'], fontsize=7, rotation=45)

                if (row == 3) and (column == 3):
                    ax.get_xaxis().set_ticklabels(['0.3', '0.35', '0.4', '0.45', '0.5', '0.55'], fontsize=7, rotation=45)
                
                if (row == 1) and (column == 4):
                    ax.get_xaxis().set_ticklabels(['0.3', '0.35', '0.4', '0.45', '0.5', '0.55'], fontsize=7, rotation=45)
                        
                if (row == 0):
                    massbox = TextArea("{:.2f}".format(float(avgmassarr[column])), textprops=dict(color='k', size=12))
                    anchored_massbox = AnchoredOffsetbox(loc=3, child=massbox, pad=0.0, frameon=False,\
                                          bbox_to_anchor=(0.3, 1.03),\
                                          bbox_transform=ax.transAxes, borderpad=0.0)
                    ax.add_artist(anchored_massbox)

                if (row == 1) and (column == 1):
                    massbox = TextArea("{:.2f}".format(float(avgmassarr[column])), textprops=dict(color='k', size=12))
                    anchored_massbox = AnchoredOffsetbox(loc=3, child=massbox, pad=0.0, frameon=False,\
                                              bbox_to_anchor=(0.3, 2.03),\
                                              bbox_transform=ax.transAxes, borderpad=0.0)
                    ax.add_artist(anchored_massbox)

                if (row == 2) and (column == 0):
                    massbox = TextArea("{:.2f}".format(float(avgmassarr[column])), textprops=dict(color='k', size=12))
                    anchored_massbox = AnchoredOffsetbox(loc=3, child=massbox, pad=0.0, frameon=False,\
                                                    bbox_to_anchor=(0.3, 3.03),\
                                                    bbox_transform=ax.transAxes, borderpad=0.0)
                    ax.add_artist(anchored_massbox)

                if column == 4:
                    colorbox = TextArea("{:.2f}".format(float(avgcolarr[4-row])), textprops=dict(color='k', size=12, rotation=270))
                    anchored_colorbox = AnchoredOffsetbox(loc=3, child=colorbox, pad=0.0, frameon=False,\
                                                          bbox_to_anchor=(1.03, 0.4),\
                                                          bbox_transform=ax.transAxes, borderpad=0.0)
                    ax.add_artist(anchored_colorbox)
                        
                if (row == 2) and (column == 3):
                    colorbox = TextArea("{:.2f}".format(float(avgcolarr[4-row])), textprops=dict(color='k', size=12, rotation=270))
                    anchored_colorbox = AnchoredOffsetbox(loc=3, child=colorbox, pad=0.0, frameon=False,\
                                                          bbox_to_anchor=(2.03, 0.4),\
                                                          bbox_transform=ax.transAxes, borderpad=0.0)
                    ax.add_artist(anchored_colorbox)

                if (row == 3) and (column == 3):
                    colorbox = TextArea("{:.2f}".format(float(avgcolarr[4-row])), textprops=dict(color='k', size=12, rotation=270))
                    anchored_colorbox = AnchoredOffsetbox(loc=3, child=colorbox, pad=0.0, frameon=False,\
                                              bbox_to_anchor=(2.03, 0.4),\
                                              bbox_transform=ax.transAxes, borderpad=0.0)
                    ax.add_artist(anchored_colorbox)

                if (row == 4) and (column == 2):
                    colorbox = TextArea("{:.2f}".format(float(avgcolarr[4-row])), textprops=dict(color='k', size=12, rotation=270))
                    anchored_colorbox = AnchoredOffsetbox(loc=3, child=colorbox, pad=0.0, frameon=False,\
                                              bbox_to_anchor=(3.03, 0.4),\
                                              bbox_transform=ax.transAxes, borderpad=0.0)
                    ax.add_artist(anchored_colorbox)

                cellcount += 1
            else:
                logging.warning("Skipping extension number %d", cellcount+2)
                cellcount += 1
                continue

    plt.savefig(savename, dpi=300)
    print "Reached end of file. Exiting."
    
    """
        # To plot it to the screen at the same position every time
        # plt.get_current_fig_manager().window.wm_geometry("600x600+20+40")
        # the arguments here are width x height + xposition + yposition
    """
    
    logging.info("Written file -- %s", savename)
    print "Written file --", savename
    
    dt = datetime.datetime
    logging.info("\n Finished at --")
    logging.info(dt.now())
