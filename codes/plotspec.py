from __future__ import division
import numpy as np
from astropy.io import fits

import datetime, sys, os
import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea

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
    
    data_path = "/Users/baj/Documents/PEARS/data_spectra_only/"
    # Get the correct filename and the number of extensions
    filename = data_path + 'h_pears_n_id' + str(pears_index) + '.fits'
    if not os.path.isfile(filename):
        filename = data_path + 'h_pears_s_id' + str(pears_index) + '.fits'

    fitsfile = fits.open(filename)
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

def rescale(ids, zs):
    medarr = np.zeros(len(ids))
    
    for k in range(len(ids)):
        
        redshift = zs[k]
        lam_em, flam_em, ferr, specname = fileprep(ids[k], redshift)
        
        # Store median of values from 4400A-4600A for each spectrum
        arg4400 = np.argmin(abs(lam_em - 4400))
        arg4600 = np.argmin(abs(lam_em - 4600))
        medarr[k] = np.median(flam_em[arg4400:arg4600+1])
    
    medval = np.median(medarr)
    
    # Return the maximum in array of median values
    return medarr, medval, np.std(medarr)

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

def shade_em_lines():

    ymin, ymax = ax.get_ylim()
    ax.fill_between(x=np.arange(4800,5200,100), y1=ymax, y2=ymin, facecolor='gray', alpha=0.5)

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
    
    skipspec = ['h_pears_n_id78220.fits',\
                'h_pears_n_id47644.fits', 'h_pears_s_id74234.fits', 'h_pears_s_id124266.fits',\
                'h_pears_n_id111054.fits',\
                'h_pears_s_id106446.fits',\
                'h_pears_n_id120710.fits',\
                'h_pears_s_id78417.fits',\
                'h_pears_s_id59792.fits','h_pears_s_id93218.fits','h_pears_n_id104213.fits',\
                'h_pears_s_id115422.fits','h_pears_s_id72113.fits','h_pears_s_id107858.fits','h_pears_s_id45223.fits','h_pears_s_id23920.fits',\
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
                'h_pears_n_id42132.fits','h_pears_n_id56602.fits', 'h_pears_n_id122827.fits',
                'h_pears_s_id85844.fits', 'h_pears_s_id36291.fits', 'h_pears_s_id116423.fits', 'h_pears_s_id22563.fits',
                'h_pears_s_id82729.fits', 'h_pears_s_id65572.fits', 'h_pears_s_id119893.fits', 'h_pears_n_id69117.fits', 
                'h_pears_n_id43079.fits', 'h_pears_s_id118193.fits', 'h_pears_s_id115683.fits', 'h_pears_s_id53743.fits',
                'h_pears_s_id82239.fits', 'h_pears_s_id93827.fits', 'h_pears_n_id76773.fits', 'h_pears_s_id109091.fits',
                'h_pears_s_id115761.fits', 'h_pears_s_id99589.fits', 'h_pears_s_id83749.fits', 'h_pears_s_id113375.fits']

    em_lines = ['h_pears_n_id90194.fits', 'h_pears_s_id17321.fits', 'h_pears_s_id106641.fits', 'h_pears_s_id106993.fits',\
    'h_pears_s_id110400.fits', 'h_pears_s_id21196.fits', 'h_pears_s_id94632.fits', 'h_pears_n_id42058.fits', 'h_pears_n_id105902.fits',\
    'h_pears_n_id118534.fits', 'h_pears_s_id26909.fits', 'h_pears_n_id113595.fits', 'h_pears_s_id16788.fits', 'h_pears_s_id64769.fits',\
    'h_pears_s_id121506.fits', 'h_pears_s_id122913.fits', 'h_pears_n_id44207.fits', 'h_pears_n_id50210.fits', 'h_pears_n_id63410.fits',\
    'h_pears_n_id72081.fits', 'h_pears_n_id84467.fits', 'h_pears_n_id85556.fits', 'h_pears_n_id108010.fits', 'h_pears_n_id107999.fits',\
    'h_pears_n_id108239.fits', 'h_pears_s_id17587.fits', 'h_pears_s_id22486.fits', 'h_pears_s_id40117.fits', 'h_pears_s_id59595.fits',\
    'h_pears_s_id63612.fits', 'h_pears_s_id64393.fits', 'h_pears_s_id65179.fits', 'h_pears_s_id68852.fits', 'h_pears_s_id94873.fits',\
    'h_pears_s_id98046.fits', 'h_pears_s_id98242.fits', 'h_pears_s_id107055.fits', 'h_pears_s_id109596.fits', 'h_pears_s_id121911.fits',\
    'h_pears_s_id109435.fits', 'h_pears_s_id119341.fits', 'h_pears_s_id122735.fits', 'h_pears_s_id129968.fits', 'h_pears_s_id17587.fits',\
    'h_pears_s_id22486.fits', 'h_pears_s_id68852.fits', 'h_pears_s_id82307.fits', 'h_pears_n_id82356.fits', 'h_pears_s_id90246.fits',\
    'h_pears_s_id94873.fits', 'h_pears_n_id35090.fits', 'h_pears_n_id42526.fits', 'h_pears_n_id66045.fits', 'h_pears_n_id82693.fits',\
    'h_pears_n_id93025.fits', 'h_pears_s_id18882.fits', 'h_pears_s_id21363.fits', 'h_pears_s_id25474.fits', 'h_pears_s_id27652.fits',\
    'h_pears_s_id40163.fits', 'h_pears_s_id41078.fits', 'h_pears_s_id43170.fits', 'h_pears_s_id46284.fits', 'h_pears_s_id50298.fits',\
    'h_pears_s_id51976.fits', 'h_pears_s_id65708.fits', 'h_pears_s_id69168.fits', 'h_pears_s_id69234.fits', 'h_pears_s_id104478.fits',\
    'h_pears_s_id104514.fits', 'h_pears_s_id105015.fits', 'h_pears_s_id107052.fits', 'h_pears_s_id113279.fits', 'h_pears_s_id117686.fits',\
    'h_pears_s_id121974.fits', 'h_pears_s_id21614.fits', 'h_pears_s_id79520.fits', 'h_pears_s_id81609.fits', 'h_pears_s_id83804.fits',\
    'h_pears_s_id90198.fits', 'h_pears_s_id91382.fits', 'h_pears_n_id39842.fits', 'h_pears_n_id47252.fits', 'h_pears_n_id61065.fits',\
    'h_pears_n_id90145.fits', 'h_pears_n_id84077.fits', 'h_pears_n_id112486.fits', 'h_pears_n_id110727.fits', 'h_pears_n_id121428.fits',\
    'h_pears_s_id31086.fits', 'h_pears_s_id32905.fits', 'h_pears_s_id118526.fits', 'h_pears_n_id75917.fits', 'h_pears_n_id54384.fits',\
    'h_pears_n_id109561.fits', 'h_pears_n_id124893.fits', 'h_pears_s_id51293.fits', 'h_pears_s_id98951.fits', 'h_pears_s_id100652.fits',\
    'h_pears_s_id124410.fits', 'h_pears_s_id125699.fits', 'h_pears_s_id133306.fits', 'h_pears_s_id78045.fits', 'h_pears_s_id87735.fits',\
    'h_pears_n_id33944.fits', 'h_pears_n_id41064.fits', 'h_pears_n_id77596.fits', 'h_pears_n_id81988.fits', 'h_pears_n_id81946.fits',\
    'h_pears_n_id84715.fits', 'h_pears_s_id52445.fits', 'h_pears_s_id104446.fits', 'h_pears_s_id107036.fits', 'h_pears_s_id118673.fits',\
    'h_pears_s_id71864.fits', 'h_pears_s_id73385.fits', 'h_pears_s_id76154.fits', 'h_pears_s_id92860.fits', 'h_pears_n_id91947.fits',\
    'h_pears_n_id101185.fits', 'h_pears_n_id119723.fits', 'h_pears_n_id125948.fits', 'h_pears_s_id35878.fits', 'h_pears_s_id38417.fits',\
    'h_pears_s_id103422.fits', 'h_pears_s_id115331.fits', 'h_pears_s_id120803.fits', 'h_pears_s_id121733.fits', 'h_pears_s_id126769.fits',\
    'h_pears_s_id21647.fits', 'h_pears_s_id80500.fits', 'h_pears_s_id83686.fits', 'h_pears_n_id42014.fits', 'h_pears_s_id109992.fits',\
    'h_pears_s_id124964.fits', 'h_pears_s_id78077.fits', 'h_pears_s_id69713.fits', 'h_pears_s_id87679.fits', 'h_pears_n_id38344.fits',\
    'h_pears_s_id71524.fits', 'h_pears_n_id123476.fits', 'h_pears_s_id119489.fits', 'h_pears_s_id109167.fits', 'h_pears_s_id110397.fits',\
    'h_pears_s_id14215.fits', 'h_pears_s_id117429.fits', 'h_pears_n_id53563.fits', 'h_pears_s_id62528.fits', 'h_pears_n_id114628.fits',\
    'h_pears_s_id114108.fits', 'h_pears_s_id124313.fits', 'h_pears_s_id119193.fits']
    
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
                if (specname in skipspec):# or (specname in em_lines):
                    #print "Skipped", specname
                    continue
                else:
                    indiv_count += 1
                    plot_spectrum_indiv(flam_em, ferr, lam_em, specname, i, j, label=False, labelmax=False) # plot individual spectrum

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
