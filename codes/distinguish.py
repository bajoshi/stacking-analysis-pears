from __future__ import division
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

import sys, glob, os, datetime

def reject_ind(iter_range, comp_spec, overall_counter):
    """
        This function takes in the range of indices to iterate over.
        It finds the indices of similar spectra and rejects them from 
        the original array of indices that were supplied for iteration
        and returns the updated array of indices.
        The program should stop when only distinguishable spectra are left in the index range.
        i.e. the list of similar indices that this function finds should be empty.
    """
    sim_spec_ind = []
    for i in iter_range:
        #if i%1000 == 0: print i, len(np.unique(sim_spec_ind))
        dist_curr_spec = comp_spec[i]
        diff = abs(dist_curr_spec - comp_spec)
        sim_spec_ind_temp = np.intersect1d(np.where(np.max(diff, axis=1) < 0.05)[0], np.where(np.mean(diff, axis=1) < 0.05)[0])
        sim_spec_ind_temp = np.setdiff1d(sim_spec_ind_temp, i) 
        # It will always find a spectrum to be similar to itself. This line is to get it to not reject the spectrum currently being tested.
        sim_spec_ind_temp = sim_spec_ind_temp[sim_spec_ind_temp > i]
        sim_spec_ind = np.append(sim_spec_ind , sim_spec_ind_temp)
        sim_spec_ind = np.unique(sim_spec_ind)

    return sim_spec_ind

def reject_ind1(iter_range, comp_spec, lam_grid_tofit):
    """
        This function takes in the range of indices to iterate over.
        It finds the indices of similar spectra and rejects them from 
        the original array of indices that were supplied for iteration
        and returns the updated array of indices.
        The program should stop when only distinguishable spectra are left in the index range.
        i.e. the list of similar indices that this function finds should be empty.
    """

    sim_list = []
    count = 0
    while 1:
        i = iter_range[count]
        dist_curr_spec = comp_spec[i]
        diff = abs(dist_curr_spec - comp_spec)
        sim_spec_ind_temp = np.intersect1d(np.where(np.max(diff, axis=1) < 0.05)[0], np.where(np.mean(diff, axis=1) < 0.05)[0])
        sim_spec_ind_temp = np.setdiff1d(sim_spec_ind_temp, i) # It will always find a spectrum to be similar to itself. This line is to get it to not reject the spectrum currently being tested.

        iter_range = np.setdiff1d(iter_range, sim_spec_ind_temp)
        sim_list.append(sim_spec_ind_temp)

        if count == len(iter_range) - 1:
            break
        else:
            count += 1

        #if count == 10: break

        #fig, ax = makefig()
        #plot_sim_spec(ax, comp_spec, lam_grid_tofit, sim_spec_ind_temp)

    return iter_range, sim_list

def makefig():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\lambda\ [\AA]$')
    ax.set_ylabel('$F_{\lambda}\ [\mathrm{arbitrary\ units}]$')
    ax.axhline(y=0,linestyle='--')

    return fig, ax

def plot_sim_spec(ax, comp_spec, lam_grid_tofit, spec_ind):
    for i in spec_ind:
        flam = comp_spec[i]
        lam_em = lam_grid_tofit
        plot_spectrum_indiv(ax, flam, lam_em, i)
    ax.legend(loc=0)
    plt.show()
    return

def plot_spectrum_indiv(ax, flam_em_indiv, lam_em_indiv, currspec_ind, label=False, labelmax=False):
    
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
        ax.plot(lam_em_indiv, flam_em_indiv, ls='-', label=str(currspec_ind))
    
    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.set_xlim(2500, 6000)

if __name__ == '__main__':

    # Start time
    dt = datetime.datetime
    print "Starting at --", dt.now()

    filename = '/Users/baj/Desktop/FIGS/new_codes/all_comp_spectra.fits'
    #filename = '/Users/baj/Desktop/FIGS/new_codes/compspec_temp.fits'
    h = fits.open(filename, memmap=False)
    print "Comparison Spectra read in.", dt.now() # takes about 3 min

    nexten = 0
    while 1:
        try:
            if h[nexten+1]:
                nexten += 1
        except IndexError:
            break
    # nexten... should be 1000 when you are using the test file -- compspec_temp.fits

    lam_step = 100
    lam_lowfit = 2500
    lam_highfit = 6500
    lam_grid_tofit = np.arange(lam_lowfit, lam_highfit, lam_step)

    comp_spec = np.zeros([nexten, len(lam_grid_tofit)], dtype=np.float64)
    for i in range(nexten):
        comp_spec[i] = h[i+1].data

    del h # probalby shouldnt delete the HDU. I'll need it later for header info for individual spectra.

    final_dist_arr, dist_arr_sim = reject_ind1(np.arange(nexten), comp_spec, lam_grid_tofit)

    f = open('dist_spec.txt', 'wa')
    f.write('#These indices in the first column correspond to the extensions (add 1 to the index shown here) of the fits file containing the comparison spectra.' + '\n')
    f.write('#The list of indices corresponds to those similar to the spectrum whose index is in the first col.' + '\n')
    for i in range(len(final_dist_arr)):
        f.write(str(final_dist_arr[i]) + ',') 
        for j in range(len(dist_arr_sim[i])): 
            # Had to include the second for loop as well because with only one loop it was putting similar spec indices on multiple lines.
            # This was causing problems with genfromtxt in the next code.
            f.write(str(dist_arr_sim[i][j]) + ' ')
        f.write('\n')

    f.close()

    print "Done at --", dt.now()
