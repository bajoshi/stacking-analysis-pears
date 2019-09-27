from __future__ import division

import numpy as np

import os
import sys
import glob

home = os.getenv('HOME')
figs_dir = home + '/Desktop/FIGS/'

stacking_analysis_dir = figs_dir + 'stacking-analysis-pears/'
massive_figures_dir = figs_dir + 'massive-galaxies-figures/'

full_pears_results_dir = massive_figures_dir + 'full_pears_results/'

selection = 'all_chab'
if selection == 'all_salp':
    selected_results_dir = full_pears_results_dir
    final_file_name = stacking_analysis_dir + 'full_pears_results.txt'
elif selection == 'all_salp_no_irac_ch3_ch4':
    selected_results_dir = full_pears_results_dir.replace('full_pears_results', 'full_pears_results_no_irac_ch3_ch4')
    final_file_name = stacking_analysis_dir + 'full_pears_results_no_irac_ch3_ch4.txt'
elif selection == 'all_salp_no_irac':
    selected_results_dir = full_pears_results_dir.replace('full_pears_results', 'full_pears_results_no_irac')
    final_file_name = stacking_analysis_dir + 'full_pears_results_no_irac.txt'
elif selection == 'all_chab':
    selected_results_dir = full_pears_results_dir.replace('full_pears_results', 'full_pears_results_chabrier')
    final_file_name = stacking_analysis_dir + 'full_pears_results_chabrier.txt'
elif selection == 'all_chab_no_irac_ch3_ch4':
    selected_results_dir = full_pears_results_dir.replace('full_pears_results', 'full_pears_results_chabrier_no_irac_ch3_ch4')
    final_file_name = stacking_analysis_dir + 'full_pears_results_chabrier_no_irac_ch3_ch4.txt'
elif selection == 'all_chab_no_irac':
    selected_results_dir = full_pears_results_dir.replace('full_pears_results', 'full_pears_results_chabrier_no_irac')
    final_file_name = stacking_analysis_dir + 'full_pears_results_chabrier_no_irac.txt'

def main():

    # Define empty lists
    PearsID = []
    Field = []
    RA = []
    DEC = []
    
    zspec = []
    zp_minchi2 = []
    zspz_minchi2 = []
    zg_minchi2 = []
    
    zp = []
    zspz = []
    zg = []
    
    zp_zerr_low = []
    zp_zerr_up = []
    zspz_zerr_low = []
    zspz_zerr_up = []
    zg_zerr_low = []
    zg_zerr_up = []
    
    zp_min_chi2 = []
    zspz_min_chi2 = []
    zg_min_chi2 = []
    
    zp_bestalpha = []
    zspz_bestalpha = []
    zg_bestalpha = []
    
    zp_model_idx = []
    zspz_model_idx = []
    zg_model_idx = []
    
    zp_age = []
    zp_tau = []
    zp_av = []
    zspz_age = []
    zspz_tau = []
    zspz_av = []
    zg_age = []
    zg_tau = []
    zg_av = []

    zp_template_ms = []
    zp_ms = []
    zp_sfr = []
    zp_uv = []
    zp_vj = []

    zspz_template_ms = []
    zspz_ms = []
    zspz_sfr = []
    zspz_uv = []
    zspz_vj = []

    zg_template_ms = []
    zg_ms = []
    zg_sfr = []
    zg_uv = []
    zg_vj = []

    # Loop over all results and store values
    for fl in glob.glob(selected_results_dir + 'redshift_fitting_results_*.txt'):
        f = np.genfromtxt(fl, dtype=None, names=True, skip_header=1)

        PearsID.append(f['PearsID'])
        Field.append(f['Field'])
        RA.append(f['RA'])
        DEC.append(f['DEC'])
        
        zspec.append(f['zspec'])
        zp_minchi2.append(f['zp_minchi2'])
        zspz_minchi2.append(f['zspz_minchi2'])
        zg_minchi2.append(f['zg_minchi2'])
        
        zp.append(f['zp'])
        zspz.append(f['zspz'])
        zg.append(f['zg'])
        
        zp_zerr_low.append(f['zp_zerr_low'])
        zp_zerr_up.append(f['zp_zerr_up'])
        zspz_zerr_low.append(f['zspz_zerr_low'])
        zspz_zerr_up.append(f['zspz_zerr_up'])
        zg_zerr_low.append(f['zg_zerr_low'])
        zg_zerr_up.append(f['zg_zerr_up'])
        
        zp_min_chi2.append(f['zp_min_chi2'])
        zspz_min_chi2.append(f['zspz_min_chi2'])
        zg_min_chi2.append(f['zg_min_chi2'])
        
        zp_bestalpha.append(f['zp_bestalpha'])
        zspz_bestalpha.append(f['zspz_bestalpha'])
        zg_bestalpha.append(f['zg_bestalpha'])
        
        zp_model_idx.append(f['zp_model_idx'])
        zspz_model_idx.append(f['zspz_model_idx'])
        zg_model_idx.append(f['zg_model_idx'])
        
        zp_age.append(f['zp_age'])
        zp_tau.append(f['zp_tau'])
        zp_av.append(f['zp_av'])
        zspz_age.append(f['zspz_age'])
        zspz_tau.append(f['zspz_tau'])
        zspz_av.append(f['zspz_av'])
        zg_age.append(f['zg_age'])
        zg_tau.append(f['zg_tau'])
        zg_av.append(f['zg_av'])
        
        zp_template_ms.append(f['zp_template_ms'])
        zp_ms.append(f['zp_ms'])
        zp_sfr.append(f['zp_sfr'])
        zp_uv.append(f['zp_uv'])
        zp_vj.append(f['zp_vj'])

        zspz_template_ms.append(f['zspz_template_ms'])
        zspz_ms.append(f['zspz_ms'])
        zspz_sfr.append(f['zspz_sfr'])
        zspz_uv.append(f['zspz_uv'])
        zspz_vj.append(f['zspz_vj'])

        zg_template_ms.append(f['zg_template_ms'])
        zg_ms.append(f['zg_ms'])
        zg_sfr.append(f['zg_sfr'])
        zg_uv.append(f['zg_uv'])
        zg_vj.append(f['zg_vj'])

    # Convert to numpy arrays
    PearsID = np.asarray(PearsID)
    Field = np.asarray(Field)
    RA = np.asarray(RA)
    DEC = np.asarray(DEC)
    
    zspec = np.asarray(zspec)
    zp_minchi2 = np.asarray(zp_minchi2)
    zspz_minchi2 = np.asarray(zspz_minchi2)
    zg_minchi2 = np.asarray(zg_minchi2)
    
    zp = np.asarray(zp)
    zspz = np.asarray(zspz)
    zg = np.asarray(zg)
    
    zp_zerr_low = np.asarray(zp_zerr_low)
    zp_zerr_up = np.asarray(zp_zerr_up)
    zspz_zerr_low = np.asarray(zspz_zerr_low)
    zspz_zerr_up = np.asarray(zspz_zerr_up)
    zg_zerr_low = np.asarray(zg_zerr_low)
    zg_zerr_up = np.asarray(zg_zerr_up)
    
    zp_min_chi2 = np.asarray(zp_min_chi2)
    zspz_min_chi2 = np.asarray(zspz_min_chi2)
    zg_min_chi2 = np.asarray(zg_min_chi2)
    
    zp_bestalpha = np.asarray(zp_bestalpha)
    zspz_bestalpha = np.asarray(zspz_bestalpha)
    zg_bestalpha = np.asarray(zg_bestalpha)
    
    zp_model_idx = np.asarray(zp_model_idx)
    zspz_model_idx = np.asarray(zspz_model_idx)
    zg_model_idx = np.asarray(zg_model_idx)
    
    zp_age = np.asarray(zp_age)
    zp_tau = np.asarray(zp_tau)
    zp_av = np.asarray(zp_av)
    zspz_age = np.asarray(zspz_age)
    zspz_tau = np.asarray(zspz_tau)
    zspz_av = np.asarray(zspz_av)
    zg_age = np.asarray(zg_age)
    zg_tau = np.asarray(zg_tau)
    zg_av = np.asarray(zg_av)
    
    zp_template_ms = np.asarray(zp_template_ms)
    zp_ms = np.asarray(zp_ms)
    zp_sfr = np.asarray(zp_sfr)
    zp_uv = np.asarray(zp_uv)
    zp_vj = np.asarray(zp_vj)

    zspz_template_ms = np.asarray(zspz_template_ms)
    zspz_ms = np.asarray(zspz_ms)
    zspz_sfr = np.asarray(zspz_sfr)
    zspz_uv = np.asarray(zspz_uv)
    zspz_vj = np.asarray(zspz_vj)

    zg_template_ms = np.asarray(zg_template_ms)
    zg_ms = np.asarray(zg_ms)
    zg_sfr = np.asarray(zg_sfr)
    zg_uv = np.asarray(zg_uv)
    zg_vj = np.asarray(zg_vj)

    # Now save to a text file
    data = np.array(zip(PearsID, Field, RA, DEC, zspec, zp_minchi2, zspz_minchi2, zg_minchi2, \
        zp, zspz, zg, zp_zerr_low, zp_zerr_up, zspz_zerr_low, zspz_zerr_up, zg_zerr_low, zg_zerr_up, \
        zp_min_chi2, zspz_min_chi2, zg_min_chi2, zp_bestalpha, zspz_bestalpha, zg_bestalpha, \
        zp_model_idx, zspz_model_idx, zg_model_idx, zp_age, zp_tau, zp_av, \
        zspz_age, zspz_tau, zspz_av, zg_age, zg_tau, zg_av, \
        zp_template_ms, zp_ms, zp_sfr, zp_uv, zp_vj, \
        zspz_template_ms, zspz_ms, zspz_sfr, zspz_uv, zspz_vj, \
        zg_template_ms, zg_ms, zg_sfr, zg_uv, zg_vj), \
    dtype=[('PearsID', int), ('Field', '|S7'), ('RA', float), ('DEC', float), ('zspec', float), ('zp_minchi2', float), ('zspz_minchi2', float), ('zg_minchi2', float), \
        ('zp', float), ('zspz', float), ('zg', float), ('zp_zerr_low', float), ('zp_zerr_up', float), ('zspz_zerr_low', float), \
        ('zspz_zerr_up', float), ('zg_zerr_low', float), ('zg_zerr_up', float), \
        ('zp_min_chi2', float), ('zspz_min_chi2', float), ('zg_min_chi2', float), ('zp_bestalpha', float), ('zspz_bestalpha', float), ('zg_bestalpha', float), \
        ('zp_model_idx', int), ('zspz_model_idx', int), ('zg_model_idx', int), ('zp_age', float), ('zp_tau', float), ('zp_av', float), \
        ('zspz_age', float), ('zspz_tau', float), ('zspz_av', float), ('zg_age', float), ('zg_tau', float), ('zg_av', float), \
        ('zp_template_ms', float), ('zp_ms', float), ('zp_sfr', float), ('zp_uv', float), ('zp_vj', float), \
        ('zspz_template_ms', float), ('zspz_ms', float), ('zspz_sfr', float), ('zspz_uv', float), ('zspz_vj', float), \
        ('zg_template_ms', float), ('zg_ms', float), ('zg_sfr', float), ('zg_uv', float), ('zg_vj', float)])

    hdr = "  PearsID  Field  RA  DEC  zspec  zp_minchi2  zspz_minchi2  zg_minchi2" + \
        "  zp  zspz  zg  zp_zerr_low  zp_zerr_up  zspz_zerr_low  zspz_zerr_up  zg_zerr_low  zg_zerr_up" + \
        "  zp_min_chi2  zspz_min_chi2  zg_min_chi2  zp_bestalpha  zspz_bestalpha  zg_bestalpha" + \
        "  zp_model_idx  zspz_model_idx  zg_model_idx  zp_age  zp_tau  zp_av" + \
        "  zspz_age  zspz_tau  zspz_av  zg_age  zg_tau  zg_av" + \
        "  zp_template_ms  zp_ms  zp_sfr  zp_uv  zp_vj" + \
        "  zspz_template_ms  zspz_ms  zspz_sfr  zspz_uv  zspz_vj" + \
        "  zg_template_ms  zg_ms  zg_sfr  zg_uv  zg_vj"
    np.savetxt(final_file_name, data, \
        fmt=['%d', '%s', '%.7f', '%.6f', '%.4f', '%.2f', '%.2f', '%.2f', \
        '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', \
        '%.2f', '%.2f', '%.2f', '%.3e', '%.3e', '%.3e', \
        '%d', '%d', '%d', '%.3e', '%.3e', '%.2f', \
        '%.3e', '%.3e', '%.2f', '%.3e', '%.3e', '%.2f', \
        '%.4f', '%.3e', '%.3e', '%.4f', '%.4f', \
        '%.4f', '%.3e', '%.3e', '%.4f', '%.4f', \
        '%.4f', '%.3e', '%.3e', '%.4f', '%.4f'], delimiter=' ', header=hdr)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)