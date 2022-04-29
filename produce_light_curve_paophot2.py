#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:39:52 2021

@author: ed
"""

import argparse as ap
import os
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import sys
import pandas as pd
import logging
from scipy.interpolate import InterpolatedUnivariateSpline as ius

def ParseArgs():
    parser = ap.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--tic', type=int)
    parser.add_argument('--actions', type=int, nargs='*')
    parser.add_argument('--aper', type=float, nargs='*', default=None)
    parser.add_argument('--bad_comp_tics', type=int, nargs='*', default=None)
    parser.add_argument('--bad_comp_inds', type=int, nargs='*', default=None)
    parser.add_argument('--night', type=str, default=None)
    parser.add_argument('--ti', type=float, default=None)
    parser.add_argument('--te', type=float, default=None)
    parser.add_argument('--dmb', type=float, default=0.5)
    parser.add_argument('--dmf', type=float, default=3.5)
    parser.add_argument('--dmag', type=float, default=0.5)
    return parser.parse_args()

def custom_logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    log_format = logging.Formatter(format_string)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def lb(time, flux, err, bin_width):
    '''
    Function to bin the data into bins of a given width. time and bin_width
    must have the same units

    Input - time, flux, err, bin_width
    Return - time_bin, flux_bin, err_bin
    '''

    edges = np.arange(np.min(time), np.max(time), bin_width)
    dig = np.digitize(time, edges)
    time_binned = (edges[1:] + edges[:-1]) / 2
    flux_binned = np.array([np.nan if len(flux[dig == i]) == 0 else flux[dig == i].mean() for i in range(1, len(edges))])
    err_binned = np.array([np.nan if len(flux[dig == i]) == 0 else np.sqrt(np.sum(err[dig==i]**2))/len(err[dig==i]) for i in range(1, len(edges))])
    time_bin = time_binned[~np.isnan(err_binned)]
    err_bin = err_binned[~np.isnan(err_binned)]
    flux_bin = flux_binned[~np.isnan(err_binned)]

    return time_bin, flux_bin, err_bin

def calc_noise(r_aper, exptime, dc_pp_ps, scint, lc, gain=2.):
    """
    Work out the additonal noise sources for the error bars
    Parameters
    ----------
    r_aper : float
        Radius of the photometry aperture
    epxtime : float
        Current exposure time
    dc_pp_ps : float
        Dark current per pixel per second
    lc : array-like
        The photometry containing star + sky light
    Returns
    -------
    lc_err_new : array-like
        A new error array accounting for other sources of noise
    Raises
    ------
    None
    """
    read_noise = 14.0
    npix = np.pi*r_aper**2
    dark_current = dc_pp_ps*npix*exptime
    # scintillation?
    lc_err_new = np.sqrt(lc/gain + dark_current + npix*read_noise**2 + (scint*lc)**2)
    return lc_err_new

def estimate_scintillation_noise(airmass, exptime):
    """
    Calculate the level of scintillation noise
    Assuming W = 1.75
             airmass = airmass of obs
             height = 2400 m
    Parameters
    ----------
    airmass : array-like
        List of airmass values
    epxtime : float
        Exposure time of the observations
    Returns
    -------
    Nsc : array-like
        Scintillation noise array
    Raises
    ------
    None
    """
    # old youngs/dravins equation
    #Nsc = (0.09*20**(-2./3.)) * (airmass**1.75) * (np.exp(2500./8000.)) * ((2.*exptime)**-0.5)

    # new method from osborn 2015 for paranal
    # convert airmass to zenith distance - in radians
    zd = np.arccos(1./np.array(airmass))
    # 1.56 is the coeff for paranal
    sig_scint = np.sqrt(10E-6 * 1.56**2 * 0.20**(-4./3.) * exptime**-1 * np.cos(zd)**-3. * np.exp(-2*2400./8000.))
    return sig_scint

def find_bad_comp_stars(comp_fluxes, airmass, comp_mags0,
                        sig_level=3., dmag=0.5):
    comp_star_rms = np.array([])
    Ncomps = comp_fluxes.shape[0]
    for i in range(Ncomps):
        comp_flux = np.copy(comp_fluxes[i])
        airmass_cs = np.polyfit(airmass, comp_flux, 1)
        airmass_mod = np.polyval(airmass_cs, airmass)
        comp_flux_corrected = comp_flux / airmass_mod
        comp_flux_norm = comp_flux_corrected / np.median(comp_flux_corrected)
        comp_star_rms = np.append(comp_star_rms, np.std(comp_flux_norm))
    comp_star_mask = np.array([True for cs in comp_star_rms])
    i = 0.
    while True:
        i += 1.
        comp_mags = np.copy(comp_mags0[comp_star_mask])
        comp_rms = np.copy(comp_star_rms[comp_star_mask])
        N1 = len(comp_mags)
        edges = np.arange(comp_mags.min(), comp_mags.max()+dmag, dmag)
        dig = np.digitize(comp_mags, edges)
        mag_nodes = (edges[:-1]+edges[1:]) / 2.
        std_medians = np.array([np.nan if len(comp_rms[dig==i])==0. 
                                else np.median(comp_rms[dig==i]) 
                                for i in range(1, len(edges))])
        cut = np.isnan(std_medians)
        mag_nodes = mag_nodes[~cut]
        std_medians = std_medians[~cut]
        spl = ius(mag_nodes, std_medians)
        mod = spl(comp_mags)
        mod0 = spl(comp_mags0)
        std = np.std(comp_rms - mod)
        comp_star_mask = (comp_star_rms <= mod0 + std * sig_level)
        N2 = np.sum(comp_star_mask)
        if N1 == N2:
            break
        elif i > 10.:
            break
    return comp_star_mask, comp_star_rms, i
    
if __name__ == "__main__":
    args = ParseArgs()
    tic = args.tic
    name = args.name
    bsdir = '/ngts/scratch/brightstars/PAOPhot2/'+name+'/'
    if not os.path.exists(bsdir):
        os.system('mkdir '+bsdir)
    outdir = bsdir+'analyse_outputs/'
    root_dir = '/ngts/scratch/PAOPhot2/'
    if not os.path.exists(outdir):
        os.system('mkdir '+outdir)
        os.system('mkdir '+outdir+'comp_star_check_plots/')
        os.system('mkdir '+outdir+'ind_tel_lcs/')
        os.system('mkdir '+outdir+'data_files/')
        os.system('mkdir '+outdir+'logs/')
    if not os.path.exists(bsdir+'action_summaries/'):
        os.system('mkdir '+bsdir+'action_summaries/')
    night = args.night
    if night is None:
        night = input('Enter night of obs: ')
    if not os.path.exists(outdir+'comp_star_check_plots/'+night):
        os.system('mkdir '+outdir+'comp_star_check_plots/'+night)
    if not os.path.exists(outdir+'ind_tel_lcs/'+night):
        os.system('mkdir '+outdir+'ind_tel_lcs/'+night)
    if not os.path.exists(outdir+'data_files/'+night):
        os.system('mkdir '+outdir+'data_files/'+night)
    if not os.path.exists(outdir+'logs/'+night):
        os.system('mkdir '+outdir+'logs/'+night)
    logger_main = custom_logger(outdir+'logs/'+night+'/'+name+'_'+night+'_main.log')
    actions = args.actions
    r_ap = args.aper
    if r_ap is None:
        ap_id = np.loadtxt(root_dir+f'target_catalogues/TIC-{tic}_apstring.dat', dtype=int)
        r_ap = [np.arange(2.0, 10.5, 0.5)[ap_id]]
    ap_ids = [int(2*r - 4) for r in r_ap]
    
    star_cat = pyfits.getdata(root_dir+f'target_catalogues/TIC-{tic}.fits')
    star_mask= pyfits.getdata(root_dir+f'target_catalogues/TIC-{tic}_mask.fits')
    comp_mask= np.array([m[0] for m in star_mask])
    tic_ids = np.array(star_cat.tic_id)
    idx = np.array(star_cat.PAOPHOT_IDX)
    logger_main.info(f'User rejected comp stars (tics): {args.bad_comp_tics}')
    logger_main.info(f'User rejected comp stars (inds): {args.bad_comp_inds}')
    if args.bad_comp_tics is not None:
        comp_mask_2 = [False if t in args.bad_comp_tics else True
                       for t in tic_ids[idx==2]]
        comp_mask &= comp_mask_2
    elif args.bad_comp_inds is not None:
        comp_mask_2 = [False if i in args.bad_comp_inds else True
                       for i in range(100)]
        comp_mask &= comp_mask_2
    tmags_full = np.array(star_cat.Tmag)
    tmag_target = tmags_full[0]
    tmags_comps = tmags_full[idx==2]
    logger_main.info('Checking comp star brightness')
    bad_mag_inds = []
    bad_mag_tics = []
    for i, tici in zip(range(len(tmags_comps)), tic_ids[idx==2]):
        tmag = tmags_comps[i]
        if tmag < tmag_target-args.dmb or tmag > tmag_target + args.dmf:
            comp_mask[i] = False
            bad_mag_inds.append(i)
            bad_mag_tics.append(tici)
    logger_main.info(f'Comps rejected by brightness (inds): {bad_mag_inds}')
    logger_main.info(f'Comps rejected by brightness (tics): {bad_mag_tics}')
    
    ac_apers_min_target = np.array([])
    ac_apers_min_master = np.array([])
    for ac in actions:
        logger = custom_logger(outdir+'logs/'+night+'/'+name+f'_action{ac}.log')
        print(' ')
        print(' ')
        logger.info(f'Running for Action{ac}...')
        os.system('cp '+root_dir+f'action_summaries/{ac}_TIC-{tic}.png '+bsdir+'action_summaries/')
        phot_file_root = root_dir+f'photometry/action{ac}/ACITON_{ac}_'
        bjds = pyfits.getdata(phot_file_root+'BJD.fits')
        fluxes = pyfits.getdata(phot_file_root+'FLUX.fits')
        skybgs = pyfits.getdata(phot_file_root+'FLUX_BKG.fits')
        target_fluxes_full = np.copy(fluxes[0])
        target_bjd = np.copy(bjds[0])
        target_skys_full = np.copy(skybgs[0])
        airmass = pyfits.getdata(phot_file_root+'AIRMASS.fits')
        scint_noise = estimate_scintillation_noise(airmass, 10.)
        phot_csv_file = outdir+'data_files/'+night+f'/action{ac}_paophot2_dat.csv'
        if os.path.exists(phot_csv_file):
            logger.info('Phot CSV file already exists: '+phot_csv_file)
            logger.info('Adding "new" data to existing file...')
            df = pd.read_csv(phot_csv_file, index_col='NExposure')
        else:
            logger.info('No existing phot csv.')
            logger.info('Creating new phot file: '+phot_csv_file)
            df = pd.DataFrame(np.column_stack((target_bjd, airmass)),
                          columns=['BJD','Airmass'])
        
        comp_fluxes_full = np.copy(fluxes[idx==2][comp_mask])
        comp_skys_full = np.copy(skybgs[idx==2][comp_mask])
        comp_bjds = np.copy(bjds[idx==2][comp_mask])
        comp_tics_full = np.copy(tic_ids[idx==2][comp_mask])
        comp_tmags_full = np.copy(tmags_comps[comp_mask])
        Ncomps_full = len(comp_mask)
        comp_inds_full = np.linspace(0, Ncomps_full-1, Ncomps_full, dtype=int)[comp_mask]
        
        ap_target_rms = np.array([])
        ap_comp_rms = np.array([])
        
        for r, idr in zip(r_ap, ap_ids):
            print(' ')
            logger.info(f'Running for: Action {ac};  Aper - {r:.1f} pix')
            target_flux = np.copy(target_fluxes_full[:, idr])
            target_sky = np.copy(target_skys_full[:, idr])
            df.loc[:, f'RawA{r}'] = target_flux
            df.loc[:, f'SkyBgA{r}'] = target_sky
            comp_fluxes = np.vstack(([np.copy(cf[:, idr])
                                      for cf in comp_fluxes_full]))
            comp_skys = np.vstack(([np.copy(cf[:, idr])
                                    for cf in comp_skys_full]))
            Ncomps = comp_fluxes.shape[0]
            logger.info('Finding bad comp stars...')
            comp_star_mask_r, comp_star_rms, Niter = find_bad_comp_stars(
                comp_fluxes, airmass, comp_tmags_full, dmag=args.dmag)
            comp_fluxes_good = np.copy(comp_fluxes[comp_star_mask_r])
            
            logger.info(f'Searched through {Niter:.0f} iterations.')
            logger.info(f'Number of bad_comp_stars (A{r:.1f}): {np.sum(~comp_star_mask_r)}')
            logger.info(f'Bad_comp_stars (A{r:.1f}; inds): {comp_inds_full[~comp_star_mask_r]}')
            logger.info(f'Bad_comp_stars (A{r:.1f}; tics): {comp_tics_full[~comp_star_mask_r]}')
            
            logger.info(f'Number of good comp stars (A{r:.1f}): {np.sum(comp_star_mask_r)}')
            logger.info(f'Good comp stars (A{r:.1f}; inds): {comp_inds_full[comp_star_mask_r]}')
            logger.info(f'Good comp stars (A{r:.1f}; tics): {comp_tics_full[comp_star_mask_r]}')
            
            plt.figure()
            plt.semilogy(comp_tmags_full[~comp_star_mask_r],
                         comp_star_rms[~comp_star_mask_r] * 100,
                         '.r', zorder=1)
            plt.semilogy(comp_tmags_full[comp_star_mask_r],
                         comp_star_rms[comp_star_mask_r] * 100,
                         '.k', zorder=2)
            for i, j, flag in zip(range(Ncomps), comp_inds_full, comp_star_mask_r):
                if flag:
                    c='black'
                else:
                    c='red'
                plt.gca().annotate(int(j),
                                   (comp_tmags_full[i]+0.01, 100 * comp_star_rms[i]+0.01),
                                   color=c)
            plt.xlabel('Tmag')
            plt.ylabel('RMS (% per exposure)')
            plt.title(name+f'  Action {ac}   Aper {r} pix')
            plt.savefig(outdir+'comp_star_check_plots/'+night+f'/action{ac}_A{r}_mag_vs_rms.png')
            plt.close()
            
     #       fig, axes = plt.subplots(int((Ncomps+1)/2), 2, sharex=True,
     #                                figsize=(12, 3*int((Ncomps+1)/2)))
     #       axes = axes.reshape(-1)
     #       for i, j, flag in zip(range(Ncomps), comp_inds_full, comp_star_mask_r):
     #           if flag:
     #               c='k'
     #           else:
     #               c='r'
     #           ax=axes[i]
     #           comp_tic = comp_tics_full[i]
     #           comp_flux = np.copy(comp_fluxes[i])
     #           airmass_cs = np.polyfit(airmass, comp_flux, 1)
     #           airmass_mod = np.polyval(airmass_cs, airmass)
     #           comp_flux_corrected = comp_flux / airmass_mod
     #           comp_flux_norm = comp_flux_corrected / np.median(comp_flux_corrected)
     #           ax.plot(comp_bjds[i], comp_flux_norm, '.'+c, alpha=0.5,
     #                   label=f'{j}:  TIC-{comp_tic}')
     #           leg = ax.legend(loc='upper center', frameon=False)
     #           plt.setp(leg.get_texts(), color=c)
     #       fig.subplots_adjust(hspace=0., wspace=0.)
     #       plt.savefig(outdir+'comp_star_check_plots/'+night+f'/action{ac}_A{r}_comp_star_airmassdt_lcs.png')
     #       plt.close()
            
            
            fig, axes = plt.subplots(int((Ncomps+1)/2), 2, sharex=True,
                                     figsize=(12, 3*int((Ncomps+1)/2)))
            axes = axes.reshape(-1)
            comp_flux0 = np.copy(comp_fluxes[0])
            for i, j, flag in zip(range(Ncomps), comp_inds_full, comp_star_mask_r):
                if flag:
                    c='k'
                else:
                    c='r'
                ax=axes[i]
                comp_tic = comp_tics_full[i]
                comp_flux = np.copy(comp_fluxes[i])
                comp_flux_corrected = comp_flux / comp_flux0
            #    comp_flux_norm = comp_flux_corrected / np.median(comp_flux_corrected)
                ax.plot(comp_bjds[i], comp_flux_corrected, '.'+c, alpha=0.5,
                        label=f'{j}:  TIC-{comp_tic}')
                leg = ax.legend(loc='upper center', frameon=False)
                plt.setp(leg.get_texts(), color=c)
            fig.subplots_adjust(hspace=0., wspace=0.)
            plt.savefig(outdir+'comp_star_check_plots/'+night+f'/action{ac}_A{r}_comp_star_comp0dt_lcs.png')
            plt.close()
            
     #       fig, axes = plt.subplots(int((Ncomps+1)/2), 2, sharex=True,
     #                                figsize=(12, 3*int((Ncomps+1)/2)))
     #       axes = axes.reshape(-1)
     #       for i, j, flag in zip(range(Ncomps), comp_inds_full, comp_star_mask_r):
     #           if flag:
     #               c='k'
     #           else:
     #               c='r'
     #           ax=axes[i]
     #           comp_tic = comp_tics_full[i]
     #           comp_flux = np.copy(comp_fluxes[i])
     #           ax.plot(comp_bjds[i], comp_flux, '.'+c, alpha=0.5,
     #                   label=f'{j}:  TIC-{comp_tic}')
     #           leg = ax.legend(loc='upper center', frameon=False)
     #           plt.setp(leg.get_texts(), color=c)
     #       fig.subplots_adjust(hspace=0., wspace=0.)
     #       plt.savefig(outdir+'comp_star_check_plots/'+night+f'/action{ac}_A{r}_comp_star_rawcounts.png')
     #       plt.close()
                        
            master_comp = np.sum(comp_fluxes_good, axis=0)
            cs_mc_am = np.polyfit(airmass, master_comp, 1)
            airmass_mc_mod = np.polyval(cs_mc_am, airmass)
            ap_comp_rms = np.append(ap_comp_rms, np.std(master_comp / airmass_mc_mod))
            logger.info(f'A{r} - master comp rms: {np.std(master_comp / airmass_mc_mod)*100:.3f} %')
            comp_skys_good = comp_skys[comp_star_mask_r]
            comp_errs = np.vstack(([calc_noise(r, 10, 1.0, scint_noise, cfi+csi)
                                    for cfi, csi in zip(comp_fluxes_good, comp_skys_good)]))
            master_comp_err = np.sqrt(np.sum(comp_errs**2, axis=0))
            lightcurve = target_flux / master_comp
            target_err = calc_noise(r, 10, 1.0, scint_noise, target_flux+target_sky)
            err_factor = np.sqrt((target_err/target_flux)**2 + (master_comp_err/master_comp)**2)
            lightcurve_err = lightcurve * err_factor
            
            t0 = int(target_bjd[0])
            bjd0 = target_bjd-t0
            ti, te = args.ti, args.te
            if ti is None and te is None:
                oot = bjd0 > 0.
            elif ti is None and te is not None:
                oot = bjd0 > te
            elif ti is not None and te is None:
                oot = bjd0 < ti
            else:
                oot = (bjd0 < ti)| (bjd0 > te)
            
            lc_med = np.median(lightcurve[oot])
            
            norm_flux = lightcurve / lc_med
            norm_err = lightcurve_err / lc_med
            
            ap_target_rms = np.append(ap_target_rms, np.std(norm_flux[oot]))
            logger.info(f'A{r} - target rms: {np.std(norm_flux[oot])*100:.3f} %')
            
            df.loc[:, f'FluxA{r}'] = lightcurve
            df.loc[:, f'FluxErrA{r}'] = lightcurve_err
            df.loc[:, f'FluxNormA{r}'] = norm_flux
            df.loc[:, f'FluxNormErrA{r}'] = norm_err
            df.loc[:, f'MasterCompA{r}'] = master_comp
            df.loc[:, f'MasterCompErrA{r}'] = master_comp_err
            
            t0 = int(target_bjd[0])
            plt.figure()
            plt.plot(target_bjd-t0, norm_flux, '.k')
            
            norm_sig = np.std(norm_flux)
            idbin = ((norm_flux > 1-6.*norm_sig) & (norm_flux < 1+6.*norm_sig))
            tb, fb, eb = lb(target_bjd[idbin], norm_flux[idbin], norm_err[idbin], 5/1440.)
            plt.errorbar(tb-t0, fb, yerr=eb, fmt='bo')
            
            plt.ylabel('Norm Flux')
            plt.xlabel(f'Time (BJD - {t0})')
            
            plt.savefig(outdir+'ind_tel_lcs/'+night+f'/action{ac}_A{r}_lc.png')
            plt.close()
            
            df.to_csv(phot_csv_file,
                      index_label='NExposure')
            
        plt.figure()
        plt.plot(np.array(r_ap), ap_target_rms*100, 'ko', label=name)
        plt.ylabel('LC Precision (%)')
        plt.xlabel('Aperture Radius (pixels)')
        ax2 = plt.twinx()
        ax2.plot(np.array(r_ap), ap_comp_rms * 100, 'ro', label='MasterComp')
        ax2.set_ylabel('MasterComp Precision (%)', color='red')
        ax2.tick_params(axis='y', colors='red')
        plt.savefig(outdir+'ind_tel_lcs/'+night+f'/action{ac}_aperture_precisions.png')
        plt.close()
        
        ac_apers_min_target = np.append(ac_apers_min_target, np.array(r_ap)[ap_target_rms.argmin()])
        ac_apers_min_master = np.append(ac_apers_min_master, np.array(r_ap)[ap_comp_rms.argmin()])
    
    bjd, flux_t, err_t = np.array([]), np.array([]), np.array([])
    flux_mc, err_mc = np.array([]), np.array([])
    for ac, rt, rc in zip(actions, ac_apers_min_target, ac_apers_min_master):
        logger.info(f'Action {ac} - "Best" apers -  Target: {rt} pix; Comp: {rc} pix')
        dat = pd.read_csv(outdir+'data_files/'+night+f'/action{ac}_paophot2_dat.csv',
                          index_col='NExposure')
        bjd = np.append(bjd, np.array(dat.BJD))
        flux_t = np.append(flux_t, np.array(dat.loc[:, f'FluxNormA{rt}']))
        err_t = np.append(err_t, np.array(dat.loc[:, f'FluxNormErrA{rt}']))
        flux_mc = np.append(flux_mc, np.array(dat.loc[:, f'FluxNormA{rc}']))
        err_mc = np.append(err_mc, np.array(dat.loc[:, f'FluxNormErrA{rc}']))
    
    tbin_t, fbin_t, ebin_t = lb(bjd, flux_t, err_t, 5/1440.)
    tbin_mc, fbin_mc, ebin_mc = lb(bjd, flux_mc, err_mc, 5/1440.)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    ax1.plot(bjd-t0, flux_t, '.k', alpha=0.3, zorder=1)
    ax1.errorbar(tbin_t-t0, fbin_t, yerr=ebin_t, fmt='bo', ecolor='cyan', zorder=2)
    ax1.set_ylabel('Norm Flux - Target Apers', fontsize=14)
    ax1.set_title(name+'  Night: '+night+f'\nActions: {actions}\nApers: {ac_apers_min_target}')
    
    ax2.plot(bjd-t0, flux_mc, '.k', alpha=0.3, zorder=1)
    ax2.errorbar(tbin_mc-t0, fbin_mc, yerr=ebin_mc, fmt='bo', ecolor='cyan', zorder=2)
    ax2.set_ylabel('Norm Flux - Comp Apers', fontsize=14)
    ax2.set_title(name+'  Night: '+night+f'\nActions: {actions}\nApers: {ac_apers_min_master}')
    
    ax2.set_xlabel(f'Time (BJD - {t0}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(outdir+'/'+name+'_NGTS_'+night+'_paophot2_tmp_lc.png')
    plt.show(block=False)
    save = input('Save over autosaved plot? [y/n] :  ')
    if save == 'y':
        plt.savefig(outdir+'/'+name+'_NGTS_'+night+'_paophot2_tmp_lc.png')
    plt.close()
