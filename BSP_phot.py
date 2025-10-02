#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 09:40:07 2025

@author: emb
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import BSP_utils as bspu
import BSP_db as bspd

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

def find_comp_star_rms(comp_fluxes, airmass, comp_mags0):
    comp_star_rms = np.array([])
    Ncomps = comp_fluxes.shape[0]
    for i in range(Ncomps):
        comp_flux = np.copy(comp_fluxes[i])
        airmass_cs = np.polyfit(airmass, comp_flux, 1)
        airmass_mod = np.polyval(airmass_cs, airmass)
        comp_flux_corrected = comp_flux / airmass_mod
        comp_flux_norm = comp_flux_corrected / np.median(comp_flux_corrected)
        comp_star_rms_val = np.std(comp_flux_norm)
        if np.isfinite(comp_star_rms_val):
            comp_star_rms = np.append(comp_star_rms, comp_star_rms_val)
        else:
            comp_star_rms = np.append(comp_star_rms, 99.)
    return comp_star_rms

def find_bad_comp_stars(logger, comp_fluxes, airmass, comp_mags0,
                        sig_level=3., dmag=0.5):
    comp_star_rms = find_comp_star_rms(comp_fluxes, airmass, comp_mags0)
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


def get_aperture_radii(cmd_args, tmag_target):
    r_ap = cmd_args.aper
    if r_ap is None:
        if tmag_target >= 10.:
            r_ap = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        elif 9.0 <= tmag_target < 10.0:
            r_ap = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        elif tmag_target < 9.0:
            r_ap = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    ap_ids = [int(2*r - 4) for r in r_ap]
    return r_ap, ap_ids

def obtain_initial_comparison_star_list(logger, cmd_args, obs_night, ac_id,
                                        tic_ids, idx, comp_mask_ngpipe,
                                        target_tmag, comp_star_tmags):
    if cmd_args.force_comp_stars:
        logger.info(f'Night {obs_night}  Action {ac_id}: Using user defined comparison stars.')
        if cmd_args.comp_inds is not None:
            comp_mask = np.array([True if i in cmd_args.comp_inds else False 
                                  for i in range(np.sum(idx == 2))])
            logger.info(f'Night {obs_night}  Action {ac_id}: Using {np.sum(comp_mask)} user defined comparison stars.')
            logger.info(f'Night {obs_night}  Action {ac_id}: Using these comparison stars (inds): {cmd_args.comp_inds}')
        elif cmd_args.comp_tics is not None:
            comp_mask = np.array([True if t in cmd_args.comp_tics else False 
                                  for t in tic_ids[idx==2]])
            logger.info(f'Night {obs_night}  Action {ac_id}: Using {np.sum(comp_mask)} user defined comparison stars.')
            logger.info(f'Night {obs_night}  Action {ac_id}: Using these comparison stars (tics): {cmd_args.comp_tics}')
        else:
            logger.info('ERROR - If user defined comp stars (--force_comp_stars) I need comparison star IDs (--comp_inds) or TIC IDs (--comp_tics).')
            raise ValueError('If user defined comp stars (--force_comp_stars) I need comparison star IDs (--comp_inds) or TIC IDs (--comp_tics).')        
    
    else:
        comp_mask= np.array([bool(m) for m in comp_mask_ngpipe])
        logger.info(f'Night {obs_night}  Action {ac_id}: User rejected comp stars (tics): {cmd_args.bad_comp_tics}')
        logger.info(f'Night {obs_night}  Action {ac_id}: User rejected comp stars (inds): {cmd_args.bad_comp_inds}')
        if cmd_args.bad_comp_tics is not None:
            comp_mask_2 = [False if t in cmd_args.bad_comp_tics else True
                           for t in tic_ids[idx==2]]
            comp_mask &= comp_mask_2
        elif cmd_args.bad_comp_inds is not None:
            comp_mask_2 = [False if i in cmd_args.bad_comp_inds else True
                           for i in range(np.sum(idx == 2))]
            comp_mask &= comp_mask_2
        logger.info(f'Night {obs_night}  Action {ac_id}: Checking comp star brightness')
        bad_mag_inds = []
        bad_mag_tics = []
        for i, tici in zip(range(len(comp_star_tmags)), tic_ids[idx==2]):
            tmag = comp_star_tmags[i]
            if tmag < target_tmag-cmd_args.dmb or tmag > target_tmag + cmd_args.dmf:
                comp_mask[i] = False
                bad_mag_inds.append(i)
                bad_mag_tics.append(tici)
        logger.info(f'Night {obs_night}  Action {ac_id}: Comps rejected by brightness (inds): {bad_mag_inds}')
        logger.info(f'Night {obs_night}  Action {ac_id}: Comps rejected by brightness (tics): {bad_mag_tics}')
    
    return logger, comp_mask

def get_ngpipe_photometric_data(logger, logger_main, cmd_args, phot_file_dir, ngpipe_output_dir,
                                ac_id, idx, comp_mask, cat_tic_ids, comp_star_tmags):
     if os.path.exists(phot_file_dir):
         logger.info('Found photometry directory')
         phot_file_root = phot_file_dir + f'ACTION_{ac_id}_'
     else:
         logger.info(f'Can\'t find photometry for Action {ac_id}')
         logger.info('Trying "old" directory')
         phot_file_dir = ngpipe_output_dir + f'old/photometry_old/action{ac_id}/'
         if os.path.exists(phot_file_dir):
             logger.info('Found "old" photometry directory')
             phot_file_root = phot_file_dir + f'ACTION_{ac_id}_'
         else:
             logger_main.info(f'No photometry for Action {ac_id}')
             logger_main.info(f'Skipping Action {ac_id}.')
             return None, None, None, None, None, None, None, None
     try:
         bjds = pyfits.getdata(phot_file_root+'BJD.fits.bz2') / 86400. + 2456658.5
     except:
         bjds = pyfits.getdata(phot_file_root+'BJD.fits') / 86400. + 2456658.5
     try:
         fluxes = pyfits.getdata(phot_file_root+'FLUX.fits.bz2')
     except:
         fluxes = pyfits.getdata(phot_file_root+'FLUX.fits')
     try:
         skybgs = pyfits.getdata(phot_file_root+'FLUX_BKG.fits.bz2')
     except:
         skybgs = pyfits.getdata(phot_file_root+'FLUX_BKG.fits')
     try:
         psfs = pyfits.getdata(phot_file_root+'PSF.fits.bz2')
     except:
         psfs = pyfits.getdata(phot_file_root+'PSF.fits')
     
     #try:
     #    airmass_0 = pyfits.getdata(phot_file_root+'IMAGELIST.fits.bz2')
     airmass_hdu_0 = pyfits.open(phot_file_root+'IMAGELIST.fits')
     airmass_0 = np.array(airmass_hdu_0[1].data['AIRMASS'])
     
     target_bjd0 = np.copy(bjds[0])
     bjd_int = int(target_bjd0[0])
     ignore = cmd_args.ignore_bjd
     ignore1, ignore2 = ignore[0]+bjd_int, ignore[1]+bjd_int
     target_flux0 = np.copy(fluxes[0][:, 0])    # select target array for 1 aperture to check for bad images
     bjd_keep_time_range = (target_bjd0 <= ignore1) | (target_bjd0 >= ignore2) 
     bjd_keep_good_timestamps = (target_flux0 > 0.)
     airmass_keep = airmass_0 > 0.99
     
     if np.sum(airmass_keep) <= 0.2 * len(airmass_0):
         logger.info('Fewer than 20% of the airmass array is above 1.')
         logger.info('Skipping action.')
         return None, None, None, None, None, None, None, None
     
     keep = bjd_keep_time_range & bjd_keep_good_timestamps & airmass_keep
     target_bjd = np.copy(bjds[0])[keep]
     target_fluxes_full = np.copy(fluxes[0])[keep]
     target_skys_full = np.copy(skybgs[0])[keep]
     sep_centre_fwhm = np.copy(psfs[1])[keep]
     tl_centre_fwhm = np.mean(psfs[[14, 15]], axis=0)[keep]
     rgw_fwhm = np.copy(psfs[-3])[keep]
     airmass = np.copy(airmass_0)[keep]
     
     scint_noise = estimate_scintillation_noise(airmass, float(cmd_args.exptime))
     
     comp_fluxes_full = np.copy(fluxes[idx==2][comp_mask][:, keep, :])
     comp_skys_full = np.copy(skybgs[idx==2][comp_mask][:, keep, :])
     comp_bjds = np.copy(bjds[idx==2][comp_mask][:, keep])
     comp_tics_full = np.copy(cat_tic_ids[idx==2][comp_mask])
     comp_tmags_full = np.copy(comp_star_tmags[comp_mask])
     Ncomps_full = len(comp_mask)
     comp_inds_full = np.linspace(0, Ncomps_full-1, Ncomps_full, dtype=int)[comp_mask]
     
     comp_fluxes_bad0 = np.copy(fluxes[idx==2][~comp_mask][:, keep, :])
     comp_bjds_bad0 = np.copy(bjds[idx==2][~comp_mask][:, keep])
     comp_tics_bad0 = np.copy(cat_tic_ids[idx==2][~comp_mask])
     comp_inds_bad0 = np.linspace(0, Ncomps_full-1, Ncomps_full, dtype=int)[~comp_mask]
     Ncomps_bad0 = len(comp_tics_bad0)
     
     return target_bjd, target_fluxes_full, target_skys_full, \
             sep_centre_fwhm, tl_centre_fwhm, rgw_fwhm, airmass, scint_noise, \
                 comp_fluxes_full, comp_skys_full, comp_bjds, comp_tics_full, \
                     comp_inds_full, comp_tmags_full, comp_fluxes_bad0, comp_bjds_bad0, \
                         comp_tics_bad0, comp_inds_bad0, Ncomps_bad0
                         
def make_global_reject_comp_star_plot(global_reject_comp_bjds,
                                      global_reject_comp_fluxes_all_apers,
                                      aper_id, Nglobal_reject_comps, 
                                      global_reject_comp_inds, global_reject_comp_tics,
                                      airmass_vals, ac_id, aper_rad, output_dir):
    comp_fluxes_bad = np.vstack(([np.copy(cfb[:, aper_id]) 
                                  for cfb in global_reject_comp_fluxes_all_apers]))
    #print('Starting Plotting')
    fig, axes = plt.subplots(int((Nglobal_reject_comps+1)/2), 2, sharex=True,
                             figsize=(12, 3*int((Nglobal_reject_comps+1)/2)))
    #print('Made the figure instance')
    axes = axes.reshape(-1)
    for i, j in zip(range(Nglobal_reject_comps), global_reject_comp_inds):
        ax=axes[i]
        comp_tic = global_reject_comp_tics[i]
        comp_flux = np.copy(comp_fluxes_bad[i])
        cs = np.polyfit(airmass_vals, comp_flux, 1)
        airmass_mod = np.polyval(cs, airmass_vals)
        comp_flux_corrected = comp_flux / airmass_mod
        ax.plot(global_reject_comp_bjds[i], comp_flux_corrected, '.k', alpha=0.5,
                label=f'{j}:  TIC-{comp_tic}', rasterized=True)
        leg = ax.legend(loc='upper center', frameon=False)
        plt.setp(leg.get_texts(), color='k')
    fig.subplots_adjust(hspace=0., wspace=0.)
    #print('Finished Plotting')
    plt.savefig(output_dir+f'comp_star_check_plots/action{ac_id}_A{aper_rad}_global_rejected_comp_stars_airmassDT_lcs.png')
    plt.close()

def make_comp_star_plots(comp_tmags, comp_rms_vals, comp_inds, comp_tics,
                         comp_bjds, comp_fluxes, comp_star_mask, airmass_vals,
                         output_dir, obj_name, obs_night, ac_id, aper_radius):
    Nrej_comp_stars = np.sum(~comp_star_mask)
    plt.figure()
    if Nrej_comp_stars > 0:
        plt.semilogy(comp_tmags[~comp_star_mask], comp_rms_vals[~comp_star_mask] * 100,
                     '.r', zorder=2, rasterized=True)
    plt.semilogy(comp_tmags[comp_star_mask], comp_rms_vals[comp_star_mask] * 100,
                 '.k', zorder=1, rasterized=True)
    Ncomps = len(comp_tmags)
    for i, j, flag in zip(range(Ncomps), comp_inds, comp_star_mask):
        if flag:
            c='black'
        else:
            c='red'
        plt.gca().annotate(int(j),
                           (comp_tmags[i]+0.01, 100 * comp_rms_vals[i]+0.01),
                           color=c)
    plt.xlabel('Tmag')
    plt.ylabel('RMS (% per exposure)')
    plt.title(obj_name+'   Night '+obs_night+f'   Action {ac_id}   Aper {aper_radius} pix')
    plt.savefig(output_dir+f'comp_star_check_plots/action{ac_id}_A{aper_radius}_mag_vs_rms.png')
    plt.close()
    
    fig, axes = plt.subplots(int((Ncomps+1)/2), 2, sharex=True,
                             figsize=(12, 3*int((Ncomps+1)/2)))
    axes = axes.reshape(-1)
    comp_flux0 = np.copy(comp_fluxes[0])
    for i, j, flag in zip(range(Ncomps), comp_inds, comp_star_mask):
        if flag:
            c='k'
        else:
            c='r'
        ax=axes[i]
        comp_tic = comp_tics[i]
        comp_flux = np.copy(comp_fluxes[i])
        comp_flux_corrected = comp_flux / comp_flux0
        ax.plot(comp_bjds[i], comp_flux_corrected, '.'+c, alpha=0.5,
                label=f'{j}:  TIC-{comp_tic}', rasterized=True)
        leg = ax.legend(loc='upper center', frameon=False)
        plt.setp(leg.get_texts(), color=c)
    fig.subplots_adjust(hspace=0., wspace=0.)
    plt.savefig(output_dir+f'comp_star_check_plots/action{ac_id}_A{aper_radius}_comp_star_comp0DT_lcs.png')
    plt.close()
    
    fig, axes = plt.subplots(int((Ncomps+1)/2), 2, sharex=True,
                             figsize=(12, 3*int((Ncomps+1)/2)))
    axes = axes.reshape(-1)
    comp_flux0 = np.copy(comp_fluxes[0])
    for i, j, flag in zip(range(Ncomps), comp_inds, comp_star_mask):
        if flag:
            c='k'
        else:
            c='r'
        ax=axes[i]
        comp_tic = comp_tics[i]
        comp_flux = np.copy(comp_fluxes[i])
        cs = np.polyfit(airmass_vals, comp_flux, 1)
        airmass_mod = np.polyval(cs, airmass_vals)
        comp_flux_corrected = comp_flux / airmass_mod
        ax.plot(comp_bjds[i], comp_flux_corrected, '.'+c, alpha=0.5,
                label=f'{j}:  TIC-{comp_tic}', rasterized=True)
        leg = ax.legend(loc='upper center', frameon=False)
        plt.setp(leg.get_texts(), color=c)
    fig.subplots_adjust(hspace=0., wspace=0.)
    plt.savefig(output_dir+f'comp_star_check_plots/action{ac_id}_A{aper_radius}_comp_star_airmassDT_lcs.png')
    plt.close()

def compute_differential_phot_lc(logger, cmd_args, obj_name, obs_night, phot_df, 
                                 ac_id, aper_rad, ap_comp_rms, ap_target_rms,
                                 target_bjd, target_flux, target_sky, target_scint_noise,
                                 comp_fluxes, comp_skys, airmass_vals, outdir):
    master_comp = np.sum(comp_fluxes, axis=0)
    cs_mc_am = np.polyfit(airmass_vals, master_comp, 1)
    airmass_mc_mod = np.polyval(cs_mc_am, airmass_vals)
    ap_comp_rms = np.append(ap_comp_rms, np.std(master_comp / airmass_mc_mod))
    logger.info(f'Action{ac_id}; A{aper_rad} - master comp rms: {np.std(master_comp / airmass_mc_mod)*100:.3f} %')
    comp_errs = np.vstack(([calc_noise(aper_rad, 10, 1.0, target_scint_noise, cfi+csi)
                            for cfi, csi in zip(comp_fluxes, comp_skys)]))
    master_comp_err = np.sqrt(np.sum(comp_errs**2, axis=0))
    lightcurve = target_flux / master_comp
    target_err = calc_noise(aper_rad, 10, 1.0, target_scint_noise, target_flux+target_sky)
    err_factor = np.sqrt((target_err/target_flux)**2 + (master_comp_err/master_comp)**2)
    lightcurve_err = lightcurve * err_factor
    
    # Handle not needing ti and te as arguments
    t0 = int(target_bjd[0])
    bjd0 = target_bjd-t0
    ti, te = cmd_args.ti, cmd_args.te
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
    logger.info(f'Action{ac_id}; A{aper_rad} - target rms: {np.std(norm_flux[oot])*100:.3f} %')
    
    phot_df.loc[:, f'FluxA{aper_rad}'] = lightcurve
    phot_df.loc[:, f'FluxErrA{aper_rad}'] = lightcurve_err
    phot_df.loc[:, f'FluxNormA{aper_rad}'] = norm_flux
    phot_df.loc[:, f'FluxNormErrA{aper_rad}'] = norm_err
    phot_df.loc[:, f'MasterCompA{aper_rad}'] = master_comp
    phot_df.loc[:, f'MasterCompErrA{aper_rad}'] = master_comp_err
    
    t0 = int(target_bjd[0])
    plt.figure()
    plt.plot(target_bjd-t0, norm_flux, '.k', rasterized=True)
    
    norm_sig = np.std(norm_flux)
    idbin = ((norm_flux > 1-6.*norm_sig) & (norm_flux < 1+6.*norm_sig))
    tb, fb, eb = bspu.lb(target_bjd[idbin], norm_flux[idbin], norm_err[idbin], 5/1440.)
    plt.errorbar(tb-t0, fb, yerr=eb, fmt='bo')
    
    plt.ylabel('Norm Flux')
    plt.xlabel(f'Time (BJD - {t0})')
    
    plt.title(obj_name+'   Night '+obs_night+f'   Action {ac_id}   Aper {aper_rad} pix')
    
    plt.savefig(outdir+f'ind_tel_lcs/action{ac_id}_A{aper_rad}_lc.png')
    plt.close()
    
    return logger, phot_df, ap_comp_rms, ap_target_rms

def run_BSP_process_single_action(logger_main, cmd_args, object_name, ngpipe_op_dir,
                                  ac_apers_min_target, ac_apers_min_master,
                                  missing_actions, output_file_names, 
                                  target_tic_id, ac, ns, outdir):
    # Set up a new logger for each action
    logger = bspu.custom_logger(outdir+'logs/'+object_name+'_night'+ns+f'_action{ac}.log')
    print(' ')
    print(' ')
    
    # Query the target catalogue files for each field
    phot_file_root_dir = ngpipe_op_dir + f'bs_photometry/action{ac}/'
    logger, tic_ids_full_cat, cat_idx_vals, comp_star_mask_ngpipe, \
        target_tmag_val, comps_tmag_vals = bspd.query_target_catalogues(
                logger, target_tic_id, phot_file_root_dir, ac)
      
    # Determine the set of aperture radii to run the BSP process for
    aper_radii, aper_radii_ids = get_aperture_radii(cmd_args, target_tmag_val)

    # Obtain the initial comparison star selection
    logger, comp_star_mask_initial = obtain_initial_comparison_star_list(
                                            logger, cmd_args, ns, ac,
                                            tic_ids_full_cat, cat_idx_vals,
                                            comp_star_mask_ngpipe,
                                            target_tmag_val, comps_tmag_vals)
        
    # Get photometric data from ngpipe
    target_bjd_vals, target_fluxes_all_apers, target_skys_all_apers, \
        sep_centre_fwhm, tl_centre_fwhm, rgw_fwhm, airmass, scint_noise_vals, \
            comp_fluxes_good0_all_apers, comp_skys_good0_all_apers, comp_bjd_vals_good0, comp_tics_good0, \
                comp_inds_good0, comp_tmags_good0, comp_fluxes_bad0, comp_bjds_bad0, \
                    comp_tics_bad0, comp_inds_bad0, Ncomps_bad0 = \
            get_ngpipe_photometric_data(logger, logger_main, cmd_args, phot_file_root_dir, 
                                        ngpipe_op_dir, ac, cat_idx_vals,
                                        comp_star_mask_initial, tic_ids_full_cat,
                                        comps_tmag_vals)
    # If target_bjd is None it is because we are skipping this action
    if target_bjd_vals is None:
        missing_actions = np.append(missing_actions, ac)
        return ac_apers_min_target, ac_apers_min_master, missing_actions, output_file_names
    
    # Ensure we have the correct DataFrame to save the data to
    df_full_dir = outdir+'data_files/'
    phot_csv_file = df_full_dir+f'action{ac}_bsproc_dat.csv'
    if os.path.exists(phot_csv_file):
        logger.info('Phot CSV file already exists: '+phot_csv_file)
        logger.info('Adding "new" data to existing file...')
        df = pd.read_csv(phot_csv_file, index_col='NExposure')
        if cmd_args.update_bjd:
            # Optional flag to update the BJD in the BSP csv files
            # Used in cases where the ngpipe BJD correction failed due to web query issues
            df.loc[:, 'BJD'] = target_bjd_vals
    else:
        logger.info('No existing phot csv.')
        logger.info('Creating new phot file: '+phot_csv_file)
        df = pd.DataFrame(np.column_stack((target_bjd_vals, airmass,
                                           sep_centre_fwhm,
                                           tl_centre_fwhm,
                                           rgw_fwhm)),
                          columns=['BJD','Airmass','FWHM_SEP',
                                   'FWHM_TL','FWHM_RGW'])

    ap_target_rms_store = np.array([])
    ap_comp_rms_store = np.array([])
    
    for r, idr in zip(aper_radii, aper_radii_ids):
        print(' ')
        logger.info(f'Running for: Action {ac};  Aper - {r:.1f} pix')
        
        target_flux_vals = np.copy(target_fluxes_all_apers[:, idr])
        target_sky_vals = np.copy(target_skys_all_apers[:, idr])
        df.loc[:, f'RawA{r}'] = target_flux_vals
        df.loc[:, f'SkyBgA{r}'] = target_sky_vals
        comp_fluxes_good0 = np.vstack(([np.copy(cf[:, idr])
                                  for cf in comp_fluxes_good0_all_apers]))
        comp_skys_good0 = np.vstack(([np.copy(cf[:, idr])
                                for cf in comp_skys_good0_all_apers]))
        
        if Ncomps_bad0 == 0:
            logger.info('No globally rejected comparison stars exist. Skipping plot.')
        else:
            logger.info(f'{Ncomps_bad0} globally rejected comparison stars. Plotting...')
            make_global_reject_comp_star_plot(comp_bjds_bad0, comp_fluxes_bad0,
                                              idr, Ncomps_bad0, comp_inds_bad0,
                                              comp_tics_bad0, airmass, ac, r, outdir)
            
        if cmd_args.force_comp_stars:
            logger.info('User defined comparisons. Skipping bad comp rejection.')
            comp_fluxes_good = np.copy(comp_fluxes_good0)
            comp_skys_good = np.copy(comp_skys_good0)
            comp_star_rms_good = find_comp_star_rms(comp_fluxes_good0, airmass, comp_tmags_good0)
            
            make_comp_star_plots(comp_tmags_good0, comp_star_rms_good,
                                 comp_inds_good0, comp_tics_good0,
                                 comp_bjd_vals_good0, comp_fluxes_good0, airmass,
                                 outdir, object_name, ns, ac, r)
            
        else:
            logger.info('Finding bad comp stars...')
            comp_star_mask_r, comp_star_rms_good, Niter = find_bad_comp_stars(
                logger, comp_fluxes_good0, airmass, comp_tmags_good0, dmag=cmd_args.dmag)
            comp_fluxes_good = np.copy(comp_fluxes_good0[comp_star_mask_r])
            comp_skys_good = np.copy(comp_skys_good0[comp_star_mask_r])
            
            logger.info(f'Searched through {Niter:.0f} iterations.')
            logger.info(f'Number of bad_comp_stars (Action{ac}; A{r:.1f}): {np.sum(~comp_star_mask_r)}')
            logger.info(f'Bad_comp_stars (Action{ac}; A{r:.1f}; inds): {comp_inds_good0[~comp_star_mask_r]}')
            logger.info(f'Bad_comp_stars (Action{ac}; A{r:.1f}; tics): {comp_tics_good0[~comp_star_mask_r]}')
            
            logger.info(f'Number of good comp stars (Action{ac}; A{r:.1f}): {np.sum(comp_star_mask_r)}')
            logger.info(f'Good comp stars (Action{ac}; A{r:.1f}; inds): {comp_inds_good0[comp_star_mask_r]}')
            logger.info(f'Good comp stars (Action{ac}; A{r:.1f}; tics): {comp_tics_good0[comp_star_mask_r]}')
            
            make_comp_star_plots(comp_tmags_good0, comp_star_rms_good,
                                 comp_inds_good0, comp_tics_good0,
                                 comp_bjd_vals_good0, comp_fluxes_good0,
                                 comp_star_mask_r, airmass,
                                 outdir, object_name, ns, ac, r)

        
        logger, df, ap_comp_rms_store, ap_target_rms_store = \
            compute_differential_phot_lc(logger, cmd_args, object_name, ns, df,
                                         ac, r, ap_comp_rms_store, ap_target_rms_store,
                                         target_bjd_vals, target_flux_vals,
                                         target_sky_vals, scint_noise_vals,
                                         comp_fluxes_good, comp_skys_good, airmass,
                                         outdir
                                         )
        
    df.to_csv(phot_csv_file,
              index_label='NExposure')
        
    plt.figure()
    plt.plot(np.array(aper_radii), ap_target_rms_store*100, 'ko', label=object_name)
    plt.ylabel('LC Precision (%)')
    plt.xlabel('Aperture Radius (pixels)')
    plt.title(object_name+'   Night '+ns+f'   Action {ac}')
    ax2 = plt.twinx()
    ax2.plot(np.array(aper_radii), ap_comp_rms_store * 100, 'ro', label='MasterComp')
    ax2.set_ylabel('MasterComp Precision (%)', color='red')
    ax2.tick_params(axis='y', colors='red')
    plt.savefig(outdir+f'ind_tel_lcs/action{ac}_aperture_precisions.png')
    plt.close()
    
    ac_apers_min_target = np.append(
        ac_apers_min_target, np.array(aper_radii)[ap_target_rms_store.argmin()])
    ac_apers_min_master = np.append(
        ac_apers_min_master, np.array(aper_radii)[ap_comp_rms_store.argmin()])
    output_file_names.append(phot_csv_file)
    
    return ac_apers_min_target, ac_apers_min_master, missing_actions, output_file_names

def run_BSP_process(logger_main, cmd_args, object_name, ngpipe_op_dir, action_ids,
                    obs_night_store, ind_nights_outdir_store, target_tic_id):
    # Set up containers for full loop
    ac_apers_min_target_store = np.array([], dtype=float)
    ac_apers_min_master_store = np.array([], dtype=float)
    missing_action_store = np.array([], dtype=int)
    output_file_name_store = []
    
    # Run the BSP process for each action
    for action_id, night_str, outdir in zip(action_ids, obs_night_store, 
                                            ind_nights_outdir_store):
        ac_apers_min_target_store, ac_apers_min_master_store, missing_action_store, \
            output_file_name_store = run_BSP_process_single_action(
                logger_main, cmd_args, object_name, ngpipe_op_dir,
                ac_apers_min_target_store, ac_apers_min_master_store,
                missing_action_store, output_file_name_store,
                target_tic_id, action_id, night_str, outdir)
    
    return ac_apers_min_target_store, ac_apers_min_master_store, \
            missing_action_store, output_file_name_store

def collect_best_aperture_photometry(logger_main, ac_apers_min_target, ac_apers_min_master,
                                     missing_actions, output_file_names, action_ids, obs_night_store,
                                     obs_nights, obj_name, target_tic, cmd_args, outdir):
    action_store = np.array([], dtype=int)
    airmass_store = np.array([])
    fwhm_sep, fwhm_tl, fwhm_rgw = np.array([]), np.array([]), np.array([])
    bjd, flux_t, err_t = np.array([]), np.array([]), np.array([])
    flux_mc, err_mc = np.array([]), np.array([])
    flux0_t, err0_t, skybg_t = np.array([]), np.array([]), np.array([])
    flux0_mc, err0_mc, skybg_mc = np.array([]), np.array([]), np.array([])
    ac_map = np.array([True if not ac in missing_actions else False for ac in action_ids])
    
    for ac, ns, fn, rt, rc in zip(action_ids[ac_map], np.array(obs_night_store)[ac_map],
                                  output_file_names, ac_apers_min_target, ac_apers_min_master):
        logger_main.info(f'Action {ac} - "Best" apers -  Target: {rt} pix; Comp: {rc} pix')
        dat = pd.read_csv(fn, index_col='NExposure')
        action_store = np.append(action_store, np.array([ac for i in range(len(dat))], dtype=int))
        bjd = np.append(bjd, np.array(dat.BJD))
        airmass_store = np.append(airmass_store, np.array(dat.Airmass))
        fwhm_sep = np.append(fwhm_sep, np.array(dat.FWHM_SEP))
        fwhm_tl  = np.append(fwhm_tl,  np.array(dat.FWHM_TL))
        fwhm_rgw = np.append(fwhm_rgw, np.array(dat.FWHM_RGW))
        flux_t = np.append(flux_t, np.array(dat.loc[:, f'FluxNormA{rt}']))
        err_t = np.append(err_t, np.array(dat.loc[:, f'FluxNormErrA{rt}']))
        flux_mc = np.append(flux_mc, np.array(dat.loc[:, f'FluxNormA{rc}']))
        err_mc = np.append(err_mc, np.array(dat.loc[:, f'FluxNormErrA{rc}']))
        flux0_t = np.append(flux0_t, np.array(dat.loc[:, f'FluxA{rt}']))
        err0_t = np.append(err0_t, np.array(dat.loc[:, f'FluxErrA{rt}']))
        flux0_mc = np.append(flux0_mc, np.array(dat.loc[:, f'FluxA{rc}']))
        err0_mc = np.append(err0_mc, np.array(dat.loc[:, f'FluxErrA{rc}']))
        skybg_t = np.append(skybg_t, np.array(dat.loc[:, f'SkyBgA{rt}']))
        skybg_mc = np.append(skybg_mc, np.array(dat.loc[:, f'SkyBgA{rc}']))
        
        
    opt  = np.column_stack((action_store, bjd, airmass_store,
                            flux_t, err_t,
                            flux0_t, err0_t, skybg_t,
                            fwhm_sep, fwhm_tl, fwhm_rgw))
    opmc = np.column_stack((action_store, bjd, airmass_store,
                            flux_mc, err_mc,
                            flux0_mc, err0_mc, skybg_mc,
                            fwhm_sep, fwhm_tl, fwhm_rgw))
    
    ns1 = ''
    ns2 = ''
    for n in obs_nights:
        ns1 += ' '+n
        ns2 += '_'+n

    if len(ns2) > 55:
        ns2 = '{} to {}'.format(ns2[:10],ns2[-10])
    
    if cmd_args.camera is None:
        camstr = ''
    else:
        camstr= '_cam'+cmd_args.camera
    headert =  ' Object: '+obj_name+f'  (TIC-{target_tic})       '+camstr + \
               '\n Night(s): '+ns1 + \
              f'\n Actions: {action_ids[ac_map]}' + \
              f'\n Aperture Radii: {ac_apers_min_target} pixels' + \
               '\n Note these apertures minimise the target flux RMS' + \
               '\n ActionID   BJD   Airmass   FluxNorm   FluxNormErr   Flux   FluxErr  SkyBg   FWHM_SEP   FWHM_TL   FWHM_RGW'
    
    headermc =  ' Object: '+obj_name+f'  (TIC-{target_tic})       '+camstr + \
                '\n Night(s): '+ns1 + \
               f'\n Actions: {action_ids[ac_map]}' + \
               f'\n Aperture Radii: {ac_apers_min_master} pixels' + \
                '\n Note these apertures minimise the master comparison flux RMS' + \
                '\n ActionID   BJD   Airmass   FluxNorm   FluxNormErr   Flux   FluxErr  SkyBg   FWHM_SEP   FWHM_TL   FWHM_RGW'
                
    np.savetxt(outdir+'/'+obj_name+'_NGTS'+ns2+camstr+'_target_apers_bsproc_lc.dat',
               opt, header=headert,
               fmt='%i %.8f %.6f %.8f %.8f %.8f %.8f %.3f %.4f %.4f %.4f', delimiter=' ')
    
    np.savetxt(outdir+'/'+obj_name+'_NGTS'+ns2+camstr+'_master_apers_bsproc_lc.dat',
               opmc, header=headermc,
               fmt='%i %.8f %.6f %.8f %.8f %.8f %.8f %.3f %.4f %.4f %.4f', delimiter=' ')
    
    t0 = np.int(bjd[0])
    tbin_t, fbin_t, ebin_t = bspu.lb(bjd, flux_t, err_t, 5/1440.)
    tbin_mc, fbin_mc, ebin_mc = bspu.lb(bjd, flux_mc, err_mc, 5/1440.)
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    
    ax2.plot(bjd-t0, flux_mc, '.k', alpha=0.3, zorder=1, rasterized=True)
    ax2.errorbar(tbin_mc-t0, fbin_mc, yerr=ebin_mc, fmt='bo', ecolor='cyan', zorder=2)
    ax2.set_ylabel('Norm Flux - Comp Apers', fontsize=14)
    ax2.set_title(obj_name+'  Night(s): '+ns1+f'\nActions: {action_ids[ac_map]}\Apers: {ac_apers_min_master}')
    
    ax2.set_xlabel(f'Time (BJD - {t0})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(outdir+'/'+obj_name+'_NGTS'+ns2+camstr+'_bsproc_tmp_lc.png')
    plt.show()
    plt.close()