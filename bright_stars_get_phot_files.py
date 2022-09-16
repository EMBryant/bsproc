#!/usr/local/python3/bin/python
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
import pymysql
import pymysql.cursors

def ParseArgs():
    parser = ap.ArgumentParser()
    parser.add_argument('name', type=str, default=None,
                        help='Object name to produce light curve(s) for. REQUIRED')
    parser.add_argument('night', type=str, nargs='*', default=None,
                        help='Night(s) on which the observations were taken. Must be format YYYY-MM-DD. REQUIRED')
 #   parser.add_argument('--tic', type=int)
 #   parser.add_argument('--campaign', type=str, default=None)
 #   parser.add_argument('--actions', type=int, nargs='*', default=None)
    parser.add_argument('--output', type=str, default='./bsproc_outputs/',
                        help='Name of directory to save the test outputs in. Default is ./bsproc_outputs/')
    parser.add_argument('--aper', type=float, nargs='*', default=None,
                        help='Apertures to get light curves using. Full or half integers (eg. 2.0 3.0 4.5 6.0). OPTIONAL')
    parser.add_argument('--camera', type=str, default=None,
                        help='Camera to get light curves for. OPTIONAL')
    parser.add_argument('--bad_comp_tics', type=int, nargs='*', default=None,
                        help='TIC IDs of comparison stars to manually exclude. OPTIONAL')
    parser.add_argument('--bad_comp_inds', type=int, nargs='*', default=None,
                        help='BSProc IDs of comparison stars to manually exclude. OPTIONAL')
    parser.add_argument('--force_comp_stars', action='store_true',
                        help='This overrides the auto comparison star rejection. OPTIONAL. If used must also provide --comp_inds or --comp_tics')
    parser.add_argument('--comp_inds', type=int, nargs='*', default=None,
                        help='BSProc IDs of comparison stars to use. REQUIRED if --force_comp_stars used')
    parser.add_argument('--comp_tics', type=int, nargs='*', default=None,
                        help='TIC IDs of comparison stars to use. REQUIRED if --force_comp_stars used')
    parser.add_argument('--ti', type=float, default=None,
                        help='Time of ingress. Used for normalisation. OPTIONAL.')
    parser.add_argument('--te', type=float, default=None,
                        help='Time of egress. Used for normalisation. OPTIONAL')
    parser.add_argument('--dmb', type=float, default=0.5,
                        help='Comparison stars brighter than the target, with a Tmag difference greater than this are excluded. OPTIONAL. Default is 0.5mag')
    parser.add_argument('--dmf', type=float, default=3.5,
                        help='Comparison stars fainter than the target, with a Tmag difference greater than this are excluded. OPTIONAL. Default is 3.5mag')
    parser.add_argument('--dmag', type=float, default=0.5,
                        help='Node spacing for comparison star rejection spline. OPTIONAL. Default is 0.5mag.')
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

def find_comp_star_rms(comp_fluxes, airmass, comp_mags0):
    comp_star_rms = np.array([])
    Ncomps = comp_fluxes.shape[0]
    for i in range(Ncomps):
        comp_flux = np.copy(comp_fluxes[i])
        airmass_cs = np.polyfit(airmass, comp_flux, 1)
        airmass_mod = np.polyval(airmass_cs, airmass)
        comp_flux_corrected = comp_flux / airmass_mod
        comp_flux_norm = comp_flux_corrected / np.median(comp_flux_corrected)
        comp_star_rms = np.append(comp_star_rms, np.std(comp_flux_norm))
    return comp_star_rms

def find_bad_comp_stars(comp_fluxes, airmass, comp_mags0,
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

if __name__ == "__main__":
    args = ParseArgs()
 #   tic = args.tic
    name = args.name
    if name is None:
        raise ValueError('I need an object name. (--name <object_name>) ')
  #  if args.testing is not None:
  #      bsdir = '/ngts/scratch/brightstars/PAOPhot2/'+args.testing+'/'+name+'/'
  #      if not os.path.exists('/ngts/scratch/brightstars/PAOPhot2/'+args.testing):
  #          os.system('mkdir /ngts/scratch/brightstars/PAOPhot2/'+args.testing)
    bsdir = args.output
    if not os.path.exists(bsdir):
        os.system('mkdir '+bsdir)
    objdir = bsdir+'/'+name+'/'
    if not os.path.exists(objdir):
        os.system('mkdir '+objdir)
    outdir_main = objdir+'analyse_outputs/'
    root_dir = '/ngts/scratch/PAOPhot2/'
    if not os.path.exists(outdir_main):
        os.system('mkdir '+outdir_main)
        os.system('mkdir '+outdir_main+'master_logs/')
    if not os.path.exists(objdir+'action_summaries/'):
        os.system('mkdir '+objdir+'action_summaries/')
    nights = args.night
    if nights is None:
        raise ValueError('I need night(s) for the observations. (--night <YYYY-MM-DD>)')
    for night in nights:
        ymd = night.split('-')
        if not len(ymd) == 3:
            raise ValueError('Night in wrong format. Must be YYYY-MM-DD')
        y, m, d = ymd[0], ymd[1], ymd[2]
        if not len(y) == 4:
            raise ValueError('Night in wrong format. Must be YYYY-MM-DD')
        if not len(m) == 2:
            raise ValueError('Night in wrong format. Must be YYYY-MM-DD')
        if not len(d) == 2:
            raise ValueError('Night in wrong format. Must be YYYY-MM-DD')
        if int(m) > 12.5:
            raise ValueError('Night in wrong format. Must be YYYY-MM-DD')
        outdir = outdir_main+night+'/'
        if not os.path.exists(outdir):
            os.system('mkdir '+outdir)
            os.system('mkdir '+outdir+'comp_star_check_plots/')
            os.system('mkdir '+outdir+'ind_tel_lcs/')
            os.system('mkdir '+outdir+'data_files/')
            os.system('mkdir '+outdir+'logs/')
            os.system('mkdir '+outdir+'phot_files/')
     
    logger_main = custom_logger(outdir_main+'master_logs/'+name+'_bsproc_main.log')
 #   actions = args.actions
    actions = np.array([], dtype=int)
    night_store = []
    #Find campaign name
    if name[:3] == 'TOI':
        toiid = int(name.split('-')[-1])
        camp_id = f'TOI-{toiid:05d}'
    elif name[:3] == 'TIC':
        ticid = str(name.split('-')[-1])
        camp_id = 'TIC-'+ticid
    for night in nights:     
        connection = pymysql.connect(host='ngtsdb', db='ngts_ops', user='pipe')
        if args.camera is None:
            qry = "select action_id,num_images,status from action_summary_log where night='"+night+"' and campaign like '%"+camp_id+"%'"
        else:
            qry = "select action_id,num_images,status from action_summary_log where night='"+night+"' and campaign like '%"+camp_id+"%' and camera_id="+args.camera
        with connection.cursor() as cur:
            cur.execute(qry)
            res = cur.fetchall()
        if len(res) < 0.5:
            if name[:3] == 'TOI':
                camp_id2 = f'TOI-{toiid}'
                if args.camera is None:
                    qry = "select action_id,num_images,status from action_summary_log where night='"+night+"' and campaign like '%"+camp_id2+"%'"
                else:
                    qry = "select action_id,num_images,status from action_summary_log where night='"+night+"' and campaign like '%"+camp_id2+"%' and camera_id="+args.camera
                with connection.cursor() as cur:
                    cur.execute(qry)
                    res = cur.fetchall()
                if len(res) < 0.5:
                    logger_main.info('I found no actions for '+name+' for night '+night)
                    logger_main.info('Checking for other actions for night '+night)
                    if args.camera is None:
                        qry2 = "select campaign,action_id,num_images,status from action_summary_log where night='"+night+"' and action_type='observeField'"
                        qry3 = "select night,action_id,num_images,status from action_summary_log where campaign like '%"+camp_id+"%'"
                    else:
                        qry2 = "select campaign,action_id,num_images,status from action_summary_log where night='"+night+"' and action_type='observeField' and camera_id="+args.camera
                        qry3 = "select night,action_id,num_images,status from action_summary_log where campaign like '%"+camp_id+"%' and camera_id="+args.camera
                    with connection.cursor() as cur:
                        cur.execute(qry2)
                        res2 = cur.fetchall()
                    if len(res2) < 0.5:
                        logger_main.info('Found no actions for other objects for night '+night)
                    else:
                        logger_main.info('Found actions for other objects from night '+night+':')
                        for r in res2:
                            logger_main.info(f'{r}')
                    logger_main.info('Checking for other actions for '+name+' on different nights')
                    with connection.cursor() as cur:
                        cur.execute(qry3)
                        res3 = cur.fetchall()
                    if len(res3) < 0.5:
                        logger_main.info('Found no actions for '+name+' on other nights')
                    else:
                        logger_main.info('Found actions for '+name+' on other nights:')
                        for r in res3:
                            logger_main.info(f'{r}')
                    if len(res2) < 0.5 and len(res3) < 0.5:
                        raise ValueError('Found no actions for '+name+' or on '+night+'. Check your inputs.')
                        
                    elif len(res2) > 0.5 and len(res3) < 0.5:
                        raise ValueError('Found no actions at all for '+name+' but found actions for other objects on '+night+'. Check your inputs.')
                    
                    elif len(res2) < 0.5 and len(res3) > 0.5:
                        raise ValueError('Found actions for '+name+' on other nights but none on '+night+'. Check your inputs.')
                    else:
                        raise ValueError('Found actions for '+name+' on other nights and actions for other objects on '+night+'. Check your inputs.')

        logger_main.info(f'Found {len(res)} actions for '+name+' for night '+night)
        for r in res:
            logger_main.info(f'{r}')
        logger_main.info('Checking actions...')
        for r in res:
            action = int(r[0])
            num_ims= r[1]
            status = r[2]
            if status == 'completed':
                actions = np.append(actions, action)
                night_store.append(night)
            elif status == 'aborted' and num_ims > 100:
                actions = np.append(actions, action)
                night_store.append(night)
    
    logger_main.info(f'Found total of {len(actions)} good actions for {len(nights)} nights ({nights}).')
    
    tic = None
    if name[:3] == 'TIC':
        tic = int(name.split('-')[-1])
        logger_main.info(f'Object is TIC-{tic}')
    elif name[:3] == 'TOI':
        logger_main.info('Finding TIC ID from TOI ID...')
        db_toiid = str(int(name.split('-')[-1]))+'.01'
        connection = pymysql.connect(host='ngtsdb', db='tess', user='pipe')
        qry = "select tic_id from tois where toi_id = "+db_toiid+";"
        with connection.cursor() as cur:
            cur.execute(qry)
            res = cur.fetchone()
            if len(res) < 0.5:
                raise ValueError('Couldn\'t find a TIC ID for '+name+'.')
            else:
                tic = int(res[0])
        logger_main.info(f'Object is TIC-{tic}')
                
    star_cat = pyfits.getdata(root_dir+f'target_catalogues/TIC-{tic}.fits')
    star_df = pd.DataFrame(star_cat).set_index('tic_id')
    star_df.to_csv(outdir+f'phot_files/TIC-{tic}_paophot2_catalogue.csv', index_label='tic_id')
    star_mask= pyfits.getdata(root_dir+f'target_catalogues/TIC-{tic}_mask.fits')
    tic_ids = np.array(star_cat.tic_id)
    idx = np.array(star_cat.PAOPHOT_IDX)
    tmags_full = np.array(star_cat.Tmag)
    tmag_target = tmags_full[0]
    tmags_comps = tmags_full[idx==2]
    
    if args.force_comp_stars:
        logger_main.info(f'Nights {nights}: Using user defined comparison stars.')
        if args.comp_inds is not None:
            comp_mask = np.array([True if i in args.comp_inds else False 
                                  for i in range(100)])
            logger_main.info(f'Nights {nights}: Using {np.sum(comp_mask)} user defined comparison stars.')
            logger_main.info(f'Nights {nights}: Using these comparison stars (inds): {args.comp_inds}')
        elif args.comp_tics is not None:
            comp_mask = np.array([True if t in args.comp_tics else False 
                                  for t in tic_ids[idx==2]])
            logger_main.info(f'Nights {nights}: Using {np.sum(comp_mask)} user defined comparison stars.')
            logger_main.info(f'Nights {nights}: Using these comparison stars (tics): {args.comp_tics}')
        else:
            raise ValueError('If user defined comp stars (--force_comp_stars) I need comparison star IDs (--comp_inds) or TIC IDs (--comp_tics).')        
    
    else:
        comp_mask= np.array([m[0] for m in star_mask])
        logger_main.info(f'Nights {nights}: User rejected comp stars (tics): {args.bad_comp_tics}')
        logger_main.info(f'Nights {nights}: User rejected comp stars (inds): {args.bad_comp_inds}')
        if args.bad_comp_tics is not None:
            comp_mask_2 = [False if t in args.bad_comp_tics else True
                           for t in tic_ids[idx==2]]
            comp_mask &= comp_mask_2
        elif args.bad_comp_inds is not None:
            comp_mask_2 = [False if i in args.bad_comp_inds else True
                           for i in range(100)]
            comp_mask &= comp_mask_2
        logger_main.info(f'Nights {nights}: Checking comp star brightness')
        bad_mag_inds = []
        bad_mag_tics = []
        for i, tici in zip(range(len(tmags_comps)), tic_ids[idx==2]):
            tmag = tmags_comps[i]
            if tmag < tmag_target-args.dmb or tmag > tmag_target + args.dmf:
                comp_mask[i] = False
                bad_mag_inds.append(i)
                bad_mag_tics.append(tici)
        logger_main.info(f'Nights {nights}: Comps rejected by brightness (inds): {bad_mag_inds}')
        logger_main.info(f'Nights {nights}: Comps rejected by brightness (tics): {bad_mag_tics}')
    
    r_ap = args.aper
    if r_ap is None:
        if tmag_target >= 10.:
            r_ap = [2.0, 2.5, 3.0, 3.5, 4.0]
        elif 9.0 <= tmag_target < 10.0:
            r_ap = [3.0, 3.5, 4.0, 4.5, 5.0]
        elif tmag_target < 9.0:
            r_ap = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    ap_ids = [int(2*r - 4) for r in r_ap]


    ac_apers_min_target = np.array([])
    ac_apers_min_master = np.array([])
    missing_actions = np.array([])
    for ac, ns in zip(actions, night_store):
        logger = custom_logger(outdir+'logs/'+name+'_night'+ns+f'_action{ac}.log')
        print(' ')
        print(' ')
    #    if not os.path.exists(root_dir+f'action_summaries/{ac}_TIC-{tic}.png'):
    #        logger.info(f'Can\'t find action summary for Action {ac}')
    #        logger_main.info(f'No action summary for Action {ac}')
    #        logger_main.info(f'Skipping Action {ac}.')
    #        missing_actions = np.append(missing_actions, ac)
    #        continue
        logger.info('Night '+ns+f': Running for Action{ac}...')
        os.system('cp '+root_dir+f'action_summaries/{ac}_TIC-{tic}.png '+objdir+'action_summaries/')
        
        phot_file_root = root_dir+f'photometry/action{ac}/ACITON_{ac}_'
        try:
            bjds = pyfits.getdata(phot_file_root+'BJD.fits.bz2')
        except:
            bjds = pyfits.getdata(phot_file_root+'BJD.fits')
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
        
        target_fluxes_full = np.copy(fluxes[0])
        target_bjd = np.copy(bjds[0])
        target_skys_full = np.copy(skybgs[0])
        sep_centre_fwhm = np.copy(psfs[:, 1])
        tl_centre_fwhm = np.mean(psfs[:, [14, 15]], axis=1)
        rgw_fwhm = np.copy(psfs[:, -3])
        
        try:
            airmass = pyfits.getdata(phot_file_root+'AIRMASS.fits.bz2')
        except:
            airmass = pyfits.getdata(phot_file_root+'AIRMASS.fits')
        scint_noise = estimate_scintillation_noise(airmass, 10.)
        phot_csv_file = outdir+f'phot_files/action{ac}_bsproc_phot.csv'
        df = pd.DataFrame(np.column_stack((target_bjd, airmass,
                                               sep_centre_fwhm,
                                               tl_centre_fwhm,
                                               rgw_fwhm)),
                          columns=['BJD','Airmass','FWHM_SEP',
                                   'FWHM_TL','FWHM_RGW'])
        
        comp_fluxes_full = np.copy(fluxes[idx==2][comp_mask])
        comp_skys_full = np.copy(skybgs[idx==2][comp_mask])
        comp_bjds = np.copy(bjds[idx==2][comp_mask])
        comp_tics_full = np.copy(tic_ids[idx==2][comp_mask])
        comp_tmags_full = np.copy(tmags_comps[comp_mask])
        Ncomps_full = len(comp_mask)
        comp_inds_full = np.linspace(0, Ncomps_full-1, Ncomps_full, dtype=int)[comp_mask]
        
        comp_fluxes_bad0 = np.copy(fluxes[idx==2][~comp_mask])
        comp_bjds_bad0 = np.copy(bjds[idx==2][~comp_mask])
        comp_tics_bad0 = np.copy(tic_ids[idx==2][~comp_mask])
        Ncomps_bad0 = len(comp_tics_bad0)
        comp_inds_bad0 = np.linspace(0, Ncomps_full-1, Ncomps_full, dtype=int)[~comp_mask]
    
        ap_target_rms = np.array([])
        ap_comp_rms = np.array([])
        
        for r, idr in zip(r_ap, ap_ids):
            print(' ')
            logger.info(f'Running for: Action {ac};  Aper - {r:.1f} pix')
            target_flux = np.copy(target_fluxes_full[:, idr])
            target_sky = np.copy(target_skys_full[:, idr])
            df.loc[:, f'TargetCountsA{r}'] = target_flux
            df.loc[:, f'SkyBgA{r}'] = target_sky
            comp_fluxes = np.vstack(([np.copy(cf[:, idr])
                                      for cf in comp_fluxes_full]))
            comp_skys = np.vstack(([np.copy(cf[:, idr])
                                    for cf in comp_skys_full]))
            Ncomps = comp_fluxes.shape[0]
            
            comp_fluxes_bad = np.vstack(([np.copy(cfb[:, idr])
                                          for cfb in comp_fluxes_bad0]))
            
            if args.force_comp_stars:
                logger.info('User defined comparisons. Skipping bad comp rejection.')
                comp_fluxes_good = np.copy(comp_fluxes)
                comp_skys_good = np.copy(comp_skys)
                comp_star_rms = find_comp_star_rms(comp_fluxes, airmass, comp_tmags_full)
                
            
            else:
                logger.info('Finding bad comp stars...')
                comp_star_mask_r, comp_star_rms, Niter = find_bad_comp_stars(
                    comp_fluxes, airmass, comp_tmags_full, dmag=args.dmag)
                comp_fluxes_good = np.copy(comp_fluxes[comp_star_mask_r])
                comp_skys_good = np.copy(comp_skys[comp_star_mask_r])
                comp_tics_good = np.copy(comp_tics_full[comp_star_mask_r])
            
            target_err = calc_noise(r, 10, 1.0, scint_noise, target_flux+target_sky)
            df.loc[:, f'TargetErrA{r}'] = target_err
    
            comp_fluxes_good
            comp_errs = np.vstack(([calc_noise(r, 10, 1.0, scint_noise, cfi+csi)
                                    for cfi, csi in zip(comp_fluxes_good, comp_skys_good)]))
            for i in range(len(comp_tics_good)):
                tic = comp_tics_good[i]
                df.loc[:, f'TIC-{tic}_RawFluxA{r}'] = comp_fluxes_good[i]
                df.loc[:, f'TIC-{tic}_RawFluxErrA{r}'] = comp_errs[i]
        
        
        df.to_csv(phot_csv_file,
                      index_label='NExposure')
            
