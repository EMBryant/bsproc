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
    parser.add_argument('--ignore_bjd', type=float, nargs=2, default=[0.2, 0.21],
                        help='BJD Timespan (fractional day) to ignore within data e.g. due to clouds')
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
    root_dir = '/ngts/PAOPhot2/'
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
    elif name == 'HIP-41378':
        camp_id = 'HIP41378'
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
    elif name == 'HIP-41378':
        tic = 366443426
        logger_main.info(f'Object is TIC-{tic}')
                
    star_cat = pyfits.getdata(root_dir+f'target_catalogues/TIC-{tic}.fits')
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
        if not os.path.exists(root_dir+f'photometry/action{ac}/ACITON_{ac}_BJD.fits.bz2'):
            logger.info(f'Can\'t find photometry for Action {ac}')
            logger_main.info(f'No photometry for Action {ac}')
            logger_main.info(f'Skipping Action {ac}.')
            missing_actions = np.append(missing_actions, ac)
            continue
        logger.info('Night '+ns+f': Running for Action{ac}...')
#        os.system('cp '+root_dir+f'action_summaries/{ac}_TIC-{tic}.png '+objdir+'action_summaries/')
        
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
        
        target_bjd0 = np.copy(bjds[0])
        bjd_int = int(target_bjd0[0])
        ignore = args.ignore
        ignore1, ignore2 = ignore[0]+bjd_int, ignore[1]+bjd_int
        bjd_keep = (target_bjd0 <= ignore1) | (target_bjd0 >= ignore2)
        target_bjd = np.copy(bjds[0])[bjd_keep]
        target_fluxes_full = np.copy(fluxes[0])[bjd_keep]
        target_skys_full = np.copy(skybgs[0])[bjd_keep]
        sep_centre_fwhm = np.copy(psfs[:, 1])[bjd_keep]
        tl_centre_fwhm = np.mean(psfs[:, [14, 15]], axis=1)[bjd_keep]
        rgw_fwhm = np.copy(psfs[:, -3])[bjd_keep]
        
        try:
            airmass = pyfits.getdata(phot_file_root+'AIRMASS.fits.bz2')[bjd_keep]
        except:
            airmass = pyfits.getdata(phot_file_root+'AIRMASS.fits')[bjd_keep]
        scint_noise = estimate_scintillation_noise(airmass, 10.)
        phot_csv_file = outdir+f'data_files/action{ac}_bsproc_dat.csv'
        if os.path.exists(phot_csv_file):
            logger.info('Phot CSV file already exists: '+phot_csv_file)
            logger.info('Adding "new" data to existing file...')
            df = pd.read_csv(phot_csv_file, index_col='NExposure')
        else:
            logger.info('No existing phot csv.')
            logger.info('Creating new phot file: '+phot_csv_file)
            df = pd.DataFrame(np.column_stack((target_bjd, airmass,
                                               sep_centre_fwhm,
                                               tl_centre_fwhm,
                                               rgw_fwhm)),
                          columns=['BJD','Airmass','FWHM_SEP',
                                   'FWHM_TL','FWHM_RGW'])
        
        comp_fluxes_full = np.copy(fluxes[idx==2][comp_mask][bjd_keep])
        comp_skys_full = np.copy(skybgs[idx==2][comp_mask][bjd_keep])
        comp_bjds = np.copy(bjds[idx==2][comp_mask][bjd_keep])
        comp_tics_full = np.copy(tic_ids[idx==2][comp_mask])
        comp_tmags_full = np.copy(tmags_comps[comp_mask])
        Ncomps_full = len(comp_mask)
        comp_inds_full = np.linspace(0, Ncomps_full-1, Ncomps_full, dtype=int)[comp_mask]
        
        comp_fluxes_bad0 = np.copy(fluxes[idx==2][~comp_mask][bjd_keep])
        comp_bjds_bad0 = np.copy(bjds[idx==2][~comp_mask][bjd_keep])
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
            df.loc[:, f'RawA{r}'] = target_flux
            df.loc[:, f'SkyBgA{r}'] = target_sky
            comp_fluxes = np.vstack(([np.copy(cf[:, idr])
                                      for cf in comp_fluxes_full]))
            comp_skys = np.vstack(([np.copy(cf[:, idr])
                                    for cf in comp_skys_full]))
            Ncomps = comp_fluxes.shape[0]
            
            comp_fluxes_bad = np.vstack(([np.copy(cfb[:, idr])
                                          for cfb in comp_fluxes_bad0]))
            #print('Starting Plotting')
            fig, axes = plt.subplots(int((Ncomps_bad0+1)/2), 2, sharex=True,
                                     figsize=(12, 3*int((Ncomps_bad0+1)/2)))
            #print('Made the figure instance')
            axes = axes.reshape(-1)
            comp_flux0 = np.copy(comp_fluxes[0])
            for i, j in zip(range(Ncomps_bad0), comp_inds_bad0):
                ax=axes[i]
                comp_tic = comp_tics_bad0[i]
                comp_flux = np.copy(comp_fluxes_bad[i])
                comp_flux_corrected = comp_flux / comp_flux0
            #    comp_flux_norm = comp_flux_corrected / np.median(comp_flux_corrected)
                ax.plot(comp_bjds_bad0[i], comp_flux_corrected, '.k', alpha=0.5,
                        label=f'{j}:  TIC-{comp_tic}')
                leg = ax.legend(loc='upper center', frameon=False)
                plt.setp(leg.get_texts(), color='k')
            fig.subplots_adjust(hspace=0., wspace=0.)
            #print('Finished Plotting')
            plt.savefig(outdir+f'comp_star_check_plots/action{ac}_A{r}_global_rejected_comp_stars_comp0dt_lcs.png')
            plt.close()
            if args.force_comp_stars:
                logger.info('User defined comparisons. Skipping bad comp rejection.')
                comp_fluxes_good = np.copy(comp_fluxes)
                comp_skys_good = np.copy(comp_skys)
                comp_star_rms = find_comp_star_rms(comp_fluxes, airmass, comp_tmags_full)
                
                plt.figure()
                plt.semilogy(comp_tmags_full, comp_star_rms * 100,
                             '.k', zorder=2)
                for i, j in zip(range(Ncomps), comp_inds_full):
                    plt.gca().annotate(int(j),
                                       (comp_tmags_full[i]+0.01, 100 * comp_star_rms[i]+0.01),
                                       color='black')
                plt.xlabel('Tmag')
                plt.ylabel('RMS (% per exposure)')
                plt.title(name+'   Night '+ns+f'   Action {ac}   Aper {r} pix')
                plt.savefig(outdir+f'comp_star_check_plots/action{ac}_A{r}_mag_vs_rms.png')
                plt.close()
                
                fig, axes = plt.subplots(int((Ncomps+1)/2), 2, sharex=True,
                                         figsize=(12, 3*int((Ncomps+1)/2)))
                axes = axes.reshape(-1)
                comp_flux0 = np.copy(comp_fluxes[0])
                for i, j in zip(range(Ncomps), comp_inds_full):
                    ax=axes[i]
                    comp_tic = comp_tics_full[i]
                    comp_flux = np.copy(comp_fluxes[i])
                    comp_flux_corrected = comp_flux / comp_flux0
                #    comp_flux_norm = comp_flux_corrected / np.median(comp_flux_corrected)
                    ax.plot(comp_bjds[i], comp_flux_corrected, '.k', alpha=0.5,
                            label=f'{j}:  TIC-{comp_tic}')
                    leg = ax.legend(loc='upper center', frameon=False)
                    plt.setp(leg.get_texts(), color='k')
                fig.subplots_adjust(hspace=0., wspace=0.)
                plt.savefig(outdir+f'comp_star_check_plots/action{ac}_A{r}_comp_star_comp0dt_lcs.png')
                plt.close()
            
            else:
                logger.info('Finding bad comp stars...')
                comp_star_mask_r, comp_star_rms, Niter = find_bad_comp_stars(
                    comp_fluxes, airmass, comp_tmags_full, dmag=args.dmag)
                comp_fluxes_good = np.copy(comp_fluxes[comp_star_mask_r])
                comp_skys_good = np.copy(comp_skys[comp_star_mask_r])
                
                logger.info(f'Searched through {Niter:.0f} iterations.')
                logger.info(f'Number of bad_comp_stars (Action{ac}; A{r:.1f}): {np.sum(~comp_star_mask_r)}')
                logger.info(f'Bad_comp_stars (Action{ac}; A{r:.1f}; inds): {comp_inds_full[~comp_star_mask_r]}')
                logger.info(f'Bad_comp_stars (Action{ac}; A{r:.1f}; tics): {comp_tics_full[~comp_star_mask_r]}')
                
                logger.info(f'Number of good comp stars (Action{ac}; A{r:.1f}): {np.sum(comp_star_mask_r)}')
                logger.info(f'Good comp stars (Action{ac}; A{r:.1f}; inds): {comp_inds_full[comp_star_mask_r]}')
                logger.info(f'Good comp stars (Action{ac}; A{r:.1f}; tics): {comp_tics_full[comp_star_mask_r]}')
                
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
                plt.title(name+'   Night '+ns+f'   Action {ac}   Aper {r} pix')
                plt.savefig(outdir+f'comp_star_check_plots/action{ac}_A{r}_mag_vs_rms.png')
                plt.close()
                
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
                plt.savefig(outdir+f'comp_star_check_plots/action{ac}_A{r}_comp_star_comp0dt_lcs.png')
                plt.close()
    
            master_comp = np.sum(comp_fluxes_good, axis=0)
            cs_mc_am = np.polyfit(airmass, master_comp, 1)
            airmass_mc_mod = np.polyval(cs_mc_am, airmass)
            ap_comp_rms = np.append(ap_comp_rms, np.std(master_comp / airmass_mc_mod))
            logger.info(f'Action{ac}; A{r} - master comp rms: {np.std(master_comp / airmass_mc_mod)*100:.3f} %')
            comp_errs = np.vstack(([calc_noise(r, 10, 1.0, scint_noise, cfi+csi)
                                    for cfi, csi in zip(comp_fluxes_good, comp_skys_good)]))
            master_comp_err = np.sqrt(np.sum(comp_errs**2, axis=0))
            lightcurve = target_flux / master_comp
            target_err = calc_noise(r, 10, 1.0, scint_noise, target_flux+target_sky)
            err_factor = np.sqrt((target_err/target_flux)**2 + (master_comp_err/master_comp)**2)
            lightcurve_err = lightcurve * err_factor
            
            # Handle not needing ti and te as arguments
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
            logger.info(f'Action{ac}; A{r} - target rms: {np.std(norm_flux[oot])*100:.3f} %')
            
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
            
            plt.title(name+'   Night '+ns+f'   Action {ac}   Aper {r} pix')
            
            plt.savefig(outdir+f'ind_tel_lcs/action{ac}_A{r}_lc.png')
            plt.close()
            
            df.to_csv(phot_csv_file,
                      index_label='NExposure')
            
        plt.figure()
        plt.plot(np.array(r_ap), ap_target_rms*100, 'ko', label=name)
        plt.ylabel('LC Precision (%)')
        plt.xlabel('Aperture Radius (pixels)')
        plt.title(name+'   Night '+ns+f'   Action {ac}   Aper {r} pix')
        ax2 = plt.twinx()
        ax2.plot(np.array(r_ap), ap_comp_rms * 100, 'ro', label='MasterComp')
        ax2.set_ylabel('MasterComp Precision (%)', color='red')
        ax2.tick_params(axis='y', colors='red')
        plt.savefig(outdir+f'ind_tel_lcs/action{ac}_aperture_precisions.png')
        plt.close()
        
        ac_apers_min_target = np.append(ac_apers_min_target, np.array(r_ap)[ap_target_rms.argmin()])
        ac_apers_min_master = np.append(ac_apers_min_master, np.array(r_ap)[ap_comp_rms.argmin()])
    
    action_store = np.array([], dtype=int)
    airmass_store = np.array([])
    fwhm_sep, fwhm_tl, fwhm_rgw = np.array([]), np.array([]), np.array([])
    bjd, flux_t, err_t = np.array([]), np.array([]), np.array([])
    flux_mc, err_mc = np.array([]), np.array([])
    flux0_t, err0_t, skybg_t = np.array([]), np.array([]), np.array([])
    flux0_mc, err0_mc, skybg_mc = np.array([]), np.array([]), np.array([])
    ac_map = np.array([True if not ac in missing_actions else False for ac in actions])
    for ac, ns, rt, rc in zip(actions[ac_map], np.array(night_store)[ac_map], ac_apers_min_target, ac_apers_min_master):
        logger.info(f'Action {ac} - "Best" apers -  Target: {rt} pix; Comp: {rc} pix')
        dat = pd.read_csv(outdir+f'data_files/action{ac}_bsproc_dat.csv',
                          index_col='NExposure')
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
    for n in nights:
        ns1 += ' '+n
        ns2 += '_'+n

    if len(ns2) > 55:
        ns2 = '{} to {}'.format(ns2[:10],ns2[-10])
    
    if args.camera is None:
        camstr = ''
    else:
        camstr= '_cam'+args.camera
    headert =  ' Object: '+args.name+f'  (TIC-{tic})       '+camstr + \
               '\n Night(s): '+ns1 + \
              f'\n Actions: {actions}' + \
              f'\n Aperture Radii: {ac_apers_min_target} pixels' + \
               '\n Note these apertures minimise the target flux RMS' + \
               '\n ActionID   BJD   Airmass   FluxNorm   FluxNormErr   Flux   FluxErr  SkyBg   FWHM_SEP   FWHM_TL   FWHM_RGW'
    
    headermc =  ' Object: '+args.name+f'  (TIC-{tic})       '+camstr + \
                '\n Night(s): '+ns1 + \
               f'\n Actions: {actions}' + \
               f'\n Aperture Radii: {ac_apers_min_master} pixels' + \
                '\n Note these apertures minimise the master comparison flux RMS' + \
                '\n ActionID   BJD   Airmass   FluxNorm   FluxNormErr   Flux   FluxErr  SkyBg   FWHM_SEP   FWHM_TL   FWHM_RGW'
                
    np.savetxt(outdir+'/'+name+'_NGTS'+ns2+camstr+'_target_apers_bsproc_lc.dat',
               opt, header=headert,
               fmt='%i %.8f %.6f %.8f %.8f %.8f %.8f %.3f %.4f %.4f %.4f', delimiter=' ')
    
    np.savetxt(outdir+'/'+name+'_NGTS'+ns2+camstr+'_master_apers_bsproc_lc.dat',
               opmc, header=headermc,
               fmt='%i %.8f %.6f %.8f %.8f %.8f %.8f %.3f %.4f %.4f %.4f', delimiter=' ')
    
    tbin_t, fbin_t, ebin_t = lb(bjd, flux_t, err_t, 5/1440.)
    tbin_mc, fbin_mc, ebin_mc = lb(bjd, flux_mc, err_mc, 5/1440.)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    ax1.plot(bjd-t0, flux_t, '.k', alpha=0.3, zorder=1)
    ax1.errorbar(tbin_t-t0, fbin_t, yerr=ebin_t, fmt='bo', ecolor='cyan', zorder=2)
    ax1.set_ylabel('Norm Flux - Target Apers', fontsize=14)
    ax1.set_title(name+'  Night(s): '+ns1+f'\nActions: {actions[ac_map]}\nApers: {ac_apers_min_target}')
    
    ax2.plot(bjd-t0, flux_mc, '.k', alpha=0.3, zorder=1)
    ax2.errorbar(tbin_mc-t0, fbin_mc, yerr=ebin_mc, fmt='bo', ecolor='cyan', zorder=2)
    ax2.set_ylabel('Norm Flux - Comp Apers', fontsize=14)
    ax2.set_title(f'Apers: {ac_apers_min_master}')
    
    ax2.set_xlabel(f'Time (BJD - {t0})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(outdir+'/'+name+'_NGTS'+ns2+camstr+'_bsproc_tmp_lc.png')
    plt.show()
   # plt.show(block=False)
   # save = input('Save over autosaved plot? [y/n] :  ')
   # if save == 'y':
   #     plt.savefig(outdir+'/'+name+'_NGTS'+ns2+camstr+'_bsproc_tmp_lc.png')
    plt.close()
