#!/usr/local/python3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 09:36:58 2025

Contains the functions relating to querying databases and target catalogues
for BSP pipeline runs

@author: Edward M. Bryant
"""
import pymysql
import pymysql.cursors
import astropy.io.fits as pyfits
import numpy as np
import os

def get_target_catalogue_from_database(tic_id):
    """
    Function to extract the ngpipe photometry target catalogue from the SQL databases
    NOTE - this function is not used currently as ngpipe produces the target catalogues
             as fits files for each action
             
    Parameters
    ----------
    tic_id : int
        TIC ID of the target

    Returns
    -------
    tic_ids : array; int
        TIC IDs for each star in the ngpipe target catalogue
    idx : array; int
        Indices from ngpipe differentiating the target star, the nearby stars, and the comparison stars.
    mask : array; boolean
        Comparison star rejection mask from ngpipe.
    tmags : array; float
        TESS mag values for all stars in the ngpipe target catalogue
    
    """
    # This first query checks whether there is information within the ngpipe target
    #   catalogue for the target star
    # If the information does not exist we exit and BSP knows to fail gracefully
    qry = 'SELECT target_tic_id FROM ngts_wcs.ngts_target_photometry_catalogue WHERE target_tic_id={:};'.format(tic_id)
    connection = pymysql.connect(host='ngtsdb', user='pipe')
    with connection.cursor() as cur:
        cur.execute(qry)
    output = cur.fetchall()
    if len(output)==0: 
        return None, None, None, None
    
    # This second query now pulls the relevant info for the target, nearby, and
    #    comparison stars from the ngpipe target catalogue and TIC SQL databases
    qry = 'SELECT a.tic_id, a.obj_type, a.mask, b.Tmag, a.metric FROM ngts_wcs.ngts_target_photometry_catalogue a LEFT JOIN catalogues.tic8 b ON a.tic_id=b.tic_id WHERE a.target_tic_id={:} ORDER BY a.obj_type,a.metric;'.format(tic_id)
    with connection.cursor() as cur:
        cur.execute(qry)
    output = cur.fetchall()
    tic_ids = np.array([i[0] for i in output])
    idx     = np.array([i[1] for i in output])
    mask    = np.array([i[2] for i in output])
    tmags   = np.array([i[3] for i in output])
    return tic_ids, idx, mask, tmags

def check_output_directories(bs_root_dir, obs_nights, obj_name):
    """
    Function to ensure the correct output directories exist
    If the relevant directories do not exist then they will be generated
    
    Parameters
    ----------
    bs_root_dir : str
        Path to the root directory for the BSP outputs for all objects
    obs_nights : list; str
        NGTS observation nights considered in the BSP pipeline run
    obj_name : str
        Name of the target star

    Returns
    -------
    outdir_main : str
        Path to the specific root directory for BSP pipeline outputs for this object
    ind_night_outdirs : list; str
        Paths to directories for the BSP outputs for each observation night considered
            in this BSP pipeline run
    
    Raises
    ------
    ValueError
        If an observation night(s) is not provided to the command line or is provided 
            in an incorrect format
        
    """
    # Check for whether BSP output root directory exists
    # If it does not, create the directory
    if not os.path.exists(bs_root_dir):
        os.system('mkdir '+bs_root_dir)
    # Check whether the BSP output directory exists for the object
    # If not, create the directory with correct structure
    objdir = bs_root_dir+'/'+obj_name+'/'
    if not os.path.exists(objdir):
        os.system('mkdir '+objdir)
    outdir_main = objdir+'analyse_outputs/'
    
    if not os.path.exists(outdir_main):
        os.system('mkdir '+outdir_main)
        os.system('mkdir '+outdir_main+'master_logs/')
    if not os.path.exists(objdir+'action_summaries/'):
        os.system('mkdir '+objdir+'action_summaries/')
    if obs_nights is None:
        raise ValueError('I need night(s) for the observations. (--night <YYYY-MM-DD>)')
    
    # Create a sub-directory within the BSP outputs directory for each observation night
    ind_night_outdirs = []
    for night in obs_nights:
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
        ind_night_outdirs.append(outdir)
            
    return outdir_main, ind_night_outdirs

def find_action_ids(logger_main, cmd_args, obj_name, obs_nights, ind_night_outdirs):
    """
    Function to identify the relevant NGTS action IDs from the SQL databases
        for the object and observation nights we are running the BSP pipeline for

    Parameters
    ----------
    logger_main : logger
        Logger governing the log file for the overall BSP run.
    cmd_args : 
        Store for the command line arguments.
    obj_name : str
        Name of the target.
    obs_nights : list; str
        NGTS observation nights.
    ind_night_outdirs : list; str
        Paths to BSP output directories corresponding to each observation night.

    Returns
    -------
    logger_main : logger
        Logger governing the log file for the overall BSP run.
    actions : array; int
        NGTS action IDs
    night_store : list; str
        NGTS observation nights for each NGTS action
    ind_night_outdir_store : list; str
        BSP output paths to use for each NGTS action
    
    Raises
    ------
    ValueError
        If no campaign name can be determine for the given object name
        If no actions for the given object can be found on the given night

    """
    
    actions = np.array([], dtype=int)
    night_store = []
    ind_night_outdir_store = []
    # First we determine the NGTS campaign name to use
    if obj_name[:3] == 'TOI':
        toiid = int(obj_name.split('-')[-1])
        camp_id = f'TOI-{toiid:05d}'
    elif obj_name[:3] == 'TIC':
        ticid = str(obj_name.split('-')[-1])
        camp_id = 'TIC-'+ticid
### If you have a non TIC or TOI campaign name add some code here: ####
### This will allow bsproc to find the correct actions
### Code should be of the format:
### elif obj_name == <Name of your object>:  (use this name on the command line)
###     camp_id = <Campign name prefix for your object>
### See examples for HIP-41378 and WASP-47
    elif obj_name == 'HIP-41378':
        camp_id = 'HIP41378'
    elif obj_name == 'WASP-47' or obj_name == 'WASP47':
        camp_id = 'WASP47'
    else:
        raise ValueError("No campaign name can be determined for Object Name :  "+obj_name+". Please check your inputs.")
    for night, ind_night_outdir in zip(obs_nights, ind_night_outdirs):  
        # SQL queries to identify actions associated with the campaign name for 
        #   the NGTS observation nights considered here
        connection = pymysql.connect(host='ngtsdb', db='ngts_ops', user='pipe')
        if cmd_args.camera is None:
            qry = "select action_id,num_images,status from action_summary_log where night='"+night+"' and campaign like '%"+camp_id+"%'"
        else:
            qry = "select action_id,num_images,status from action_summary_log where night='"+night+"' and campaign like '%"+camp_id+"%' and camera_id="+cmd_args.camera
        with connection.cursor() as cur:
            cur.execute(qry)
            res = cur.fetchall()
        if len(res) < 0.5:
            # If we find no action IDs we test for an alternative campaign name style
            if obj_name[:3] == 'TOI':
                camp_id2 = f'TOI-{toiid}'
                if cmd_args.camera is None:
                    qry = "select action_id,num_images,status from action_summary_log where night='"+night+"' and campaign like '%"+camp_id2+"%'"
                else:
                    qry = "select action_id,num_images,status from action_summary_log where night='"+night+"' and campaign like '%"+camp_id2+"%' and camera_id="+cmd_args.camera
                with connection.cursor() as cur:
                    cur.execute(qry)
                    res = cur.fetchall()
                if len(res) < 0.5:
                    logger_main.info('I found no actions for '+obj_name+' for night '+night)
                    logger_main.info('Checking for other actions for night '+night)
                    # If we find no actions we check for other actions on the same night
                    # If we find other actions we print these to the screen and then exit gracefully
                    # This allows the user to check if they have entered the object name incorrectly
                    #    resulting in an incorrect campaign name
                    # We also check for other actions for the campaign name on other nights
                    # Any such actions are printed to the screen and then BSP exits gracefully
                    # This allows the user to check if they have entered the wrong observation night
                    if cmd_args.camera is None:
                        qry2 = "select campaign,action_id,num_images,status from action_summary_log where night='"+night+"' and action_type='observeField'"
                        qry3 = "select night,action_id,num_images,status from action_summary_log where campaign like '%"+camp_id+"%'"
                    else:
                        qry2 = "select campaign,action_id,num_images,status from action_summary_log where night='"+night+"' and action_type='observeField' and camera_id="+cmd_args.camera
                        qry3 = "select night,action_id,num_images,status from action_summary_log where campaign like '%"+camp_id+"%' and camera_id="+cmd_args.camera
                    with connection.cursor() as cur:
                        cur.execute(qry2)
                        res2 = cur.fetchall()
                    if len(res2) < 0.5:
                        logger_main.info('Found no actions for other objects for night '+night)
                    else:
                        logger_main.info('Found actions for other objects from night '+night+':')
                        for r in res2:
                            logger_main.info(f'{r}')
                    logger_main.info('Checking for other actions for '+obj_name+' on different nights')
                    with connection.cursor() as cur:
                        cur.execute(qry3)
                        res3 = cur.fetchall()
                    if len(res3) < 0.5:
                        logger_main.info('Found no actions for '+obj_name+' on other nights')
                    else:
                        logger_main.info('Found actions for '+obj_name+' on other nights:')
                        for r in res3:
                            logger_main.info(f'{r}')
                    if len(res2) < 0.5 and len(res3) < 0.5:
                        raise ValueError('Found no actions for '+obj_name+' or on '+night+'. Check your inputs.')
                        
                    elif len(res2) > 0.5 and len(res3) < 0.5:
                        raise ValueError('Found no actions at all for '+obj_name+' but found actions for other objects on '+night+'. Check your inputs.')
                    
                    elif len(res2) < 0.5 and len(res3) > 0.5:
                        raise ValueError('Found actions for '+obj_name+' on other nights but none on '+night+'. Check your inputs.')
                    else:
                        raise ValueError('Found actions for '+obj_name+' on other nights and actions for other objects on '+night+'. Check your inputs.')

        # If we find actions for the object on the provided night we add these to the overall BSP run log file
        logger_main.info(f'Found {len(res)} actions for '+obj_name+' for night '+night)
        for r in res:
            logger_main.info(f'{r}')
        logger_main.info('Checking actions...')
        # For each action we check whether it is worth extracting photometry from
        # We check for whether the action has
        #    1. completed
        #    2. been aborted but where > 100 images were taken before this point
        for r in res:
            action = int(r[0])
            num_ims= r[1]
            status = r[2]
            if status == 'completed':
                actions = np.append(actions, action)
                night_store.append(night)
                ind_night_outdir_store.append(ind_night_outdir)
            elif status == 'aborted' and num_ims > 100:
                actions = np.append(actions, action)
                night_store.append(night)
                ind_night_outdir_store.append(ind_night_outdir)
    
    logger_main.info(f'Found total of {len(actions)} good actions for {len(obs_nights)} nights ({obs_nights}).')
    
    return logger_main, actions, night_store, ind_night_outdir_store

def get_target_tic_id(logger_main, obj_name):
    """
    Function to find the TIC ID for the target

    Parameters
    ----------
    logger_main : logger
        Logger governing the log file for the overall BSP run.
    obj_name : str
        Name of the target.
    
    Returns
    -------
    logger_main : logger
        Logger governing the log file for the overall BSP run.
    tic_id : int
        TIC ID of the target
    
    Raises
    ------
    ValueError
        If no TIC ID can be found for the given object    

    """
    
    tic = None
    if obj_name[:3] == 'TIC':
        # If the provided object name is in the form of a TIC ID we simply use this
        tic = int(obj_name.split('-')[-1])
        logger_main.info(f'Object is TIC-{tic}')
    elif obj_name[:3] == 'TOI':
        # If the provided object name is in the form of a TOI ID we find the TIC ID
        #   using the relevant SQL database
        logger_main.info('Finding TIC ID from TOI ID...')
        db_toiid = str(int(obj_name.split('-')[-1]))+'.01'
        connection = pymysql.connect(host='ngtsdb', db='tess', user='pipe')
        qry = "select tic_id from tois where toi_id = "+db_toiid+";"
        with connection.cursor() as cur:
            cur.execute(qry)
            res = cur.fetchone()
            if len(res) < 0.5:
                logger_main.info('ERROR - no TIC ID found in DB for '+obj_name)
                raise ValueError('Couldn\'t find a TIC ID for '+obj_name+'.')
            else:
                tic = int(res[0])
        logger_main.info(f'Object is TIC-{tic}')
### If you have a non TIC or TOI campaign name add some code here: ####
### This will allow you to manually provide bsproc with the correct TIC ID
### Code should be of the format:
### elif name == <Name of your object>:  (use this name on the command line)
###     tic = <TIC ID of your object>
###     logger_main.info(f'Object is TIC-{tic}')
### See examples for HIP-41378 and WASP-47
    elif obj_name == 'HIP-41378':
        tic = 366443426
        logger_main.info(f'Object is TIC-{tic}')
    elif obj_name == 'WASP47' or obj_name == 'WASP-47':
        tic = 102264230
        logger_main.info(f'Object is TIC-{tic}')
    if tic is None:
        logger_main.info('ERROR - no TIC ID found for '+obj_name+'. Quitting.')
        raise ValueError('No TIC ID found for '+obj_name+'. Quitting.')
    
    return logger_main, tic

def query_target_catalogues(logger, obj_tic, phot_file_dir, ac_id):
    """
    Function to query the target catalogue for the NGTS action

    Parameters
    ----------
    logger : logger
        Logger governing the log file for the single action BSP run
    obj_tic : int
        TIC ID of the target.
    phot_file_dir : str
        Path to the ngpipe output directory for the NGTS action
    ac_id : int
        NGTS action ID
    
    Returns
    -------
    logger : logger
        Logger governing the log file for the single action BSP run
    tic_ids : array; int
        TIC IDs for all stars in the ngpipe photometry catalogue
    idx : array; int
        Indices from ngpipe differentiating the target star, the nearby stars, and the comparison stars.
    star_mask : array; boolean
        Comparison star rejection mask from ngpipe.
    tmag_target : float
        TESS mag of the target star.
    tmags_comps : array; float
        TESS mag values for the comparison stars.
    
    Raises
    ------
    ValueError
        If no ngpipe target catalogue information can be found for the NGTS action   

    """
    logger.info(f'Querying target catalogue for action {ac_id}...')
    target_cat_fits_path = phot_file_dir + f'ACTION_{ac_id}_PHOTOMETRY_CATALOGUE.fits'
    if os.path.exists(target_cat_fits_path):
        # If the ngpipe photometry catalogue for the action exists in the correct
        #    directory we use this to extract the target catalogue information
        star_cat  = pyfits.getdata(target_cat_fits_path)
        tic_ids = np.array(star_cat.tic_id)
        idx = np.array(star_cat.phot_type)
        star_mask = 1 - np.array(star_cat.mask, dtype=int)[idx==2]
        tmags_full = np.array(star_cat.Tmag)
        tmag_target = tmags_full[0]
        tmags_comps = tmags_full[idx==2]
    else:
        # If the ngpipe photometry catalogue file does not exist we instead query
        #   the ngpipe target catalogue from the SQL databases
        # NOTE - this functionality is mostly unused these days following updates to ngpipe
        tic_ids, idx, star_mask0, tmags_full = get_target_catalogue_from_database(obj_tic)
        if tic_ids is None:
            # If no target catalogue information can be found BSP exits gracefully
            # Solution here is to ensure the ngpipe photometry reduction has completed for the observations
            logger.info('ERROR - Couldn\'t find any target catalogue information for TIC '+str(obj_tic))
            raise ValueError('Couldn\'t find any target catalogue information for TIC '+str(obj_tic))
        tmag_target = tmags_full[0]
        tmags_comps = tmags_full[idx==2]
        star_mask = 1 - star_mask0[idx==2]
    
    return logger, tic_ids, idx, star_mask, tmag_target, tmags_comps
