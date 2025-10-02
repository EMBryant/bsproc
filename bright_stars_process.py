#!/usr/local/python3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:39:52 2021

@author: ed
"""

import argparse as ap
import BSP_utils as bspu
import BSP_db as bspd
import BSP_phot as bspp

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
                        help='Name of directory to save the outputs in. Default is ./bsproc_outputs/')
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
    parser.add_argument('--exptime', type=float, default=10.,
                        help='Exposure time of observations')
    parser.add_argument('--update_bjd', action='store_true',
                        help='Provide this to over write the BJD saved in an existing BSP csv output with the new ngpipe values')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse initial arguments
    args = ParseArgs()
    object_name = args.name
    if object_name is None:
        raise ValueError('I need an object name. (--name <object_name>) ')
        
    # Set up output directories
    bsdir = args.output
    ngpipe_op_dir = '/ngts/PAOPhot2/'
    observation_nights = args.night
    outdir_main, individual_night_outdirs = bspd.check_output_directories(
            bsdir, observation_nights, object_name)
    
    #Initialise overall logger
    logger_main = bspu.custom_logger(outdir_main+'master_logs/'+object_name+'_bsproc_main.log')
    
    # Find the relevant action IDs from the SQL databases
    logger_main, actions, night_store, individual_night_outdir_store = bspd.find_action_ids(
            logger_main, args, object_name, observation_nights, individual_night_outdirs)
    
    # Find the TIC ID for the target
    logger_main, obj_ticid = bspd.get_target_tic_id(logger_main, object_name)

    ac_apers_min_target_store, ac_apers_min_master_store,  missing_action_store, \
        output_file_name_store = bspp.run_BSP_process(logger_main, args, object_name,
                                                 ngpipe_op_dir, actions, night_store, 
                                                 individual_night_outdir_store,
                                                 obj_ticid)
    
    bspp.collect_best_aperture_photometry(
        logger_main, ac_apers_min_target_store, ac_apers_min_master_store,
        missing_action_store, output_file_name_store, actions, night_store,
        observation_nights, object_name, obj_ticid, args, outdir_main
        )