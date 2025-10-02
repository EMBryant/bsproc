#!/usr/local/python3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 09:22:09 2025

Contains functions of general/miscellaneous actions within BSP pipeline runs

@author: Edward M. Bryant
"""
import logging
import sys
import numpy as np
from datetime import datetime, timezone

def custom_logger(logger_name, level=logging.DEBUG):
    """
    This function is used to set up the log files to track important information
      from the BSP runs
    """
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

def set_up_overall_logger(obj_name, logger_name, level=logging.DEBUG):
    """
    This function ensures the correct logging file for the overall logger is used
    It also adds a line to the logging file to denote the start of a new run
    """
    logger = custom_logger(logger_name, level=level)
    logger.info('#############################################################')
    logger.info('NEW BSP RUN for '+obj_name)
    now = datetime.now(timezone.utc)
    now_string = now.strftime('%Y-%m-%dT%H:%M:%S')
    logger.info('Run starting at UTC date and time: '+now_string)
    logger.info('#############################################################')
    
    return logger

def set_up_ind_action_logger(obj_name, action_id, 
                             logger_name, level=logging.DEBUG):
    """
    This function ensures the correct logging file for the single action logger is used
    It also adds a line to the logging file to denote the start of a new run
    """
    logger = custom_logger(logger_name, level=level)
    logger.info('#############################################################')
    logger.info('NEW BSP RUN for '+obj_name+f'  Action {action_id}')
    now = datetime.now(timezone.utc)
    now_string = now.strftime('%Y-%m-%dT%H:%M:%S')
    logger.info('Run starting at UTC date and time: '+now_string)
    logger.info('#############################################################')
    
    return logger

def lb(time, flux, err, bin_width):
    '''
    Function to bin the data into bins of a given width. time and bin_width
    must have the same units

    Parameters
    ----------
    time : array; float
        Array of the observation time stamps
    flux : array; float
        Flux time series values
    err : array; float
        Error values for the flux values
    bin_width : float
        Width of the bins to use (in same units as time)
        
    Returns
    -------
    time_bin : array; float
        Time stamps for the binned flux points
    flux_bin : array; float
        Binned flux data points
    err_bin : array; float
        Uncertainties on the binned flux values
    '''

    edges = np.arange(np.min(time), np.max(time), bin_width)
    dig = np.digitize(time, edges)
    time_binned = (edges[1:] + edges[:-1]) / 2
    flux_binned = np.array([np.nan if len(flux[dig == i]) == 0 else flux[dig == i].mean() 
                            for i in range(1, len(edges))])
    err_binned = np.array([np.nan if len(flux[dig == i]) == 0 else np.sqrt(np.sum(err[dig==i]**2))/len(err[dig==i]) 
                           for i in range(1, len(edges))])
    time_bin = time_binned[~np.isnan(err_binned)]
    err_bin = err_binned[~np.isnan(err_binned)]
    flux_bin = flux_binned[~np.isnan(err_binned)]

    return time_bin, flux_bin, err_bin