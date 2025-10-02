#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 09:22:09 2025

@author: emb
"""
import logging
import sys
import numpy as np

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