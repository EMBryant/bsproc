#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:55:44 2021

@author: ed
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse as ap

def ParseArgs():
    parser = ap.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--night', type=str)
    parser.add_argument('--actions', type=int, nargs='*')
    parser.add_argument('--apers', type=float, nargs='*', default=None)
    parser.add_argument('--output', type=str, default='./bsproc_outputs/')
    return parser.parse_args()

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

if __name__ == "__main__":
    args = ParseArgs()
    name = args.name
 #   root_dir = '/ngts/scratch/brightstars/PAOPhot2/'+name+'/'
    root_dir = args.output+'/'+name+'/'
    filedir = root_dir+'analyse_outputs/'+args.night+'/data_files/'
    opdir = root_dir+'analyse_outputs/'+args.night+'/'
    bjd, flux, err = np.array([]), np.array([]), np.array([])
    flux0, err0 = np.array([]), np.array([])
    actions = np.array([])
    airmass = np.array([])
    skybg = np.array([])
    if len(args.apers) == 1:
        apers = [args.apers[0] for ac in args.actions]
        file_name = opdir+name+f'_NGTS_'+args.night+f'_A{args.apers[0]}_bsproc_lc.dat'
    else:
        apers = args.apers
        file_name = opdir+name+f'_NGTS_'+args.night+f'_multiap_bsproc_lc.dat'
    for ac, rap in zip(args.actions, apers):
        df = pd.read_csv(filedir+f'action{ac}_bsproc_dat.csv',
                         index_col='NExposure')
        t = np.array(df.BJD)
        f = np.array(df.loc[:, f'FluxNormA{rap}'])
        e = np.array(df.loc[:, f'FluxNormErrA{rap}'])
        f0 = np.array(df.loc[:, f'FluxA{rap}'])
        e0 = np.array(df.loc[:, f'FluxErrA{rap}'])
        
        bjd = np.append(bjd, t)
        flux = np.append(flux, f)
        err = np.append(err, e)
        flux0 = np.append(flux0, f0)
        err0 = np.append(err0, e0)
        actions = np.append(actions, np.zeros_like(t)+ac)
        
        am = np.array(df.Airmass)
        bg = np.array(df.loc[:, f'SkyBgA{rap}'])
        
        airmass = np.append(airmass, am)
        skybg = np.append(skybg, bg)
    
    sig = np.std(flux)
    idbin = ((flux > 1-6.*sig) & (flux < 1+6.*sig))
    tb, fb, eb = lb(bjd[idbin], flux[idbin], err[idbin], 5/1440.)
    
    plt.plot(bjd, flux, '.k', alpha=0.2)
    plt.errorbar(tb, fb, yerr=eb, fmt='bo')
    plt.xlabel('Time (BJD)')
    plt.ylabel('Norm Flux')
    
    plt.savefig(opdir+name+f'_NGTS_'+args.night+f'_A{rap}_bsproc_lc.png')
    plt.show(block=False)
    save = input('Save over autosaved plot? [y/n] :  ')
    if save == 'y':
        plt.savefig(opdir+name+f'_NGTS_'+args.night+f'_A{rap}_bsproc_lc.png')
    plt.close()
    
    op = np.column_stack((actions, bjd, airmass, flux, err, flux0, err0, skybg))
    
    header =  'Object: '+args.name + \
              '\n Night: '+args.night + \
             f'\n Actions: {args.actions}' + \
             f'\n Aperture Radii: {apers} pixels' + \
              '\n ActionID   BJD   Airmass   FluxNorm   FluxNormErr   Flux   FluxErr  SkyBg'
    
    np.savetxt(file_name,
               op, header=header,
               fmt='%i %.8f %.6f %.8f %.8f %.8f %.8f %.3f', delimiter=' ')