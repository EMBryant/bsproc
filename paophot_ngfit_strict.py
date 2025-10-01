#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:18:15 2021

@author: ed
"""

##########################################################
import argparse as ap
import sys
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
import batman as bm
import pandas as pd
from tqdm import tqdm
import os
##########################################################
rootdir = '/media/ed/data/ngts/BrightStarsWorkingGroup/PAOPhot2/'

def ParseArgs():
    '''
    Function to parse the command line arguments
    '''
    parser = ap.ArgumentParser()
    parser.add_argument('file_name', type=str)
    parser.add_argument('--night', type=str, default='20XX-YY-ZZ')
    parser.add_argument('--obj', type=str, default='tmp')
    parser.add_argument('--tc', type=float, default=None)
    parser.add_argument('--t1', type=float, default=None)
    parser.add_argument('--t2', type=float, default=None)
    parser.add_argument('--per', type=float, default=5.)
    parser.add_argument('--rp', type=float, default=0.05)
    parser.add_argument('--a', type=float, default=15)
    parser.add_argument('--inc', type=float, default=89.)
    parser.add_argument('--detrend', type=str, default='airmass')
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--end', type=float, default=1.0)
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

def lnprior(theta, tc1, tc2):
    tc, rp = theta[0], theta[1]
    if not 0.0 <= rp <= 1.0:
        return -np.inf
    if not tc1 <= tc <= tc2:
        return -np.inf
    else:
        return 0.
def lnlike(theta, params, model, sap, err, detrend, aids, actions, quad_dt=False):
    params.t0, params.rp = theta[0], theta[1]
    lc_bm = model.light_curve(params)
    lc_oot = np.array([])
    for ac, i in zip(aids, range(len(aids))):
        detrend_ac = detrend[actions==ac]
        if quad_dt:
            lc_oot = np.append(lc_oot, theta[2+3*i] + theta[3+3*i]*detrend_ac + theta[4+3*i]*detrend_ac**2)
        else:
            lc_oot = np.append(lc_oot, theta[2+2*i] + theta[3+2*i]*detrend_ac)
    lc_full = lc_oot * lc_bm
    ln_likelihood = -0.5 * (np.sum(((sap - lc_full)/err)**2 + np.log(2.0 * np.pi * err**2)))
    return ln_likelihood
    
def lnprob(theta, params, model, sap, err, airmass, aids, actions, tc1, tc2, quad_dt=False):
    lp = lnprior(theta, tc1, tc2)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, params, model, sap, err, airmass, aids, actions, quad_dt)

################################################################################
######    PLOTTING FUNCTIONS    ################################################
################################################################################

def plot_1cam(bjd, pdc, pdcerr,
               pdcmed, pdcerrmed,
               model_best, model_med,
               output):
    tbin, fbin, ebin = lb(bjd, pdc, pdcerr, 5/1440.)
    tbinm, fbinm, ebinm = lb(bjd, pdcmed, pdcerrmed, 5/1440.)
    offset = int(bjd[0])
    s = bjd.argsort()
    fig1 = plt.figure(figsize=(18, 12))
    ax1 = fig1.add_subplot(211)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_best[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Best')

    ax2 = fig1.add_subplot(212)
    ax2.plot(bjd, sap, 'k.', alpha=0.4, ms=2, zorder=1, label=str(aids[0]))
    tbi, fbi, ebi = lb(bjd, sap, err, 5/1440.)
    ax2.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
    ax2.plot(bjd, lc_oot * model_best, 'C1-', zorder=3)
    ax2.legend(loc='upper left', frameon=False)
    fig1.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig1.savefig(output+obj+'_ngfit_strict_lc_bestpms.png')

    fig2 = plt.figure(figsize=(18, 12))
    ax1 = fig2.add_subplot(211)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_med[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Median')

    ax2 = fig2.add_subplot(212)
    ax2.plot(bjd, sap, 'k.', alpha=0.4, ms=2, zorder=1, label=str(aids[0]))
    tbi, fbi, ebi = lb(bjd, sap, err, 5/1440.)
    ax2.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
    ax2.plot(bjd, lc_oot * model_med, 'C1-', zorder=3)
    ax2.legend(loc='upper left', frameon=False)
    fig2.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig2.savefig(output+obj+'_ngfit_strict_lc_medpms.png')

    plt.show()
    return fig1, fig2

def plot_2cams(bjd, pdc, pdcerr,
               pdcmed, pdcerrmed,
               model_best, model_med,
               output):
    tbin, fbin, ebin = lb(bjd, pdc, pdcerr, 5/1440.)
    tbinm, fbinm, ebinm = lb(bjd, pdcmed, pdcerrmed, 5/1440.)
    offset = int(bjd[0])
    s = bjd.argsort()
    fig1 = plt.figure(figsize=(18, 12))
    ax1 = fig1.add_subplot(211)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_best[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Best')
    
    ax2 = fig1.add_subplot(223)
    ax3 = fig1.add_subplot(224)
    axes = [ax2, ax3]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot[idt], model_best[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)
    fig1.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig1.savefig(output+obj+'_ngfit_strict_lc_bestpms.png')
        
    fig2 = plt.figure(figsize=(18, 12))
    ax1 = fig2.add_subplot(211)
    ax1.plot(bjd, pdcmed, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbinm, fbinm, yerr=ebinm, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_med[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Median')
    
    ax2 = fig2.add_subplot(223)
    ax3 = fig2.add_subplot(224)
    axes = [ax2, ax3]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot_med[idt], model_med[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)
    
    fig2.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig2.savefig(output+obj+'_ngfit_strict_lc_medpms.png')
    
    plt.show()
    return fig1, fig2

def plot_3cams(bjd, pdc, pdcerr,
               pdcmed, pdcerrmed,
               model_best, model_median,
               output):
    tbin, fbin, ebin = lb(bjd, pdc, pdcerr, 5/1440.)
    tbinm, fbinm, ebinm = lb(bjd, pdcmed, pdcerrmed, 5/1440.)
    offset = int(bjd[0])
    s = bjd.argsort()
    fig1 = plt.figure(figsize=(18, 12))
    ax1 = fig1.add_subplot(211)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_best[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Best')
    
    ax2 = fig1.add_subplot(234)
    ax3 = fig1.add_subplot(235)
    ax4 = fig1.add_subplot(236)
    axes = [ax2, ax3, ax4]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot[idt], model_best[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)
        
    fig1.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig1.savefig(output+obj+'_ngfit_strict_lc_bestpms.png')
       
    fig2 = plt.figure(figsize=(18, 12))
    ax1 = fig2.add_subplot(211)
    ax1.plot(bjd, pdcmed, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbinm, fbinm, yerr=ebinm, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_med[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Median')
    
    ax2 = fig2.add_subplot(234)
    ax3 = fig2.add_subplot(235)
    ax4 = fig2.add_subplot(236)
    axes = [ax2, ax3, ax4]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot_med[idt], model_med[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)
    
    fig2.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig2.savefig(output+obj+'_ngfit_strict_lc_medpms.png')
    
    plt.show()
    return fig1, fig2

def plot_4cams(bjd, pdc, pdcerr,
               pdcmed, pdcerrmed,
               model_best, model_median,
               output):
    tbin, fbin, ebin = lb(bjd, pdc, pdcerr, 5/1440.)
    tbinm, fbinm, ebinm = lb(bjd, pdcmed, pdcerrmed, 5/1440.)
    offset = int(bjd[0])
    s = bjd.argsort()
    fig1 = plt.figure(figsize=(18, 12))
    ax1 = fig1.add_subplot(311)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_best[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Best')
    
    ax2 = fig1.add_subplot(323)
    ax3 = fig1.add_subplot(324)
    ax4 = fig1.add_subplot(325)
    ax5 = fig1.add_subplot(326)
    axes = [ax2, ax3, ax4, ax5]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot[idt], model_best[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)
        
    fig1.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig1.savefig(output+obj+'_ngfit_strict_lc_bestpms.png')
        
    fig2 = plt.figure(figsize=(18, 12))
    ax1 = fig2.add_subplot(311)
    ax1.plot(bjd, pdcmed, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbinm, fbinm, yerr=ebinm, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_med[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Median')
    
    ax2 = fig2.add_subplot(323)
    ax3 = fig2.add_subplot(324)
    ax4 = fig2.add_subplot(325)
    ax5 = fig2.add_subplot(326)
    axes = [ax2, ax3, ax4, ax5]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot_med[idt], model_med[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)
    
    fig2.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig2.savefig(output+obj+'_ngfit_strict_lc_medpms.png')
    
    plt.show()
    return fig1, fig2

def plot_5cams(bjd, pdc, pdcerr,
               pdcmed, pdcerrmed,
               model_best, model_median,
               output):
    tbin, fbin, ebin = lb(bjd, pdc, pdcerr, 5/1440.)
    tbinm, fbinm, ebinm = lb(bjd, pdcmed, pdcerrmed, 5/1440.)
    offset = int(bjd[0])
    s = bjd.argsort()
    fig1 = plt.figure(figsize=(18, 12))
    ax1 = fig1.add_subplot(311)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_best[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Best')
    
    ax2 = fig1.add_subplot(334)
    ax3 = fig1.add_subplot(335)
    ax4 = fig1.add_subplot(336)
    ax5 = fig1.add_subplot(325)
    ax6 = fig1.add_subplot(326)
    axes = [ax2, ax3, ax4, ax5, ax6]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot[idt], model_best[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)
    
    fig1.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig1.savefig(output+obj+'_ngfit_strict_lc_bestpms.png')
            
    fig2 = plt.figure(figsize=(18, 12))
    ax1 = fig2.add_subplot(311)
    ax1.plot(bjd, pdcmed, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbinm, fbinm, yerr=ebinm, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_med[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Median')
    
    ax2 = fig2.add_subplot(334)
    ax3 = fig2.add_subplot(335)
    ax4 = fig2.add_subplot(336)
    ax5 = fig2.add_subplot(325)
    ax6 = fig2.add_subplot(326)
    axes = [ax2, ax3, ax4, ax5, ax6]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot_med[idt], model_med[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)
    
    fig2.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig2.savefig(output+obj+'_ngfit_strict_lc_medpms.png')
    
    plt.show()
    return fig1, fig2

def plot_6cams(bjd, pdc, pdcerr,
               pdcmed, pdcerrmed,
               model_best, model_median,
               output):
    tbin, fbin, ebin = lb(bjd, pdc, pdcerr, 5/1440.)
    tbinm, fbinm, ebinm = lb(bjd, pdcmed, pdcerrmed, 5/1440.)
    offset = int(bjd[0])
    s = bjd.argsort()
    fig1 = plt.figure(figsize=(18, 12))
    ax1 = fig1.add_subplot(311)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_best[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Best')
    
    ax2 = fig1.add_subplot(334)
    ax3 = fig1.add_subplot(335)
    ax4 = fig1.add_subplot(336)
    ax5 = fig1.add_subplot(337)
    ax6 = fig1.add_subplot(338)
    ax7 = fig1.add_subplot(339)
    axes = [ax2, ax3, ax4, ax5, ax6, ax7]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot[idt], model_best[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)
        
    fig1.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig1.savefig(output+obj+'_ngfit_strict_lc_bestpms.png')
       
    fig2 = plt.figure(figsize=(18, 12))
    ax1 = fig2.add_subplot(311)
    ax1.plot(bjd, pdcmed, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbinm, fbinm, yerr=ebinm, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_med[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Median')
    
    ax2 = fig2.add_subplot(334)
    ax3 = fig2.add_subplot(335)
    ax4 = fig2.add_subplot(336)
    ax5 = fig2.add_subplot(337)
    ax6 = fig2.add_subplot(338)
    ax7 = fig2.add_subplot(339)
    axes = [ax2, ax3, ax4, ax5, ax6, ax7]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot_med[idt], model_med[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)
    
    fig2.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig2.savefig(output+obj+'_ngfit_strict_lc_medpms.png')
    
    plt.show()
    return fig1, fig2

def plot_7cams(bjd, pdc, pdcerr,
               pdcmed, pdcerrmed,
               model_best, model_median,
               output):
    tbin, fbin, ebin = lb(bjd, pdc, pdcerr, 5/1440.)
    tbinm, fbinm, ebinm = lb(bjd, pdcmed, pdcerrmed, 5/1440.)
    offset = int(bjd[0])
    s = bjd.argsort()
    fig1 = plt.figure(figsize=(18, 12))
    ax1 = fig1.add_subplot(311)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_best[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Best')

    ax2 = fig1.add_subplot(345)
    ax3 = fig1.add_subplot(346)
    ax4 = fig1.add_subplot(347)
    ax5 = fig1.add_subplot(348)
    ax6 = fig1.add_subplot(337)
    ax7 = fig1.add_subplot(338)
    ax8 = fig1.add_subplot(339)
    axes = [ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot[idt], model_best[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)

    fig1.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig1.savefig(output+obj+'_ngfit_strict_lc_bestpms.png')

    fig2 = plt.figure(figsize=(18, 12))
    ax1 = fig2.add_subplot(311)
    ax1.plot(bjd, pdcmed, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbinm, fbinm, yerr=ebinm, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_med[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Median')

    ax2 = fig2.add_subplot(345)
    ax3 = fig2.add_subplot(346)
    ax4 = fig2.add_subplot(347)
    ax5 = fig2.add_subplot(348)
    ax6 = fig2.add_subplot(337)
    ax7 = fig2.add_subplot(338)
    ax8 = fig2.add_subplot(339)
    axes = [ax2, ax3, ax4, ax5, ax6, ax7, ax8] 
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot_med[idt], model_med[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)

    fig2.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig2.savefig(output+obj+'_ngfit_strict_lc_medpms.png')

    plt.show()
    return fig1, fig2

def plot_8cams(bjd, pdc, pdcerr,
               pdcmed, pdcerrmed,
               model_best, model_median,
               output):
    tbin, fbin, ebin = lb(bjd, pdc, pdcerr, 5/1440.)
    tbinm, fbinm, ebinm = lb(bjd, pdcmed, pdcerrmed, 5/1440.)
    offset = int(bjd[0])
    s = bjd.argsort()
    fig1 = plt.figure(figsize=(18, 12))
    ax1 = fig1.add_subplot(311)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_best[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Best')

    ax2 = fig1.add_subplot(345)
    ax3 = fig1.add_subplot(346)
    ax4 = fig1.add_subplot(347)
    ax5 = fig1.add_subplot(348)
    ax6 = fig1.add_subplot(349)
    ax7 = fig1.add_subplot(3,4,10)
    ax8 = fig1.add_subplot(3,4,11)
    ax9 = fig1.add_subplot(3,4,12)
    axes = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot[idt], model_best[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)

    fig1.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig1.savefig(output+obj+'_ngfit_strict_lc_bestpms.png')

    fig2 = plt.figure(figsize=(18, 12))
    ax1 = fig2.add_subplot(311)
    ax1.plot(bjd, pdcmed, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbinm, fbinm, yerr=ebinm, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_med[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Median')

    ax2 = fig2.add_subplot(345)
    ax3 = fig2.add_subplot(346)
    ax4 = fig2.add_subplot(347)
    ax5 = fig2.add_subplot(348)
    ax6 = fig2.add_subplot(349)
    ax7 = fig2.add_subplot(3,4,10)
    ax8 = fig2.add_subplot(3,4,11)
    ax9 = fig2.add_subplot(3,4,12)
    axes = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot_med[idt], model_med[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)

    fig2.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig2.savefig(output+obj+'_ngfit_strict_lc_medpms.png')

    plt.show()
    return fig1, fig2

def plot_9cams(bjd, pdc, pdcerr,
               pdcmed, pdcerrmed,
               model_best, model_median,
               output):
    tbin, fbin, ebin = lb(bjd, pdc, pdcerr, 5/1440.)
    tbinm, fbinm, ebinm = lb(bjd, pdcmed, pdcerrmed, 5/1440.)
    offset = int(bjd[0])
    s = bjd.argsort()
    fig1 = plt.figure(figsize=(18, 12))
    ax1 = fig1.add_subplot(411)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_best[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Best')

    ax2 = fig1.add_subplot(434)
    ax3 = fig1.add_subplot(435)
    ax4 = fig1.add_subplot(436)
    ax5 = fig1.add_subplot(437)
    ax6 = fig1.add_subplot(438)
    ax7 = fig1.add_subplot(439)
    ax8 = fig1.add_subplot(4,3,10)
    ax9 = fig1.add_subplot(4,3,11)
    ax10= fig1.add_subplot(4,3,12)
    axes = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot[idt], model_best[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)

    fig1.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig1.savefig(output+obj+'_ngfit_strict_lc_bestpms.png')

    fig2 = plt.figure(figsize=(18, 12))
    ax1 = fig2.add_subplot(411)
    ax1.plot(bjd, pdcmed, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbinm, fbinm, yerr=ebinm, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_med[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Median')

    ax2 = fig1.add_subplot(434)
    ax3 = fig1.add_subplot(435)
    ax4 = fig1.add_subplot(436)
    ax5 = fig1.add_subplot(437)
    ax6 = fig1.add_subplot(438)
    ax7 = fig1.add_subplot(439)
    ax8 = fig1.add_subplot(4,3,10)
    ax9 = fig1.add_subplot(4,3,11)
    ax10= fig1.add_subplot(4,3,12)
    axes = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot_med[idt], model_med[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)

    fig2.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig2.savefig(output+obj+'_ngfit_strict_lc_medpms.png')

    plt.show()
    return fig1, fig2

def plot_10cams(bjd, pdc, pdcerr,
               pdcmed, pdcerrmed,
               model_best, model_median,
               output):
    tbin, fbin, ebin = lb(bjd, pdc, pdcerr, 5/1440.)
    tbinm, fbinm, ebinm = lb(bjd, pdcmed, pdcerrmed, 5/1440.)
    offset = int(bjd[0])
    s = bjd.argsort()
    fig1 = plt.figure(figsize=(18, 12))
    ax1 = fig1.add_subplot(411)
    ax1.plot(bjd, pdc, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbin, fbin, yerr=ebin, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_best[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Best')

    ax2 = fig1.add_subplot(445)
    ax3 = fig1.add_subplot(446)
    ax4 = fig1.add_subplot(447)
    ax5 = fig1.add_subplot(448)
    ax6 = fig1.add_subplot(449)
    ax7 = fig1.add_subplot(4,4,10)
    ax8 = fig1.add_subplot(4,4,11)
    ax9 = fig1.add_subplot(4,4,12)
    ax10= fig1.add_subplot(4,4,14)
    ax10= fig1.add_subplot(4,4,15)
    axes = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot[idt], model_best[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)

    fig1.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig1.savefig(output+obj+'_ngfit_strict_lc_bestpms.png')

    fig2 = plt.figure(figsize=(18, 12))
    ax1 = fig2.add_subplot(411)
    ax1.plot(bjd, pdcmed, 'k.', alpha=0.4, ms=2, zorder=1)
    ax1.errorbar(tbinm, fbinm, yerr=ebinm, fmt='bo', ms=5, zorder=5)
    ax1.plot(bjd[s], model_med[s], 'r-', zorder=4)
    ax1.set_xlabel(f'BJD - [{offset}]')
    ax1.set_ylabel(f'NGTS Flux')
    ax1.set_title(obj+'  NGTS  ngfit - Median')

    ax2 = fig1.add_subplot(445)
    ax3 = fig1.add_subplot(446)
    ax4 = fig1.add_subplot(447)
    ax5 = fig1.add_subplot(448)
    ax6 = fig1.add_subplot(449)
    ax7 = fig1.add_subplot(4,4,10)
    ax8 = fig1.add_subplot(4,4,11)
    ax9 = fig1.add_subplot(4,4,12)
    ax10= fig1.add_subplot(4,4,14)
    ax10= fig1.add_subplot(4,4,15)
    axes = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]
    for ax, ac in zip(axes, aids):
        idt = actions==ac
        ti, fi, ei, dti, lci = bjd[idt], sap[idt], err[idt], lc_oot_med[idt], model_med[idt]
        tbi, fbi, ebi = lb(ti, fi, ei, 5/1440.)
        ax.plot(ti, fi, '.k', alpha=0.4, ms=2, zorder=1, label=str(ac))
        ax.errorbar(tbi, fbi, yerr=ebi, fmt='bo', ms=5, zorder=5)
        ax.plot(ti, lci*dti, 'C1-', zorder=3)
        ax.legend(loc='upper left', frameon=False)

    fig2.subplots_adjust(top=0.96, bottom=0.06, right=0.98, left=0.08,
                         hspace=0.15, wspace=0.1)
    fig2.savefig(output+obj+'_ngfit_strict_lc_medpms.png')

    plt.show()
    return fig1, fig2

if __name__ == "__main__":
    args = ParseArgs()
    lc = np.loadtxt(args.file_name)
    opdir_root = '/'.join(args.file_name.split('/')[:-1])+'/fit_outputs/'
    start, end = args.start, args.end
    bjd0 = np.copy(lc[:, 1])
    bjd0 -= np.int(bjd0[0])
    keep = (bjd0 >= start) & (bjd0 <= end)
    bjd, sap, err = np.copy(lc[keep, 1]), np.copy(lc[keep, 3]), np.copy(lc[keep, 4])
    if args.detrend == 'airmass':
        detrend = np.copy(lc[keep, 2])
        quaddt = False
    elif args.detrend == 'time':
        detrend = np.copy(bjd) - np.int(bjd.min())
        quaddt = False
    elif args.detrend == 'quad':
        detrend = np.copy(bjd) - np.int(bjd.min())
        quaddt = True
    actions = np.copy(lc[keep, 0])
    aids = np.unique(actions)
    Nactions = len(aids)
    
    obj = args.obj
    night = args.night
    if not os.path.exists(opdir_root):
        os.system('mkdir '+opdir_root)
    opdir = opdir_root+night+'/'
    if not os.path.exists(opdir):
        os.system('mkdir '+opdir)
    
    tc0 = args.tc
    if tc0 is None:
        tc0 = (bjd.min() + bjd.max()) / 2
    tc1, tc2 = args.t1, args.t2
    if tc1 is None:
        tc1 = (3*bjd.min() + bjd.max()) / 4
    if tc2 is None:
        tc2 = (bjd.min() + 3*bjd.max()) / 4
    
    per = args.per
    rprs0 = args.rp
    ars0 = args.a
    inc0 = args.inc
  #  Ncycles = int((bjd[0] - t0)/per + 0.5)
  #  tc0 = t0+Ncycles*per
        
    pm = bm.TransitParams()
    pm.t0 = tc0
    pm.per = per
    pm.rp = rprs0
    pm.a = ars0
    pm.inc = inc0
    pm.ecc=0.
    pm.w=90.
    pm.u=[0.4, 0.3]
    pm.limb_dark='quadratic'
    
    m = bm.TransitModel(pm, np.array(bjd, dtype=float))
    model_initial = m.light_curve(pm)
    
    theta = [tc0, rprs0]
    labels = ['Tc','RpRs']
    
    for ac, i in zip(aids, range(Nactions)):
        idac = actions==ac
        t, dt, f = bjd[idac], detrend[idac], sap[idac]
        idt = (t < tc1) | (t > tc2)
        if quaddt:
            cs = np.polyfit(dt[idt], f[idt], 2)
            theta.append(cs[2])
            theta.append(cs[1])
            theta.append(cs[0])
            labels.append(f'c0_{i}')
            labels.append(f'c1_{i}')
            labels.append(f'c2_{i}')
        else:
            cs = np.polyfit(dt[idt], f[idt], 1)
            theta.append(cs[1])
            theta.append(cs[0])
            labels.append(f'c0_{i}')
            labels.append(f'c1_{i}')
        
    ndim = len(theta)
    nwalkers = 2*ndim + 4
    pos0 = np.column_stack(([theta[i]+0.0001*np.random.randn(nwalkers)
                             for i in range(ndim)]))
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(pm, m, sap, err, detrend, aids, actions, tc1, tc2, quaddt))
    
    print('Burn in... ')
    pos, prob, state = sampler.run_mcmc(pos0, 3000, progress=True)
    burninchains = sampler.chain
    fig, axes = plt.subplots(ndim, 1, figsize=(8, 5*ndim))
    for i in range(ndim):
        ax = axes[i]
        lbl = labels[i]
        for j in range(nwalkers):
            ax.semilogx(burninchains[j][:, i], 'k-')
            ax.set_ylabel(lbl)
    plt.savefig(opdir+obj+'_strict_burninchains.png')
    plt.close()
    sampler.reset()
    print('Sampling... ')
    pos_final, prob_final, state_final = sampler.run_mcmc(pos, 10000, progress=True)
    samples = sampler.flatchain
    loglike = sampler.flatlnprobability
    df = pd.DataFrame(np.column_stack((samples, loglike)),
                      columns=labels+['lnP'])
    df.to_csv(opdir+obj+'_ngfit_strict_samples.csv',
              index_label='NIter')
    params_best = samples[loglike.argmax()]
    params_med = np.median(samples, axis=0)
    pdc, pdcerr = np.array([]), np.array([])
    pdcmed, pdcerrmed = np.array([]), np.array([])
    lc_oot = np.array([])
    lc_oot_med = np.array([])
    for ac, i in zip(aids, range(len(aids))):
        fi, ei, dti = sap[actions==ac], err[actions==ac], detrend[actions==ac]
        if quaddt:
            lcooti = params_best[2+3*i] + params_best[3+3*i] * dti + params_best[4+3*i] * dti**2
        else:
            lcooti = params_best[2+2*i] + params_best[3+2*i] * dti
        lc_oot = np.append(lc_oot, lcooti)
        pdc = np.append(pdc, fi/lcooti)
        pdcerr = np.append(pdcerr, ei/lcooti)
        if quaddt:
            lcootim = params_med[2+3*i] + params_med[3+3*i] * dti + params_med[4 + 3*i] * dti**2
        else:
            lcootim = params_med[2+2*i] + params_med[3+2*i] * dti
        lc_oot_med = np.append(lc_oot_med, lcootim)
        pdcmed = np.append(pdcmed, fi/lcootim)
        pdcerrmed = np.append(pdcerrmed, ei/lcootim)
    
    pm.t0 = params_best[0]
    pm.rp = params_best[1]
#    pm.a = params_best[2]
#    pm.inc = params_best[3]
    model_best = m.light_curve(pm)
    
    pm.t0 = params_med[0]
    pm.rp = params_med[1]
#    pm.a = params_med[2]
#    pm.inc = params_med[3]
    model_med = m.light_curve(pm)
    
    
    vals_best = np.copy(params_best)
 #   fig3 = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
 #                        title_fmt='.6f', kwargs={'fontsize':'12'},
 #                        truths=vals_best, truth_color='C1')
 #           
 #   plt.savefig(opdir+obj+'_ngfit_strict_corner.png')
 #   plt.close()
    
    if abs(Nactions - 1) <= 0.1:
        fig1, fig2 = plot_1cam(bjd, pdc, pdcerr,
                               pdcmed, pdcerrmed,
                               model_best, model_med,
                               opdir)

    elif abs(Nactions - 2) <= 0.1:
        fig1, fig2 = plot_2cams(bjd, pdc, pdcerr,
                                pdcmed, pdcerrmed,
                                model_best, model_med,
                                opdir)
    elif abs(Nactions - 3) <= 0.1:
        fig1, fig2 = plot_3cams(bjd, pdc, pdcerr,
                                pdcmed, pdcerrmed,
                                model_best, model_med,
                                opdir)
    elif abs(Nactions - 4) <= 0.1:
        fig1, fig2 = plot_4cams(bjd, pdc, pdcerr,
                                pdcmed, pdcerrmed,
                                model_best, model_med,
                                opdir)
    elif abs(Nactions - 5) <= 0.1:
        fig1, fig2 = plot_5cams(bjd, pdc, pdcerr,
                                pdcmed, pdcerrmed,
                                model_best, model_med,
                                opdir)
    elif abs(Nactions - 6) <= 0.1:
        fig1, fig2 = plot_6cams(bjd, pdc, pdcerr,
                                pdcmed, pdcerrmed,
                                model_best, model_med,
                                opdir)
    elif abs(Nactions - 7) <= 0.1:
        fig1, fig2 = plot_7cams(bjd, pdc, pdcerr,
                                pdcmed, pdcerrmed,
                                model_best, model_med,
                                opdir)
    elif abs(Nactions - 8) <= 0.1:
        fig1, fig2 = plot_8cams(bjd, pdc, pdcerr,
                                pdcmed, pdcerrmed,
                                model_best, model_med,
                                opdir)
    elif abs(Nactions - 9) <= 0.1:
        fig1, fig2 = plot_9cams(bjd, pdc, pdcerr,
                                pdcmed, pdcerrmed,
                                model_best, model_med,
                                opdir)
    elif abs(Nactions - 10) <= 0.1:
        fig1, fig2 = plot_10cams(bjd, pdc, pdcerr,
                                pdcmed, pdcerrmed,
                                model_best, model_med,
                                opdir)
