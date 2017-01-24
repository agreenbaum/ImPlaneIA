#! /usr/bin/env python

"""
May 05 2014
Update - user etc June 2016
Author: Alex Greenbaum agreenba@pha.jhu.edu

generate exposure times for GPI observations
estimate peak counts based on GPI "ETC" & sky data
"""

import numpy as np
import sys, os
from astropy.io import fits
import matplotlib.pyplot as plt
import cPickle as pickle

home = "/Users/agreenba/"

# LenoxSTScI_delivery_APOD_NRM10.xlsx has relative area calculations: 6.2% Area of NRM compared to "direct"
"""
From "Allen's Astrophysical Quantities", 4th ed. (2000) .

     band   l/um      dl/um    W m-2         Jy       photons s-1
                                um-1                    m-2 um-1       R     zero mag countrate

       Y    1.0400     0.20    2.00e-09     1441.0    1.50e+10         5.20  3.00e+09 photons m-2 s-1
       J    1.2150     0.26    3.31e-09     1630.0    2.02e+10         4.67  5.25e+09 photons m-2 s-1
       H    1.6540     0.29    1.15e-09     1050.0    9.56e+09         5.70  2.77e+09 photons m-2 s-1
      K1    2.0400     0.32    4.30e-10      667.0    4.66e+09         6.38  1.49e+09 photons m-2 s-1
      K2    2.2600     0.32    4.30e-10      667.0    4.66e+09         7.06  1.49e+09 photons m-2 s-1
      Ks    2.1570     0.32    4.30e-10      667.0    4.66e+09         6.74  1.49e+09 photons m-2 s-1
       K    2.1790     0.41    4.14e-10      655.0    4.53e+09         5.31  1.86e+09 photons m-2 s-1
       L    3.5470     0.57    6.59e-11      276.0    1.17e+09         6.22  6.67e+08 photons m-2 s-1
      L'    3.7610     0.65    5.26e-11      248.0    9.94e+08         5.79  6.46e+08 photons m-2 s-1
       M    4.7690     0.45    2.11e-11      160.0    5.06e+08        10.60  2.28e+08 photons m-2 s-1
     8.7    8.7560     1.20    1.96e-12       50.0    8.62e+07         7.30  1.03e+08 photons m-2 s-1
       N   10.4720     5.19    9.63e-13       35.2    5.07e+07         2.02  2.63e+08 photons m-2 s-1
    11.7   11.6530     1.20    6.31e-13       28.6    3.69e+07         9.71  4.43e+07 photons m-2 s-1
       Q   20.1300     7.80    7.18e-14        9.7    7.26e+06         2.58  5.66e+07 photons m-2 s-1

"""
zmagcr = {'Y': 1.50e+09, 'J':2.02e+10, 'H':9.56e+09, 'K1':4.66e+09, 'K2':4.66e+09} # photons m-2 s-1 um-1
dkam = {'Y':0.20, 'J':0.26, 'H':0.29, 'K1':0.32, 'K2':0.32}
'''
cr = Atel * zmagcr[band] * pow(10.0, -mag/2.5) * dlam # counts/s
'''

def overhead(time, coadds, frames):
	"""
	overhead estimation from:
	http://www.gemini.edu/sciops/instruments/gpi/status-and-availability?q=node/11551
	time in seconds
	"""
	return frames*(((time +3.0)*coadds) + 15.0)

def empirical_countcalculator(band, mode, band_mag,itime):
	gamma = 60000 # empirically selected
	throughput= {'NRM_Y':1.0,'NRM_J':1,'NRM_H':1,'NRM_K1':1, 'NRM_K2':0.75}
	modet = {'PRISM': 1.0, 'WOLLASTON':6/17.}
	counts = gamma*(throughput[band]/modet[mode])*(10**(band_mag/-2.5))*itime
	return counts

def empirical_etc(band, mode, band_mag):
	throughput= {'NRM_Y':1,'NRM_J':1,'NRM_H':1,'NRM_K1':1, 'NRM_K2':0.75}
	modet = {'PRISM': 1.0, 'WOLLASTON':6/17.}

	gamma = 60000
	pkcnts = 14000 # do not want to exceed this number
	rawexptime = pkcnts / (gamma*(throughput[band]/modet[mode])*(10**(band_mag/-2.5)))
	ntquants = rawexptime//1.49
	gpiexptime = ntquants*1.49
	print "GPI exposure time:",gpiexptime
	print "peak counts expected:", gpiexptime*gamma*(throughput[band]/modet[mode])*(10**(band_mag/-2.5))
	return gpiexptime

def main(argv):
    """
    e.g. running:

        python gpinrm_etc.py J -disp prism -m 5.8 -t 15 -ne 8

    will calculate counts/exposure and total integration and acquisition overheads 
    for J band pol mode target with Jmag = 5.8, and 8 15s exposures

    optionally one can try:

        python gpinrm_etc.py K1 -oid TW_Hya, -disp wollaston

    to query simbad for "TW_Hya" and look for Kmag. Since no time is specified 
    it will calculate exp time needed for max counts (14000) and use default 10 
    exposures per wollaston HWPA
    """

    import argparse
    from astroquery.simbad import Simbad
    parser = argparse.ArgumentParser(description = "This script provides an estimate of exposure times including overheads for GPI NRM observations")

    # Instrument settings (band, mode)
    parser.add_argument("band", type=str, help="GPI filter to be used. Must be one of Y, J, H, K1, or K2", choices=["Y", "y", "J", "j", "H", "h" "K1", "k1", "K2", "k2"], default=None)
    parser.add_argument("-disp", "--disperser", type=str, help="Disperser mode, options are PRISM or WOLLASTON", choices=["PRISM", "prism", "WOLLASTON", "wollaston"])
    # What is the apparent magnitude:
    parser.add_argument("-m","--appmagnitude", type=float, help="Apparent magnitude of unresolved object of interest in observing band")
    # How much time do you want?
    parser.add_argument("-t", "--time", type=float, help="Time per exposure")
    parser.add_argument("-ne", "--numExp", type=int, help="number of exposures requested. (This will be multiplied by 4 for wollaston exposures to get numExp in each HWPA). Default is 10", default=10)

    # Optional -- Target name -- query from simbad??
    parser.add_argument("-oid", "--objectid", type=str, help="If you want to query simbad, provide object identifier")

    args = parser.parse_args(argv)
    #print argv
    #print args

    if (args.disperser=="wollaston" or args.disperser=="WOLLASTON"):
        mode = "WOLLASTON"
        nexp = args.numExp*4
    elif (args.disperser=="prism" or args.disperser=="PRISM"):
        mode = "PRISM"
        nexp = args.numExp
    else:
        print "Disperser not specified, providing exposure times for spectral mode ('PRISM')"
        mode = "PRISM"
        nexp = args.numExp

    if (args.band=='y' or args.band=='Y'):
        filt = "NRM_Y"
    elif (args.band=='j' or args.band=='J'):
        filt = "NRM_J"
    elif (args.band=='h' or args.band=='H'):
        filt = "NRM_H"
    elif (args.band=='k1' or args.band=='K1'):
        filt = "NRM_K1"
    elif (args.band=='k2' or args.band=='K2'):
        filt = "NRM_K2"
    else:
        print "band not specified, default in H band"
        filt="NRM_H"

    # Check Simbad if obj id entered:
    if args.appmagnitude is not None:
        mag = args.appmagnitude
        print "given magnitude:", mag
    elif args.objectid is not None:
        print "Input object:", args.objectid, ", Searching with Simbad..."
        bandkey = filt[4]
        try:
            Simbad.add_votable_fields("flux({0})".format(bandkey))
            simobj = Simbad.query_object(args.objectid)
            mag = simobj['FLUX_{0}'.format(bandkey)]
        except:
            print "No {0} band magnitude in Simbad,".format(bandkey),
            print "provide an estimate with flag -m if you want this"
            sys.exit()
    else:
        print "************************   FAILED   ****************************"
        print "BECAUSE:"
        print "Must specify either apparent magnitude in band of interest with -m flag ",
        print "OR an object identified for a Simbad query. ",
        print "If Simbad doesn't have the flux you want, provide an estimate with -m flag."
        sys.exit()
   

    print "========================================="
    print "Exposure limit (~14K counts):"
    print "------------------------------"
    maxexptime = empirical_etc(filt, mode, mag)
    print "========================================="

    if args.time is not None:
        t = args.time
    else:
        t = maxexptime

    counts = empirical_countcalculator(filt, mode, mag, t)
    print "========================================="
    print "A {0}s exposure has ~{1} counts".format(t, counts)
    print "------------------------------"
    # exposures can only be in multiples of ~1.5 s
    tquant = (t//1.459)*1.459
    print "for", nexp, tquant, "s exposures, with overhead this will take:"
    total = overhead(tquant, 1, nexp) + 10*60 # 10 minutes for target acquisition
    print total, "s or", total//60, "minutes and", total%60, "s including target aquisition (10 min)"
    print "========================================="

if __name__ == "__main__":

    main(sys.argv[1:])

