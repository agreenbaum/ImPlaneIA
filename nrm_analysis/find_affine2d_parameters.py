#! /usr/bin/env python

"""
by A. Greenbaum & A. Sivaramakrishnan 
2018 08 15 initial code

"""


from __future__ import print_function
# Standard imports
import os, sys, time
import numpy as np
from astropy.io import fits
from astropy import units as u


# Module imports
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.misctools import utils 

um = 1.0e-6


def create_afflist_rot(rotdegs, mx, my, sx,sy, xo,yo):

    alist = []
    for nrot, rotd in enumerate(rotdegs):
        rotd_ = utils.avoidhexsingularity(rotd)
        alist.append(utils.Affine2d(rotradccw=np.pi*rotd_/180.0, name="{0:.3f}".format(rotd_)))

    return alist


def find_rotation(imagedata,
                  rotdegs, mx, my, sx,sy, xo,yo,              # for Affine2d
                  pixel, npix, bandpass, over, holeshape, outdir=None):  # for nrm_model
    """ AS AZG 2018 08 Ann Arbor Develop the rotation loop first """

    print("Before Loop: ", rotdegs)

    "Extend this name to include multi-paramater searches?"
    psffmt = 'psf_nrm_{0:d}_{1:s}_{2:.3f}um_r{3:.3f}deg.fits' 

    if hasattr(rotdegs, '__iter__') is False:
        rotdegs = (rotdegs,)

    affine2d_list = create_afflist_rot(rotdegs, mx, my, sx,sy, xo,yo)
    crosscorr_rots = []


    for (rot,aff) in zip(rotdegs,affine2d_list):

        print(aff.name + "...")
        jw = NRM_Model(mask='jwst', holeshape=holeshape, 
                       over=over,
                       affine2d=aff)
        jw.set_pixelscale(pixel)
        jw.simulate(fov=npix, bandpass=bandpass, over=over)
        psffn = psffmt.format(npix, holeshape, bandpass/um, rot)
        if outdir: 
            fits.PrimaryHDU(data=jw.psf).writeto(outdir+"/"+psffn, overwrite=True)
            fits.writeto(psffn, jw.psf, overwrite=True)
            header = fits.getheader(psffn)
            utils.affinepars2header(header, aff)
            fits.update(psffn,jw.psf, header=header)

        crosscorr_rots.append(utils.rcrosscorrelate(imagedata, jw.psf).max())
        del jw


    print("Debug: ", crosscorr_rots, rotdegs)
    rot_measured_d, max_cor = utils.findpeak_1d(crosscorr_rots, rotdegs)
    print("Rotation measured: max correlation {1:.3e}", rot_measured_d, max_cor)

    # return convenient affine2d
    return utils.Affine2d(rotradccw=np.pi*rot_measured_d/180.0, 
                          name="{0:.4f}".format(rot_measured_d))
