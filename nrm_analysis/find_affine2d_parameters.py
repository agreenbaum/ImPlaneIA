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


rad2mas = 1000.0/u.arcsec.to(u.rad)


# Module imports
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.misctools import utils 

um = 1.0e-6

VERBOSE = False
def vprint(*args):  # VERBOSE mode printing
    if VERBOSE: print("-----------------------------------------", *args)
    pass

def create_afflist_rot(rotdegs, mx, my, sx,sy, xo,yo):
    """ create a list of affine objects with various rotations to use 
        in order to go through and find which fits some image plane data best """
    alist = []
    for nrot, rotd in enumerate(rotdegs):
        rotd_ = utils.avoidhexsingularity(rotd)
        alist.append(utils.Affine2d(rotradccw=np.pi*rotd_/180.0, name="{0:.3f}".format(rotd_)))
    return alist


def create_afflist_scales(scales, mx, my, sx,sy, xo,yo):
    """ create a list of affine objects with various uniform pixel scale factors
        (typically close to unity) to use in order to go through and find which
         fits some image plane data best 
         
         if pixelscale in """
    alist = []
    for nscl, scl in enumerate(scales):
        alist.append(utils.Affine2d(mx*scl, my*scl, sx*scl,sy*scl, xo*scl,yo*scl,
                                    name="scale_{0:.4f}".format(scl)))
    return alist


def find_scale(imagedata, 
               affine_best, # best current guess at data geometry cf analytical ideal
               scales, # scales are near-unity
               pixel, npix, bandpass, over, holeshape, outdir=None):  # for nrm_model
    """  Preserve incoming "pixel" value, put the scale correction into the Affine2d object
         Is that kosher???  Should we change the pixel scale and leave affine2d the same?
         Affine2d can also incorporate unequal x and y scales, shears...
         For now place scale corrections into the Affine2d object     
         
         Note - placing isotropic scale change into Affine2d is equivalent to changing
         the effective image distance in the optical train while insisting that the
         mask physical geometry does not change, and the wavelength is perfectly knowm


         AS 2018 10  """

    affine_best.show("\tfind_scale")
    vprint("\tBefore Loop: ", scales)

    #Extend this name to include multi-paramater searches?
    psffmt = 'psf_nrm_{0:d}_{1:s}_{2:.3f}um_scl{3:.3f}.fits' 
    # expect (npix, holeshape, bandpass/um, scl)

    if hasattr(scales, '__iter__') is False:
        scales = (scales,)

    affine2d_list = create_afflist_scales(scales, 
                                          affine_best.mx, affine_best.my, 
                                          affine_best.sx,affine_best.sy, 
                                          affine_best.xo,affine_best.yo)
    crosscorrs = []


    for (scl,aff) in zip(scales,affine2d_list):

        vprint(aff.name + "...")
        jw = NRM_Model(mask='jwst', holeshape=holeshape, 
                       over=over,
                       affine2d=aff)
        jw.set_pixelscale(pixel)
        jw.simulate(fov=npix, bandpass=bandpass, over=over)
        psffn = psffmt.format(npix, holeshape, bandpass/um, scl)
        if outdir: 
            fits.PrimaryHDU(data=jw.psf).writeto(outdir+"/"+psffn, overwrite=True)
            fits.writeto(psffn, jw.psf, overwrite=True)
            header = fits.getheader(psffn)
            utils.affinepars2header(header, aff)
            fits.update(psffn,jw.psf, header=header)

        crosscorrs.append(utils.rcrosscorrelate(imagedata, jw.psf).max())
        del jw


    vprint("\t*******************")
    vprint("\tDebug: crosscorrelations", crosscorrs)
    vprint("\tDebug:            scales", scales)
    scl_measured, max_cor = utils.findpeak_1d(crosscorrs, scales)
    vprint("\tScale factor measured {0:.5f}  Max correlation {1:.3e}".format(scl_measured, max_cor))
    vprint("\tpixel pitch from header  {0:.3f} mas".format(pixel*rad2mas))
    vprint("\tpixel pitch  {0:.3f} mas (implemented using affine2d)".format(scl_measured*pixel*rad2mas))

    # return convenient affine2d
    return utils.Affine2d( affine_best.mx*scl_measured, affine_best.my*scl_measured, 
                           affine_best.sx*scl_measured, affine_best.sy*scl_measured, 
                           affine_best.xo*scl_measured, affine_best.yo*scl_measured,
                           name="scale_{0:.4f}".format(scl))


def find_rotation(imagedata,
                  rotdegs, mx, my, sx,sy, xo,yo,              # for Affine2d
                  pixel, npix, bandpass, over, holeshape, outdir=None):  # for nrm_model
    """ AS AZG 2018 08 Ann Arbor Develop the rotation loop first """

    vprint("Before Loop: ", rotdegs)

    #Extend this name to include multi-paramater searches?
    psffmt = 'psf_nrm_{0:d}_{1:s}_{2:.3f}um_r{3:.3f}deg.fits' 
    # expect (npix, holeshape, bandpass/um, scl)

    if hasattr(rotdegs, '__iter__') is False:
        rotdegs = (rotdegs,)

    affine2d_list = create_afflist_rot(rotdegs, mx, my, sx,sy, xo,yo)
    crosscorr_rots = []


    for (rot,aff) in zip(rotdegs,affine2d_list):

        vprint(aff.name + "...")
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


    vprint("Debug: ", crosscorr_rots, rotdegs)
    rot_measured_d, max_cor = utils.findpeak_1d(crosscorr_rots, rotdegs)
    vprint("Rotation measured: max correlation {1:.3e}", rot_measured_d, max_cor)

    # return convenient affine2d
    return utils.Affine2d(rotradccw=np.pi*rot_measured_d/180.0, 
                          name="{0:.4f}".format(rot_measured_d))
