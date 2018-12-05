#! /usr/bin/env  python 
# Heritage mathematia nb from Alex & Laurent
# Python by Alex Greenbaum & Anand Sivaramakrishnan Jan 2013
# updated May 2013 to include hexagonal envelope

from __future__ import print_function
import numpy as np
import scipy.special
import sys
from nrm_analysis.misctools import utils # April 2016 trying to get imports right
import hextransformEE # change to rel imports!

from astropy import units as u
from astropy.units import cds
cds.enable()

"""
def Jinc(x, y):
    " dimensionless argument r = sqrt(x^2 + y^2)"
    R = (Jinc.d/Jinc.lam) * Jinc.pitch *  \
        np.sqrt((x-Jinc.offx)*(x-Jinc.offx) + (y-Jinc.offy)*(y-Jinc.offy))
    return replacenan(scipy.special.jv(1, np.pi*R) / (2.0*R))
"""

def Jinc(x, y, **kwargs): # LG++
    """ D/m diam, lam/m, pitch/rad , returns real tfm of circular aperture at wavelength
        Peak value Unity, first zero occurs when theta = 1.22 lambda/D,
        Dimensionless argument rho =  pi * theta D / lambda
        Jinc = 2 J1(rho) / rho
        TBD Extend to unequal x and y pitch by
            arg = pi sqrt((xpitch xdist)^2 + (ypitch ydist)^2) D / lambda
        Use centerpoint(s): return (0.5*s[0] - 0.5,  0.5*s[1] - 0.5)
        correct for Jinc, hextransform to place peak in:
            central pixel (odd array) 
            pixel corner (even array)
        use c[0] - 1 to move peak *down* in ds9
        use c[1] - 1 to move peak *left* in ds9

        As it stands here - 
        LG+  Jinc(0) = pi/4
        LG++ Jinc(0) = 1
    """
    c = kwargs['c'] 
    pitch = kwargs['pitch']
    D = kwargs['D']
    lam = kwargs['lam']

    rho = pitch * np.pi * (D / lam) *  np.sqrt( (x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) )

    J = 2.0 * scipy.special.jv(1, rho) / rho
    nanposition=np.where(np.isnan(J))
    if len(nanposition[0] == 0):  J[nanposition]=1.0
    return J

def phasor(kx, ky, hx, hy, lam, phi_m, pitch, tilt=1):
    # %%%%%%%%%%%%phi is in units of waves%%%%%%%%%%%%%% -pre July 2014
    # -------EDIT-------- changes phi to units of m -- way more physical for
    #             broadband simulations!!
    # kx ky image plane coords in units of of sampling pitch (oversampled, or not)
    # hx, hy hole centers in meters
    # pitch is sampling pitch in radians in image plane
    # ===========================================
    # k in units of "radians" hx/lam in units of "waves," requiring the 2pi
    # Example calculation -- JWST longest baseline ~6m
    #            Nyquist sampled for 64mas at 4um
    # hx/lam = 1.5e6
    # for one full cycle, when does 2pi k hx/lam = 2pi?
    # k = lam/hx = .66e-6 rad x ~2e5 = ~1.3e-1 as = 2 x ~65 mas, which is Nyquist
    # The 2pi needs to be there! That also means phi/lam is in waves
    # ===========================================
    return np.exp(-2*np.pi*1j*((pitch*kx*hx + pitch*ky*hy)/lam + (phi_m /lam) ))

"""
def interf(kx, ky):
    interference = 0j
    for hole, ctr in enumerate(interf.ctrs):
        interference = phasor((kx-interf.offx), (ky-interf.offy),
                        ctr[0], ctr[1],
                interf.lam, interf.phi[hole], interf.pitch) + interference
    return interference
"""

def interf(kx, ky, **kwargs):
    """  returns complex amplitude of fringes.  
         Use k to remind us that it is spatial frequency (angles)
         in image space 
         this works in oversampled space
    """
    psfctr = kwargs['c'] # the center of the PSF, in simulation pixels (ie oversampled)
    ctrs = kwargs['ctrs'] # hole centers
    phi = kwargs['phi']
    lam = kwargs['lam']
    pitch = kwargs['pitch'] # detpixscale/oversample

    #print((" psfctr ", psfctr))
    #print((" ctrs ", ctrs))
    #print((" phi ",  phi))
    #print((" lam ", lam))
    #print((" pitch ", pitch))

    fringe_complexamp = 0j
    for hole, ctr in enumerate(ctrs):
        fringe_complexamp += phasor((kx - psfctr[0]), (ky - psfctr[1]), 
                                    ctr[0], ctr[1], lam, phi[hole], pitch)
    return fringe_complexamp.T


def ASF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset, verbose=False):
    """ returns real array 
        psf_offset in oversampled pixels,  phi/m """
    """ Update with appropriate corrections from hexeeLGplusplus 2017 AS 
        straighten out transposes """
    """ 
        2018 01 22  switch offsets x for y to move envelope same way as fringes:
        Envelopes don't get transposed, fringes do
        BTW ctrs are not used, but left in for identical calling sequence of these 
        kinds of fromfunction feeders...
    """
    pitch = detpixel/float(oversample)
    ImCtr = np.array( utils.centerpoint((oversample*fov,oversample*fov)) ) + \
            np.array((psf_offset[1],psf_offset[0]))*oversample # note flip 1 and 0
    #print("ASF ImCtr {0}".format(ImCtr))
    return np.fromfunction(Jinc, (oversample*fov,oversample*fov),
                           c=ImCtr, 
                           D=d, 
                           lam=lam, 
                           pitch=pitch)


def ASFfringe(detpixel, fov, oversample, ctrs, lam, phi, psf_offset, 
              verbose=False):
    " returns real +/- array "
    pitch = detpixel/float(oversample)
    ImCtr = np.array( utils.centerpoint((oversample*fov,oversample*fov)) ) + \
            np.array(psf_offset)*oversample # no flip of 1 and 0, because of transpose in interf
    #print("ASFfringe ImCtr {0}".format(ImCtr))
    return np.fromfunction(interf, (oversample*fov,oversample*fov), 
                           c=ImCtr,
                           ctrs=ctrs, 
                           phi=phi,
                           lam=lam, 
                           pitch=pitch)

def ASFhex(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset): 
    " returns real +/- array "
    """ 
        2018 01 22  switch offsets x for y to move envelope same way as fringes:
        Envelopes don't get transposed, fringes do
        BTW ctrs are not used, but left in for identical calling sequence of these 
        kinds of fromfunction feeders...
    """
    pitch = detpixel/float(oversample)
    d = d
    ImCtr = np.array( utils.centerpoint((oversample*fov,oversample*fov)) ) + \
            np.array((psf_offset[1],psf_offset[0]))*oversample # note flip 1 and 0
    #print("ASFhex ImCtr {0}".format(ImCtr))
    return hextransformEE.hextransform(
                           s=(oversample*fov,oversample*fov), 
                           c=ImCtr, 
                           d=d, 
                           lam=lam, 
                           pitch=pitch)


def PSF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset,
        shape ='circ', verbose=False):
    """
        detpixel/rad 
        fov/detpixels 
        oversample 
        ctrs/m 
        d/m, 
        lam/m, 
        phi/m
        -----------------
        OLD LG+?: image_center (x,y) in oversampled pixels, 
                  used as fov*oversample/2.0 - image_center[x or y]
        -----------------
        NEW LG++: 
        psf_offset (x,y) in detector pixels, used as an offset, 
        the actual psf center is fov/2.0 + psf_offset[x or y] (in detector pixels)
    """

    #We presume we'll use the fringe pattern of hole centers treated as delta functions
    asf_fringe = ASFfringe(detpixel, fov, oversample, ctrs, lam, phi, psf_offset)

    if shape == 'circ':
        asf = ASF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset) * asf_fringe
    elif shape == 'circonly':
        asf = ASF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset)
    elif shape == 'hex':
        asf = ASFhex(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset) * asf_fringe
    elif shape == 'hexonly':
        asf = ASFhex(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset)
    elif shape == 'fringe':
        asf = asf_fringe
    else:
        raise ValueError("pupil shape %s not supported" % shape)
    if verbose:
        print("-----------------")
        print(" PSF Parameters:")
        print("-----------------")
        print("pitch: {0}, fov: {1}, oversampling: {2}, centers: {3}".format(pixel,
            fov, oversample, ctrs) + \
            "d: {0}, wavelength: {1}, pistons: {2}, shape: {3}".format(d, lam, 
            phi, shape))

    return (asf*asf.conj()).real

######################################################################
#  New in LG++ - harmonic fringes
#  New in LG++ - model_array(), ffc, ffs moved here from leastsqnrm.py
######################################################################
def harmonicfringes(**kwargs):
    """  returns sine and cosine fringes.  in image space 
         this works in the oversampled space that is each slice of the model
         switch to pitch for calc here in calls to ffc ffs
    """
    fov = kwargs['fov'] # in detpix
    pitch = kwargs['pitch'] # detpixscale
    psf_offset = kwargs['psf_offset'] # the PSF ctr, detpix
    baseline = kwargs['baseline'] # hole centers' vector, m
    lam = kwargs['lam'] # m
    oversample = kwargs['oversample']

    cpitch = pitch/oversample
    ImCtr = np.array( utils.centerpoint((oversample*fov,oversample*fov)) ) + \
            np.array(psf_offset)*oversample # first  no flip of 1 and 0, no transpose

    if 0:
        print(" ImCtr {}".format( ImCtr), end="" )
        print(" lam {}".format( lam) )
        print(" detpix pitch {}".format( pitch) )
        print(" pitch for calculation {}".format( pitch/oversample) )
        print(" over  {}".format( oversample), end="" )
        print(" fov/detpix  {}".format( fov), end="" )

    return (np.fromfunction(ffc, (fov*oversample, fov*oversample), c=ImCtr,
                                                                   baseline=baseline,
                                                                   lam=lam, pitch=cpitch),
            np.fromfunction(ffs, (fov*oversample, fov*oversample), c=ImCtr,
                                                                   baseline=baseline,
                                                                   lam=lam, pitch=cpitch))



def ffc(kx, ky, **kwargs):
    ko = kwargs['c'] # the PSF ctr
    baseline = kwargs['baseline'] # hole centers' vector
    lam = kwargs['lam'] # m
    pitch = kwargs['pitch'] # pitch for calcn = detpixscale/oversample
    return 2*np.cos(2*np.pi*pitch*((kx - ko[0])*baseline[0] + 
                                   (ky - ko[1])*baseline[1]) / lam)

def ffs(kx, ky, **kwargs):
    ko = kwargs['c'] # the PSF ctr
    baseline = kwargs['baseline'] # hole centers' vector
    lam = kwargs['lam'] # m
    pitch = kwargs['pitch'] # pitch for calcn = detpixscale/oversample
    # print("*****  pitch for ffc ffs {}".format( pitch) )
    return 2*np.sin(2*np.pi*pitch*((kx - ko[0])*baseline[0] + 
                                   (ky - ko[1])*baseline[1]) / lam)



def model_array(ctrs, lam, oversample, pitch, fov, d, psf_offset=(0,0),
                shape ='circ', verbose=False):
    # pitch is detpixel
    # psf_offset in detpix

    nholes = ctrs.shape[0]
    phi = np.zeros((nholes,)) # no phase errors in the model slices...
    modelshape = (fov*oversample, fov*oversample)  # spatial extent of image model - the oversampled array
    
    if verbose:
        print("------------------")
        print(" Model Parameters:")
        print("------------------")
        print("pitch: {0}, fov: {1}, oversampling: {2}, centers: {3}".format(pitch,
            fov, oversample, ctrs) + \
            " d: {0}, wavelength: {1}, shape: {2}".format(d, lam, shape) +\
            "\ncentering:{0}\n {1}".format(centering, off))

    # calculate primary beam envelope (non-negative real)
    # ASF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset) * asf_fringe
    if shape=='circ':
        asf_pb = ASF(   pitch, fov, oversample, ctrs, d, lam, phi, psf_offset)
    elif shape=='hex':
        asf_pb = ASFhex(pitch, fov, oversample, ctrs, d, lam, phi, psf_offset)
    else:
        raise KeyError("Must provide a valid hole shape. Current supported shapes are" \
                " 'circ' and 'hex'.")
    primary_beam = asf_pb * asf_pb
    

    alist = []
    for i in range(nholes - 1):
        for j in range(nholes - 1):
            if j + i + 1 < nholes:
                alist = np.append(alist, i)
                alist = np.append(alist, j + i + 1)
    alist = alist.reshape((len(alist)//2, 2))
    #print("alist length {0} alist {1}".format(len(alist), alist))

    ffmodel = []
    ffmodel.append(nholes * np.ones(modelshape))
    for basepair in alist:
        #print("i", int(basepair[0]), end="")
        #print("  j", int(basepair[1]), end="")
        baseline = ctrs[int(basepair[0])] - ctrs[int(basepair[1])]
        #print(baseline)
        cosfringe, sinfringe = harmonicfringes(fov=fov, pitch=pitch, psf_offset=psf_offset,
                                               baseline=baseline,
                                               oversample=oversample,
                                               lam=lam)
        ffmodel.append( np.transpose(cosfringe) )
        ffmodel.append( np.transpose(sinfringe) )
    #print("shape ffmodel", np.shape(ffmodel))

    return primary_beam, ffmodel


def multiplyenv(env, fringeterms):
    # The envelope is size (fov, fov). This multiplies the envelope by each of the 43 slices
    # (if 7 holes) in the fringe model
    full = np.ones((np.shape(fringeterms)[1], np.shape(fringeterms)[2], np.shape(fringeterms)[0]+1))
    for i in range(len(fringeterms)):
        full[:,:,i] = env * fringeterms[i]
    print("Total # fringe terms:", i)
    return full

# ------------------------

# Various AS tests:
##############################################################

def test_circtransform(s=None, c=None, D=None, lam=None, pitch=None, DEBUG=False, odir=None):

    if c is None: # ctr for odd & even arrays
        c = (float(s[0])/2.0  - 0.5,  float(s[1])/2.0  - 0.5)
    print(("analyticnrm2.test_circtransform: Center:",c, "Shape:", s))

    aa =  np.fromfunction(Jinc, s,  c=c, D=D, lam=lam, pitch=pitch)

    aa[np.where(np.isnan(aa))] = 1.0
    if odir:
        fnroot = "Jinc_D%.1fm_lam%.1fuma_offx%.1f_offy_%.1f" %  \
                 (kwargs['D'], kwargs['lam'], kwargs['offx'], kwargs['offy'])
        fits.PrimaryHDU(data=aa).writeto(odir+fnroot+".fits", overwrite=True)
        pl.imshow(aa, interpolation='nearest')
        pl.savefig(odir+fnroot+".png")
        pl.show()
    return aa

def func(x,y):
    """ 
    called with fromfunction((5,7),...) 
    func() creates an array of the form:
    anand@aether.local:83  ./00_fromfunc.py

Python array:
     y = 0   1    2    3    4    5    6
 x
 0  [[  0.   1.   2.   3.   4.   5.   6.]
 1   [ 10.  11.  12.  13.  14.  15.  16.]
 2   [ 20.  21.  22.  23.  24.  25.  26.]
 3   [ 30.  31.  32.  33.  34.  35.  36.]
 4   [ 40.  41.  42.  43.  44.  45.  46.]]



DS9 display:

     y = 0   1    2    3    4    5    6
 x
 4   [ 40.  41.  42.  43.  44.  45.  46.]]
 3   [ 30.  31.  32.  33.  34.  35.  36.]
 2   [ 20.  21.  22.  23.  24.  25.  26.]
 1   [ 10.  11.  12.  13.  14.  15.  16.]
 0  [[  0.   1.   2.   3.   4.   5.   6.]

    """

    return 10*x + y

def test_fromfunction(s, odir=None):
    aa =  np.fromfunction(func, s)
    if odir:
        fits.PrimaryHDU(data=aa).writeto(odir+"fromfunc_5x7.fits", overwrite=True)
        pl.imshow(aa, interpolation='nearest')
        pl.savefig(odir+"fromfunc_5x7.png")
        #pl.show()
    return aa


def test_circ(outdir):
    a = test_fromfunction((5,7), odir=outdir)

    D = (6.5 * u.m).value
    lam = (4.3e-6 * u.m).value
    reselt = lam/D * u.rad
    gamma = 4
    pitch = reselt.value / gamma # if Nyq/10, expect first zero at 10 * 1.22 pixel radius = 12 pixels

    # examine centering for odd, even, x and y offsets
    nn=10
    s = (nn,nn)
    ctr = utils.centerpoint(s)

    a = test_circtransform(s=s, D=D, lam=lam, pitch=pitch, c=ctr, DEBUG=True)
    fits.PrimaryHDU(data=a).writeto(outdir+"/jinc%d_gamma%d.fits"%(nn,gamma), overwrite=True)


    # OFFSETs from centerpoint: ctrpt - offset (watch the sign!)
    nn = 11
    s = (nn,nn)
    ctr = utils.centerpoint(s)

    a = test_circtransform(s=s, D=D, lam=lam, pitch=pitch, c=ctr, DEBUG=True)
    fits.PrimaryHDU(data=a).writeto(outdir+"/jinc%d_gamma%d.fits"%(nn,gamma), overwrite=True)

    # check your bessel zero... 122 pixels radius?
    gamma = 100 # if Nyq/10, expect first zero at 10 * 1.22 pixel radius = 12 pixels.  If gamma=100, zero at r=122 pixels
    pitch = reselt.value / gamma
    nn = 255
    s = (nn,nn)
    ctr = utils.centerpoint(s)
    a = test_circtransform(s=s, D=D, lam=lam, pitch=pitch, c=ctr, DEBUG=True)
    fits.PrimaryHDU(data=a*a).writeto(outdir+"/jinc%d_gamma%d.fits"%(nn,gamma), overwrite=True)
    # answer - Dia = 250 - 6 pixels = 244 pixels, half is 122 pixels.  Bingo!  Checks out fine!


def test_PSFs():

    outdir = "/Users/anand/gitsrc/nrm_analysis/rundir/test_analyticnrm2/"

    fov = 101
    gamma = 2.3

    D = (6.5 * u.m).value
    d = (0.8 * u.m).value
    lam = (4.3e-6 * u.m).value
    reselt = lam/D * u.rad
    detpixel = reselt.value / gamma 
    s = (fov, fov)
    oversample=1

    # copy out hole creation from LG_Model, use 2 holes
    ctrs_asdesigned = np.array( [ 
                        [ 0.00000000,  -2.640000],       #B4 -> B4  as-designed -> as-built mapping
                        [ 1.1431500 ,  1.9800000]    ] ) #C1 -> C6
    ctrs_asbuilt = ctrs_asdesigned.copy()
    ctrs_asbuilt[:,0] *= -1
    ctrs = ctrs_asbuilt
    phi = np.array((0.0, 0.0))

    """def PSF(detpixel, fov, oversample, ctrs, d, lam, phi, psf_offset, shape = 'circ', verbose=False):"""


    psf_offsets = ((0,0), )
    psf_offsets = ((0,0), (0,1), (1,0))  
    # 2018.01.22 anand@stsci.edu
    # this loop checks out in DS9 -         centers 51,51, 52,51, 51,52 respectively for fringes (which call transpose)
    # circonly moves x for y and y for x... centers 51,51  51,52  52,51  no transpose called - 
    # hexonly moves x for y and y for x...  centers 51,51  51,52  52,51   no transpose called flat top & bottom of hextfm looks correct though
    # Solution to hexonly & jinconly is to switch x,y and use transpose?  or just move centers appropriately?  Do the latter
    # since we do not put any phase berrations into this calculation.
    ### THIS FIXED THE OFFSETTING to be the same for fringe pattern as well as envelope pattern. anand@stsci.edu
    # Note that nnow shifting the PSFD() unit pixels creates the same peak intrensities much better than before.  This could
    # explain primary-beam-shaped residuals in LG amd LG+.
    

    for psfo in psf_offsets:
        mnem = "off_%.1f_%.1f"%psfo
        #test w/fringeonly
        psf = PSF(detpixel, fov, oversample, ctrs, d, lam, phi, psfo, shape='fringeonly', verbose=True)
        fits.PrimaryHDU(data=psf).writeto(outdir+"/_"+mnem+"_hb4c6_fringeOnly_psf.fits", overwrite=True)

        #test w/circ
        psf = PSF(detpixel, fov, oversample, ctrs, d, lam, phi, psfo, shape='circ', verbose=True)
        fits.PrimaryHDU(data=psf).writeto(outdir+"/"+mnem+"_hb4c6_circ_psf.fits", overwrite=True)
        psf = PSF(detpixel, fov, oversample, ctrs, d, lam, phi, psfo, shape='circonly', verbose=True)
        fits.PrimaryHDU(data=psf).writeto(outdir+"/_"+mnem+"_hb4c6_circleOnly_psf.fits", overwrite=True)

        #test w/hex
        psf = PSF(detpixel, fov, oversample, ctrs, d, lam, phi, psfo, shape='hex', verbose=True)
        fits.PrimaryHDU(data=psf).writeto(outdir+"/"+mnem+"_hb4c6_hex_psf.fits", overwrite=True)
        psf = PSF(detpixel, fov, oversample, ctrs, d, lam, phi, psfo, shape='hexonly', verbose=True)
        fits.PrimaryHDU(data=psf).writeto(outdir+"/_"+mnem+"_hb4c6_hexOnly_psf.fits", overwrite=True)

        #test pistonsigns  w/hex, finer pixellation... advance B4 wrt C6
        # 2018.01.22 anand@stsci.edu
        # quarter wave piston steps repeat after one wave.  positive phi_m is a phase ADVANCE
        # - see that positive B4 piston moves fringe pattern towards negative C6 piston here.
        # After a full wave, peak pixel repeats at central pixel 51,51 exactly

        # The same thing happens with the (1,0) offset.
        # The same thing happens with the (0,1) offset.

        phi = np.array((0, 0))
        psf = PSF(detpixel*0.5, fov, oversample, ctrs, d, lam, phi, psfo, shape='hex', verbose=True)
        fits.PrimaryHDU(data=psf).writeto(outdir+"/0.00wave_"+mnem+"_hb4c6_hpsf.fits", overwrite=True)

        phi = np.array((lam/8, -lam/8))
        psf = PSF(detpixel*0.5, fov, oversample, ctrs, d, lam, phi, psfo, shape='hex', verbose=True)
        fits.PrimaryHDU(data=psf).writeto(outdir+"/0.25qwave_"+mnem+"_hb4c6_hpsf.fits", overwrite=True)

        phi = np.array((lam/4, -lam/4))
        psf = PSF(detpixel*0.5, fov, oversample, ctrs, d, lam, phi, psfo, shape='hex', verbose=True)
        fits.PrimaryHDU(data=psf).writeto(outdir+"/0.50wave_"+mnem+"_hb4c6_hpsf.fits", overwrite=True)

        phi = np.array((3.0*lam/8, -3.0*lam/8))
        psf = PSF(detpixel*0.5, fov, oversample, ctrs, d, lam, phi, psfo, shape='hex', verbose=True)
        fits.PrimaryHDU(data=psf).writeto(outdir+"/0.75wave_"+mnem+"_hb4c6_hpsf.fits", overwrite=True)

        phi = np.array((lam/2, -lam/2))
        psf = PSF(detpixel*0.5, fov, oversample, ctrs, d, lam, phi, psfo, shape='hex', verbose=True)
        fits.PrimaryHDU(data=psf).writeto(outdir+"/1.00wave_"+mnem+"_hb4c6_hpsf.fits", overwrite=True)


def test_model_array():

    outdir = "/Users/anand/gitsrc/nrm_analysis/rundir/test_analyticnrm2/"

    fov = 101
    gamma = 2.3

    D = (6.5 * u.m).value
    d = (0.8 * u.m).value
    lam = (4.3e-6 * u.m).value
    reselt = lam/D * u.rad
    detpixel = reselt.value / gamma 
    s = (fov, fov)
    oversample=1

    # copy-paste from LG_Model.py:
    ctrs_asdesigned = np.array( [ 
                        [ 0.00000000,  -2.640000],       #B4 -> B4  as-designed -> as-built mapping
                        [-2.2863100 ,  0.0000000],       #C5 -> C2
                        [ 2.2863100 , -1.3200001],       #B3 -> B5
                        [-2.2863100 ,  1.3200001],       #B6 -> B2
                        [-1.1431500 ,  1.9800000],       #C6 -> C1
                        [ 2.2863100 ,  1.3200001],       #B2 -> B6
                        [ 1.1431500 ,  1.9800000]    ] ) #C1 -> C6
    ctrs_asbuilt = ctrs_asdesigned.copy()
    ctrs_asbuilt[:,0] *= -1
    ctrs = ctrs_asbuilt

    pitch = detpixel 
    psf_offset=(0,0),
    shape ='circ'

    mnem = "modelarray_off_%.1f_%.1f"%(0,0)
    pb, ff = model_array(ctrs, lam, oversample, detpixel, fov, d, psf_offset=(0,0),
                         shape ='hex', verbose=False)
    fits.PrimaryHDU(data=ff*pb).writeto(outdir+"/"+mnem+"_hb4c6_hpsf.fits", overwrite=True)

    mnem = "modelarray_off_%.1f_%.1f"%(1,0)
    pb, ff = model_array(ctrs, lam, oversample, detpixel, fov, d, psf_offset=(1,0),
                         shape ='hex', verbose=False)
    fits.PrimaryHDU(data=ff*pb).writeto(outdir+"/"+mnem+"_hb4c6_hpsf.fits", overwrite=True)

    mnem = "modelarray_off_%.1f_%.1f"%(0,1)
    pb, ff = model_array(ctrs, lam, oversample, detpixel, fov, d, psf_offset=(0,1),
                         shape ='hex', verbose=False)
    fits.PrimaryHDU(data=ff*pb).writeto(outdir+"/"+mnem+"_hb4c6_hpsf.fits", overwrite=True)


if __name__ == "__main__":
    #if 0: # works for nrmanalysis2 unittest: 
    #    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../')) # change to rel imports!
    #    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../')) # change to rel imports!
    #    import misctools  # change to rel imports!
    #    from misctools import misctools.utils  # change to rel imports
    #    import hextransformEE # change to rel imports!


    test_model_array()
    test_PSFs()
    test_fromfunction((5,7), odir="/Users/azgreenb/pythonmodules/nrm_analysis/rundir/test_analyticnrm2/")
    test_circ("/Users/azgreenb/pythonmodules/nrm_analysis/rundir/test_analyticnrm2/")
