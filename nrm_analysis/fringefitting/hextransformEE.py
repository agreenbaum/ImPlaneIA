#! /usr/bin/env python
from __future__ import print_function
import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as pl
from astropy import units as u
from astropy.units import cds
cds.enable()
#>>> distance = 42.0 * u.meter; then: dfloat = distance.value
#>>> x = 1.0 * u.parsec
#>>> x.to(u.km)  
# cds.enable()
#>>> r = 1 * u.rad # <Quantity 1.0 rad>
#>>> r.value # 1.0
#>>> from astropy.units import cds
#>>> cds.enable()
#>>> d = r.to(u.arcsec) # <Quantity 206264.80624709636 arcsec>
#>>> e = r.to(u.marcsec)# <Quantity 206264806.24709633 marcsec>

""" Python implementation: anand@stsci.edu 6 Mar 2013
    Algorithm: eelliott@stsci.edu -  Applied Optics, Vol 44, No. 8 10 March 2005  Sabatke et al.
    Erin Elliott's analytical hexagon-aperture PSF, page 1361 equation 5
    Coordinate center at center of symmetry, and flat edge along xi axis
        ---  eta
      /     \ ^ 
      \     / |
        ---   -> xi
    hex(xi,eta) = g(xi,eta) + g(-xi,eta)
"""

def gfunction(xi, eta, **kwargs):  #was g_eeAG
    c = kwargs['c']
    pixel = kwargs['pixel']
    d = kwargs['d']
    lam = kwargs['lam']
    xi = (d/lam)*pixel*(xi - c[0])
    eta = (d/lam)*pixel*(eta - c[1])

    if kwargs['minus'] is True:
        xi = -1*xi
    i = 1j
    Pi = np.pi
    g = np.exp(-i*Pi*(2*eta/np.sqrt(3) + xi)) * (\
            (np.sqrt(3)*eta - 3*xi) *
        (np.exp(i*Pi*np.sqrt(3)*eta) - np.exp(i*Pi*(4*eta/np.sqrt(3) + xi)))  + \
        (np.sqrt(3)*eta + 3*xi) * (np.exp(i*Pi*eta/np.sqrt(3)) - np.exp(i*Pi*xi)) ) \
        / (4*Pi*Pi*(eta*eta*eta - 3*eta*xi*xi))

    return g

def glimit(xi, eta, **kwargs):
    c = kwargs['c']
    pixel = kwargs['pixel']
    d = kwargs['d']
    lam = kwargs['lam']
    xi = (d/lam)*pixel*(xi - c[0])
    eta = (d/lam)*pixel*(eta - c[1])

    if kwargs['minus'] is True:
        xi = -1*xi
    Pi = np.pi
    g = ( np.exp(-1j*Pi*xi)/(2*np.sqrt(3)*Pi*Pi*xi*xi) ) * \
        ( -1 + 1j*Pi*xi + np.exp(1j*Pi*xi)-2j*Pi*xi*np.exp(1j*Pi*xi) )

    return g

def centralpix(xi, eta, **kwargs):
    c = kwargs['c']
    pixel = kwargs['pixel']
    d = kwargs['d']
    lam = kwargs['lam']
    xi = (d/lam)*pixel*(xi - c[0])
    eta = (d/lam)*pixel*(eta - c[1])

    if kwargs['minus'] is True:
        xi = -1*xi
    i = 1j
    Pi = np.pi
    g = np.sqrt(3) / 4.0

    return g

#ef hex_eeAG(s=(121,121), c=None, d=0.80, lam=4.3e-6, pitch=mas2rad(65)):
def hextransform(s=None, c=None, d=None, lam=None, pitch=None, DEBUG=False, odir=None): # was hex_eeAG
    """
        returns a forced REAL array, so the assumption is no shift in pupil plane
        d/m flat-to-flat distance, lam/m, pitch/rad
        c is the center of the PSF (if no phase aberrations), NOT an offset from array center.
    """
    if c is None:
        """  anand@stsci.edu Wed Mar 22 11:54:54 EDT 2017
        Further poking 2017 for LG++
        pre-2017 code tried to make fft2 slopes zero in recovered pupils... but this 
        is probably an incorrect interpretation of zero frequency in the analytical-to-FFT algo translation.
        Now both ASF modulus' are centered on pixel (odd array) /corner (even array),
        and the imaginary parts of the hex ASF are machine-noise close to zero.

        Test results:

        Center: (2.0, 2.0) Shape: (5, 5)
            hex imag mean max min 0.0 9.36750677027e-17 -9.36750677027e-17
        Center: (2.5, 2.5) Shape: (6, 6)
            hex imag mean max min 0.0 3.12250225676e-16 -3.12250225676e-16
        Center: (50.0, 50.0) Shape: (101, 101)
            hex imag mean max min 0.0 8.22258927613e-16 -8.22258927613e-16
        Center: (49.5, 49.5) Shape: (100, 100)
            hex imag mean max min 0.0 3.60952587264e-15 -3.60952587264e-15
        """
        c = (float(s[0])/2.0  - 0.5,  float(s[1])/2.0  - 0.5)
    print(("Center:",c, "Shape:", s))

    # ?? 2017 g = np.fromfunction(gfunction, s, d=d, c=c, lam=lam, pixel=pitch, minus=False)
    hex_complex = np.fromfunction(gfunction, s, d=d, c=c, lam=lam, pixel=pitch, minus=False) + \
                  np.fromfunction(gfunction, s, d=d, c=c, lam=lam, pixel=pitch, minus=True)
    # There will be a strip of NaNs down the middle (eta-axis)
    (xnan, ynan)= np.where(np.isnan(hex_complex))
    # The "yval" will be the same for all points
    # loop over the xi values
    for index in range(len(xnan)):
        """
        Replace NaN strip with limiting behavior. Calls from function glimit.
        """
        hex_complex[xnan[index], ynan[index]] = \
            glimit(xnan[index], ynan[index], d=d, c=c, lam=lam, pixel=pitch, minus=False) + \
            glimit(xnan[index], ynan[index], d=d, c=c, lam=lam, pixel=pitch, minus=True)
        """ changed 2017        v
            glimit(xnan[index], xnan[index], d=d, c=c, lam=lam, pixel=pitch, minus=False) + \
            glimit(xnan[index], xnan[index], d=d, c=c, lam=lam, pixel=pitch, minus=True)
                                ^
        """
    (xnan, ynan)= np.where(np.isnan(hex_complex))
    #print "wherenan", xnan, ynan

    for index in range(len(xnan)):
        """
        Replace NaN strip with limiting behavior. Calls from function glimit, which 
        doesn't give the right values at the moment for the origin.
        """
        hex_complex[xnan[index], ynan[index]] =  \
            centralpix(xnan[index], ynan[index], d=d, c=c, lam=lam, pixel=pitch, minus=False) + \
            centralpix(xnan[index], xnan[index], d=d, c=c, lam=lam, pixel=pitch, minus=True)

    if DEBUG:
        hr = hex_complex.real
        hi = hex_complex.imag
        print(("hex real mean max min",  hr.mean(), hr.max(), hr.min()))
        print(("hex imag mean max min",  hi.mean(), hi.max(), hi.min(), " should be very small"))

        """  anand@stsci.edu Wed Mar 22 11:54:54 EDT 2017  DEBUG on..."""
        """  anand@stsci.edu Thu Nov  9 11:00:10 EST 2017
             This fft pupil recovery is approximate, because of differing 
             centering interpretations of FFT cf analytical calculation of ASF (probably) 
             The sanity check is to see the 'spatially-filtered by finite fov' pupil
             that matches the assumed hex geometry (flat up, not point-up)
        """
        hexhole = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(hex_complex)))
        fits.PrimaryHDU(data=np.abs(hexhole) ).writeto(odir+"%d_hexpupilrecovered_mod.fits"%s[0], overwrite=True)
        maxmod = np.abs(hexhole).max()
        maskpha = np.where(np.abs(hexhole)<maxmod/10.0)
        """ phase via this fft is misleading - anand@stsci.edu """

    # The assumption here is tht the imaginary component is machine-noise zero.
    # This assumption checks out OK.
    # Endnote to this tfm - if the complex part is always tiny->zero,
    # why pass it a center at all?  Why not calculate c here and get it right?
    # Think through how this is actually used after the call.
    # 
    return hex_complex.real


def test_hextransform(s, odir="", pitch=None, c=None):
    modasf = hextransform(s=s,  c=c, d=0.80, lam=4.3e-6, pitch=pitch, DEBUG=True, odir=odir)
    #pl.imshow(modasf, interpolation='nearest')
    fits.PrimaryHDU(data=modasf).writeto(odir+"%d_HexModASFlimtest.fits"%s[0], overwrite=True)
    #pl.show()


def main():

    # image plane sampled to large extent - less pupil-mushing due to finite fov
    # Check the recovered pupils are 'flat side up', not 'point-up'
    ODIR = "./test_hexee_dbg/10x_" # image plane sampled to large extent - less pupil-mushing due to finite fov
    pitch = 10.0 * ((65*u.marcsec).to(u.rad)).value
    test_hextransform(s=(101,101), odir=ODIR, pitch=pitch)
    test_hextransform(s=(100,100), odir=ODIR, pitch=pitch)

    ODIR = "./test_hexee_dbg/"
    pitch = ((65*u.marcsec).to(u.rad)).value
    test_hextransform(s=(5,5),     odir=ODIR, pitch=pitch)  # easy to see centering in modulus of ASF fits files
    test_hextransform(s=(6,6),     odir=ODIR, pitch=pitch)  # easy to see centering in modulus of ASF fits files
    test_hextransform(s=(101,101), odir=ODIR, pitch=pitch)
    test_hextransform(s=(100,100), odir=ODIR, pitch=pitch)

    # center of ASF is always float(shape[0]/2.0 - 0.5,...
    nn = 11
    imcx, imcy = (0.5*nn - 0.5,  0.5*nn - 0.5)
    pitch = 10.0 * ((65*u.marcsec).to(u.rad)).value
    
    ODIR = "./test_hexee_dbg/dx0dy0_" # move center of hex transform to see ds9 directions
    test_hextransform(s=(nn,nn), odir=ODIR, pitch=pitch, c=(imcx, imcy))

    ODIR = "./test_hexee_dbg/dx1dy0_" # move center of hex transform to see ds9 directions
    test_hextransform(s=(nn,nn), odir=ODIR, pitch=pitch, c=(imcx - 1.0, imcy))

    ODIR = "./test_hexee_dbg/dx0dy1_" # move center of hex transform to see ds9 directions
    test_hextransform(s=(nn,nn), odir=ODIR, pitch=pitch, c=(imcx, imcy - 1.0))

if __name__ == "__main__":

    main()

