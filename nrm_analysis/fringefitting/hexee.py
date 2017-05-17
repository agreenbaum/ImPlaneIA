#! /usr/bin/env python
import numpy as np
from astropy.io import fits
import os
#import matplotlib.pyplot as pl
DEBUG = True
DEBUG = False
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


def mas2rad(mas):
    rad = mas*(10**(-3)) / (3600*180/np.pi)
    return rad

def g_ee(xi, eta, **kwargs):
    D = kwargs['D']
    c = kwargs['c']
    xi = xi - c[0]
    eta = eta - c[1]
    if kwargs['minus'] is True:
        xi = -1*xi
    i = 1j
    Pi = np.pi
    g = np.exp(-i*Pi*D*(2*eta/np.sqrt(3) + xi)) * (\
        (np.sqrt(3)*eta - 3*xi) *
        (np.exp(i*Pi*D*np.sqrt(3)*eta) - np.exp(i*Pi*D*(4*eta/np.sqrt(3) + xi)))  + \
        (np.sqrt(3)*eta + 3*xi) * (np.exp(i*Pi*D*eta/np.sqrt(3)) - np.exp(i*Pi*D*xi)) ) \
        / (4*Pi*Pi*(eta*eta*eta - 3*eta*xi*xi))
    g[np.isnan(g)] = 0.0
    return g

def g_eeAG(xi, eta, **kwargs):
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
    """
    if eta==0:
        if xi==0:
            g = np.sqrt(3)*d*d / 4.0
        else:
            g = (np.exp(-1j*Pi*xi)/(2*np.sqrt(3)*Pi*Pi*xi*xi) )* \
            (-1 + 1j*Pi*xi + np.exp(1j*Pi*xi)-2j*Pi*xi*np.exp(1j*Pi*xi))
    else:
    """
    g = np.exp(-i*Pi*(2*eta/np.sqrt(3) + xi)) * (\
            (np.sqrt(3)*eta - 3*xi) *
        (np.exp(i*Pi*np.sqrt(3)*eta) - np.exp(i*Pi*(4*eta/np.sqrt(3) + xi)))  + \
        (np.sqrt(3)*eta + 3*xi) * (np.exp(i*Pi*eta/np.sqrt(3)) - np.exp(i*Pi*xi)) ) \
        / (4*Pi*Pi*(eta*eta*eta - 3*eta*xi*xi))

    #g[np.where(g==np.nan)] = ( (-1 + 1j*Pi*xi + np.exp(1j*Pi*xi)-2j*Pi*xi*np.exp(1J*Pi*xi)) )
    # (np.exp(-1j*Pi)/(2*np.sqrt(3)*Pi*Pi*xi)) *
    #g[np.where(g==np.isnan(g))] = (np.exp(-1j*Pi)/(2*np.sqrt(3)*Pi*Pi*xi)) * (\
    #               (-1 + 1j*Pi*xi + np.exp(1j*Pi*xi)*(1-2j*Pi*xi)) )
    #g[np.isnan(g)] = 0.0
    return g
"""

def g_eeGEN(xi, eta, **kwargs):

    i = 1j
    Pi = np.pi

    g = np.exp(-i*Pi*(2*eta/np.sqrt(3) + xi)) * (\
            (np.sqrt(3)*eta - 3*xi) *
        (np.exp(i*Pi*np.sqrt(3)*eta) - np.exp(i*Pi*(4*eta/np.sqrt(3) + xi)))  + \
        (np.sqrt(3)*eta + 3*xi) * (np.exp(i*Pi*eta/np.sqrt(3)) - np.exp(i*Pi*xi)) ) \
        / (4*Pi*Pi*(eta*eta*eta - 3*eta*xi*xi))
    #g[np.where(g==np.nan)] = (np.exp(-1j*Pi)/(2*np.sqrt(3)*Pi*Pi*xi)) * \
                    (-1 + 1j*xi + np.exp(1j*Pi*xi)*(1-2j*Pi*xi))
    #g[np.where(np.isnan(g))] = np.sqrt(3)*np.cos(np.pi*xi)/(3*np.pi*np.pi*xi*xi) - \
    #                 np.sin(np.pi*xi)/(np.sqrt(3)*np.pi*xi)
    #g[np.isnan(g)] = 0
    return g
"""

def g_eeGEN(xi, eta, **kwargs):
    D = kwargs['D']

    i = 1j
    Pi = np.pi

    g = np.exp(-i*Pi*D*(2*eta/np.sqrt(3) + xi)) * (\
        (np.sqrt(3)*eta - 3*xi) *
        (np.exp(i*Pi*D*np.sqrt(3)*eta) - np.exp(i*Pi*D*(4*eta/np.sqrt(3) + xi)))  + \
        (np.sqrt(3)*eta + 3*xi) * (np.exp(i*Pi*D*eta/np.sqrt(3)) - np.exp(i*Pi*D*xi)) ) \
        / (4*Pi*Pi*(eta*eta*eta - 3*eta*xi*xi))
    #g[np.where(g==np.nan)] = (np.exp(-1j*D*Pi)/(2*np.sqrt(3)*Pi*Pi*xi)) * \
    #               (-1 + 1j*D*xi + np.exp(1j*d*Pi*xi)*(1-2j*D*Pi*xi))
    #g[np.where(np.isnan(g))] = np.sqrt(3)*np.cos(np.pi*xi)/(3*np.pi*np.pi*xi*xi) - \
    #                 np.sin(np.pi*xi)/(np.sqrt(3)*np.pi*xi)
    #g[np.isnan(g)] = 0
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
    i = 1j
    Pi = np.pi
    g = (np.exp(-1j*Pi*xi)/(2*np.sqrt(3)*Pi*Pi*xi*xi) )* \
        (-1 + 1j*Pi*xi + np.exp(1j*Pi*xi)-2j*Pi*xi*np.exp(1J*Pi*xi))

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



# hex_ee(s=(201,201), c=None, D=0.0201) 
#   - first dark ring r=58 pixels, centered on 101.5,101.5
#     even though c = (100.5, 100.5
# Problem
#   - if s=(200,200) division by zero occurs along eta=0 axis
#   - need to find limiting value and fix nan's after the call 
    
def hex_ee(s=(201,201), c=None, D=0.0201):
    if c is None:
        c = (float(s[0])/2.0, float(s[1])/2.0)
    hex_complex =  np.fromfunction(g_ee, s, D=D, c=c, minus=False)  + \
                   np.fromfunction(g_ee, s, D=D, c=c, minus=True)
    if DEBUG:
        hr = hex_complex.real
        hi = hex_complex.imag
        print "hex real mean max min",  hr.mean(), hr.max(), hr.min()
        print "hex imag mean max min",  hi.mean(), hi.max(), hi.min()
    return np.abs(hex_complex)

def hex_eeAG(s=(121,121), c=None, d=0.80, lam=4.3e-6, pitch=mas2rad(65.6), verbose=False):
    if c is None:
        c = float(s[0])/2.0-0.5, float(s[1])/2.0 - 0.5
    if verbose: 
        print "Center:",c, s

    g = np.fromfunction(g_eeAG, s, d=d, c=c, lam=lam, pixel=pitch, minus=False)
    hex_complex = np.fromfunction(g_eeAG, s, d=d, c=c, lam=lam, pixel=pitch, minus=False) + \
              np.fromfunction(g_eeAG, s, d=d, c=c, lam=lam, pixel=pitch, minus=True)
    # There will be a strip of NaNs down the middle (eta-axis)
    (xnan, ynan)= np.where(np.isnan(hex_complex))
    # The "yval" will be the same for all points
    # loop over the xi values
    for index in range(len(xnan)):
        """
        Replace NaN strip with limiting behavior. Calls from function glimit.
        """
        hex_complex[xnan[index], ynan[index]] = glimit(xnan[index], xnan[index], d=d, \
                    c=c, lam=lam, pixel=pitch, minus=False) + \
                    glimit(xnan[index], xnan[index], d=d, c=c, \
                     lam=lam, pixel=pitch, minus=True)

    (xnan, ynan)= np.where(np.isnan(hex_complex))
    #print "wherenan", xnan, ynan

    for index in range(len(xnan)):
        """
        Replace NaN strip with limiting behavior. Calls from function glimit, which 
        doesn't give the right values at the moment for the origin.
        """
        hex_complex[xnan[index], ynan[index]] = centralpix(xnan[index], ynan[index], d=d, \
                    c=c, lam=lam, pixel=pitch, minus=False) + \
                    centralpix(xnan[index], xnan[index], d=d, c=c, \
                     lam=lam, pixel=pitch, minus=True)


    if DEBUG:
        hr = hex_complex.real
        hi = hex_complex.imag
        print "hex real mean max min",  hr.mean(), hr.max(), hr.min()
        print "hex imag mean max min",  hi.mean(), hi.max(), hi.min()
    return np.abs(hex_complex)

def test_hex_ee():
    fits.PrimaryHDU(data=hex_ee()).writeto(ODIR+"hex_ee.fits", clobber=True)
    fits.PrimaryHDU(data=hex_ee(s=(201,201), D=0.201)).writeto(ODIR+"hex_ee2.fits", clobber=True)

def test_hex_eeAG():
    pl.imshow(hex_eeAG(), interpolation='nearest')
    fits.PrimaryHDU(data=hex_eeAG()).writeto(ODIR+"data/Hexlimtest.fits", clobber=True)
    pl.show()

if __name__ == "__main__":


    if os.getenv('USER') == "anand":
        ODIR = "/Users/anand/PythonModules/testdata/"
    else:
        ODIR = "/Users/alexandragreenbaum/"
    os.chdir(ODIR)
    print "Current working directory: " + os.getcwd()

    test_hex_eeAG()
