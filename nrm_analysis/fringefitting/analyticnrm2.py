#! /usr/bin/env  python 
# Heritage mathematia nb from Alex & Laurent
# Python by Alex Greenbaum & Anand Sivaramakrishnan Jan 2013
# updated May 2013 to include hexagonal envelope

import numpy as np
import scipy.special
import sys
from nrm_analysis.misctools import utils # April 2016 trying to get imports right
import hexee

def flip(holearray):
	fliparray= holearray.copy()
	fliparray[:,1] = -1*holearray[:,1]
	return fliparray

def rotatevectors(vectors, thetarad):
	"""
	vectors is a list of vectors - e.g. nrm hole  centers
	positive x decreases under slight rotation
	positive y increases under slight rotation
	"""
	c, s = (np.cos(thetarad), np.sin(thetarad))
	ctrs_rotated = []
	for vector in vectors:
		ctrs_rotated.append([c*vector[0] - s*vector[1], 
		                     s*vector[0] + c*vector[1]])
	return np.array(ctrs_rotated)

def mas2rad(mas):
	rad = mas*(10**(-3)) / (3600*180/np.pi)
	return rad

def replacenan(array):
	nanpos=np.where(np.isnan(array))
	array[nanpos]=np.pi/4
	return array

def rad2mas(rad):
	mas = rad * (3600*180/np.pi) * 10**3
	return mas

def Jinc(x, y):
	" dimensionless argument r = sqrt(x^2 + y^2)"
	R = (Jinc.d/Jinc.lam) * Jinc.pitch *  \
	    np.sqrt((x-Jinc.offx)*(x-Jinc.offx) + (y-Jinc.offy)*(y-Jinc.offy))
	return replacenan(scipy.special.jv(1, np.pi*R) / (2.0*R))

def Hex(x,y, **kwargs):
	c = kwargs['c']
	pitch = kwargs['pitch']
	d = kwargs['d']
	lam = kwargs['lam']
	
	xi = (x-c[0]) #(d/lam) * pitch * (x - c[0])
	eta = (y-c[1]) #(d/lam) * pitch* (y - c[1])

	myhex = hexee.g_eeGEN(xi, eta, D=(d*pitch/lam)) \
	       + hexee.g_eeGEN(-xi, eta, D=(d*pitch/lam))
	return myhex

def hex_correction(x, y, **kwargs):
	which = kwargs['which']
	c = kwargs['c']
	pitch = kwargs['pitch']
	d = kwargs['d']
	lam = kwargs['lam']
	xi = (x-c[0]) #(d/lam) * pitch * (x - c[0])
	eta = (y-c[1]) #(d/lam) * pitch* (y - c[1])

	if which=='ETA':
		return hexee.glimit(xi, eta, c=c, d=d, lam=lam, pixel=pitch, minus=False) \
			+ hexee.glimit(xi, eta, c=c, d=d, lam=lam, pixel=pitch, minus=True)
	if which == 'XI':
		return np.sqrt(3)*d*d / 2.0

def phasor(kx, ky, hx, hy, lam, phi, pitch, tilt=1):
	# %%%%%%%%%%%%phi is in units of waves%%%%%%%%%%%%%% -pre July 2014
	# -------EDIT-------- changes phi to units of m -- way more physical for
	# 		      broadband simulations!!
	# kx ky image plane coords in units of of sampling pitch (oversampled, or not)
	# hx, hy hole centers in meters
	# pitch is sampling pitch in radians in image plane
	# ===========================================
	# k in units of "radians" hx/lam in units of "waves," requiring the 2pi
	# Example calculation -- JWST longest baseline ~6m
	# 			 Nyquist sampled for 64mas at 4um
	# hx/lam = 1.5e6
	# for one full cycle, when does 2pi k hx/lam = 2pi?
	# k = lam/hx = .66e-6 rad x ~2e5 = ~1.3e-1 as = 2 x ~65 mas, which is Nyquist
	# The 2pi needs to be there! That also means phi/lam is in waves
	# ===========================================
	return np.exp(-2*np.pi*1j*((pitch*kx*hx + pitch*ky*hy)/lam + (phi /lam) ))

def interf(kx, ky):
	interference = 0j
	for hole, ctr in enumerate(interf.ctrs):
		interference = phasor((kx-interf.offx), (ky-interf.offy),
		                ctr[0], ctr[1],
				interf.lam, interf.phi[hole], interf.pitch) + interference
	return interference

def ASF(pixel, fov, oversample, ctrs, d, lam, phi, centering = (0.5, 0.5), verbose=False):

	print "centering ", centering

	if centering is 'PIXELCENTERED':
		off_x = 0.5
		off_y = 0.5
	elif centering is 'PIXELCORNER':
		off_x = 0.0
		off_y = 0.0
	else:
		off_x, off_y = centering

	if verbose:
		print "ASF centering ", centering
		print "ASF offsets ", off_x, off_y
		print "total x,y cent ", oversample*fov/2.0 - off_x,\
			oversample*fov/2.0 - off_y

	# Jinc parameters
	Jinc.lam = lam
	Jinc.offx = oversample*fov/2.0 - off_x # in pixels
	Jinc.offy = oversample*fov/2.0 - off_y
	Jinc.pitch = pixel/float(oversample)
	Jinc.d = d

	primarybeam = np.fromfunction(Jinc, (int((oversample*fov)), int((oversample*fov))))

	# Sept 2015 -- to fix fromfunc coordinate confusion
	primarybeam = primarybeam.transpose()

	# interference terms' parameters
	interf.lam = lam
	interf.offx = oversample*fov/2.0 - off_x # in pixels
	interf.offy = oversample*fov/2.0 - off_y
	interf.pitch = pixel/float(oversample)
	interf.ctrs = ctrs
	interf.phi = phi

	fringing = np.fromfunction(interf,  (int((oversample*fov)), int((oversample*fov))))

	# Sept 2015 -- to fix fromfunc coordinate confusion
	fringing = fringing.transpose()

	return primarybeam*fringing

def ASFfringe(pixel, fov, oversample, ctrs, d, lam, phi, centering = (0.5, 0.5), verbose=False):

	print "centering ", centering

	if centering is 'PIXELCENTERED':
		off_x = 0.5
		off_y = 0.5
	elif centering is 'PIXELCORNER':
		off_x = 0.0
		off_y = 0.0
	else:
		off_x, off_y = centering

	if verbose:
		print "ASF centering ", centering
		print "ASF offsets ", off_x, off_y

	# Jinc parameters
	Jinc.lam = lam
	Jinc.offx = oversample*fov/2.0 - off_x # in pixels
	Jinc.offy = oversample*fov/2.0 - off_y
	Jinc.pitch = pixel/float(oversample)
	Jinc.d = d

	primarybeam = np.fromfunction(Jinc, (int((oversample*fov)), int((oversample*fov))))

	# Sept 2015 -- to fix fromfunc coordinate confusion
	primarybeam = primarybeam.transpose()

	# interference terms' parameters
	interf.lam = lam
	interf.offx = oversample*fov/2.0 - off_x # in pixels
	interf.offy = oversample*fov/2.0 - off_y
	interf.pitch = pixel/float(oversample)
	interf.ctrs = ctrs
	interf.phi = phi

	fringing = np.fromfunction(interf,  (int((oversample*fov)), int((oversample*fov))))

	# Sept 2015 -- to fix fromfunc coordinate confusion
	fringing = fringing.transpose()

	return fringing

def ASFhex(pixel, fov, oversample, ctrs, d, lam, phi, centering = 'PIXELCENTERED'):

	print "centering ", centering

	exp =0

	if centering is 'PIXELCENTERED':
		off_x = 0.5 + exp
		off_y = 0.5 + exp
	elif centering is 'PIXELCORNER':
		off_x = 0.0 + exp
		off_y = 0.0 + exp
	else:
		off_x, off_y = centering
	"""
	print "ASF centering ", centering
	print "ASF offsets ", off_x, off_y
	"""

	#Hex kwargs
	offx = (float(oversample*fov)/2.0) - off_x # in pixels
	offy = (float(oversample*fov)/2.0) - off_y
	print "offx, offy:", offx, offy 

	pitch = pixel/float(oversample)
	d = d

	# interference terms' parameters
	interf.lam = lam
	interf.offx = (oversample*fov)/2.0 - off_x # in pixels
	interf.offy = (oversample*fov)/2.0 - off_y
	interf.pitch = pixel/float(oversample)
	interf.ctrs = ctrs
	interf.phi = phi


	primarybeam = hexee.hex_eeAG(s=(oversample*fov,oversample*fov), 
					c=(offx,offy), d=d, lam=lam, pitch=pitch)

	# Sept 2015 -- to fix fromfunc coordinate confusion
    # July 2016, commented out transpose to be consistent w/simulation orientation
	#primarybeam = primarybeam.transpose()

	fringing = np.fromfunction(interf,  (int((oversample*fov)), int((oversample*fov))))

	# Sept 2015 -- to fix fromfunc coordinate confusion
	fringing = fringing.transpose()

	return primarybeam *fringing

def PSF(pixel, fov, oversample, ctrs, d, lam, phi, centering = 'PIXELCENTERED', shape = 'circ',
	verbose=False):
	"pixel/rad fov/pixels oversample ctrs/m d/m, lam/m, phi/m"

	if shape == 'circ':
		asf = ASF(pixel, fov, oversample, ctrs, d, lam, phi, centering)
	elif shape == 'hex':
		asf = ASFhex(pixel, fov, oversample, ctrs, d, lam, phi, centering)
	elif shape == 'fringe':
		asf = ASFfringe(pixel, fov, oversample, ctrs, d, lam, phi, centering)
	else:
		raise ValueError("pupil shape %s not supported" % shape)
	if verbose:
		print "-----------------"
		print " PSF Parameters:"
		print "-----------------"
		print "pitch: {0}, fov: {1}, oversampling: {2}, centers: {3}".format(pixel,
			fov, oversample, ctrs) + \
			"d: {0}, wavelength: {1}, pistons: {2}, shape: {3}".format(d, lam, 
			phi, shape)

	PSF_ = asf*asf.conj()
	return PSF_.real

def average_pixels(posx, posy):
	pixel_positions = np.zeros((match_niriss.fov * match_niriss.over, match_niriss.fov * match_niriss.over,))
	pixel_positions[match_niriss.over*posx:match_niriss.over*posx + match_niriss.over, \
			match_niriss.over*posy:match_niriss.over*posy + match_niriss.over] =  1
	pixelavg = np.sum(pixel_positions * match_niriss.img) / (match_niriss.over**2)
	return pixelavg

def match_niriss(img, fov, oversample):
	match_niriss.fov = fov
	match_niriss.over = oversample
	match_niriss.img = img
	size = match_niriss.fov, match_niriss.fov

	# I would like to use numpy's fromfunction (for average_pixels) here, but it doesn't seem to work.
	# The error I get is "invalid slice"
	niriss_scaled = np.zeros((fov,fov))
	for i in range(match_niriss.fov):
		for j in range(match_niriss.fov):
			niriss_scaled[i,j] = average_pixels(i,j)
	return niriss_scaled

if __name__ == "__main__":

	import pylab as pl
	from astropy.io import fits 
	np.set_printoptions(precision=2)

	print 'script'
	#print phasor(1+0j,1+0j,1+0j,1+0j,1+0j,0j)

	JWlam = 4.3e-6
	JWd = 0.80
	JWD = 6.5
	JWpixel = mas2rad(65) # radians
	fov_pix = 123 #  pixels aka nb's 't' - number of detector pixels (NaN if even number)
	oversample = 3 # sample this many times per detector pixel (linearly)

	# in meters below:
	JWctrs = np.array( [[ 0.0, -3.25],
				        [ 0.0, 3.25]    ] )
	JWctrs = np.array( [	[ 0.00000000,  -2.640000],
				[-2.2863100 ,  0.0000000],
				[ 2.2863100 , -1.3200001],
				[-2.2863100 ,  1.3200001],
				[-1.1431500 ,  1.9800000],
				[ 2.2863100 ,  1.3200001],
				[ 1.1431500 ,  1.9800000]    ] )
	phi_zero = np.zeros(np.shape(JWctrs)[0])
	phi_nb = np.array( [0.028838669455909766, -0.061516214504502634, 0.12390958557781348, \
				-0.020389361461019516, 0.016557347248600723, -0.03960017912525625, \
				-0.04779984719154552] )
	# make phi_nb into m, defined by these angular sizes at band center of F430M
	phi_nb = phi_nb*4.3e-6

	print "lambda/Tel_D (in arcsec)"
	print (JWlam/JWD)* (3600*180/np.pi)
	print "pixel scale (radians)"
	print JWpixel
	print " # pixels per primary beam resolution element (lamda/hole_d)"
	print (JWlam/JWd) / JWpixel
	print " # pixels per telescope resolution element (lamda/hole_D)"
	print (JWlam/JWD) / JWpixel

	center = fov_pix/2.0 
	print "center", center
	perfectpsf = PSF(JWpixel, fov_pix, oversample, JWctrs, JWd, JWlam, phi_zero)
	binnedpsf = utils.rebin(perfectpsf, (oversample, oversample))

	perfectpsf_corner = PSF(JWpixel, fov_pix, oversample, JWctrs, JWd, JWlam, phi_zero, centering = (0, 0))
	binnedpsf_corner = utils.rebin(perfectpsf_corner, (oversample, oversample))

	print "array size", np.shape(perfectpsf)
	print "max pos", np.where(perfectpsf == np.max(perfectpsf))
	print "max", perfectpsf.max()
	print perfectpsf
	pl.figure(1)
	pl.imshow(np.sqrt(perfectpsf), interpolation='nearest',  cmap = 'hot')
	pl.figure(2)
	pl.imshow(np.sqrt(binnedpsf), interpolation='nearest', cmap = 'hot')
	pl.figure(3)
	pl.imshow(np.sqrt(perfectpsf_corner), interpolation='nearest',  cmap = 'hot')
	pl.figure(4)
	pl.imshow(np.sqrt(binnedpsf_corner), interpolation='nearest', cmap = 'hot')

	#pl.show()

	two_slice = np.zeros((2, np.shape(binnedpsf)[0], np.shape(binnedpsf)[1]))
	two_slice[0] = binnedpsf
	two_slice[1] = binnedpsf_corner
	hdu = fits.PrimaryHDU()
	hdu.data = two_slice
	hdu.header.update("SLICE0", 0.5,"offset for centering")
	hdu.header.update("SLICE1", 2.0,"offset for centering")
	hdu.writeto("/Users/alexandragreenbaum/data/two_centerings.fits", clobber = True)
