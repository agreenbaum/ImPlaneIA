#! /usr/bin/env python

import numpy as np
import pyfits
import pylab as pl
import os
from scipy.misc import comb
from analyticnrm2 import mas2rad, rad2mas
from argparse import ArgumentParser

HOME = '/Users/agreenba/'

um = 1.0e-6

def dist(vec1, vec2):
	return float(np.sqrt( (vec1[0] - vec2[0])**2 + (vec2[1] - vec2[1])**2 ))

def baselines3(triangle):
	"""
	arrange u,v coordinates give a triplet of holes to be used in a single 
	closure phase calculation. 
	triangle has the form [ctr,coord]
	e.g. the y coordinate of the 3rd center is triangle[2,1]
	     the x coordinate of the 3rd center is triangle[2,0]
	"""
	D12 = dist(triangle[0,:], triangle[1,:])
	D23 = dist(triangle[1,:], triangle[2,:])
	D31 = dist(triangle[2,:], triangle[0,:])
	
	x1x2 = ( (triangle[0,0] - triangle[1,0]) , (triangle[0,1] - triangle[1,1]) )
	x2x3 = ( (triangle[1,0] - triangle[2,0]) , (triangle[1,1] - triangle[2,1]) )
	x3x1 = ( (triangle[2,0] - triangle[0,0]) , (triangle[2,1] - triangle[0,1]) )
	#print "in baselines:", x1x2, x2x3, x3x1
	return x1x2, x2x3, x3x1

def baselines4(quad):
	"""
	arrange u,v coordinates give a 4 holes to be used in a single 
	closure amplitude calculation. 
	quadruple has the form [ctr,coord]
	e.g. the y coordinate of the 3rd center is triangle[2,1]
	     the x coordinate of the 3rd center is triangle[2,0]
	"""

	D12 = dist(quad[0,:], quad[1,:])
	D23 = dist(quad[1,:], quad[2,:])
	D31 = dist(quad[2,:], quad[0,:])
	
	x1x2 = ( (quad[0,0] - quad[1,0]) , (quad[0,1] - quad[1,1]) )
	x2x3 = ( (quad[1,0] - quad[2,0]) , (quad[1,1] - quad[2,1]) )
	x3x4 = ( (quad[2,0] - quad[3,0]) , (quad[2,1] - quad[3,1]) )
	x4x1 = ( (quad[3,0] - quad[0,0]) , (quad[3,1] - quad[0,1]) )
	return x1x2, x2x3, x3x4, x4x1

def visphase(baseline, ratio, position):
	"""
	From David L.'s NRM code V(u,v) = {1 + r exp[-i2pi(alpha.u + delta.v)]}/(1+r)
	2011 Sparse Aperture Masking at the VLT

	r: flux ratio between sources
	alpha, delta: RA, DEC (radians)
	u, v: orthogonal spatial frequency vectors

	Returns: the resulting phase given a baseline, flux ratio, and separation.
	To do: figure out the scaling & coordinates for the baseline.
	"""
	alpha, delta = position
	u,v = baseline

	visibility = ( 1 + ratio * (np.exp((-1j*2*np.pi*(alpha*u +delta*v))) ) ) \
			/(1+ratio)
	phase = np.angle(visibility)
	return phase

def visamp(baseline, ratio, position):
	"""
	From David L.'s NRM code V(u,v) = {1 + r exp[-i2pi(alpha.u + delta.v)]}/(1+r)

	r: flux ratio between sources
	alpha, delta: RA, DEC (radians)
	u, v: orthogonal spatial frequency vectors

	Returns: the resulting amplitude given a baseline, flux ratio, and separation.
	To do: figure out the scaling & coordinates for the baseline.
	"""
	alpha, delta = position
	u,v = baseline

	visibility = ( 1 + ratio * (np.exp((-1j*2*np.pi*(alpha*u +delta*v))) ) ) \
			/(1+ratio)
	amplitude = np.absolute(visibility)
	return amplitude

def return_vis(ctrs, ratio, position,scaling):
	N = len(ctrs)
	phases = np.zeros(N*(N-1)/2)
	amps = phases.copy()
	n=0
	ctrs = scaling*ctrs
	for g in range(N-1):
		for q in range(N-g-1):
			phases[n+q] = visphase((ctrs[g,0]-ctrs[q+g+1,0], ctrs[g,1]-ctrs[q+g+1,1]),
						ratio,position)
			amps[n+q] = visamp((ctrs[g,0]-ctrs[q+g+1,0], ctrs[g,1]-ctrs[q+g+1,1]),
						ratio,position)
		n = n + q + 1
	return phases, amps

# Added Feb 16 2015: 
def return_vis_uv(uvs, ratio, position, scaling):
	"""
	1/lambda scaling
	"""
	nbl = len(uvs)
	phases = np.zeros(nbl)
	amps = phases.copy()
	uvs = scaling*uvs
	for g in range(nbl):
		phases[g] = visphase((uvs[g,0], uvs[g,1]),
						ratio,position)
		amps[g] = visamp((uvs[g,0], uvs[g,1]),
						ratio,position)
	return phases, amps


def return_cps(ctrs, ratio, position, scaling):
	N = len(ctrs)
	cps = np.zeros(int(comb(N,3)))
	n=0
	for k in range(N-2):
		for i in range(N-k-2):
			for j in range(N-k-i-2):
				side1, side2, side3 = baselines3(np.array([ctrs[k,:]*scaling,
									ctrs[i+k+1,:]*scaling,
									ctrs[j+i+k+2,:]*scaling]))
				cps[n+j] = visphase(side1, ratio, position) + \
					       visphase(side2, ratio, position) + \
					       visphase(side3, ratio, position)
			n= n+j+1
	return cps#,np.mean(cps), np.var(cps) * (comb(N,3)/((N*(N-1)/2) - 1 ))
