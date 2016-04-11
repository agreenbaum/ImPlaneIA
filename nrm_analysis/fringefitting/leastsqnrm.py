#! /usr/bin/env  python 
# Mathematica nb from Alex & Laurent

import numpy as np
import scipy.special
import numpy.linalg as linalg
import sys
import analyticnrm2 as analytic
import driverutils as utils
import hexee
from scipy.misc import comb

m = 1.0
mm = 1.0e-3 * m
um = 1.0e-6 * m

# 2analyticnrm.py creates image plane intensity from analytic function

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
	# meant to replace the center of Jinc function with its limit pi/4
	nanpos=np.where(np.isnan(array))
	array[nanpos]=np.pi/4
	return array

def primarybeam(kx, ky):
	# Calculates the envelope intensity for circular holes and monochromatic light
	" dimensionless argument r = sqrt(x^2 + y^2)"
	R = (primarybeam.d/primarybeam.lam) * primarybeam.pitch *  \
	    np.sqrt((kx-primarybeam.offx)*(kx-primarybeam.offx) + \
	    (ky-primarybeam.offy)*(ky-primarybeam.offy))
	pb = replacenan(scipy.special.jv(1, np.pi*R) / (2.0*R))

	#  Sept 2015 -- to fix coordinate confusion
	pb = pb.transpose()

	return pb * pb.conj()

def hexpb():
	pb = hexee.hex_eeAG(s=hexpb.size, c=(hexpb.offx, hexpb.offy), \
			      d=hexpb.d, lam=hexpb.lam, pitch=hexpb.pitch)

	#  Sept 2015 -- to fix coordinate confusion
	pb = pb.transpose()

	return pb * pb.conj()

def ffc(kx, ky):
	return 2*np.cos(2*np.pi*ffc.pitch*((kx - ffc.offx)*(ffc.ri[0] - ffc.rj[0]) + 
					   (ky - ffc.offy)*(ffc.ri[1] - ffc.rj[1]))/ffc.lam)

def ffs(kx, ky):
	return -2*np.sin(2*np.pi*ffs.pitch*((kx - ffs.offx)*(ffs.ri[0] - ffs.rj[0]) + 
					    (ky - ffs.offy)*(ffs.ri[1] - ffs.rj[1]))/ffs.lam)


def model_array(ctrs, lam, oversample, pitch, fov, d, centering ='PIXELCENTERED', shape ='circ', 
		verbose=False):
	
	if centering =='PIXELCORNER':
		off = np.array([0.0, 0.0])
	elif centering =='PIXELCENTERED':
		off = np.array([0.5, 0.5])
	else:
		off = centering

	if verbose:
		print "------------------"
		print " Model Parameters:"
		print "------------------"
		print "pitch: {0}, fov: {1}, oversampling: {2}, centers: {3}".format(pitch,
			fov, oversample, ctrs) + \
			" d: {0}, wavelength: {1}, shape: {2}".format(d, lam, shape) +\
			"\ncentering:{0}\n {1}".format(centering, off)

	
	# primary beam parameters:
	primarybeam.shape=shape
	primarybeam.lam = lam
	primarybeam.size = (oversample * fov, oversample * fov)
	primarybeam.offx = oversample*fov/2.0 - off[0] # in pixels
	primarybeam.offy = oversample*fov/2.0 - off[1]
	primarybeam.pitch = pitch/float(oversample)
	primarybeam.d = d

	hexpb.shape=shape
	hexpb.lam = lam
	hexpb.size = (oversample * fov, oversample * fov)
	hexpb.offx = oversample*fov/2.0 - off[0] # in pixels
	hexpb.offy = oversample*fov/2.0 - off[1]
	hexpb.pitch = pitch/float(oversample)
	hexpb.d = d

	# model fringe matrix parameters:
	ffc.N = len(ctrs) # number of holes
	ffc.lam = lam
	ffc.over = oversample
	ffc.pitch = pitch / float(oversample)
	ffc.size = (oversample * fov, oversample * fov)
	ffc.offx = oversample*fov / 2.0 - off[0]
	ffc.offy = oversample*fov / 2.0 - off[1]

	ffs.N = len(ctrs) # number of holes
	ffs.lam = lam
	ffs.over = oversample
	ffs.pitch = pitch / float(oversample)
	ffs.size = (oversample * fov, oversample * fov)
	ffs.offx = oversample*fov / 2.0 - off[0]
	ffs.offy = oversample*fov / 2.0 - off[1]
	
	alist = []
	for i in range(ffc.N - 1):
		for j in range(ffc.N - 1):
			if j + i + 1 < ffc.N:
				alist = np.append(alist, i)
				alist = np.append(alist, j + i + 1)
	alist = alist.reshape(len(alist)/2, 2)

	print ffc.size
	ffmodel = []
	ffmodel.append(ffc.N * np.ones(ffc.size))
	for q,r in enumerate(alist):
		# r[0] and r[1] are holes i and j, x-coord: 0, y-coord: 1
		ffc.ri = ctrs[r[0]]
		ffc.rj = ctrs[r[1]]
		ffs.ri = ctrs[r[0]]
		ffs.rj = ctrs[r[1]]

		# Sept 2015 -- added in transpose to fix coordinate confusion
		ffmodel.append( np.transpose(np.fromfunction(ffc, ffc.size)) )
		ffmodel.append( np.transpose(np.fromfunction(ffs, ffs.size)) )
	print "shape ffmodel", np.shape(ffmodel)

	if shape=='circ':
		return np.fromfunction(primarybeam,ffc.size), ffmodel
	elif shape=='hex':
		return hexpb(), ffmodel
	else:
		raise KeyError("Must provide a valid hole shape. Current supported shapes are" \
				" 'circ' and 'hex'.")

def scaling(img, photons):
	# img gives a perfect psf to count its total flux
	# photons is the desired number of photons (total flux in data)
	total = np.sum(img)
	print "total", total
	return photons / total

def matrix_operations(img, model, flux = None, verbose=False):
	# least squares matrix operations to solve A x = b, where A is the model, b is the data (image), and x is the coefficient vector we are solving for. In 2-D data x = inv(At.A).(At.b) 

	flatimg = img.reshape(np.shape(img)[0] * np.shape(img)[1])
	#print "img shape before:", flatimg.shape
	nanlist = np.where(np.isnan(flatimg))
	#print "NaNs:", np.shape(nanlist),nanlist
	flatimg = np.delete(flatimg, nanlist)
	#print "img shape after:", flatimg.shape
	if flux is not None:
		flatimg = flux * flatimg / flatimg.sum()
	#print "model shape:", np.shape(model)

	# A
	flatmodel_nan = model.reshape(np.shape(model)[0] * np.shape(model)[1], np.shape(model)[2])
	#flatmodel = model.reshape(np.shape(model)[0] * np.shape(model)[1], np.shape(model)[2])
	flatmodel = np.zeros((len(flatimg), np.shape(model)[2]))
	print "flat model dimensions ", np.shape(flatmodel)
	print "flat image dimensions ", np.shape(flatimg)
	for fringe in range(np.shape(model)[2]):
		flatmodel[:,fringe] = np.delete(flatmodel_nan[:,fringe], nanlist)
	# At (A transpose)
	flatmodeltransp = flatmodel.transpose()
	# At.A (makes square matrix)
	modelproduct = np.dot(flatmodeltransp, flatmodel)
	# At.b
	data_vector = np.dot(flatmodeltransp, flatimg)
	# inv(At.A)
	inverse = linalg.inv(modelproduct)
	cond = np.linalg.cond(inverse)

	x = np.dot(inverse, data_vector)
	res = np.dot(flatmodel, x) - flatimg
	naninsert = nanlist[0] - np.arange(len(nanlist[0]))
	res = np.insert(res, naninsert, np.nan)
	res = res.reshape(img.shape[0], img.shape[1])

	if verbose:
		print 'model flux', flux
		print 'data flux', flatimg.sum()
		print "flat model dimensions ", np.shape(flatmodel)
		print "model transpose dimensions ", np.shape(flatmodeltransp)
		print "flat image dimensions ", np.shape(flatimg)
		print "transpose * image data dimensions", np.shape(data_vector)
		print "flat img * transpose dimensions", np.shape(inverse)

	
	return x, res, cond

def weighted_operations(img, model, weights, verbose=False):
	# least squares matrix operations to solve A x = b, where A is the model, b is the data (image), and x is the coefficient vector we are solving for. In 2-D data x = inv(At.A).(At.b) 

	clist = weights.reshape(weights.shape[0]*weights.shape[1])**2
	flatimg = img.reshape(np.shape(img)[0] * np.shape(img)[1])
	nanlist = np.where(np.isnan(flatimg))
	flatimg = np.delete(flatimg, nanlist)
	clist = np.delete(clist, nanlist)
	# A
	flatmodel_nan = model.reshape(np.shape(model)[0] * np.shape(model)[1], np.shape(model)[2])
	#flatmodel = model.reshape(np.shape(model)[0] * np.shape(model)[1], np.shape(model)[2])
	flatmodel = np.zeros((len(flatimg), np.shape(model)[2]))
	for fringe in range(np.shape(model)[2]):
		flatmodel[:,fringe] = np.delete(flatmodel_nan[:,fringe], nanlist)
	# At (A transpose)
	flatmodeltransp = flatmodel.transpose()
	# At.C.A (makes square matrix)
	CdotA = flatmodel.copy()
	for i in range(flatmodel.shape[1]):
		CdotA[:,i] = clist * flatmodel[:,i]
	modelproduct = np.dot(flatmodeltransp, CdotA)
	# At.C.b
	Cdotb = clist * flatimg
	data_vector = np.dot(flatmodeltransp, Cdotb)
	# inv(At.C.A)
	inverse = linalg.inv(modelproduct)
	cond = np.linalg.cond(inverse)

	x = np.dot(inverse, data_vector)
	res = np.dot(flatmodel, x) - flatimg
	naninsert = nanlist[0] - np.arange(len(nanlist[0]))
	res = np.insert(res, naninsert, np.nan)
	res = res.reshape(img.shape[0], img.shape[1])

	if verbose:
		print "flat model dimensions ", np.shape(flatmodel)
		print "model transpose dimensions ", np.shape(flatmodeltransp)
		print "flat image dimensions ", np.shape(flatimg)
		print "transpose * image data dimensions", np.shape(data_vector)
		print "flat img * transpose dimensions", np.shape(inverse)

	
	return x, res,cond

def multiplyenv(env, fringeterms):
	# The envelope is size (fov, fov). This multiplies the envelope by each of the 43 slices in the fringe model
	full = np.ones((np.shape(fringeterms)[1], np.shape(fringeterms)[2], np.shape(fringeterms)[0]+1))
	for i in range(len(fringeterms)):
		full[:,:,i] = env * fringeterms[i]
	print "Total # fringe terms:", i
	return full

def deltapistons(pistons):
	# This function is used for comparison to calculate relative pistons from given pistons (only deltapistons are measured in the fit)
	N = len(pistons)
	# same alist as above to label holes
	alist = []
	for i in range(N - 1):
		for j in range(N - 1):
			if j + i + 1 < N:
				alist = np.append(alist, i)
				alist = np.append(alist, j + i + 1)
	alist = alist.reshape(len(alist)/2, 2)
	delta = np.zeros(len(alist))
	for q,r in enumerate(alist):
		delta[q] = pistons[r[0]] - pistons[r[1]]
	return delta

def sin2deltapistons(coeffs, verbose=False):
	# coefficients of sine terms mulitiplied by 2*pi
	delta = np.zeros((len(coeffs) -1)/2)
	for q in range((len(coeffs) - 1)/2):
		delta[q] = np.arcsin(coeffs[2*q+2]) / (np.pi*2.0)
	if verbose:
		print "shape coeffs", np.shape(coeffs)
		print "shape delta", np.shape(delta)

	return delta

def tan2visibilities(coeffs, verbose=False):
	"""
	Technically the fit measures phase AND amplitude, so to retrieve
	the phase we need to consider both sin and cos terms. Consider one fringe:
	A { cos(kx)cos(dphi) +  sin(kx)sin(dphi) } = 
	A(a cos(kx) + b sin(kx)), where a = cos(dphi) and b = sin(dphi)
	and A is the fringe amplitude, therefore coupling a and b
	In practice we measure A*a and A*b from the coefficients, so:
	Ab/Aa = b/a = tan(dphi)
	call a' = A*a and b' = A*b (we actually measure a', b')
	(A*sin(dphi))^2 + (A*cos(dphi)^2) = A^2 = a'^2 + b'^2

	Edit 10/2014: pistons now returned in units of radians!!
	"""
	# coefficients of sine terms mulitiplied by 2*pi
	delta = np.zeros((len(coeffs) -1)/2)
	amp = np.zeros((len(coeffs) -1)/2)
	for q in range((len(coeffs) - 1)/2):
		delta[q] = (np.arctan2(coeffs[2*q+2], coeffs[2*q+1])) 
		amp[q] = np.sqrt(coeffs[2*q+2]**2 + coeffs[2*q+1]**2)
	if verbose:
		print "shape coeffs", np.shape(coeffs)
		print "shape delta", np.shape(delta)

	# returns fringe amplitude & phase
	return amp, delta

def cos2deltapistons(coeffs, verbose=False):
	# coefficients of cosine terms (multiplied by 2*pi)
	delta = np.zeros((len(coeffs)-1)/2)
	for q in range((len(coeffs)-1)/2):
		if coeffs[2*q+2]<0:
			sgn = -1
		else:
			sgn = 1
		delta[q] = sgn * np.arccos(coeffs[2*q+1]) / (np.pi*2.0)

	if verbose:
		print "shape coeffs", np.shape(coeffs)
		print "shape delta", np.shape(delta)

	return delta

def fixeddeltapistons(coeffs, verbose=False):
	delta = np.zeros((len(coeffs) -1)/2)
	for q in range((len(coeffs) - 1)/2):
		delta[q] = np.arcsin((coeffs[2*q+1] + coeffs[2*q+2]) / 2) / (np.pi*2.0)
	if verbose:
		print "shape coeffs", np.shape(coeffs)
		print "shape delta", np.shape(delta)

	return delta	

def populate_antisymmphasearray(deltaps, N=7):
	fringephasearray = np.zeros((N,N))
	step=0
	n=N-1
	for h in range(n):
		"""
		fringephasearray[0,q+1:] = coeffs[0:6]
		fringephasearray[1,q+2:] = coeffs[6:11]
		fringephasearray[2,q+3:] = coeffs[11:15]
		fringephasearray[3,q+4:] = coeffs[15:18]
		fringephasearray[4,q+5:] = coeffs[18:20]
		fringephasearray[5,q+6:] = coeffs[20:]
		"""
		fringephasearray[h, h+1:] = deltaps[step:step+n]
		step= step+n
		n=n-1
	fringephasearray = fringephasearray - fringephasearray.T
	return fringephasearray

def populate_symmamparray(amps, N=7):
	fringeamparray = np.zeros((N,N))
	step=0
	n=N-1
	for h in range(n):
		fringeamparray[h,h+1:] = amps[step:step+n]
		step = step+n
		n=n-1
	fringeamparray = fringeamparray + fringeamparray.T
	return fringeamparray

def redundant_cps(deltaps, N = 7):
	fringephasearray = populate_antisymmphasearray(deltaps, N=N)
	cps = np.zeros(int(comb(N,3)))
	nn=0
	for kk in range(N-2):
		for ii in range(N-kk-2):
			for jj in range(N-kk-ii-2):
				cps[nn+jj] = fringephasearray[kk, ii+kk+1] \
					   + fringephasearray[ii+kk+1, jj+ii+kk+2] \
					   + fringephasearray[jj+ii+kk+2, kk]
			nn = nn+jj+1
	return cps

def closurephase(deltap, N=7):
	# N is number of holes in the mask
	# 7 and 10 holes available (JWST & GPI)

	# p is a triangular matrix set up to calculate closure phases
	if N == 7:
		p = np.array( [ deltap[:6], deltap[6:11], deltap[11:15], \
				deltap[15:18], deltap[18:20], deltap[20:] ] )
	elif N == 10:
		p = np.array( [ deltap[:9], deltap[9:17], deltap[17:24], \
				deltap[24:30], deltap[30:35], deltap[35:39], \
				deltap[39:42], deltap[42:44], deltap[44:] ] )
		
	else:
		print "invalid hole number"

	# calculates closure phases for general N-hole mask (with p-array set up properly above)
	cps = np.zeros((N - 1)*(N - 2)/2)
	for l1 in range(N - 2):
		for l2 in range(N - 2 - l1):
			cps[int(l1*((N + (N-3) -l1) / 2.0)) + l2] = \
				p[l1][0] + p[l1+1][l2] - p[l1][l2+1]
	return cps

def return_CAs(amps, N=7):
	fringeamparray = populate_symmamparray(amps, N=N)
	nn=0
	CAs = np.zeros(int(comb(N,4)))
	for ii in range(N-3):
		for jj in range(N-ii-3):
			for kk in range(N-jj-ii-3):
				for ll  in range(N-jj-ii-kk-3):
					CAs[nn+ll] = fringeamparray[ii,jj+ii+1] \
						   * fringeamparray[ll+ii+jj+kk+3,kk+jj+ii+2] \
			/ (fringeamparray[ii,kk+ii+jj+2]*fringeamparray[jj+ii+1,ll+ii+jj+kk+3])
				nn=nn+ll+1
	return CAs
	
def rebin(a = None, rc=(2,2), verbose=None):

	"""  
	anand@stsci.edu
	Perform simple-minded flux-conserving binning... clip trailing
	size mismatch: eg a 10x3 array binned by 3 results in a 3x1 array
	"""

	r, c = rc

	R = a.shape[0]
	C = a.shape[1]

	nr = int(R / r)
	nc = int(C / c)

	b = a[0:nr, 0:nc].copy()
	b = b * 0

	for ri in range(0, nr):
		Rlo = ri * r
		if verbose:
			print "row loop"
		for ci in range(0, nc):
			Clo = ci * c
			b[ri, ci] = a[Rlo:Rlo+r, Clo:Clo+c].sum()
			if verbose:
				print "    [%d:%d, %d:%d]" % (Rlo,Rlo+r, Clo,Clo+c),
				print "%4.0f"  %   a[Rlo:Rlo+r, Clo:Clo+c].sum()
	return b

if __name__ == "__main__":

	import pylab as pl
	import NRM_mask_definitions as NRM
	# np.set_printoptions(precision=2)

	JWlam = 4.3e-6
	JWd = 0.80
	JWD = 6.5
	JWpixel = mas2rad(65) # radians
	fov_pix = 121 #  pixels aka nb's 't' - number of detector pixels (NaN if even number)
	oversample = 3 # sample this many times per detector pixel (linearly)

	phi_nb = np.array( [0.028838669455909766, -0.061516214504502634, 0.12390958557781348, \
				-0.020389361461019516, 0.016557347248600723, -0.03960017912525625, \
				-0.04779984719154552] )


	JWctrs = np.array( [	[ 0.00000000,  -2.640000],
				[-2.2863100 ,  0.0000000],
				[ 2.2863100 , -1.3200001],
				[-2.2863100 ,  1.3200001],
				[-1.1431500 ,  1.9800000],
				[ 2.2863100 ,  1.3200001],
				[ 1.1431500 ,  1.9800000]    ] )

	test = deltapistons(phi_nb)
	cptest = closurephase(test)
	print "These should be zero:", cptest

	#draws out fringe model and envelope in fov
	pb, modelfringes = model_array(JWctrs, JWlam, oversample, JWpixel, fov_pix, JWd)
	modelmat = multiplyenv(pb, modelfringes)
	#rebin model
	modelmat_niriss = np.zeros((fov_pix, fov_pix, np.shape(modelmat)[2]))
	for sl in range(np.shape(modelmat)[2]):
		modelmat_niriss[:,:,sl] = utils.rebin(modelmat[:,:,sl], (3,3))

	print np.shape(modelmat_niriss)
	pl.figure(1)
	pl.imshow(np.sum(modelmat_niriss, axis = 2), interpolation = 'nearest', cmap = 'hot')
	pl.figure(2)
	pl.imshow(np.sum(modelmat, axis = 2), interpolation = 'nearest', cmap = 'hot')
	#pl.imshow(pb, interpolation = 'nearest', cmap = 'hot')

	# "data" image taken from function in 2analyticnrm.py
	data = analytic.PSF(JWpixel, fov_pix, oversample, JWctrs, JWd, JWlam, phi_nb)
	data_niriss = rebin(data, (oversample, oversample))

	#solutions = matrix_operations(data, modelmat)
	solutions_niriss, resid_niriss = matrix_operations(data_niriss, modelmat_niriss)
	#delts = sin2deltapistons(solutions)
	delts_niriss = sin2deltapistons(solutions_niriss)
	#print "cos & sin coefficients: ", solutions
	print "cos & sin coefficients at niriss scale: ", solutions_niriss
	print "original input relative pistons", deltapistons(phi_nb)
	#print "calculated relative pistons", delts
	print "calculated relative pistons rebinned", delts_niriss
	print "closure phases from calculated pistons", closurephase(delts_niriss)
	#print "residuals: ", res

	sys.exit()

	N = 10
	OVER = 3
	GPIpixel = analytic.mas2rad(14.3)
	phi_0 = np.zeros((N))
	gpi_d, gpi_ctrs = NRM.gpi_g10s40_asmanufactured(0.64/mm)
	gpi_ctrs = np.array(gpi_ctrs)
	nrm = NRM.NRM_mask_definitions('gpi_g10s40')
	date = '2013_Feb27'
	slc = 11
	Hwl = nrm.get_scale(band='Hband', date = date, fldr = '/Users/alexandragreenbaum/data/GPIPSF/')
	wlH = Hwl[slc] * um
	fov = 221

	params = [GPIpixel, gpi_ctrs, gpi_d, phi_0]

	pb, modelfringes = model_array(rotatevectors(flip(gpi_ctrs), 0.0), wlH, OVER, GPIpixel, fov, gpi_d)
	modelmat = multiplyenv(pb, modelfringes)
	#rebin model
	modelmat_niriss = np.zeros((fov, fov, np.shape(modelmat)[2]))
	for sl in range(np.shape(modelmat)[2]):
		modelmat_niriss[:,:,sl] = utils.rebin(modelmat[:,:,sl], (3,3))
	data = analytic.PSF(GPIpixel, fov, OVER, rotatevectors(flip(gpi_ctrs), 0.0), gpi_d, wlH, phi_0)
	data_rebin = utils.rebin(data, (OVER,OVER))

	#solutions = matrix_operations(data, modelmat)
	solutions_rebin, resid_rebin = matrix_operations(data_rebin, modelmat_niriss)
	#delts = sin2deltapistons(solutions)
	delts_rebin = sin2deltapistons(solutions_rebin)
	#print "cos & sin coefficients: ", solutions
	print "cos & sin coefficients at niriss scale: ", solutions_rebin
	#print "calculated relative pistons", delts
	print "calculated relative pistons rebinned", delts_rebin
	print "closure phases from calculated pistons", closurephase(delts_rebin)

	print params

	sys.exit()


	#draws out fringe model and envelope in fov
	pb, modelfringes = model_array(JWctrs, JWlam, oversample, JWpixel, fov_pix, JWd)
	modelmat = multiplyenv(pb, modelfringes)
	#rebin model
	modelmat_niriss = np.zeros((fov_pix, fov_pix, np.shape(modelmat)[2]))
	for sl in range(np.shape(modelmat)[2]):
		modelmat_niriss[:,:,sl] = rebin(modelmat[:,:,sl], (3,3))

	print np.shape(modelmat_niriss)
	pl.figure(1)
	pl.imshow(np.sum(modelmat_niriss, axis = 2), interpolation = 'nearest', cmap = 'hot')
	pl.figure(2)
	pl.imshow(np.sum(modelmat, axis = 2), interpolation = 'nearest', cmap = 'hot')
	#pl.imshow(pb, interpolation = 'nearest', cmap = 'hot')

	# "data" image taken from function in 2analyticnrm.py
	data = analytic.PSF(JWpixel, fov_pix, oversample, JWctrs, JWd, JWlam, phi_nb)
	data_niriss = rebin(data, (oversample, oversample))

	#solutions = matrix_operations(data, modelmat)
	solutions_niriss, resid_niriss = matrix_operations(data_niriss, modelmat_niriss)
	#delts = sin2deltapistons(solutions)
	delts_niriss = sin2deltapistons(solutions_niriss)
	#print "cos & sin coefficients: ", solutions
	print "cos & sin coefficients at niriss scale: ", solutions_niriss
	print "original input relative pistons", deltapistons(phi_nb)
	#print "calculated relative pistons", delts
	print "calculated relative pistons rebinned", delts_niriss
	print "closure phases from calculated pistons", closurephase(delts_niriss)
	#print "residuals: ", res

	pl.show()
