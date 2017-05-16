#! /usr/bin/env python
import numpy as np
import numpy.fft as fft
from astropy.io import fits
import os, sys
import cPickle as pickle
from scipy.misc import comb

m_ = 1.0
mm_ =  m_/1000.0
um_ = mm_/1000.0
nm_ = um_/1000.0


def mas2rad(mas):
	rad = mas*(10**(-3)) / (3600*180/np.pi)
	return rad

def rad2mas(rad):
	mas = rad * (3600*180/np.pi) * 10**3
	return mas

def makedisk(N, R, ctr=(0,0), array=None):
	
	if N%2 == 1:
		M = (N-1)/2
		xx = np.linspace(-M-ctr[0],M-ctr[0],N)
		yy = np.linspace(-M-ctr[1],M-ctr[1],N)
	if N%2 == 0:
		M = N/2
		xx = np.linspace(-M-ctr[0],M-ctr[0]-1,N)
		yy = np.linspace(-M-ctr[1],M-ctr[1]-1,N)
	(x,y) = np.meshgrid(xx, yy.T)
	r = np.sqrt((x**2)+(y**2))
	array = np.zeros((N,N))
	array[r<R] = 1
	return array

def makehex(N, s, ctr=(0,0)):
	"""
	A. Greenbaum Sept 2015 - probably plenty of routines like this but I couldn't find
	one quickly.
	makes a hexagon with side length s in array size s at given rotation in degrees
	and centered at ctr coordinates. 
	0 rotation is flat side down
	  ---   y
	/     \ ^ 
	\     / |
	  ---   -> x
	"""
	array = np.zeros((N, N))
	if N%2 == 1:
		M = (N-1)/2
		xx = np.linspace(-M-ctr[0],M-ctr[0],N)
		yy = np.linspace(-M-ctr[1],M-ctr[1],N)
	if N%2 == 0:
		M = N/2
		xx = np.linspace(-M-ctr[0],M-ctr[0]-1,N)
		yy = np.linspace(-M-ctr[1],M-ctr[1]-1,N)
	(x,y) = np.meshgrid(xx, yy.T)
	h = np.zeros((N,N))
	d = np.sqrt(3)

	array[(y<(d*s/2.))*(y>(-d*s/2.))*\
	  (y>(d*x)-(d*s))*\
	  (y<(d*x)+(d*s))*\
	  (y<-(d*x)+(d*s))*\
	  (y>-(d*x)-(d*s))] = 1
	return array


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

def lambdasteps(lam, percent, steps=4):
	# not really percent, just fractional bandwidth
	frac = percent/2.0
	steps = steps/ 2.0
	# add some very small number to the end to include the last number.
	lambdalist = np.arange( -1 *  frac * lam + lam, frac*lam + lam + 10e-10, frac*lam/steps)
	return lambdalist

def tophatfilter(lam_c, frac_width, npoints=10):
	wllist = lambdasteps(lam_c, frac_width, steps=npoints)
	filt = []
	for ii in range(len(wllist)):
		filt.append(np.array([1.0, wllist[ii]]))
	return filt

def crosscorrelatePSFs(a, A, ov, verbose=False):
    """
    Assume that A is oversampled and padded by one pixel each side
    """
    print "\nInfo: performing cross correlations\n"
    print "\ta.shape is ", a.shape
    print "\tA.shape is ", A.shape
    #print "\tov is", ov

    p,q = a.shape
    padshape = (2*p, 2*q)  # pad the arrays to be correlated to avoid 
                           # 'aliasing bleed' from the edges
    cormat = np.zeros((2*ov, 2*ov))


    for x in range(2*ov):
        if verbose: print x, ":", 
        for y in range(2*ov):
            binA = krebin(A[x:x+p*ov, y:y+q*ov], a.shape)
            #print "\tbinA.shape is ", binA.shape  #DT 12/05/2014, binA.shape is  (5, 5)
            apad, binApad = (np.zeros(padshape), np.zeros(padshape))
            apad[:p,:q] = a         #shape (10,10) ,data is in the bottom left 5x5 corner 
            binApad[:p,:q] = binA   #shape (10,10) ,35x35 slice of perfect PSF 
                                      #binned down to 5x5 is in the bottom left corner
            cormat[x,y] = np.max(rcrosscorrelate(apad, binApad, verbose=False))
                       #shape of utils.rcrosscorrelate(apad, binApad, verbose=False is 10x10
            if verbose: print "%.3f" % cormat[x,y],
        if verbose: print "\n"
    return cormat

"""
def rebin(a, shape): # Klaus P's fastrebin from web
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)
"""

def centerit(img, r=img.shape[0]/2):
	print 'Before cropping:', img.shape
	lo = 140-40
	hi = 140+40
	ann = makedisk(img.shape[0], 11)
	#peakx, peaky = np.where(img==np.ma.masked_invalid(img[lo:hi,lo:hi]).max())
	peakx, peaky = np.where(img==np.ma.masked_invalid(img[ann==1]).max())
	#print 'peaking on: ',np.ma.masked_invalid(img[lo:hi,lo:hi]).max()
	print 'peaking on: ',np.ma.masked_invalid(img[ann==1]).max()
	print 'peak x,y:', peakx,peaky
	cropped = img[int(peakx-r):int(peakx+r+1),int(peaky-r):int(peaky+r+1)]
	print 'Cropped image shape:',cropped.shape
	print 'value at center:', cropped[r,r]
	print np.where(cropped == cropped.max())
	#pl.imshow(cropped, interpolation='nearest', cmap='bone')
	#pl.show()
	return cropped

def deNaN(s, datain):
	## Get rid of NaN values with nearest neighbor median
	fov=datain.shape[0]
	a2 = np.zeros((2*fov, 2*fov))
	print "FOV:", fov, "selection shape:", (fov//2,3*fov//2,fov//2,3*fov/2)
	a2[fov//2:fov+fov//2, fov//2 : fov+fov//2 ] = datain
	xnan, ynan = np.where(np.isnan(a2))
	for qq in range(len(a2[np.where(np.isnan(a2))])):
		a2[xnan[qq], ynan[qq]] = neighbor_median((xnan[qq],ynan[qq]), s, a2)
	return a2[fov//2:fov +fov//2, fov//2: fov+fov//2]


def neighbor_median(ctr, s, a2):
	# take the median of nearest neighbors within box side s
	#med = np.median(np.ma.masked_invalid(a2[ctr[0]-s:ctr[0]+s+1,
	#print a2[ctr[0]-s:ctr[0]+s+1, ctr[1]-s:ctr[1]+s+1]
	atmp = a2[ctr[0]-s:ctr[0]+s+1, ctr[1]-s:ctr[1]+s+1]
	med = np.median(atmp[np.isnan(atmp)==False])
	#print med


def get_fits_filter(fitsheader, ):
	wavestring = "WAVE"
	weightstring = "WGHT"
	filterlist = []
	print fitsheader[:]
	j =0
	for j in range(len(fitsheader)):
		if wavestring+str(j) in fitsheader:
			wght = fitsheader[weightstring+str(j)]
			wavl = fitsheader[wavestring+str(j)]  
			print "wave", wavl
			filterlist.append(np.array([wght,wavl]))
	print filterlist
	return filterlist


# From webbpsf:
def specFromSpectralType(sptype, return_list=False):
	"""Get Pysynphot Spectrum object from a spectral type string.

	"""
	lookuptable = {
	"O3V":   (50000, 0.0, 5.0),
	"O5V":   (45000, 0.0, 5.0),
	"O6V":   (40000, 0.0, 4.5),
	"O8V":   (35000, 0.0, 4.0),
	"O5I":   (40000, 0.0, 4.5),
	"O6I":   (40000, 0.0, 4.5),
	"O8I":   (34000, 0.0, 4.0),
	"B0V":   (30000, 0.0, 4.0),
	"B3V":   (19000, 0.0, 4.0),
	"B5V":   (15000, 0.0, 4.0),
	"B8V":   (12000, 0.0, 4.0),
	"B0III": (29000, 0.0, 3.5),
	"B5III": (15000, 0.0, 3.5),
	"B0I":   (26000, 0.0, 3.0),
	"B5I":   (14000, 0.0, 2.5),
	"A0V":   (9500, 0.0, 4.0),
	"A1V":   (9200, 0.0, 4.1),
	"A3V":   (8250, 0.0, 4.2),
	"A2V":   (8840, 0.0, 4.2),
	"A4V":   (8270, 0.0, 4.3),
	"A5V":   (8250, 0.0, 4.5),
	"A6V":   (8000, 0.0, 4.6),
	"A7V":   (7800, 0.0, 4.7),
	"A8V":   (7500, 0.0, 4.7),
	"A9V":   (7440, 0.0, 4.7),
	"A0I":   (9750, 0.0, 2.0),
	"A5I":   (8500, 0.0, 2.0),
	"F0V":   (7250, 0.0, 4.5),
	"F5V":   (6500, 0.0, 4.5),
	"F6III": (6500, 0.0, 5.0),
	"F0I":   (7750, 0.0, 2.0),
	"F5I":   (7000, 0.0, 1.5),
	"G0V":   (6000, 0.0, 4.5),
	"G5V":   (5750, 0.0, 4.5),
	"G0III": (5750, 0.0, 3.0),
	"G5III": (5250, 0.0, 2.5),
	"G0I":   (5500, 0.0, 1.5),
	"G5I":   (4750, 0.0, 1.0),
	"K0V":   (5250, 0.0, 4.5),
	"K2V":   (4750, 0.0, 4.5),
	"K4V":   (4500, 0.0, 4.5),
	"K5V":   (4250, 0.0, 4.5),
	"K7V":   (4000, 0.0, 4.5),
	"K0III": (4750, 0.0, 2.0),
	"K5III": (4000, 0.0, 1.5),
	"K0I":   (4500, 0.0, 1.0),
	"K5I":   (3750, 0.0, 0.5),
	"M0V":   (3750, 0.0, 4.5),
	"M1V":   (3680, 0.0, 4.5),
	"M2V":   (3500, 0.0, 4.64),
	"M3V":   (3500, 0.0, 4.7),
	"M4V":   (3500, 0.0, 4.8),
	"M5V":   (3500, 0.0, 4.94),
	"M6V":   (3500, 0.0, 5.0),
	"M0III": (3750, 0.0, 1.5),
	"M0I":   (3750, 0.0, 0.0),
	"M2I":   (3500, 0.0, 0.0)}


	if return_list:
		sptype_list = lookuptable.keys()
		def sort_sptype(typestr):
			letter = typestr[0]
			lettervals = {'O':0, 'B': 10, 'A': 20,'F': 30, 'G':40, 'K': 50, 'M':60}
			value = lettervals[letter]*1.0
			value += int(typestr[1])
			if "III" in typestr: value += .3
			elif "I" in typestr: value += .1
			elif "V" in typestr: value += .5
			return value
		sptype_list.sort(key=sort_sptype)
		sptype_list.insert(0,"Flat spectrum in F_nu")
		sptype_list.insert(0,"Flat spectrum in F_lambda")
		return sptype_list

	#try:
	keys = lookuptable[sptype]
	print "Spectral type exists in table ({0}):".format(sptype), keys
	return pysynphot.Icat('ck04models',keys[0], keys[1], keys[2])

def combine_transmission(filt, SRC):
	''' SRC is a spectral type string, e.g. A0V 
		not the neatest, but gets the job done.'''
	import pysynphot
	filt_wls = np.zeros(len(filt))
	filt_wght = np.zeros(len(filt))
	for ii in range(len(filt)):
		filt_wls[ii] = filt[ii][1] # in m
		filt_wght[ii] = filt[ii][0]
	src = specFromSpectralType(SRC)
	src = src.resample(np.array(filt_wls)*1.0e10) # converts to angstrom for pysynphot
	specwavl, specwghts = src.getArrays()
	totalwght = specwghts * filt_wght
	transmissionlist = []
	ang_to_m = 1.0e-10
	for ii in range(len(filt_wls)):
		transmissionlist.append((totalwght[ii], filt_wls[ii]))
	return transmissionlist


def makeA(nh, verbose=False):
	""" 
	Writes the "NRM matrix" that gets pseudo-inverterd to provide
	(arbitrarily constrained) zero-mean phases of the holes.

	makeA taken verbatim from Anand's pseudoinverse.py

	 input: nh - number of holes in NR mask
	 input: verbose - True or False
	 output: A matrix, nh columns, nh(nh-1)/2 rows  (eg 21 for nh=7)

	Ax = b  where x are the nh hole phases, b the nh(nh-1)/2 fringe phases,
	and A the NRM matrix

	Solve for the hole phases:
		Apinv = np.linalg.pinv(A)
		Solution for unknown x's:
		x = np.dot(Apinv, b)

	Following Noah Gamper's convention of fringe phases,
	for holes 'a b c d e f g', rows of A are 

	    (-1 +1  0  0  ...)
	    ( 0 -1 +1  0  ...)
	

	which is implemented in makeA() as:
		matrixA[row,h2] = -1
		matrixA[row,h1] = +1

	To change the convention just reverse the signs of the 'ones'.

	When tested against Alex'' NRM_Model.py "piston_phase" text output of fringe phases, 
	these signs appear to be correct - anand@stsci.edu 12 Nov 2014

	anand@stsci.edu  29 Aug 2014
		"""

	print "\nmakeA(): "
	#                   rows         cols
	ncols = (nh*(nh-1))//2
	nrows = nh
	matrixA = np.zeros((ncols, nrows))
	if verbose: print matrixA
	row = 0
	for h2 in range(nh):
		if verbose: print
		for h1 in range(h2+1,nh):
			if h1 >= nh:
				break
			else:
				if verbose:
					print "R%2d: "%row, 
					print "%d-%d"%(h1,h2)
				matrixA[row,h2] = -1
				matrixA[row,h1] = +1
				row += 1
	if verbose: print
	return matrixA


def fringes2pistons(fringephases, nholes):
	"""
	For NRM_Model.py to use to extract pistons out of fringes, given its hole bookkeeping,
	which apparently matches that of this module, and is the same as Noah Gamper's
	anand@stsci.edu  12 Nov 2014
	input: 1D array of fringe phases, and number of holes
	returns: pistons in same units as fringe phases
	"""
	Anrm = makeA(nholes)
	Apinv = np.linalg.pinv(Anrm)
	return np.dot(Apinv, fringephases)

def makeK(nh, verbose=False):
	""" 
	As above, write the "kernel matrix" that converts fringe phases
	to closure phases. This can be psuedo-inverted to provide a 
	subset of "calibrated" fringe phases (hole-based noise removed)

	 input: nh - number of holes in NR mask
	 input: verbose - True or False
	 output: L matrix, nh(nh-1)/2 columns, comb(nh, 3) rows  (eg 35 for nh=7)

	Kx = b, where: 
		- x are the nh(nh-1)/2 calibrated fringe phases 
		- b the comb(nh, 3) closure phases,
	and K the kernel matrix

	Solve for the "calibrated" phases:
		Kpinv = np.linalg.pinv(K)
		Solution for unknown x's:
		x = np.dot(Kpinv, b)

	Following the convention of fringe phase ordering above, which should look like:
	h12, h13, h14, ..., h23, h24, ....
	rows of K should look like:

	    (+1 -1  0  0  0  0  0  0 +1 ...) e.g., h12 - h13 + h23
	    (+1 +1  0 +1  ...)
	

	which is implemented in makeK() as:
		matrixK[n_cp, f12] = +1
		matrixK[n_cp, f13] = -1
		matrixK[n_cp, f23] = +1

	need to define the row selectors
	 k is a list that looks like [9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
	 -----up to nh*(nh-1)/2
	 i is a list that looks like [0,9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
	 -----up to nh*(nh-1)/2 -1
	because there are 9 fringe phases per single hole (decreasing by one to avoid repeating)
	hope that helps explain this!

	agreenba@pha.jhu.edu  22 Aug 2015
		"""

	print "\nmakeK(): "
	nrow = comb(nh, 3)
	ncol = nh*(nh-1)/2

	# first define the row selectors
	# k is a list that looks like [9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
	# -----up to nh*(nh-1)/2
	# i is a list that looks like [0,9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
	# -----up to nh*(nh-1)/2 -1
	countk=[]
	val=0
	for q in range(nh-1):
		val = val + (nh-1)-q
		countk.append(val)
	counti = [0,]+countk[:-1]
	# MatrixK
	row=0
	matrixK = np.zeros((nrow, ncol))
	for ii in range(nh-2):
		for jj in range(nh-ii-2):
			for kk in range(nh-ii-jj-2):
				matrixK[row+kk, counti[ii]+jj] = 1
				matrixK[row+kk, countk[ii+jj]+kk] = 1
				matrixK[row+kk, counti[ii]+jj+kk+1] = -1
			row=row+kk+1
	if verbose: print

	return matrixK


def create_ifneed(dir_):
	""" http://stackoverflow.com/questions/273192/check-if-a-directory-exists-and-create-it-if-necessary
    kanja
	"""
	if not os.path.exists(dir_):
		os.mkdir(dir_)


def nb_pistons(multiplier=1.0, debug=False):
	""" copy nb values from NRM_Model.py and play with them using this function, as a sanity check to compare to CV2 data
	    Define phi at the center of F430M band: lam = 4.3*um
		Scale linearly by "multiplier" factor if requested
		Returns *zero-meaned* pistons in meters for caller to interpret (typically as wavefront, i.e. OPD)
	"""
	if debug:
		phi_nb_ = np.array( [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] ) * 50.0 * nm_  # -150 nm to 150 nm OPDs in 50 nm steps
		return (phi_nb_ - phi_nb_.mean())
	else:
		phi_nb_ = multiplier * np.array( [0.028838669455909766, -0.061516214504502634, 0.12390958557781348, \
		                                 -0.020389361461019516, 0.016557347248600723, -0.03960017912525625, \
		                                  -0.04779984719154552] ) # phi in waves
		phi_nb_ = phi_nb_ - phi_nb_.mean() # phi in waves, zero mean
		wl = 4.3 * um_
		print "std dev  of piston OPD: %.1e um" % (phi_nb_.std() * wl/um_)
		print "variance of piston OPD: %.1e rad" % (phi_nb_.var() * (4.0*np.pi*np.pi))
		return phi_nb_ * 4.3*um_ # phi_nb in m


def get_webbpsf_filter(filtfile, specbin=None, trim=False):
	"""
	Returns array of [weight, wavelength_in_meters] empirically tested... 2014 Nov
	specbin: integer, bin spectrum down by this factor
	trim: (lambda_c, extend of 'live' spectrum), e.g. (2.77e-6, 0.4) trims below 0.8 and
	      above 1.2 lambda_c
	"""
	W = 1 # remove confusion - wavelength index
	T = 0 # remove confusion - trans index after reding in...
	f = fits.open(filtfile)
	thru = f[1].data
	f.close()
	tmp_array = np.zeros((len(thru), 2))
	for i in range(len(thru)):	
		tmp_array[i,W] = thru[i][0] * 1.0e-10   # input in angst  _ANGSTROM = 1.0e-10
		tmp_array[i,T] = thru[i][1]             # weights (peak unity, unnormalized sum)
		if 0:
			if i == len(thru)//2:
				print "input line: %d " % i
				print "raw input spectral wavelength: %.3e " % thru[i][0] 
				print "cvt input spectral wavelength: %.3e " % tmp_array[i,W] 

	# remove leading and trailing throughput lines with 'flag' array of indices
	flag = np.where(tmp_array[:,T]!=0)[0]
	spec = np.zeros((len(flag), 2))
	spec[:,:] = tmp_array[flag, :]

	# rebin as desired - fewer wavelengths for debugginng quickly
	if specbin:
		smallshape = spec.shape[0]//specbin
		print "bin by",  specbin, "  from ", spec.shape[0], " to",  smallshape
		spec = spec[:smallshape*specbin, :]  # clip trailing 
		spec = krebin(spec, (smallshape,2))
		#print "          wl/um", spec[:,W]
		spec[:,W] = spec[:,W] / float(specbin) # krebin added up waves
		spec[:,T] = spec[:,T] / float(specbin) # krebin added up trans too
		#print "wl/um / specbin",spec[:,W]
		#print "specbin - spec.shape", spec.shape

	if trim:
		print "TRIMming"
		wl = spec[:,W].copy()
		tr = spec[:,T].copy()
		idx = np.where((wl > (1.0 - 0.5*trim[1])*trim[0]) & (wl < (1.0 + 0.5*trim[1])*trim[0]))
		wl = wl[idx]
		tr = tr[idx]
		spec = np.zeros((len(idx[0]),2))
		spec[:,1] = wl
		spec[:,0] = tr
		print "post trim - spec.shape", spec.shape

	print "post specbin - spec.shape", spec.shape
	print "%d spectral samples " % len(spec[:,0]) + \
	      "between %.3f and %.3f um" % (spec[0,W]/um_,spec[-1,W]/um_)
 
	return spec

def trim_webbpsf_filter(filt, specbin=None, plot=False):
	print "================== " + filt + " ==================="
	beta = {"F277W":0.6, "F380M":0.15, "F430M":0.17, "F480M":0.2}
	lamc = {"F277W":2.70e-6, "F380M":3.8e-6, "F430M":4.24e-6, "F480M":4.8e-6}
	filterdirectory = os.getenv('WEBBPSF_PATH')+"/NIRISS/filters/" 
	band = get_webbpsf_filter(filterdirectory+filt+"_throughput.fits", 
	                          specbin=specbin, 
	                          trim=(lamc[filt], beta[filt]))
	print "filter", filt, "band.shape", band.shape,  "\n", band
	wl = band[:,1]
	tr = band[:,0]
	if plot: plt.plot(wl/um_, np.log10(tr), label=filt)
	return band

def testmain():
	import matplotlib.pyplot as plt
	fig = plt.figure(1,figsize=(6,4.5),dpi=150)
	trim_webbpsf_filter("F277W", specbin=6, plot=True)
	trim_webbpsf_filter("F380M", plot=True)
	trim_webbpsf_filter("F480M", plot=True)
	trim_webbpsf_filter("F430M", plot=True)
	ax = plt.axes()
	plt.setp(ax, xlim=(1.9, 5.1), ylim=(-4.9, 0.1))
	plt.ylabel("log10 ( Transmission )", fontsize=12)
	plt.xlabel("wavelength / micron", fontsize=12)
	plt.title("JWST NIRISS filters in WebbPSF data", fontsize=14)
	plt.legend(loc="lower left")
	plt.savefig("JWSTNIRISSfiltercurves.pdf", bbox_inches='tight')
	#plt.show()
	
	#########################################################################################
	##################################### from ##############################################
	################################## gpipoppy2.py #########################################
	#########################################################################################
	#########################################################################################


def krebin(a, shape): # Klaus P's fastrebin from web - return array of 'shape'
	sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
	return a.reshape(sh).sum(-1).sum(1)


def  rebin(a = None, rc=(2,2), verbose=None):  # Thinly-wrapped krebin 
	"""  
	anand@stsci.edu
	Perform simple-minded flux-conserving binning... clip trailing
	size mismatch: eg a 10x3 array binned by 3 results in a 3x1 array
	"""
	return krebin(a, (a.shape[0]//rc[0],a.shape[1]//rc[1]))


# used in NRM_Model.py
def rcrosscorrelate(a=None, b=None, verbose=True):

	""" Calculate cross correlation of two identically-shaped real arrays,
		returning a new array  that is the correlation of the two input 
		arrays.
	"""

	c = crosscorrelate(a=a, b=b, verbose=verbose) / (np.sqrt((a*a).sum())*np.sqrt((b*b).sum()))
	return  c.real.copy()


def crosscorrelate(a=None, b=None, verbose=True):

	""" Calculate cross correlation of two identically-shaped real or complex arrays,
		returning a new complex array  that is the correl of the two input 
		arrays. 
		ACF(f) = FT(powerspectrum(f))
	"""

	if a.shape != b.shape:
		print "crosscorrelate: need identical arrays"
		return None

	fac = np.sqrt(a.shape[0] * a.shape[1])
	if verbose: print "fft factor = %.5e" % fac

	A = fft.fft2(a)/fac
	B = fft.fft2(b)/fac
	if verbose: print "\tPower2d -  a: %.5e  A: %.5e " % ( (a*a.conj()).real.sum() ,  (A*A.conj()).real.sum())

	c = fft.ifft2(A * B.conj())  * fac * fac

	if verbose: print "\tPower2d -  A: %.5e  B: %.5e " % ( (A*A.conj()).real.sum() ,  (B*B.conj()).real.sum())
	if verbose: print "\tPower2d -  c=A_correl_B: %.5e " % (c*c.conj()).real.sum()

	if verbose: print "\t(a.sum %.5e x b.sum %.5e) = %.5e   c.sum = %.5e  ratio %.5e  " % (a.sum(), b.sum(), a.sum()*b.sum(), c.sum().real,  a.sum()*b.sum()/c.sum().real)

	if verbose: print 
	return fft.fftshift(c)

# used in NRM_Model.py
def quadratic(p, x):
	"  max y = -b^2/4a occurs at x = -b^2/2a"
	print "Max y value %.5f"%(-p[1]*p[1] /(4.0*p[0]) + p[2])
	print "occurs at x = %.5f"%(-p[1]/(2.0*p[0]))
	return -p[1]/(2.0*p[0]), -p[1]*p[1] /(4.0*p[0]) + p[2], p[0]*x*x + p[1]*x + p[2]


# used in NRM_Model.py
def findmax(mag, vals, mid=1.0):
	p = np.polyfit(mag, vals, 2)
	fitr = np.arange(0.95*mid, 1.05*mid, .01) 
	maxx, maxy, fitc = quadratic(p, fitr)
	return maxx, maxy


def findmax_detail(mag, vals, start=0.9,stop=1.1):
	''' mag denotes x values (like magnifications tested) vals is for y values.'''
	## e.g. polyfit returns highest degree first
	## e.g. p[0].x^2 +p[1].x + p[2]
	p = np.polyfit(mag, vals, 2)
	fitr = np.arange(start, stop, 0.001) 
	maxx, maxy, fitc = quadratic(p, fitr)
	error = quadratic(p,np.array(mag))[2] - vals
	return maxx, maxy,fitr, fitc, error

def jdefault(o):
    return o.__dict__

def save(nrmobj, outputname, savdir = ""):
	"""
	Probably don't need to use this unless have run a fit.
	This is only to save fitting parameters and results right now.
	"""
	import json
	class savobj: 
		def __init__(self):
			return None
	savobj.test = 1
	with open(r"{0}{1}.json".format(savdir, outputname), "wb") as output_file:
		json.dump(savobj, output_file, default=jdefault)
	print "success!"

	ffotest = json.load(open(outputname+".json", 'r'))
	
	print hasattr(ffotest, 'test')
	print ffotest['test']


	# init stuff
	savobj.pscale_rad, savobj.pscale_mas = nrmobj.pixel, rad2mas(nrmobj.pixel)
	savobj.holeshape, savobj.ctrs, savobj.d, savobj.D, savobj.N, \
		savobj.datapath, savobj.refdir 	= 	nrmobj.holeshape, nrmobj.ctrs, nrmobj.d, \
											nrmobj.D, nrmobj.N, nrmobj.datapath, nrmobj.refdir

	if hasattr(nrmobj, "refpsf"):
		savobj.refpsf, savobj.rot_best = nrmobj.refpsf, nrmobj.rot_measured
	if hasattr(nrmobj, "fittingmodel"):
		# details
		savobj.weighted, savobj.pixweight, savobj.bestcenter, \
			savobj.bandpass, savobj.modelctrs, savobj.over,\
			savobj.modelpix  =	nrmobj.weighted, nrmobj.pixweight, nrmobj.bestcenter, \
								nrmobj.bandpass, nrmobj.modelctrs, nrmobj.over, nrmobj.modelpix
		# resulting arrays
		savobj.modelmat, savobj.soln, \
			savobj.residual, savobj.cond, \
			savobj.rawDC, savobj.flux, \
			savobj.fringeamp, savobj.fringephase,\
			savobj.cps, savobj.cas 	= 	nrmobj.fittingmodel, nrmobj.soln, nrmobj.residual, \
								nrmobj.cond, nrmobj.rawDC, nrmobj.flux, nrmobj.fringeamp, \
								nrmobj.fringephase, nrmobj.redundant_cps, nrmobj.redundant_cas
		if not hasattr(nrmobj, "modelpsf"):
			nrmobj.plot_model()
		savobj.modelpsf = nrmobj.modelpsf
	with open(r"{0}.ffo".format(savdir, outputname), "wb") as output_file:
		json.dump(savobj, output_file, default=jdefault)

	return savdir+outputname+".ffo"

############### More useful functions ##################
def baselinify(ctrs):
	N = len(ctrs)
	uvs = np.zeros((N*(N-1)//2, 2))
	label = np.zeros((N*(N-1)//2, 2))
	bllengths = np.zeros(N*(N-1)//2)
	nn=0
	for ii in range(N-1):
		for jj in range(N-ii-1):
			uvs[jj+nn, 0] = ctrs[ii,0] - ctrs[ii+jj+1,0]
			uvs[jj+nn, 1] = ctrs[ii,1] - ctrs[ii+jj+1,1]
			bllengths[jj+nn] = np.sqrt((ctrs[ii,0]-ctrs[ii+jj+1,0])**2 +\
					    (ctrs[ii,1]-ctrs[ii+jj+1,1])**2)
			label[jj+nn,:] = np.array([ii, ii+jj+1])
		nn = nn+jj+1
	return uvs, bllengths, label

def count_cps(ctrs):
	from scipy.misc import comb
	N = len(ctrs)
	ncps = comb(N,3)
	cp_label = np.zeros((ncps, 3))
	u1 = np.zeros(ncps)
	v1 = np.zeros(ncps)
	u2 = np.zeros(ncps)
	v2 = np.zeros(ncps)
	nn=0
	for ii in range(N-2):
		for jj in range(N-ii-2):
			for kk in range(N-ii-jj-2):
				#print '---------------'
				#print nn+kk
				cp_label[nn+kk,:] = np.array([ii, ii+jj+1, ii+jj+kk+2])
				u1[nn+kk] = ctrs[ii,0] - ctrs[ii+jj+1, 0]
				v1[nn+kk] = ctrs[ii,1] - ctrs[ii+jj+1, 1]
				u2[nn+kk] = ctrs[ii+jj+1,0] - ctrs[ii+jj+kk+2, 0]
				#print ii
				#print ii+jj+1, ii+jj+kk+2
				v2[nn+kk] = ctrs[ii+jj+1,1] - ctrs[ii+jj+kk+2, 1]
				#print u1[nn+kk], u2[nn+kk], v1[nn+kk], v2[nn+kk]
			nn = nn+kk+1
	return u1, v1, u2, v2, cp_label

def baseline_info(mask, pscale_mas, lam_c):
	mask.ctrs = np.array(mask.ctrs)
	uvs, bllengths, label = baselinify(mask.ctrs)

	print "========================================"
	print "All the baseline sizes:"
	print bllengths
	print "========================================"
	print "Longest baseline:"
	print bllengths.max(), "m"
	print "corresponding to lam/D =", rad2mas(lam/bllengths.max()), "mas at {0} m \n".format(lam_c)
	print "Shortest baseline:"
	print bllengths.min(), "m"
	print "corresponding to lam/D =", rad2mas(lam/bllengths.min()), "mas at {0} m".format(lam_c)
	print "========================================"
	print "Mask is Nyquist at",
	print mas2rad(2*pscale_mas)*(bllengths.max())

def corner_plot(pickfile, nbins = 100, save="my_calibrated/triangle_plot.png"):
	"""
	Make a corner plot after the fact using the pickled results from the mcmc
	"""
	import corner
	import matplotlib.pyplot as plt

	mcmc_results = pickle.load(open(pickfile))
	keys = mcmc_results.keys()
	print keys
	chain = np.zeros((len(mcmc_results[keys[0]]), len(keys)))
	print len(keys)
	print chain.shape
	for ii,key in enumerate(keys):
		chain[:,ii] = mcmc_results[key]

	fig = corner.corner(chain, labels = keys, bins=nbins, show_titles=True)
	plt.savefig("triangle_plot.pdf")
	plt.show()

def populate_symmamparray(amps, N=7):
    fringeamparray = np.zeros((N,N))
    step=0
    n=N-1
    for h in range(n):
        #print "step", step, "step+n", step+n
        #print "h", h, "h+1", h+1, "and on"
        #print fringeamparray[h,h+1:].shape, amps[step:step+n].shape
        fringeamparray[h,h+1:] = amps[step:step+n]
        step = step+n
        n=n-1
    fringeamparray = fringeamparray + fringeamparray.T
    return fringeamparray

def t3vis(vis, N=7):
    """ provided visibilities, this put these into triple product"""
    amparray = populate_symmamparray(vis, N=N)
    t3vis = np.zeros(int(comb(N,3)))
    nn=0
    for kk in range(N-2):
        for ii in range(N-kk-2):
            for jj in range(N-kk-ii-2):
                t3vis[nn+jj] = amparray[kk,ii+kk+1] \
                * amparray[ii+kk+1,jj+ii+kk+2] \
                * amparray[jj+ii+kk+2,kk]
            nn=nn+jj+1
    return t3vis

def t3err(viserr, N=7):
    """ provided visibilities, this put these into triple product"""
    amparray = populate_symmamparray(viserr, N=N)
    t3viserr = np.zeros(int(comb(N,3)))
    nn=0
    for kk in range(N-2):
        for ii in range(N-kk-2):
            for jj in range(N-kk-ii-2):
                t3viserr[nn+jj] = np.sqrt(amparray[kk,ii+kk+1]**2 \
                + amparray[ii+kk+1,jj+ii+kk+2]**2 \
                + amparray[jj+ii+kk+2,kk]**2 )
            nn=nn+jj+1
    return t3viserr

if __name__ == "__main__":

	testmain()

	
