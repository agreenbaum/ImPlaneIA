#! /usr/bin/env python
import numpy as np

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

def weightpixels(array, weightarray):

	if np.shape(weightarray)[0] !=np.shape(weightarray)[1]:
		raise ValueError("Pixel Weight Array Is Not Square")

	oversample = np.shape(weightarray)[0]
	shapex = np.shape(array)[0]
	shapey = np.shape(array)[1]
	b = array.copy()
	b = b.reshape(shapex//oversample, oversample, shapey//oversample, oversample)
	d = b.copy()

	for i in range(np.shape(weightarray)[0]):
		for j in range(np.shape(weightarray)[1]):
			d[:,i,:,j] = d[:,i,:,j]* weightarray[i,j]

	"""
	# e.g 1 in the center, 0.8 on the edges:
	d[:,1, :, 1] = d[:,1, :, 1]

	d[:,0, :, :] = 0.8 * d[:,0, :, :] 
	d[:,2, :, :] = 0.8 * d[:,2, :, :] 
	d[:,:, :, 0] = 0.8 * d[:, :, :, 0] 
	d[:,:, :, 2] = 0.8 * d[:, :, :, 2] 

	for i in range(shapex):
		for j in range(shapey):
			d[i,:,j,:] = b[i,:,j,:] * weightarray
	"""

	return d.sum(-1).sum(1)

def pixelpowerprof(s = np.array([3,3]), power = 4, ctr = None ):
	shape = np.array(s)
	center = shape / 2
	print center
	y,x = np.indices(s)
	pix = (shape[0] /2.0)*np.ones(shape) - np.abs( (x - center[0])**power + (y - center[1])**power )**(1/float(power))
	return pix

if __name__ == "__main__":

	pixw = np.array([ [0.8, 0.8, 0.8],
			  [0.8, 1.0, 0.8],
			  [0.8, 0.8, 0.8] ])

