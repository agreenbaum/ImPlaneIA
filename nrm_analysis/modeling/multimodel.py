#! /usr/bin/env python

import numpy as np
import os
from nrm_analysis.misctools.utils import mas2rad

#######################################################################
def vis(baseline, ratio, separation, angle):
    """
    *** modified for N multiple sources (length of any of the params)***

    From David L.'s NRM code V(u,v) = {1 + r exp[-i2pi(alpha.u + delta.v)]}/(1+r)
    2011 Sparse Aperture Masking at the VLT

    r: flux ratio between sources
    alpha, delta: RA, DEC (radians)
    u, v: orthogonal spatial frequency vectors
    These baseline vectors should be scaled by 1/wavelength

    Inputs in mas and degrees
    Returns: the resulting visibility (amplitude, phase) given
             a baseline, flux ratio, separation, and angle.
    """
    # Basline info doesn't care about scene
    u = baseline[0]
    v = baseline[1]
    # For multiple sources ratio, separation, & angle are lists or arrays
    visnum = 1+0j
    ratio = np.array(ratio)
    for ii in range(len(ratio)):

        alpha = mas2rad(separation[ii])*np.cos(np.pi*angle[ii]/180.0)
        delta = mas2rad(separation[ii])*np.sin(np.pi*angle[ii]/180.0)
        visnum += ratio[ii]*(np.exp((-1j*2*np.pi*(alpha*u + delta*v))) )
    visibility = visnum / (1+ ratio.sum())

    phase = np.angle(visibility)
    amplitude = np.absolute(visibility)

    return amplitude, phase
#######################################################################

def model_bispec_uv(tri_uv, ratio, separation, pa, inv_wavl):
    """
    Takes 3 cp uv coords, or a set of them and returns model cps based on
    contrast ratio and position specified, at some wavelength. 
    tri_uv = array( [[u1, v1], [u2, v2], [u3, v3],]  )
    shape: [2, 3, ncp, nwavl]
    contrast, separation, pa all should be len(inv_wavl) or scalar
    """
    #ratio = np.array(ratio)
    #separation = np.array(separation)
    #pa = np.array(pa)
    uvs = inv_wavl*tri_uv
    model_cps = vis(uvs[:,0, ...], ratio, separation, pa)[0] + \
                vis(uvs[:,1, ...], ratio, separation, pa)[0] + \
                vis(uvs[:,2, ...], ratio, separation, pa)[0]
    model_t3amp = vis(uvs[:,0, ...], ratio, separation, pa)[1] * \
                vis(uvs[:,1, ...], ratio, separation, pa)[1] * \
                vis(uvs[:,2, ...], ratio, separation, pa)[1]
    # Return in deg to match oifits standard
    return model_t3amp, 180.*model_cps/np.pi
