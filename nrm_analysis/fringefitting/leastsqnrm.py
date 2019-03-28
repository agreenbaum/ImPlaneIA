#! /usr/bin/env  python 
# Mathematica nb from Alex & Laurent
# anand@stsci.edu major reorg as LG++ 2018 01
# python3 required (int( (len(coeffs) -1)/2 )) because of  float int/int result change from python2

import numpy as np
import scipy.special
import numpy.linalg as linalg
import sys
from scipy.misc import comb
import os, pickle
from uncertainties import unumpy  # pip install if you need

m = 1.0
mm = 1.0e-3 * m
um = 1.0e-6 * m


def scaling(img, photons):  # RENAME this function
    # img gives a perfect psf to count its total flux
    # photons is the desired number of photons (total flux in data)
    total = np.sum(img)
    print("total", total)
    return photons / total


def matrix_operations(img, model, flux = None, verbose=False, linfit=False):
    # least squares matrix operations to solve A x = b, where A is the model,
    # b is the data (image), and x is the coefficient vector we are solving for.
    # In 2-D data x = inv(At.A).(At.b) 

    flatimg = img.reshape(np.shape(img)[0] * np.shape(img)[1])
    nanlist = np.where(np.isnan(flatimg))
    flatimg = np.delete(flatimg, nanlist)
    if flux is not None:
        flatimg = flux * flatimg / flatimg.sum()

    # A
    flatmodel_nan = model.reshape(np.shape(model)[0] * np.shape(model)[1], np.shape(model)[2])
    flatmodel = np.zeros((len(flatimg), np.shape(model)[2]))

    if verbose:
        print("flat model dimensions ", np.shape(flatmodel))
        print("flat image dimensions ", np.shape(flatimg))

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
        print('model flux', flux)
        print('data flux', flatimg.sum())
        print("flat model dimensions ", np.shape(flatmodel))
        print("model transpose dimensions ", np.shape(flatmodeltransp))
        print("flat image dimensions ", np.shape(flatimg))
        print("transpose * image data dimensions", np.shape(data_vector))
        print("flat img * transpose dimensions", np.shape(inverse))

    if linfit: 
        try:
            from linearfit import linearfit

            # dependent variables
            M = np.mat(flatimg)

            # photon noise
            noise = np.sqrt(np.abs(flatimg))

            # this sets the weights of pixels fulfilling condition to zero
            weights = np.where(np.abs(flatimg)<=1.0, 0.0, 1.0/(noise**2))    

            # uniform weight
            wy = weights
            S = np.mat(np.diag(wy));
            # matrix of independent variables
            C = np.mat(flatmodeltransp)

            # initialize object
            result = linearfit.LinearFit(M,S,C)

            # do the fit
            result.fit()

            # delete inverse_covariance_matrix to reduce size of pickled file
            result.inverse_covariance_matrix = []

            linfit_result = result
            print("Returned linearfit result")

        except ImportError:
            linfit_result = None
    #         if verbose:
            print("linearfit module not imported, no covariances saved.")
    else:
        linfit_result = None
        if verbose:
            print("linearfit module not imported, no covariances saved.")

    return x, res, cond, linfit_result
    

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
        print("flat model dimensions ", np.shape(flatmodel))
        print("model transpose dimensions ", np.shape(flatmodeltransp))
        print("flat image dimensions ", np.shape(flatimg))
        print("transpose * image data dimensions", np.shape(data_vector))
        print("flat img * transpose dimensions", np.shape(inverse))
    
    return x, res,cond


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
    Edit 05/2017: J. Sahlmann added support of uncertainty propagation
    """
    if type(coeffs[0]).__module__ != 'uncertainties.core':
        # if uncertainties not present, proceed as usual
        
        # coefficients of sine terms mulitiplied by 2*pi

        delta = np.zeros(int( (len(coeffs) -1)/2 ))  # py3
        amp = np.zeros(int( (len(coeffs) -1)/2 ))  # py3
        for q in range(int( (len(coeffs) -1)/2 )):  # py3
            delta[q] = (np.arctan2(coeffs[2*q+2], coeffs[2*q+1])) 
            amp[q] = np.sqrt(coeffs[2*q+2]**2 + coeffs[2*q+1]**2)
        if verbose:
            print("shape coeffs", np.shape(coeffs))
            print("shape delta", np.shape(delta))

        # returns fringe amplitude & phase
        return amp, delta
    
    else:
        #         propagate uncertainties
        qrange = np.arange(int( (len(coeffs) -1)/2 ))  # py3
        fringephase = unumpy.arctan2(coeffs[2*qrange+2], coeffs[2*qrange+1])
        fringeamp = unumpy.sqrt(coeffs[2*qrange+2]**2 + coeffs[2*qrange+1]**2)
        return fringeamp, fringephase


def fixeddeltapistons(coeffs, verbose=False):
    delta = np.zeros(int( (len(coeffs) -1)/2 ))  # py3
    for q in range(int( (len(coeffs) -1)/2 )):  # py3
        delta[q] = np.arcsin((coeffs[2*q+1] + coeffs[2*q+2]) / 2) / (np.pi*2.0)
    if verbose:
        print("shape coeffs", np.shape(coeffs))
        print("shape delta", np.shape(delta))

    return delta    


def populate_antisymmphasearray(deltaps, N=7):
    if type(deltaps[0]).__module__ != 'uncertainties.core':
        fringephasearray = np.zeros((N,N))
    else:
        fringephasearray = unumpy.uarray(np.zeros((N,N)),np.zeros((N,N)))    
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

    if type(amps[0]).__module__ != 'uncertainties.core':
        fringeamparray = np.zeros((N,N))
    else:
        fringeamparray = unumpy.uarray(np.zeros((N,N)),np.zeros((N,N)))
        
    step=0
    n=N-1
    for h in range(n):
        fringeamparray[h,h+1:] = amps[step:step+n]
        step = step+n
        n=n-1
    fringeamparray = fringeamparray + fringeamparray.T
    return fringeamparray


def phases_and_amplitudes(solution_coefficients, N=7):

    #     number of solution coefficients
    Nsoln = len(solution_coefficients)    
    
    # normalise by intensity
    soln = np.array([solution_coefficients[i]/solution_coefficients[0] for i in range(Nsoln)])

    # compute fringe quantitites
    fringeamp, fringephase = tan2visibilities( soln )    
    
#     import pdb
#     pdb.set_trace()

    # compute closure phases
    if type(solution_coefficients[0]).__module__ != 'uncertainties.core':
        redundant_closure_phases = redundant_cps(np.array(fringephase), N=N)
    else:
        redundant_closure_phases, fringephasearray = redundant_cps(np.array(fringephase), N=N)
    
    # compute closure amplitudes
    redundant_closure_amplitudes = return_CAs(np.array(fringephase), N=N)

    return fringephase, fringeamp, redundant_closure_phases, redundant_closure_amplitudes


def redundant_cps(deltaps, N = 7):
    fringephasearray = populate_antisymmphasearray(deltaps, N=N)
    if type(deltaps[0]).__module__ != 'uncertainties.core':
        cps = np.zeros(int(comb(N,3)))
    else:
        cps = unumpy.uarray( np.zeros(np.int(comb(N,3))),np.zeros(np.int(comb(N,3))) )    
    nn=0
    for kk in range(N-2):
        for ii in range(N-kk-2):
            for jj in range(N-kk-ii-2):
                cps[nn+jj] = fringephasearray[kk, ii+kk+1] \
                       + fringephasearray[ii+kk+1, jj+ii+kk+2] \
                       + fringephasearray[jj+ii+kk+2, kk]
            nn = nn+jj+1
    if type(deltaps[0]).__module__ != 'uncertainties.core':
        return cps
    else:
        return cps, fringephasearray

        
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
        print("invalid hole number")

    # calculates closure phases for general N-hole mask (with p-array set up properly above)
    cps = np.zeros(int((N - 1)*(N - 2)/2)) #py3
    for l1 in range(N - 2):
        for l2 in range(N - 2 - l1):
            cps[int(l1*((N + (N-3) -l1) / 2.0)) + l2] = \
                p[l1][0] + p[l1+1][l2] - p[l1][l2+1]
    return cps


def return_CAs(amps, N=7):
    fringeamparray = populate_symmamparray(amps, N=N)            
    nn=0
    
    if type(amps[0]).__module__ != 'uncertainties.core':
        CAs = np.zeros(int(comb(N,4)))
    else:
        CAs = unumpy.uarray( np.zeros(np.int(comb(N,4))),np.zeros(np.int(comb(N,4))) )
        
    for ii in range(N-3):
        for jj in range(N-ii-3):
            for kk in range(N-jj-ii-3):
                for ll  in range(N-jj-ii-kk-3):
                    CAs[nn+ll] = fringeamparray[ii,jj+ii+1] \
                               * fringeamparray[ll+ii+jj+kk+3,kk+jj+ii+2] \
            / (fringeamparray[ii,kk+ii+jj+2]*fringeamparray[jj+ii+1,ll+ii+jj+kk+3])
                nn=nn+ll+1
    return CAs
