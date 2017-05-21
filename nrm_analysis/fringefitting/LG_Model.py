#! /usr/bin/env  python 
# Heritage mathematia nb from Alex & Laurent
# Heritage python by Alex Greenbaum & Anand Sivaramakrishnan Jan 2013
# updated May 2013 to include hexagonal envelope

# Goals: a more convenient module for analytic simulation & model fitting
"""
Imports:
"""
import numpy as np
import scipy.special
import hexee
import math
import nrm_analysis.misctools.utils as utils
from nrm_analysis.misctools.utils import rebin
import sys, os
import time
from astropy.io import fits
import logging
from argparse import ArgumentParser
#import leastsqnrm_smallangle as leastsqnrm
#import leastsqnrm_kp as leastsqnrm
import leastsqnrm as leastsqnrm
import analyticnrm2
#_log.setLevel(logging.INFO)
#import IPSmap as IPS
import subpix

_default_log = logging.getLogger('NRM_Model')
#_default_log.setLevel(logging.INFO)
_default_log.setLevel(logging.ERROR)

"""

====================
NRM_Model
====================

A module for conveniently dealing with an "NRM object"
This should be able to take an NRM_mask_definitions object for mask geometry

Defines mask geometry and detector-scale parameters
Simulates PSF (broadband or monochromatic)
Builds a fringe model - either by user definition, or automated to data
Fits model to data by least squares

Masks:
  * gpi_g10s40
  * jwst
  * visir?

Methods:

simulate
make_model
fit_image
plot_model
perfect_from_model
new_and_better_scaling_routine
auto_find_center
save


First written by Alexandra Greenbaum 2013-2014
Algorithm documented in 
    Greenbaum, A. Z., Pueyo, L. P., Sivaramakrishnan, A., and Lacour, S., 
    Astrophysical Journal vol. 798, Jan 2015.
Developed with NASA APRA (AS, AZG), NSF GRFP (AZG), NASA Sagan (LP), and French taxpayer (SL)  support

"""
phi_nb = np.array( [0.028838669455909766, -0.061516214504502634, \
     0.12390958557781348, -0.020389361461019516, 0.016557347248600723, \
    -0.03960017912525625, -0.04779984719154552] ) # phi in waves
# define phi at the center of F430M band:
phi_nb = phi_nb *4.3e-6 # phi_nb in m
m = 1.0
mm = 1.0e-3 * m
um = 1.0e-6 * m

def mas2rad(mas):
    rad = mas*(10**(-3)) / (3600*180/np.pi)
    return rad

class NRM_Model():

    def __init__(self, mask=None, holeshape="circ", pixscale=mas2rad(64), rotate=False, \
            over = 1, log=_default_log, flip=False, pixweight=None,\
            datapath="",
            scallist = None, rotlist_deg = None, phi="perfect", refdir=""):
        """
        mask will either be a string keyword for built-in values or
        an NRM_mask_geometry object.
        pixscale should be input in radians.
        """ 
        # define a handler to write log messages to stdout
        sh = logging.StreamHandler(stream=sys.stdout)

        # define the format for the log messages, here: "level name: message"
        formatter = logging.Formatter("[%(levelname)s]: %(message)s")
        sh.setFormatter(formatter)
        self.logger = log
        self.logger.addHandler(sh)

        self.holeshape = holeshape
        self.pixel = pixscale
        self.over=over
        self.maskname=mask
        self.pixweight = pixweight 
        if mask=='jwst':
            self.ctrs = np.array( [ [ 0.00000000,  -2.640000],
                        [-2.2863100 ,  0.0000000],
                        [ 2.2863100 , -1.3200001],
                        [-2.2863100 ,  1.3200001],
                        [-1.1431500 ,  1.9800000],
                        [ 2.2863100 ,  1.3200001],
                        [ 1.1431500 ,  1.9800000]    ] )
            #self.ctrs = analyticnrm2.flip(self.ctrs)
            print "CTRS FLIPPED IN INIT"
            #self.lam = 4.3*um
            self.d = 0.80
            self.D = 6.5
            # fov
            # oversample?
        else:
            try:
                print mask.ctrs
                
            except AttributeError:
                raise AttributeError("mask must be either 'jwst' \
                              or NRM_mask_geometry object")
            self.ctrs, self.d, self.D = np.array(mask.ctrs), mask.hdia, mask.activeD
        if rotate:
            print 'PROVIDING ADDITIONAL ROTATION:',rotate*180./np.pi, 'DEG'
            # Rotates vector counterclockwise in coordinates
            self.rotation = rotate
            self.ctrs = analyticnrm2.rotatevectors(self.ctrs, rotate)
        self.N = len(self.ctrs)
        self.datapath=datapath
        self.refdir = refdir
        #self.refdir = "refdir/"  #DT nov 20
        self.fmt = "%10.6e"      #DT nov 20 

        if scallist is None:
            self.scallist = np.array([ 0.995, 0.998, 1.0, 1.002, 1.005, ])
        else:
            self.scallist = scallist

        if rotlist_deg is None:
            self.rotlist_rad = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) *  np.pi / 180.0
        else:
            self.rotlist_rad = rotlist_deg *  np.pi / 180.0
        #phi added to initialization. DT, AS Feb 2016
                #Removed phi=None option
        if phi == "perfect":
            self.phi = np.zeros(len(self.ctrs))
            kwphi = "perfect"
        elif phi == 'nb':
            self.phi = phi_nb
            pl.savetxt(self.datapath+'simulationphi.txt', self.phi)
            kwphi = 'nb'
        else:
            self.phi = phi
            kwphi = 'other'

        #Added to set pistons in the call to simulate. AS Feb 2016
        def set_pistons(self,phi):
                """jwnrm.set_pistons(np.array[...])"""
                self.phi = phi

    #Removed phi from function simulate. Use phi from initializations or set using set_pistons. DT, AS February 2016
    def simulate(self, fov=None, bandpass=None, over=None, \
        pixweight=None,pixel=None,rotate=False, centering = "PIXELCENTERED"):
        """
        This method will simulate a psf using parameters input from the 
        call and already stored in the object. It also generates a simulation 
        fits header storing all of the parameters used to generate that psf.
        If the input bandpass is one number it will calculate a monochromatic
        PSF.
        Use set_pistons() to set self.phi if desired
        """
        self.simhdr = fits.PrimaryHDU().header
        # First set up conditions for choosing various parameters
        if fov == None:
            if not hasattr(self,'fov'):
                raise ValueError("Please specify a field of view value")
            else:
                self.fov_sim = self.fov
                print "using predefined FOV size:", self.fov
        else:
            self.fov_sim = fov

        if hasattr(centering, '__iter__'):
            self.simhdr['XCTR'] = (centering[0],\
                    'x center position on oversampled pixel')
            self.simhdr['YCTR'] = (centering[1],\
                    'y center position on oversampled pixel')
        self.bandpass = bandpass
        if over is not None:
            pass
        else:
            over = 1
        #if self.over == None:
        #   self.over=over
        self.simhdr['PIXWGHT']= (False, "True or false, whether there is sub-pixel weighting")
        if pixweight is not None:
            over=self.pixweight.shape[0]
            self.simhdr['PIXWGHT'] = True
        else:
            self.simhdr['PIXWGHT'] = True
        self.simhdr['OVER'] = (over, 'oversampling')

        if rotate:
            self.rotate=rotate
            self.rotctrs = analyticnrm2.rotatevectors(self.ctrs, self.rotate)
            self.simhdr['ROTRAD'] = (self.rotate, "rotation in radians")
        else:
            self.rotctrs= self.ctrs
            self.simhdr['ROTRAD'] = (0, "rotation in radians")
        if pixel==None:
            self.pixel_sim = self.pixel
        else:
            self.pixel_sim = pixel
        self.simhdr['PIXSCL'] = (self.pixel_sim, 'Pixel scale in radians')

        # The polychromatic case:
        if hasattr(self.bandpass, '__iter__'):
            self.logger.debug("------Simulating Polychromatic------")
            self.psf_over = np.zeros((over*self.fov_sim, 
                            over*self.fov_sim))
            qq=0
            for w,l in self.bandpass: # w: weight, l: lambda (wavelength)
                print "weight:", w, "wavelength:", l
                print "fov:", self.fov_sim
                print "over:",over
                self.psf_over += w*analyticnrm2.PSF(self.pixel_sim, \
                            self.fov_sim, over, \
                               self.rotctrs, self.d, l, \
                            self.phi, \
                               centering = centering, \
                            shape=self.holeshape)
                self.simhdr["WAVL{0}".format(qq)] = (l, "wavelength (m)")
                self.simhdr["WGHT{0}".format(qq)] = (w, "weight")
                qq=qq+1
            self.logger.debug("BINNING UP TO PIXEL SCALE")
        # The monochromatic case if bandpass input is a single wavelength
        else:
            self.lam=bandpass
            self.simhdr["WAVL"] = (self.lam,"Wavelength (m)")
            print  "Got one wavelength. Simulating monochromatic ... "
            self.logger.debug("Calculating Oversampled PSF")
            self.psf_over = analyticnrm2.PSF(self.pixel_sim, self.fov_sim, 
                        over, self.rotctrs, self.d, self.lam, \
                        self.phi, centering = centering, \
                        shape=self.holeshape,verbose=True)
        if self.pixweight==None:
            self.psf = rebin(self.psf_over, (over, over))
        else:
            print "using a weighted pixel",
            print "\neffective oversampling:",
            print self.pixweight.shape[0]
            self.psf=subpix.weightpixels(self.psf_over,\
                self.pixweight)
        #self.psf = self.psf[::-1,:]
        return self.psf

    def make_model(self, fov=None, bandpass=None, over=False,centering='PIXELCENTERED',
                pixweight=None, pixscale=None,rotate=False, flip=False):
        
        print "flip is", flip, "relevant to flipping coord ctrs on mask holes"
        """
        make_model generates the fringe model with the attributes of the object.
        It can take either a single wavelength or a bandpass as a list of tuples.
        The bandpass should be of the form: 
        [(weight1, wavl1), (weight2, wavl2), ...]
        """
        #self.ctrs=analyticnrm2.flip(self.ctrs)
        #if self.maskname == 'jwst':
        #   print "maskname == jwst, FLIPPING CTRS"
        #   self.ctrs = analyticnrm2.flip(self.ctrs)
        #   #self.ctrs= analyticnrm2.rotatevectors(self.ctrs, -np.pi/2)
        if fov:
            self.fov = fov
        if over is False:
            self.over=1
        else:
            self.over=over

        if pixweight is not None:
            self.over=self.pixweight.shape[0]
        if hasattr(self, 'pixscale_measured'):
            if self.pixscale_measured is not None:
                self.modelpix = self.pixscale_measured
        if pixscale==None:
            self.modelpix=self.pixel
        else:
            self.modelpix=pixscale
        if rotate:
            print "ROTATE:",rotate
            if flip == True:
                self.modelctrs = \
                analyticnrm2.flip(analyticnrm2.rotatevectors(self.ctrs,\
                                self.rot_measured))
                print "ROTATE ENGAGED-- flip rotated ctrs in make_model"
            else:
                self.modelctrs = \
                analyticnrm2.rotatevectors(self.ctrs,\
                                self.rot_measured)
            print 'MODEL USING TRUE ROTATION'
        else:
            self.modelctrs = self.ctrs

        if not hasattr(bandpass, '__iter__'):
            print "MONOCHROMATIC: single wavelength"
            self.lam = bandpass
            #self.bandpass = [(1., self.lam),]
            self.model = np.ones((self.fov, self.fov, self.N*(self.N-1)+2))
            self.model_beam, self.fringes = leastsqnrm.model_array(self.modelctrs,\
                              self.lam, self.over, \
                              self.modelpix, self.fov, \
                                  self.d, shape=self.holeshape, \
                              centering=centering,\
                              verbose=True)
            self.logger.debug("centering: {0}".format(centering))
            self.logger.debug("what primary beam has the model created?"+\
                        " {0}".format(self.model_beam))

            # this routine multiplies the envelope by each fringe "image"
            self.model_over = leastsqnrm.multiplyenv(self.model_beam, self.fringes)

            self.model = np.zeros((self.fov,self.fov, self.model_over.shape[2]))
            # loop over slices "sl" in the model
            for sl in range(self.model_over.shape[2]):
                if self.pixweight==None:
                    self.model[:,:,sl] = \
                    rebin(self.model_over[:,:,sl], \
                    (self.over, self.over))
                else:
                    print "using a weighted pixel",
                    print "\neffective oversampling:",
                    print self.pixweight.shape[0]
                    self.model[:,:,sl] = \
                    subpix.weightpixels(self.model_over[:,:,sl],\
                    self.pixweight)
            return self.model

        else:
            print "So the bandpass should have a shape:",np.shape(bandpass)
            print "And it should be iterable:", hasattr(bandpass, '__iter__')
            self.bandpass = bandpass
            print "-------Polychromatic bandpass---------"
            print self.bandpass
            
            # The model shape is (fov) x (fov) x (# solution coefficients)
            # the coefficient refers to the terms in the analytic equation
            # There are N(N-1) independent pistons, double-counted by cosine
            # and sine, one constant term and a DC offset.
            self.model = np.ones((self.fov, self.fov, self.N*(self.N-1)+2))
            self.model_beam = np.zeros((self.over*self.fov, self.over*self.fov))
            self.fringes = np.zeros((self.N*(self.N-1)+1, \
                        self.over*self.fov, self.over*self.fov))
            for w,l in self.bandpass: # w: weight, l: lambda (wavelength)
                print "weight: {0}, lambda: {1}".format(w,l)
                # model_array returns a the envelope and fringe model
                pb, ff = leastsqnrm.model_array(self.modelctrs, l, self.over, \
                                  self.modelpix,\
                                  self.fov, \
                                  self.d, \
                                  shape=self.holeshape, \
                                  centering=centering,\
                                  verbose=False)
                self.logger.debug("centering: {0}".format(centering))
                self.logger.debug("what primary beam has the model"+\
                          "created? {0}".format(pb))
                self.model_beam += pb
                self.fringes += ff

                # this routine multiplies the envelope by each fringe "image"
                self.model_over = leastsqnrm.multiplyenv(pb, ff)
                print "NRM MODEL model shape:", self.model_over.shape

                model_binned = np.zeros((self.fov,self.fov, 
                            self.model_over.shape[2]))
                # loop over slices "sl" in the model
                for sl in range(self.model_over.shape[2]):
                    if self.pixweight==None:
                        model_binned[:,:,sl] = \
                        rebin(self.model_over[:,:,sl], \
                        (self.over, self.over))
                    else:
                        print "using a weighted pixel",
                        print "\neffective oversampling:",
                        print self.pixweight.shape[0]
                        model_binned[:,:,sl] = \
                        subpix.weightpixels(self.model_over[:,:,sl],\
                        self.pixweight)

                self.model += w*model_binned
        
            return self.model

    def fit_image(self, image, reference=None, pixguess=None, rotguess=0, 
            modelin=None, weighted=False, centering = 'PIXELCENTERED',
            savepsfs=False):
        """
        fit_image will run a least-squares fit on an input image.
        Specifying a model is optional. If a model is not specified then this
        method will find the appropriate wavelength scale, rotation (and 
        hopefully centering as well -- This is not written into the object yet, 
        but should be soon).
    
        Without specifying a model, fit_image can take a reference image 
        (a cropped deNaNed version of the data) to run correlations. It is 
        recommended that the symmetric part of the data be used to avoid piston
        confusion in scaling. Good luck!
        """
        self.model_in=modelin
        self.weighted=weighted
        self.saveval = savepsfs
        #if hasattr(self, 'lam'):
        #   self.bandpass = self.lam

        if modelin is None:
            # No model provided - now perform a set of automatic routines

            # A Cleaned up version of your image to enable Fourier fitting for 
            # centering crosscorrelation with FindCentering() and
            # magnification and rotation via new_and_better_scaling_routine().
            if reference==None:
                self.reference = image
                if np.isnan(image.any()):
                    raise ValueError("Must have non-NaN image to "+\
                        "crosscorrelate for scale. Reference "+\
                        "image should also be centered. Get to it.")
            else:
                self.reference=reference

            # First find the fractional-pixel centering
            if centering== "auto":
                if hasattr(self.bandpass, "__iter__"):
                    self.auto_find_center(\
                        "centermodel_poly_{0}mas.fits".format(utils.rad2mas(self.pixel)))
                else:
                    self.auto_find_center(\
                        "centermodel_{0}m_{1}mas.fits".format(self.bandpass, \
                        utils.rad2mas(self.pixel)))
                self.bestcenter = 0.5-self.over*self.xpos, 0.5-self.over*self.ypos
            else:
                self.bestcenter = centering

            # will do a search nearby the guess for
            # PSF scaling and rotation
            if pixguess==None or rotguess==None:
                print "fit image requires a pixel scale guess " + \
                    "keyword 'pixscale' and rotation guess" + \
                    "keyword 'rotguess' if no model is provided.\n" +\
                    "So it will reward you by crashing angrily" +\
                    " and in all uppercase."
                raise ValueError("MUST SPECIFY GUESSES FOR PIX & ROT")

            self.new_and_better_scaling_routine(self.reference, 
                    scaleguess=self.pixel, rotstart=rotguess,
                    centering=self.bestcenter, fitswrite=self.saveval)
            self.pixscale_measured=self.pixscale_factor*self.pixel
            print "measured pixel scale factor:" ,self.pixscale_factor
            print "measured pixel scale (mas):", \
                    utils.rad2mas(self.pixscale_measured)
            self.fov=image.shape[0]
            self.fittingmodel=self.make_model(self.fov, bandpass=self.bandpass, 
                            over=self.over, rotate=True,
                            centering=self.bestcenter, 
                            pixscale=self.pixscale_measured)
            print "MODEL GENERATED AT PIXEL {0} mas".format(\
                utils.rad2mas(self.pixscale_measured)) +\
                ", ROTATION {0} radians".format(self.rot_measured)
        else:
            self.fittingmodel=modelin
            
        if weighted is not False:
            self.soln, self.residual = leastsqnrm.weighted_operations(image, \
                        self.fittingmodel, weights=self.weighted)
        else:
            self.soln, self.residual, self.cond,self.linfit_result = \
                leastsqnrm.matrix_operations(image, self.fittingmodel, \
                                             verbose=False)
        print "NRM_Model Raw Soln:", self.soln
        self.rawDC = self.soln[-1]
        self.flux = self.soln[0]
        self.soln = self.soln/self.soln[0]
        self.deltapsin = leastsqnrm.sin2deltapistons(self.soln)
        self.deltapcos = leastsqnrm.cos2deltapistons(self.soln)
        #self.closurephase = leastsqnrm.closurephase(self.deltapsin)*2.0*np.pi
        # fringephase now in radians
        self.fringeamp, self.fringephase = leastsqnrm.tan2visibilities(self.soln)
        self.piston = utils.fringes2pistons(self.fringephase, len(self.ctrs))   #DT added
        #self.fringeamp, self.fringephase = leastsqnrm.smallanglephi(self.soln)
        #self.closurephase = leastsqnrm.closurephase(self.fringephase, N=self.N)
        self.redundant_cps = leastsqnrm.redundant_cps(self.fringephase, N=self.N)
        self.redundant_cas = leastsqnrm.return_CAs(self.fringeamp, N=self.N)

    def plot_model(self, show=False, fits_true=0):
        """
        plot_model makes an image from the object's model and fit solutions.
        """
        try:
            self.modelpsf = np.zeros((self.fov,self.fov))
        except:
            self.modelpsf = np.zeros((self.fov_sim, self.fov_sim))
            #try:
        for ind, coeff in enumerate(self.soln):
            #self.modelpsf += self.flux*coeff * self.model[:,:,ind]
            self.modelpsf += self.flux*coeff * self.fittingmodel[:,:,ind]
        if fits_true:
            # no reason to make an hdu list if not saving a fits file
            hdulist = fits.PrimaryHDU()
            hdulist.data=self.modelpsf
            #hdulist.writeto(fits, clobber=True)
        else:
            hdulist=None
        #hdulist.header.update('PIXEL', self.pixel)
        #hdulist.header.update('WAVL', self.bandpass[0])
        #hdulist.header.update('BANDSIZE', len(self.bandpass))
        #hdulist.header.update('WAVSTEP', self.bandpass[1] - self.bandpass[0])
            #hdulist.data.update('ROTRAD', self.rot_measured)
        #except AttributeError:
        #   print "A solved model has not been created for this object"
        if show:
            import matplotlib.pyplot as plt
            plt.imshow(self.modelpsf, interpolation='nearest', cmap='gray')
            plt.show()
        return self.modelpsf, hdulist

    def perfect_from_model(self, filename="perfect_from_model.fits"):
        """
        perfect_from_model makes an image with zero pistons from the model.
        this is useful for checking that the model matches a perfect analytic
        psf simulation (e.g. like one generated from simulate_mono).
        """
        self.perfect_model= np.zeros((self.fov, self.fov))
        nterms = self.N*(self.N-1) + 2
        self.perfect_soln = np.zeros(nterms)
        self.perfect_soln[0] = 1
        self.perfect_soln[-1] = 0
        for ii in range(len(self.perfect_soln)//2-1):
            self.perfect_soln[2*ii+1] = 1.
        for ind, coeff in enumerate(self.perfect_soln):
            #self.perfect_from_model += coeff * self.model[:,:,ind]
            self.perfect_model += coeff * self.fittingmodel[:,:,ind]
        fits.PrimaryHDU(data=self.perfect_model).writeto(self.datapath + \
                                  filename, \
                                  clobber=True)
        return self.perfect_model

    def new_and_better_scaling_routine(self, img, scaleguess=None, rotstart=0.0, 
                    datapath="", centering='PIXELCENTERED', fitswrite=True):
        """
        returns pixel scale factor, rotation in addition to guess, and a 
        goodness of fit parameter than can be compared in multiple iterations.
        correlations are calculated in the image plane, in anticipation of 
        data with many bad pixels.
        """
        if not hasattr(self,'datapath'):
            self.datapath=datapath
        if not hasattr(self, 'bandpass'):
            raise ValueError("This object has no specified bandpass/wavelength")
        reffov=img.shape[0]
        #scallist = np.array([0.9,0.95, 0.98, 0.99, 0.995, 0.998, 1.0,
        #           1.002, 1.005, 1.01, 1.02, 1.05, 1.1])
        #scallist = np.array([0.93, 0.933,0.934,0.935,0.936,0.937,0.94])
        #scallist= scallist[2:-2]
        #scallist = np.array([0.5, 1.0,2.0, 2.5])
        scal_corrlist = np.zeros((len(self.scallist), reffov, reffov))
        pixscl_corrlist = scal_corrlist.copy()
        scal_corr = np.zeros(len(self.scallist))
        self.pixscl_corr = scal_corr.copy()
        
        if fitswrite:
            pixfiles = [f for f in os.listdir(self.datapath+self.refdir) if 'refpsf_pixscl' in f]
        else:
            pixfiles = []
        # ---------------------------------
        # user can specify a reference set of phases (m) at an earlier point so that 
        # all PSFs are simulated with those phase pistons (e.g. measured from data at
        # an earlier iteration
        if not hasattr(self, 'refphi'):
            self.refphi = np.zeros(len(self.ctrs))
        else:
            pass
        # ---------------------------------
        # ---------------------------------
        if len(pixfiles)>4:
            print "reading in existing reference files..."
            print 'in directory {0}'.format(self.datapath+self.refdir)
            print pixfiles
            self.pixscales = np.zeros(len(pixfiles))
            pixscl_corrlist = np.zeros((len(pixfiles), reffov, reffov))
            for q, scalfile in enumerate(pixfiles):
                print q
                f = fits.open(self.datapath+self.refdir+pixfiles[q])
                psf, filehdr=f[0].data, f[0].header
                self.test_pixscale=filehdr['PIXSCL']
                self.pixscales[q] = self.test_pixscale
                f.close()
                pixscl_corrlist[q,:,:] = run_data_correlate(img,psf)
                self.pixscl_corr[q] = np.max(pixscl_corrlist[q])
                print 'max correlation',self.pixscl_corr[q]
                if True in np.isnan(self.pixscl_corr):
                    raise ValueError("Correlation produced NaNs,"\
                             " please check your work!")

        else:
            print 'creating new reference files...'
            print 'in directory {0}'.format(self.datapath+self.refdir)
            self.pixscales = np.zeros(len(self.scallist))
            for q, scal in enumerate(self.scallist):
                print q
                self.test_pixscale = self.pixel*scal
                self.pixscales[q] = self.test_pixscale
                psf = self.simulate(bandpass=self.bandpass, fov=reffov, 
                            pixel = self.test_pixscale,
                            centering=centering)#, phi=self.refphi) #Removed phi from call to simulate. DT, AS Feb 2016
                #print "WAVELENGTH DEBUG IN NEW_AND_BETTER:"
                #print self.bandpass
                #print self.lam
                #print self.simhdr['WAVL']
                # -------------------------------
                if fitswrite:
                    temphdu = fits.PrimaryHDU(data=psf, header=self.simhdr)
                    temphdu.header["pixscale"] = (self.test_pixscale, 
                                "pixel scale (rad)")
                    temphdu.writeto(self.datapath+self.refdir+\
                        "refpsf_pixscl{0}.fits".format(self.test_pixscale), 
                        clobber=True)
                # -------------------------------
                #print temphdu.header
                #pl.show()
                pixscl_corrlist[q,:,:] = run_data_correlate(img,psf)
                self.pixscl_corr[q] = np.max(pixscl_corrlist[q])
                if True in np.isnan(self.pixscl_corr):
                    raise ValueError("Correlation produced NaNs,"\
                             " please check your work!")
        np.savetxt(self.datapath+self.refdir+"pix_correlationvalues.txt", self.pixscl_corr)
        np.savetxt(self.datapath+self.refdir+"pixelscalestested_rad.txt", self.pixscales)
        np.savetxt(self.datapath+self.refdir+"pixscalestested_val.txt", self.scallist)
        self.pixscale_optimal, scal_maxy = utils.findmax(mag=self.pixscales,
                                vals=self.pixscl_corr)
        self.pixscale_factor = self.pixscale_optimal/self.pixel
        closestpixscale = self.pixscales[self.pixscl_corr==self.pixscl_corr.max()][0]
        print 'PIXSCALE DEBUG'
        print closestpixscale

        print 'NRM_Model pixel scale:',self.pixscale_optimal
        print 'fraction of guess:', self.pixscale_factor
        #radlist = 1*(np.pi/180) * np.linspace(rotstart-4.0, rotstart+4.0, 9 )
        radlist = self.rotlist_rad  #DT, AS Jan 2016
        corrlist = np.zeros((len(radlist), reffov, reffov))
        self.corrs = np.zeros(len(radlist))

        if fitswrite:
            rotfiles = [f for f in os.listdir(self.datapath+self.refdir) if 'refpsf_rot' in f]
        else:
            rotfiles = []
        if len(rotfiles)>4:
            print "reading from file"
            self.rots = np.zeros(len(rotfiles)) 
            for q, rotfilefile in enumerate(rotfiles):
                #f = fits.open(self.datapath+self.refdir+rotfiles[q])
                f = fits.open(self.datapath+self.refdir+rotfile)
                print self.datapath+self.refdir+rotfile,
                print "rotfile", rotfile
                psf, filehdr=f[0].data, f[0].header
                self.rots[q] = filehdr['ROTRAD']
                f.close()
                corrlist[q,:,:] = run_data_correlate(psf, img)
                self.corrs[q] = np.max(corrlist[q])
                print 'max rot correlation: ',self.corrs[q]
            self.corrs = self.corrs[np.argsort(self.rots)]
            self.rots = self.rots[np.argsort(self.rots)]
        else:
            self.rots = radlist
            for q,rad in enumerate(radlist):
                print 'no rotation files found, creating new...'
                psf = self.simulate(bandpass=self.bandpass, fov=reffov, 
                #pixel = closestpixscale, rotate=rad,
                pixel = self.pixscale_optimal, rotate=rad,
                centering=centering)
                # -------------------------------
                if fitswrite:
                    temphdu = fits.PrimaryHDU(data=psf, header=self.simhdr)
                    temphdu.header["pixscale"] = (self.pixscale_optimal, \
                        "pixel scale (rad)")
                    temphdu.writeto(self.datapath+self.refdir+\
                        "refpsf_rot{0}_pixel{1}.fits".format(rad, 
                        closestpixscale),clobber=True)
                # -------------------------------
                corrlist[q,:,:] = run_data_correlate(psf, img)
                self.corrs[q] = np.max(corrlist[q])
        np.savetxt(self.datapath+self.refdir+"rot_correlationvalues.txt", self.corrs)
        np.savetxt(self.datapath+self.refdir+"rotationstested_rad.txt", self.rots)
        np.savetxt(self.datapath+self.refdir+"rotationstested_deg.txt", self.rots*180.0/np.pi, fmt=self.fmt) #DT nov 20
        self.rot_measured, maxy, = utils.findmax(mag=self.rots, vals = self.corrs)
        self.refpsf = self.simulate(bandpass=self.bandpass, 
                    pixel=self.pixscale_factor*self.pixel,
                    fov=reffov, rotate=self.rot_measured,
                    centering=centering)
        print '--------------- WHAT THE MATCHING ROUTINE FOUND ---------------'
        print 'scaling factor:', self.pixscale_factor
        print 'pixel scale (mas):', utils.rad2mas(self.pixscale_optimal)
        print 'rotation (rad):', self.rot_measured
        print '--------------- -------------- ---------------'
        try:
            self.gof = goodness_of_fit(img,self.refpsf)
        except:
            print "rewrite goodness_of_fit, it failed."
            self.gof = False
        return self.pixscale_factor, self.rot_measured,self.gof


    def auto_find_center(self, modelfitsname, overwrite=0):
        """
        This is the major method in this driver to be called. It is basically
        Deepashri's cormat driver & peak finder.

        Takes an image at the detector pixel scale and a fits file name
        * tries to read in the fits file
        * if there's no file it will go on to generate an oversampled PSF
        * if the data is a cube it will generate a cube of oversampled PSFs
        * if not it will generate a single array of oversampled PSF
        * then calculates cormat & finds the peak.

        returns center location on the detector scale.

        *************************************************************
        ***                                                       ***
        ***   NOTE: to plug this centering back into NRM_Model:   ***
        ***   ( 0.5 - model_oversampling*ctr[0],                  ***
        ***     0.5 - model_oversampling*ctr[1] )                 ***
        ***                                                       ***
        *************************************************************

        """
        self.pscale_rad = self.pixel
        self.pscale_mas = utils.rad2mas(self.pscale_rad)
        _npix = self.reference.shape[1] +2
        if ( (not os.path.isfile(modelfitsname)) or (overwrite == 1)):
            # Creates a new oversampled model, default is pixel-centered
            self.simulate(fov=_npix, over=self.over, bandpass=self.bandpass)

            hdulist=fits.PrimaryHDU(data=self.psf_over)
            hdulist.header[""] = ("", "Written from auto_find_center method")
            #hdulist.header.update()
            hdulist.writeto(self.datapath+modelfitsname, clobber=True)
        else:
            # Looks for this file to read in if it's already been written and overwrite flag not set
            try:
                self.read_model(modelfitsname)
            except:
                self.psf_over = 0
            if self.psf_over.shape[0] ==self.reference.shape[0]:
                pass
            else:
                # if overwrite flag not set, but read model doesn't match, just make a new one
                self.simulate(fov=_npix, over=self.over, bandpass=self.bandpass)

        # finds correlation matrix between oversampled and detector psf
        self.cormat = utils.crosscorrelatePSFs(self.reference, self.psf_over, self.over)
        self.find_location_of_peak()

    def read_model(self, modelfitsname):
        self.psf_over = fits.getdata(self.datapath+modelfitsname)
        return self.psf_over

    def find_location_of_peak(self):
        peak_location=np.where(self.cormat==self.cormat.max())
        y_peak=peak_location[0][0]
        x_peak=peak_location[1][0]
        y_peak_ds9=y_peak+1
        x_peak_ds9=x_peak+1
        self.x_offset =  self.over - x_peak
        self.y_offset =  self.over - y_peak
        self.xpos, self.ypos = self.x_offset/float(self.over), self.y_offset/float(self.over)
        verbose=False
        if verbose:
            print "x_peak_python,y_peak_python", x_peak,y_peak
            print "x_peak_ds9,y_peak_ds9", x_peak_ds9,y_peak_ds9
            print "first value is x, second value is y"
            print "printing offsets from the center of perfect PSF in oversampled pixels..."
            print "x_offset, y_offset", self.x_offset, self.y_offset
            print "printing offsets from the center of perfect PSF in detector pixels..."
            print "x_offset, y_offset", self.xpos,self.ypos         


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
    with open(r"{0}.ffo".format(savdir, outputname), "wb") as output_file:
        json.dump(savobj, output_file)
    print "success!"

    # init stuff
    savobj.pscale_rad, savobj.pscale_mas = nrmobj.pixel, utils.rad2mas(nrmobj.pixel)
    savobj.holeshape, savobj.ctrs, savobj.d, savobj.D, savobj.N, \
        savobj.datapath, savobj.refdir  =   nrmobj.holeshape, nrmobj.ctrs, nrmobj.d, \
                                            nrmobj.D, nrmobj.N, nrmobj.datapath, nrmobj.refdir

    if hasattr(nrmobj, "refpsf"):
        savobj.refpsf, savobj.rot_best = nrmobj.refpsf, nrmobj.rot_measured
    if hasattr(nrmobj, "fittingmodel"):
        # details
        savobj.weighted, savobj.pixweight, savobj.bestcenter, \
            savobj.bandpass, savobj.modelctrs, savobj.over,\
            savobj.modelpix  =  nrmobj.weighted, nrmobj.pixweight, nrmobj.bestcenter, \
                                nrmobj.bandpass, nrmobj.modelctrs, nrmobj.over, nrmobj.modelpix
        # resulting arrays
        savobj.modelmat, savobj.soln, \
            savobj.residual, savobj.cond, \
            savobj.rawDC, savobj.flux, \
            savobj.fringeamp, savobj.fringephase,\
            savobj.cps, savobj.cas  =   nrmobj.fittingmodel, nrmobj.soln, nrmobj.residual, \
                                nrmobj.cond, nrmobj.rawDC, nrmobj.flux, nrmobj.fringeamp, \
                                nrmobj.fringephase, nrmobj.redundant_cps, nrmobj.redundant_cas
        if not hasattr(nrmobj, "modelpsf"):
            nrmobj.plot_model()
        savobj.modelpsf = nrmobj.modelpsf
    with open(r"{0}.ffo".format(savdir, outputname), "wb") as output_file:
        pickle.dump(savobj, output_file)

def save_oifits(obj, ofn, mode="single"):
    from write_oifits import OIfits
    kws = {'path':obj.datapath,\
            'arrname':'mask', \
            'PARANG':0.0, 'PA':0.0, 'flip':False}
    oif = OIfits(obj.mask,mykeywords)
    oif.mode = mode
    # wavelength infor for a single slice
    oif.nwav = 1
    if not hasattr(bandpass, "__iter__"):
        oif.wls = np.array([obj.bandpass,])
        oif.eff_band = 0.01 # close enough for this
    else:
        # need to convert wght,wl list into wls
        oif.wls=np.array(obj.bandpass)[:,1]
        oif.eff_band = abs(oif.wls[-1] - oif.wls[0]) / oif.wls[oif.wls.shape[0]/2]
    oif.wavs = oif.wls
    oif.oiwav = oifits.OI_WAVELENGTH(oif.wavs, eff_band = oif.eff_band)
    oif.wavs = {oif.isname:oif.oiwav}
    oif.dummytables()
    # manually set v2
    oif.v2 = obj.fringeamp**2
    oif.v2_err = np.ones(oif.v2) # this is raw measurement, don't know errors yet
    oif.v2flag=np.resize(False, (len(oif.v2)))
    oif.oivis2=[]
    for qq in range(len(oif.v2)):
        vis2data = oifits.OI_VIS2(oif.timeobs, oif.int_time, oif.v2[qq],\
                                  oif.v2_err[qq], oif.v2flag[qq], oif.ucoord[qq],\
                                  oif.vcoord[qq], oif.oiwav, oif.target, \
                                  array=oif.array, station=[oif.station, oif.station])
        oif.oivis2=np.array(oif.oivis2)
    
    oif.oi_data()
    #print oif.oivis2
    #print oif.oit3
    oif.write(ofn)
    return None

def goodness_of_fit(data, bestfit, diskR=8, save=False):
    mask = np.ones(data.shape) +AG.makedisk(data.shape[0], 2) -\
                    AG.makedisk(data,shape[0], diskR)
    difference = np.ma.masked_invalid(mask*(bestfit-data))
    masked_data = np.ma.masked_invalid(mask*data)
    """
    gof = sum(abs(difference[support][data != np.nan])) / \
            sum(data[support][data != np.nan])
    """
    return abs(difference).sum() / abs(masked_data).sum()

def image_plane_correlate(data,model):
    """
    From A. Greenbaum's 'centering_correlate.py'
    Modified so that instead of throwing NaNs to 0, it masks them out.
    """
    multiply = np.ma.masked_invalid(model*data)
    if True in np.isnan(multiply):
        raise ValueError("data*model produced NaNs,"\
                    " please check your work!")
    print "masked data*model:", multiply, "model sum:", model.sum()
    return multiply.sum()/((np.ma.masked_invalid(data)**2).sum())
    #return (multiply/(np.ma.masked_invalid(data)**2))
    #multiply = np.nan_to_num(model*data)
    #return multiply.sum()/(np.nan_to_num(data).sum()**2)

def run_data_correlate(data, model):
    sci = data
    print "shape sci",np.shape(sci)
    print "shape model", np.shape(model)
    return utils.rcrosscorrelate(sci, model)

def test_simulation(logger=None, datapath = "/Users/agreenba/data/hex_tests/"):

    nrmobj = NRM_Model(mask='jwst', holeshape='hex', pixscale=mas2rad(64), rotate=np.pi/2.0,\
                log=logger)
    psf = nrmobj.simulate(fov=121, lam=4.3*um, over=False, phi=None, \
                    centering=(0.5, 0.5))
    pl.imshow(psf)
    testfile = "analytic_psf.pdf"
    pl.savefig(datapath+testfile)
    fits.PrimaryHDU(data=psf).writeto(nrmobj.datapath+testfile.replace("pdf", "fits"), clobber=True)

    """
    niriss = NIRISS()
    niriss.pupilopd
    niriss.pupil_mask = "MASK_NRM"
    result = niriss.calcPSF(oversample=1, fov_pixels=121, monochromatic=4.3*um, rebin=True)
    result.writeto(datapath+"webbpsf_psf.fits", clobber=True)
    del niriss
    del result
    """

    return nrmobj

def test_model(nrmobj, datapath = "/Users/agreenba/data/hex_tests/"):
    psf = nrmobj.simulate(fov=121, lam=4.3*um, over=False, phi = 'nb', \
                            centering=(0.0, 0.0))
    model = nrmobj.make_model(fov=121, bandpass=4.3*um)
    nrmobj.fit_image(psf, model)
    print nrmobj.soln, nrmobj.fringephase, nrmobj.closurephase
    print "Hole Shape:", nrmobj.holeshape
    print "compare to: ", nrmobj.perfect_deltap
    
    fits.PrimaryHDU(data=nrmobj.model_beam).writeto(nrmobj.datapath+"primarybeam.fits", clobber=True)
    print "SUM OF PSF", (nrmobj.model_beam.sum())
    fits.PrimaryHDU(data=nrmobj.residual).writeto(nrmobj.datapath+"analytichexres.fits", clobber=True)
    modelpsf = nrmobj.plot_model(fits=datapath+"model_test.fits")
    nrmobj.datapath="/Users/agreenba/data/hex_tests/"
    nrmobj.perfect_from_model()


def test_auto_fit():
    #generate some test data off-center... test with 5x oversampled
    """

    +   +   +   +   +   +

    +   +   +   +   +   +

    +   +   +   +   +   +
              .........
    +   +   +   +   + . +
                      *
    +   +   +   +   +   +

    +   +   +   +   +   +

    NRM_Model input calculation on detector pixel scale...
    n.b. model_oversampling is 1 since the NRM_Model thinking is now in detpix:

    ** ( 0.5 - model_oversampling*ctr[0], =  0.5detpix - (2/5)detpix =    0.5 - 0.4 = 0.1 detpix
    **   0.5 - model_oversampling*ctr[1] )=  0.5detpix - (-1/5)detpix  =  0.5 + 0.2 = 0.7 detpix
     
    ---------------------
    |                   |
    |                   |
    |*                  |
    |.                  |  (0.1, 0.7)
    |.                  |
    |.                  |
    |.                  |
    |.                  |
    |.                  |
    ..-------------------

    """
    pscale_rad = mas2rad(64)
    bandpass = 4.3*1.0e-6
    sim = NRM_Model(mask = 'jwst', pixscale=pscale_rad, datapath = "nrmmodeltests/",
                    scallist =np.array([ 0.998, 0.999, 1.0, 1.001, 1.002, ]))
    simover = 5
    overctrpos = (2, -2) # this means that the single pixel position is (2/5, -2/5)
                         # which is why there is a /simover below 
                         # (the result will not be in oversampled pixel units)
                         # 
    simctr = (0.5-(simover*overctrpos[0]/simover), 0.5-(simover*overctrpos[1]/simover))
    simpsf = sim.simulate(fov=101, over=simover, centering = simctr, bandpass = bandpass)

    print "---------------------------------------"
    print "Begin test"
    print "---------------------------------------"
    print "simulated centering:", simctr
    sim.reference = sim.psf #set the data reference to our simulated psf
    sim.auto_find_center("test_centering_mono4.3um_jwst.fits")


    print "--------------------------------------------------------"
    print "measured centering: ", 0.5-sim.xpos, 0.5-sim.ypos
    print "compared to:        ", 0.5-overctrpos[0]/float(simover),\
                          0.5-overctrpos[1]/float(simover), "simulated centering."

    print "Do these match? Then the test was successful!"
    print "--------------------------------------------------------"
    newpsf = sim.simulate(fov=101, over=simover, bandpass = bandpass, \
                 centering=(0.5-simover*sim.xpos, 0.5-simover*sim.ypos) )

    print "--------------------------------------------------------"
    print "Auto fit the data...."
    # now check the method implementaiton does the right thing.
    sim.fit_image(simpsf, pixguess = pscale_rad, centering="auto")
    #sim.new_and_better_scaling_routine(img = simpsf, scaleguess=pscale_rad, \
    #                                   centering="auto", fitswrite=False)
    print "ran auto fit, here are the results:"
    print "best center:", sim.bestcenter
    print "best pixelscale:", utils.rad2mas(sim.pixscale_measured)
    print "best rotation:", sim.rot_measured
    print "solutions:", sim.soln
    print "--------------------------------------------------------"

    import matplotlib.pyplot as plt
    if True:
        if True:
            plt.figure()
            plt.title("Pixel scale vs. correlation w/image")
            plt.plot(analyticnrm2.rad2mas(sim.pixscales), sim.pixscl_corr)
            plt.ylabel("correlation")
            plt.xlabel("pixel size (mas)")
            plt.vlines(analyticnrm2.rad2mas(sim.pixscale_optimal), sim.pixscl_corr[0],
                        sim.pixscl_corr[-1],
                        linestyles='--', color='r')
            plt.text(analyticnrm2.rad2mas(sim.pixscales[1]),sim.pixscl_corr[1],
                'best fit at {0}'.format(analyticnrm2.rad2mas(sim.pixscale_optimal)))
            plt.savefig(sim.datapath+\
                   '/pixscalecorrelation.pdf')
            plt.figure()
            plt.title("Additional rotation vs. correlation w/image")
            plt.ylabel("correlation")
            sim.rotation=0.0
            plt.xlabel("rotation + "+str(180.*sim.rotation/np.pi)+"(deg)")
            plt.plot(sim.rots, sim.corrs)
            plt.vlines(sim.rot_measured, sim.corrs[0],
                        sim.corrs[-1],
                        linestyles='--', color='r')
            plt.text(sim.rots[1],sim.corrs[1],
                'best fit at {0}'.format(sim.rot_measured))
            plt.savefig(sim.datapath+\
                   '/rotationcorrelation.pdf')

    plt.figure()
    plt.subplot2grid((1, 4), (0, 0))
    sim.plot_model()
    plt.title("solution")
    plt.imshow(sim.modelpsf, cmap="bone")
    plt.colorbar()
    plt.subplot2grid((1, 4), (0,1))
    plt.title("simulated")
    plt.imshow(simpsf, cmap="bone")
    plt.colorbar()
    plt.subplot2grid((1, 4), (0,2))
    plt.title("reference")
    plt.imshow(sim.refpsf, cmap="bone")
    plt.colorbar()
    plt.subplot2grid((1, 4), (0,3))
    plt.title("residual")
    plt.imshow(sim.residual, cmap="bone")
    plt.colorbar()
    plt.show()


    plt.figure()
    plt.subplot2grid((1, 3), (0, 0))
    plt.title("simulated PSF")
    plt.imshow(simpsf, cmap = 'bone')
    plt.subplot2grid((1, 3), (0,1))
    plt.title("Measured centering plugged back in")
    plt.imshow(newpsf, cmap = 'bone')
    plt.subplot2grid((1, 3), (0,2))
    plt.title("difference")
    plt.imshow(simpsf-newpsf, cmap = 'bone')
    plt.show()
    
if __name__ == "__main__":

    # create logger
    logger = logging.getLogger(__name__)

    # define a handler to write log messages to stdout
    sh = logging.StreamHandler(stream=sys.stdout)

    # define the format for the log messages, here: "level name: message"
    formatter = logging.Formatter("[%(levelname)s]: %(message)s")
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # create a parser object for understanding command-line arguments
    parser = ArgumentParser(description="This script tests the NRM_Model object")

    # add two boolean arguments: 'verbose' for dumping debug messages and
    #   highers, and 'quiet' for restricting to just error messages
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", 
            default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", 
            default=False, help="Be quiet! (default = False)")

    # add a required, integer argument
    parser.add_argument("-p", dest="plot", action="store_true", default=False,
            help="Generate a plot or not")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    
    logging.basicConfig(level=logging.INFO, format='%(name)-10s: %(levelname)-8s %(message)s')

    """
    THESE ARE THE TESTS
    You can set the datapath here.
    """
    
    test_auto_fit()
    #test_nrm = test_simulation(logger=logger)
    #test_model(test_nrm )
