#! /usr/bin/env  python 
# Heritage mathematia nb from Alex & Laurent
# Heritage python by Alex Greenbaum & Anand Sivaramakrishnan Jan 2013
# updated May 2013 to include hexagonal envelope

# Goals: a more convenient module for analytic simulation & model fitting
"""
Imports:
"""
from __future__ import print_function
import numpy as np
import scipy.special
import hexee
import nrm_analysis.misctools.utils as utils
from nrm_analysis.misctools.utils import rebin
import sys, os
import time
from astropy.io import fits
import logging
from argparse import ArgumentParser
from . import leastsqnrm as leastsqnrm
from . import analyticnrm2
from . import subpix
import uncertainties

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
m = 1.0
mm = 1.0e-3 * m
um = 1.0e-6 * m
mas = 1.0e-3 / (60*60*180/np.pi) # in radians

def mas2rad(mas):
    rad = mas*(10**(-3)) / (3600*180/np.pi)
    return rad

class NRM_Model():

    def __init__(self, mask=None, holeshape="circ", pixscale=None, \
            over = 1, log=_default_log, pixweight=None,\
            datapath="", refdir="",chooseholes=False, **kwargs):
        """
        mask: an NRM_mask_geometry object.
        pixscale should be input in radians; if left as None, must be updated
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
        try:
            print(mask.ctrs)
            
        except AttributeError:
            raise AttributeError("mask must be either 'jwst' \
                                  or NRM_mask_geometry object")
        self.ctrs, self.d, self.D = np.array(mask.ctrs), mask.hdia, mask.activeD

        self.N = len(self.ctrs)
        self.datapath=datapath
        self.refdir = refdir
        self.fmt = "%10.6e"      #DT nov 20 

        # AG 2018: removed phi input, rotation/scaling lists from init
        #          all this should be done OUTSIDE of init

        # From Anand for debugging:
        self.chooseholes = chooseholes
        # (Default) - use set_pistons to change. This is really only for tests.
        self.phi = np.zeros(self.N)

    def set_pistons(self, phi_m):
        """Meters of OPD at center wavelength LG++ """
        self.phi = phi_m

    def set_pixelscale(self, pixel_rad):
        """Detector pixel scale (isotropic) """
        self.pixel = pixel_rad

    def simulate(self, fov=None, bandpass=None, over=1, 
                 psf_offset=(0,0), **kwargs):
        #pixweight=None,pixel=None,rotate=False, centering = "PIXELCENTERED"):
        """
        This method will simulate a detector-scale PSF using parameters input from the 
        call and already stored in the object. It also generates a simulation 
        fits header storing all of the parameters used to generate that PSF.
        If the input bandpass is one number it will calculate a monochromatic
        PSF.
        Use set_pistons() to set self.phi if desired
        fov in detector pixels, must be specified, integer

        AS 2018:
        psf_offset (in detector pixels, PSF center offset from center of array)
            - PSF then centered at centerpoint(shape) + psf_offset
            - Works for both odd and even shapes.
            - default (0,0) - no offset from array center
            - places psf ctr at array ctr + offset, per ds9 view of offset[0],offset[1]

        over (integer) is the oversampling of the simulation  
            - defaults to no oversampling (i.e. 1)

        All mask rotations must be done outside simulate() prior to call

        Returns psf array but stores hdr and detector-scale psf in 'self'
        """
        self.simhdr = fits.PrimaryHDU().header
        self.bandpass = bandpass
        # Set up conditions for choosing various parameters
        if fov == None:
            raise ValueError("Please specify a field of view value")
        else:
            pass

        self.simhdr['PIXWGHT']= (False, "True or false, whether there is sub-pixel weighting")
        if 'pixweight' in kwargs:
            over=pixweight.shape[0]
            self.simhdr['PIXWGHT'] = True
        else:
            pass
        self.simhdr['OVER'] = (over, 'oversampling')
        self.simhdr['PIX_OV'] = (self.pixel/float(over), 'Sim pixel scale in radians')
        self.simhdr["psfoff0"] = (psf_offset[0], "psf ctr at arrayctr + off[0], detpix") # user-specified
        self.simhdr["psfoff1"] = (psf_offset[1], "psf ctr at arrayctr + off[1], detpix") # user-specified

        # Polychromatic or Monochromatic, in the same loop:
        if hasattr(self.bandpass, '__iter__'):
            self.logger.debug("------Simulating Monochromatic------")
            simbandpass = [(1.0, bandpass)]
        else:
            self.logger.debug("------Simulating Polychromatic------")
            simbandpass = bandpass

        self.psf_over = np.zeros((over*fov, over*fov))
        nspec=0
        for w,l in self.bandpass: # w: weight, l: lambda (wavelength)
            print("weight:", w, "wavelength:", l)
            print("fov:", fov)
            print("over:",over)
            print("pixel:", self.pixel)

            self.psf_over += w*analyticnrm2.PSF(self.pixel,# det pixel scale, rad
                                                fov,       # in detpix number
                                                over,      
                                                self.ctrs, # live hole centers in object
                                                self.d, l, 
                                                self.phi, 
                                                psf_offset = psf_offset, # det pixels
                                                shape=self.holeshape)
            # AS 2018: offset signs fixed to agree w/DS9, +x shifts ctr R, +y shifts up
            self.simhdr["WAVL{0}".format(nspec)] = (l, "wavelength (m)")
            self.simhdr["WGHT{0}".format(nspec)] = (w, "weight")
            nspec=nspec+1


        self.logger.debug("BINNING UP TO PIXEL SCALE")

        if self.pixweight==None:
            self.psf = utils.rebin(self.psf_over, (over, over))
        else:
            print("using a weighted pixel")
            print("\neffective oversampling:")
            print(self.pixweight.shape[0])
            self.psf=subpix.weightpixels(self.psf_over,\
                self.pixweight)

        return self.psf

    def make_model(self, fov=None, bandpass=None, over=1,psf_offset=(0,0),
                pixweight=None, pixscale=None):
        
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
        self.fov = fov
        self.over = over

        if pixweight is not None:
            self.over=self.pixweight.shape[0]
        if hasattr(self, 'pixscale_measured'):
            if self.pixscale_measured is not None:
                self.modelpix = self.pixscale_measured
        if pixscale==None:
            self.modelpix=self.pixel
        else:
            self.modelpix=pixscale

        self.modelctrs = self.ctrs

        if hasattr(self.bandpass, '__iter__') == False:
            self.logger.debug("------Simulating monochromatic------")
            simbandpass = [(1.0, bandpass)]
        else:
            self.logger.debug("------Simulating Polychromatic------")
            simbandpass = bandpass

            #self.lam = bandpass
            #self.bandpass = [(1., self.lam),]
        self.model = np.zeros((self.fov, self.fov, self.N*(self.N-1)+2))
        self.model_beam = np.zeros((self.over*self.fov, self.over*self.fov))
        self.fringes = np.zeros((self.N*(self.N-1)+1, \
                    self.over*self.fov, self.over*self.fov))
        for w,l in simbandpass: # w: weight, l: lambda (wavelength)

            print("weight: {0}, lambda: {1}".format(w,l))
            # model_array returns the envelope and fringe model (a list of oversampled fov x fov slices)
            pb, ff = analyticnrm2.model_array(self.modelctrs, l, self.over, \
                              self.modelpix,\
                              self.fov, \
                              self.d, \
                              shape=self.holeshape, \
                              psf_offset=psf_offset,\
                              verbose=False)
            self.logger.debug("Passed to model_array: psf_offset: {0}".format(psf_offset))
            self.logger.debug("Primary beam in the model created: {0}".format(pb))
            self.model_beam += pb
            self.fringes += ff

            # this routine multiplies the envelope by each fringe "image"
            self.model_over = analyticnrm2.multiplyenv(self.model_beam, self.fringes)
            print("NRM MODEL model shape:", self.model_over.shape)

            model_binned = np.zeros((self.fov,self.fov, self.model_over.shape[2]))
            # loop over slices "sl" in the model
            for sl in range(self.model_over.shape[2]):
                model_binned[:,:,sl] =  utils.rebin(self.model_over[:,:,sl],  (self.over, self.over))

            self.model += w*model_binned
        
        return self.model

    def fit_image(self, image, modelin=None, weighted=False, psf_offset=(0,0)):
        """
        fit_image will run a least-squares fit on an input image.
        Specifying a model is now required!!
        All scaling/centering/rotation must be done OUTSIDE this module!!


        fit_image computes following stored in attributes:
        - soln: Model coefficients, normalized by total flux
        - flux: multiplicative factor used to normalize soln
        - rawDC: constant term over the image (e.g. background)
        ------------------------------------------------------------------
        ### The important stuff:
        - fringeamp & fringe phase: amplitudes and phases of fringes
        - redundant_cps: All combinations of triangles for closure phase
        - redundant_cas: All combinations of squares for closure amplitude
        - All of the above _cov propogating uncertainties from fringefitting
        ------------------------------------------------------------------
        - piston: Reducing fringe phases back to hole piston (i.e. wavefront info)
        """
        self.model_in=modelin
        self.weighted=weighted
        
        self.fittingmodel=modelin
            
        if weighted is not False:
            self.soln, self.residual = leastsqnrm.weighted_operations(image, \
                        self.fittingmodel, weights=self.weighted)
        else:
            self.soln, self.residual, self.cond,self.linfit_result = \
                leastsqnrm.matrix_operations(image, self.fittingmodel, \
                                             verbose=False)
        print("NRM_Model Raw Soln:")
        print(self.soln)
        ###########################################################################
        # AG 2018: put JS's linfit solution into this step
        soln_cov_orig = uncertainties.correlated_values(self.linfit_result.p, 
                            np.array(self.linfit_result.p_normalised_covariance_matrix))
        self.fringephase_cov, \
            self.fringeamp_cov, \
            self.redundant_cps_cov, \
            self.redundant_cas_cov = \
                        leastsqnrm.phases_and_amplitudes(soln_cov_orig, N=self.N)
        ###########################################################################
        self.rawDC = self.soln[-1]
        self.flux = self.soln[0]
        self.soln = self.soln/self.soln[0]

        # fringephase now in radians
        self.fringeamp, self.fringephase = leastsqnrm.tan2visibilities(self.soln)
        self.redundant_cps = leastsqnrm.redundant_cps(self.fringephase, N=self.N)
        self.redundant_cas = leastsqnrm.return_CAs(self.fringeamp, N=self.N)
        # Handy if looking at phase errors in the pupil!
        self.piston = utils.fringes2pistons(self.fringephase, len(self.ctrs))   #DT added

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
    print("success!")

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
    print("masked data*model:", multiply, "model sum:", model.sum())
    return multiply.sum()/((np.ma.masked_invalid(data)**2).sum())
    #return (multiply/(np.ma.masked_invalid(data)**2))
    #multiply = np.nan_to_num(model*data)
    #return multiply.sum()/(np.nan_to_num(data).sum()**2)

def run_data_correlate(data, model):
    sci = data
    print("shape sci",np.shape(sci))
    print("shape model", np.shape(model))
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
    print(nrmobj.soln, nrmobj.fringephase, nrmobj.closurephase)
    print("Hole Shape:", nrmobj.holeshape)
    print("compare to: ", nrmobj.perfect_deltap)
    
    fits.PrimaryHDU(data=nrmobj.model_beam).writeto(nrmobj.datapath+"primarybeam.fits", clobber=True)
    print("SUM OF PSF", (nrmobj.model_beam.sum()))
    fits.PrimaryHDU(data=nrmobj.residual).writeto(nrmobj.datapath+"analytichexres.fits", clobber=True)
    modelpsf = nrmobj.plot_model(fits=datapath+"model_test.fits")
    nrmobj.datapath="/Users/agreenba/data/hex_tests/"
    nrmobj.perfect_from_model()
    
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
