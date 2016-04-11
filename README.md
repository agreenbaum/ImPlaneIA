# nrm_analysis README

Reduces aperture masking images to fringe observables, calibrates, does basic model fitting. Pacakage led by Alexandra Greenbaum following legacy code by Greenbaum, Anand Sivaramakrishnan, Laurent Pueyo, and with contributions from Sivaramakrishnan, Deepashri Thatte, and Johannes Sahlmann.


Necessary Python packages:
* numpy
* astropy
* pysynphot

Optional Python packages:
* matplotlib
* poppy
* a copy of Paul Boley's oifits.py in your python path

[we recommend downloading the anaconda distribution for python]


#Modules:#

* NRM_Fringes - fit fringes in the image plane. Support for masks on GPI, VISIR, and JWST-NIRISS
* NRM_Calibrate - calibrate raw frame phases, closure phase, and squared visibilities, save to oifits option
* NRM_Analyze - support for basic binary model fitting of calibrated oifits files

driver 1 - initializes NRM_model & generates best model to match the data (fine-tuning
	  centering, scaling, and rotation). Fits the data, saves to text files.
	  This driver creates a new directory for each target and saves the 
	  solution files described in the driver for each datacube slice.

driver 2 - User defines target and calibration sources in a list [target, cal, cal, ...], 
      pointing to the directories containing each source's measured fringe observables
      computed from driver 1. Calculates mean and standard error, calibrates target 
      observables with calibrator observables and does basic error propogation. Saves
      calibrated quantities to new folder "calibrated/"

driver 3 - 3 Main routines:
* Coarse search - calculates log likelihood in a coarse range of parameters and returns 
                the highest logl set, which can be used as a first guess for the mcmc fit
* widget plot - plots measured closure phases against model closure phases, tunable 
                parameter -- a handy tool to visualize the data
* mcmc - uses emcee code to find the best fit binary model to the measured fringe observables

### Basic tutorial ###
e.g., starting with a list of GPI files

gpifiles = []

        import InstrumentData
        gpidata = InstrumentData.GPI(gpifiles[1]) # just need a reference file to get header information for this observation

This creates an instance of GPI, which will read header keywords from a reference file and sets up the data according to GPI standards

    :::python
        import NRM_Fringes  

        ff =  NRM_Fringes.FringeFitter(gpidata, oversample = 5, savedir=savedir, datadir=datadir, npix=121)
        for exposure in gpifiles:
            ff.fit_fringes(exposure)

This initializes the fringe fitter with the options you want for measuring fringe observables. Then it fits each fits file's data by calling the fit_fringes method


