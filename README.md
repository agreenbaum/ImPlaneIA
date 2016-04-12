# nrm_analysis README #

Reduces aperture masking images to fringe observables, calibrates, does basic model fitting. Package development led by Alexandra Greenbaum following legacy code by Greenbaum, Anand Sivaramakrishnan, and Laurent Pueyo. Contributions from Sivaramakrishnan, Deepashri Thatte, and Johannes Sahlmann.

Necessary Python packages:

* numpy
* scipy
* astropy
* a copy of Paul Boley's oifits.py in your python path

Optional Python packages:

* matplotlib
* webbpsf
* poppy
* pysynphot

*we recommend downloading the anaconda distribution for python*


### Modules: ###

* **FringeFitter** - fit fringes in the image plane. Support for masks on GPI, VISIR, and JWST-NIRISS
* **Calibrate** - calibrate raw frame phases, closure phase, and squared visibilities, save to oifits option
* **Analyze** - support for basic binary model fitting of calibrated oifits files

### Using this package ###

**Step 1**: *fit fringes in the image plane.* 

initializes NRM_model & generates best model to match the data (fine-tuning
	  centering, scaling, and rotation). Fits the data, saves to text files.
	  This driver creates a new directory for each target and saves the 
	  solution files described in the driver for each datacube slice.

**Step 2**: *calibrate the data.* 

User defines target and calibration sources in a list [target, cal, cal, ...], 
      pointing to the directories containing each source's measured fringe observables
      computed from driver 1. Calculates mean and standard error, calibrates target 
      observables with calibrator observables and does basic error propogation. Saves
      calibrated quantities to new folder "calibrated/"

**Step 3**: *compare the data to a model (3 options).*

1. _Coarse search_ - calculates log likelihood in a coarse range of parameters and returns 
                the highest logl set, which can be used as a first guess for the mcmc fit
2. _widget plot_ - plots measured closure phases against model closure phases, tunable 
                parameter -- a handy tool to visualize the data
3. _mcmc_ - uses **emcee** code to find the best fit binary model to the measured fringe observables

## Basic tutorial ##
First step import the main modules in this package:

	from nrm_analysis import InstrumentData, nrm_core

###NIRISS Example ###
Start with test data provided in this package

	nirissfiles = [f for f in os.listdir("f430_data") if "tcube" in f] # simulated data
	nirissdata = InstrumentData.NIRISS(filt="F430M", objname="targ")

Instance of NIRISS, which will set up the data according to NIRISS standards given a filter name and the name of the observed object. Now let's say I have 1 target and 2 calibrators:

	ff =  nrm_core.FringeFitter(nirissdata, oversample = 5, savedir="targ", datadir="f430_data", npix=121)
	for exposure in nirissfiles:
		ff.fit_fringes(exposure)
		
	ff1 =  nrm_core.FringeFitter(nirissdata, oversample = 5, savedir="cal1", datadir="f430_data", npix=121)
	for exposure in nirissfiles:
		ff1.fit_fringes(exposure)
	
	ff2 =  nrm_core.FringeFitter(nirissdata, oversample = 5, savedir="cal2", datadir="f430_data", npix=121)
	for exposure in nirissfiles:
		ff2.fit_fringes(exposure)
	
This initializes the fringe fitter with the options you want for measuring fringe observables. Then it fits each fits file's data by calling the fit_fringes method. Saves the output to directory "targ", etc. in working directory. I usually name this by the object name. 

	targdir = "targ/"
	caldir = "cal1/"
	cal2dir = "cal2/"
	calib = nrm_core.Calibrate([targdir, caldir, cal2dir], nirissdata, savedir = "my_calibrated")

Instance of Calibrate, gives 3 directories containing target and any calibration sources. The first directory in the list is always assumed to be the science target. Any number of calibrators may be provided. Argument savedir default is "calibrated." Argument sub_dir_tag not provided here because there is no wavelength axis (see below in GPI example for comparison).

	calib.save_to_oifits("niriss_test.oifits")
Saves results to oifits. phaseceil keyword arg optional to set a custom dataflag. Default flag is set when phases exceed  1.0e1.

### GPI Example ###
e.g., starting with a list of GPI files

	from nrm_analysis import InstrumentData, nrm_core
	
	gpifiles = [S20130501S00{0:02d}_spdc.fits.format(q) for q in np.arange(10)] # list files in here, e.g., spectral datacubes from May 1 2013.

	gpidata = InstrumentData.GPI(gpifiles[0]) # just need a reference file to get header information for this observation. Recommended to use one of the science target files

This creates an instance of GPI, which will read header keywords from a reference file and sets up the data according to GPI standards


	ff =  nrm_core.FringeFitter(gpidata, oversample = 5, savedir="Target", datadir=datadir, npix=121)
	for exposure in gpifiles:
		ff.fit_fringes(exposure)

This initializes the fringe fitter with the options you want for measuring fringe observables. Then it fits each fits file's data by calling the fit_fringes method. Saves the output to directory "Target/" in working directory. I usually name this by the object name. 

	targdir = "Target/"
	caldir = "Calibrator1/"
	cal2dir = "Calibrator2/"
	calib = nrm_core.Calibrate([targdir, caldir, cal2dir], gpidata, savedir = "my_calibrated", sub_dir_tag = "130501")


Instance of Calibrate, gives 3 directories containing target and any calibration sources. The first directory in the list is always assumed to be the science target. Any number of calibrators may be provided. Argument savedir default is "calibrated." Argument sub_dir_tag must be provided if there is an additional axis (multiple wavelengths, or pollarizations), to save results from each slice into sub directories separated by exposure.
 
	calib.save_to_oifits("targ_vis.oifits", phaseceil = 5.0)
Saves results to oifits. phaseceil keyword arg can be used to flag phases (> 0.5 degrees in this case). Default is 1.0e1.