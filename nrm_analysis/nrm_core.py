#! /usr/bin/env python

"""
by A. Greenbaum & A. Sivaramakrishnan 
April 2016 agreenba@pha.jhu.edu

Contains 

FringeFitter - fit fringe phases and amplitudes to data in the image plane

Calibrate - Calibrate target data with calibrator data

BinaryAnalyze - Detection, mcmc modeling, visualization tools
                Much of this section is based on tools in the pymask code
                by F. Martinache, B. Pope, and A. Cheetham
                We especially thank A. Cheetham for help and advising for
                developing the analysis tools in this package. 

    LG++ anand@stsci.edu nrm_core changes:
        Removed use of 'centering' parameter, switch to psf_offsets, meant to be uniformly ds9-compatible 
        offsets from array center (regardless of even/odd array sizes).

            nrm_core.fit_image(): refslice UNTESTED w/ new utils.centroid()
            nrm_core.fit_image(): hold_centering UNTESTED w/ new utils.centroid()

"""


from __future__ import print_function
# Standard imports
import os, sys, time
import numpy as np
from astropy.io import fits
from scipy.special import comb
from scipy.stats import sem, mstats
import pickle as pickle
import matplotlib.pyplot as plt

#import nrm_analysis.oifits as oifits

# Module imports
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.misctools import utils  # AS LG++
from nrm_analysis.misctools.utils import mas2rad, baselinify, rad2mas
from nrm_analysis.modeling.binarymodel import model_cp_uv, model_allvis_uv, model_v2_uv, model_t3amp_uv
from nrm_analysis.modeling.multimodel import model_bispec_uv

from multiprocessing import Pool

class FringeFitter:
    def __init__(self, instrument_data, **kwargs):
        """
        Fit fringes in the image plane

        Takes an instance of the appropriate instrument class
        Various options can be set

        kwarg options:
        oversample - model oversampling (also how fine to measure the centering)
        psf_offset - If you already know the subpixel centering of your data, give it here
                     (not recommended except when debugging with perfectly know image placement))
        savedir - Where do you want to save the new files to? Default is working directory.
        datadir - Where is your data? Default is working directory.
        npix - How many pixels of your data do you want to use? 
               Default is the shape of a data [slice or frame].  Typically odd?
        debug - will plot the FT of your data next to the FT of a reference PSF.
                Needs poppy package to run
        verbose_save - saves more than the standard files
        interactive - default True, prompts user to overwrite/create fresh directory.  
                      False will overwrite files where necessary.

        auto_pixscale - will search for the best pixel scale value for your data given instrument geometry
        auto_rotate - will search for the best rotation value for your data given instrument geometry

        main method:
        * fit_fringes


        """
        self.instrument_data = instrument_data

        #######################################################################
        # Options
        if "oversample" in kwargs:
            self.oversample = kwargs["oversample"]
        else:
            #default oversampling is 3
            self.oversample = 3
        if "auto_rotate" in kwargs:
            # can be True/False or 1/0
            self.auto_rotate = kwargs["auto_rotate"]
        else:
            self.auto_rotate = False
        if "centering" in kwargs or "psf_offset" in kwargs: # if so do not find center of image in data
            self.hold_centering = kwargs["psf_offset"] # already-known psf offset from array ctr
            if "centering" in kwargs:
                print("**** FringeFitter class: hold_centering deprecated.  Use psf_offset ***")
        else:
            # default is auto centering, governed by hold_centering == False:
            self.hold_centering = False
        if "savedir" in kwargs:
            self.savedir = kwargs["savedir"]
        else:
            self.savedir = os.getcwd()
        #if "datadir" in kwargs:
        #    self.datadir = kwargs["datadir"]
        #else:
        #    sys.exit("""
        #    FringeFitter: datadir missing.  Where is your data?  Fragile option of 
        #    default datadir=os.getcwd() removed  - AS 12/2018 affinedev merge
        #    """)
        if "npix" in kwargs:
            self.npix = kwargs["npix"]
        else:
            self.npix = 'default'
        if "debug" in kwargs:
            self.debug=kwargs["debug"]
        else:
            self.debug=False
        if "verbose_save" in kwargs:
            self.verbose_save = kwargs["verbose_save"]
        else:
            self.verbose_save = False
        if 'interactive' in kwargs:
            self.interactive = kwargs['interactive']
        else:
            self.interactive = True
        if "save_txt_only" in kwargs:
            self.save_txt_only = kwargs["save_txt_only"]
        else:
            self.save_txt_only = False
        #######################################################################


        #######################################################################
        # Create directories if they don't already exit
        try:
            os.mkdir(self.savedir)
        except:
            if self.interactive is True:
                print(self.savedir+" Already exists, rewrite its contents? (y/n)")
                ans = input()
                if ans == "y":
                    pass
                elif ans == "n":
                    sys.exit("use alternative save directory with kwarg 'savedir' when calling FringeFitter")
                else:
                    sys.exit("Invalid answer. Stopping.")
            else:
                pass

        #######################################################################

        #--------------------------------------------------------------------------
        #05/18 2017: commented these lines out because this should all be 
        #            saved in the InstrumentData object (and it crashes for polz data
        #            which is polychrom & has a 3rd axis).
        #np.savetxt(self.savedir+"/coordinates.txt", self.instrument_data.mask.ctrs)
        #np.savetxt(self.savedir+"/wavelengths.txt", self.instrument_data.wls[0])
        #--------------------------------------------------------------------------

        #nrm = NRM_Model(mask = self.instrument_data.mask, pixscale = self.instrument_data.pscale_rad, over = self.oversample, holeshape=self.instrument_data.holeshape)
        #print nrm.holeshape
        # In future can just pass instrument_data to NRM_Model

        #plot conditions
        #if self.debug==True or self.auto_scale==True or self.auto_rotate==True:
        #    import matplotlib.pyplot as plt
        #if self.debug==True:
        #    import poppy.matrixDFT as mft

    ###
    # May 2017 J Sahlmann updates: parallelized fringe-fitting!
    ###

    def fit_fringes(self, fns, threads = 0):
        if type(fns) == str:
            fns = [fns, ]

        # Can get fringes for images in parallel
        #tore_dict = [{"object":self, "file":self.datadir+"/"+fn,"id":jj} \ # AS remove self.datadir
        store_dict = [{"object":self, "file":                 fn,"id":jj} \
                        for jj,fn in enumerate(fns)]

        t2 = time.time()
        for jj, fn in enumerate(fns):
            #it_fringes_parallel({"object":self, "file": self.datadir+"/"+fn,\ # AS remove self.datadir
            fit_fringes_parallel({"object":self, "file":                  fn,\
                                  "id":jj}, threads)
        t3 = time.time()
        print("Parallel with {0} threads took {1}s to fit all fringes".format(\
               threads, t3-t2))


    def save_output(self, slc, nrm):
        # cropped & centered PSF
        if self.save_txt_only==False:
            fits.PrimaryHDU(data=self.ctrd, \
                    header=self.scihdr).writeto(self.savedir+\
                    self.sub_dir_str+"/centered_{0}.fits".format(slc), \
                    overwrite=True)

            model, modelhdu = nrm.plot_model(fits_true=1)
            # save to fits files
            fits.PrimaryHDU(data=nrm.residual).writeto(self.savedir+\
                        self.sub_dir_str+"/residual_{0:02d}.fits".format(slc), \
                        overwrite=True)
            modelhdu.writeto(self.savedir+\
                        self.sub_dir_str+"/modelsolution_{0:02d}.fits".format(slc),\
                        overwrite=True)
        else:
            print("NOT SAVING ANY FITS FILES. SET save_txt_only=False TO SAVE.")

        # default save to text files
        np.savetxt(self.savedir+self.sub_dir_str+\
                   "/solutions_{0:02d}.txt".format(slc), nrm.soln)
        np.savetxt(self.savedir+self.sub_dir_str+\
                   "/phases_{0:02d}.txt".format(slc), nrm.fringephase)
        np.savetxt(self.savedir+self.sub_dir_str+\
                   "/amplitudes_{0:02d}.txt".format(slc), nrm.fringeamp)
        np.savetxt(self.savedir+self.sub_dir_str+\
                   "/CPs_{0:02d}.txt".format(slc), nrm.redundant_cps)
        np.savetxt(self.savedir+self.sub_dir_str+\
                   "/CAs_{0:02d}.txt".format(slc), nrm.redundant_cas)

        # optional save outputs
        if self.verbose_save:
            np.savetxt(self.savedir+self.sub_dir_str+\
                       "/condition_{0:02d}.txt".format(slc), nrm.cond)
            np.savetxt(self.savedir+self.sub_dir_str+\
                       "/flux_{0:02d}.txt".format(slc), nrm.flux)
          
        print(nrm.linfit_result)
        if nrm.linfit_result is not None:          
            # save linearfit results to pickle file
            myPickleFile = os.path.join(self.savedir+self.sub_dir_str,"linearfit_result_{0:02d}.pkl".format(slc))
            with open( myPickleFile , "wb" ) as f:
                pickle.dump((nrm.linfit_result), f) 
            print("Wrote pickled file  %s" % myPickleFile)
                       

    def save_auto_figs(self, slc, nrm):
        
        # rotation
        if self.auto_rotate==True:
            plt.figure()
            plt.plot(nrm.rots, nrm.corrs)
            plt.vlines(nrm.rot_measured, nrm.corrs[0],
                        nrm.corrs[-1], linestyles='--', color='r')
            plt.text(nrm.rots[1], nrm.corrs[1], 
                     "best fit at {0}".format(nrm.rot_measured))
            plt.savefig(self.savedir+self.sub_dir_str+\
                        "/rotationcorrelation_{0:02d}.png".format(slc))

def fit_fringes_parallel(args, threads):
    self = args['object']
    filename = args['file']
    id_tag = args['id']
    self.scidata, self.scihdr = self.instrument_data.read_data(filename)

    self.sub_dir_str = self.instrument_data.sub_dir_str
    try:
        os.mkdir(self.savedir+self.sub_dir_str)
    except:
        pass

    store_dict = [{"object":self, "slc":slc} for slc in \
                  range(self.instrument_data.nwav)]

    if threads>0:
        pool = Pool(processes=threads)
        print("Running fit_fringes in parallel with {0} threads".format(threads))
        pool.map(fit_fringes_single_integration, store_dict)
        pool.close()
        pool.join()

    else:
        for slc in range(self.instrument_data.nwav):
            fit_fringes_single_integration({"object":self, "slc":slc})

def fit_fringes_single_integration(args):
    self = args["object"]
    slc = args["slc"]
    id_tag = args["slc"]

    nrm = NRM_Model(mask=self.instrument_data.mask,
                    pixscale=self.instrument_data.pscale_rad,
                    holeshape=self.instrument_data.holeshape,
                    affine2d=self.instrument_data.affine2d,
                    over = self.oversample)

    nrm.bandpass = self.instrument_data.wls[slc]

    if self.npix == 'default':
        self.npix = self.scidata[slc,:,:].shape[0]

    DBG = False # AS testing gross psf orientatioun while getting to LG++ beta release 2018 09
    if DBG:
        nrm.simulate(fov=self.npix, bandpass=self.instrument_data.wls[slc], over=self.oversample)
        fits.PrimaryHDU(data=nrm.psf).writeto(self.savedir + "perfect.fits", overwrite=True)

    # New or modified in LG++
    # center the image on its peak pixel:
    # AS subtract 1 from "r" below  for testing >1/2 pixel offsets
    # AG 03-2019 -- is above comment still relevant?
    
    if self.instrument_data.arrname=="NIRC2_9NRM":
        self.ctrd = utils.center_imagepeak(self.scidata[slc, :,:], 
                        r = (self.npix -1)//2 - 2, cntrimg=False)  
    elif self.instrument_data.arrname=="gpi_g10s40":
        self.ctrd = utils.center_imagepeak(self.scidata[slc, :,:], 
                        r = (self.npix -1)//2 - 2, cntrimg=True)  
    else:
        self.ctrd = utils.center_imagepeak(self.scidata[slc, :,:])  
        # Old AG LG++ version
        #self.ctrd = utils.center_imagepeak(self.scidata[slc, :,:], 
        #                r = (self.npix -1)//2 - 2)  

    # returned values have offsets x-y flipped:
    # Finding centroids the Fourier way assumes no bad pixels case - Fourier domain mean slope
    centroid = utils.find_centroid(self.ctrd, self.instrument_data.threshold) # offsets from array ctr
    # use flipped centroids to update centroid of image for JWST - check parity for GPI, Vizier,...
    # pixel coordinates: - note the flip of [0] and [1] to match DS9 view
    image_center = utils.centerpoint(self.ctrd.shape) + np.array((centroid[1], centroid[0])) # info only, unused
    print(">>>> nrm_core: centroid offsets {0} from utils.centroid() <<<<".format(centroid))
    print(">>>> nrm_core: center of light in array coords (ds9) {0} <<<<".format(image_center))
    nrm.xpos = centroid[1]  # flip 0 and 1 to convert 
    nrm.ypos = centroid[0]  # flip 0 and 1

    print(">>>> nrm_core.fit_image(): refslice 6 lines commented out cf LG+ <<<<")
    """ LG++ this fails to run - not sure of what's needed - anand@stsci.edu 2018.02.11
    refslice = self.ctrd.copy()
    if True in np.isnan(refslice):
        print(">>>> nrm_core.fit_image(): refslice UNTESTED w/ new utils.centroid() <<<<")
        refslice=utils.deNaN(5, self.ctrd)
        if True in np.isnan(refslice):
            refslice = utils.deNaN(20,refslice)
    """


    nrm.reference = self.ctrd  # rename bestcenter to bestpsfoffset or similar sometime in the future
    if self.hold_centering == False:
        print("\n**** nrm.core.fit_fringes_single_integration:    <<HOLD_CENTERING>> False")
        nrm.bestcenter = nrm.xpos, nrm.ypos  ################ AS try in LG++  Works!
        print("**** nrm.bestcenter {0}  nrm.xpos {1}  nrm.ypos {2}".format(nrm.bestcenter, nrm.xpos, nrm.ypos))
        print("**** nrm.core.fit_fringes_single_integration: object.best_center updated with 'centroid' output\n")
    else:
        print(">>>> nrm_core.fit_image(): hold_centering UNTESTED w/ new utils.centroid().  psf_offset from user... <<<<")
        nrm.bestcenter = self.psf_offset # if center already known, python-style offsets from array center are here.

    nrm.make_model(fov = self.ctrd.shape[0], bandpass=nrm.bandpass, 
                   over=self.oversample,
                   psf_offset=nrm.bestcenter,  
                   pixscale=nrm.pixel)
    nrm.fit_image(self.ctrd, modelin=nrm.model, psf_offset=nrm.bestcenter)
    """
    Attributes now stored in nrm object:

    -----------------------------------------------------------------------------
    soln            --- resulting sin/cos coefficients from least squares fitting
    fringephase     --- baseline phases in radians
    fringeamp       --- baseline amplitudes (flux normalized)
    redundant_cps   --- closure phases in radians
    redundant_cas   --- closure amplitudes
    residual        --- fit residuals [data - model solution]
    cond            --- matrix condition for inversion
    -----------------------------------------------------------------------------
    """

    if self.debug==True:
        import matplotlib.pyplot as plt
        import poppy.matrixDFT as mft
        dataft = mft.matrix_dft(self.ctrd, 256, 512)
        refft = mft.matrix_dft(self.refpsf, 256, 512)
        plt.figure()
        plt.title("Data")
        plt.imshow(np.sqrt(abs(dataft)), cmap = "bone")
        plt.figure()
        plt.title("Reference")
        plt.imshow(np.sqrt(abs(refft)), cmap="bone")
        plt.show()
    
    self.save_output(slc, nrm)
    return None

class Calibrate:
    """
    Change name: NRM_calibrate


    calibrate takes a list of folders containing the fringe-fitting results 
    from each set of exposures
    The first directory should contain reduction for the target
    Subsequent folders for calibrators -- may have more than one. 

    This function will simply provide mean phase and amplitude quantities,
    calibrate with whatever calibrators are provided, and propogate the
    errors by adding in quadrature.

    Flexible for 2 different kinds of data:
    - individual measurements per exposure
    - an additional axis (e.g., wavelength or polz)

    What happens in __init__:
    - Get statistics on each set of exposures
    - Subtract calibrator phases from target phases. 
    - Divide target visibilities by cal visibilities.  

    Save methods:

    * save_to_txt
    * save_to_oifits

    """

    def __init__(self, objpaths, instrument_data, savedir=None, extra_dimension=None, **kwargs):
        """
        Initilize the class

        e.g., to run this in a driver 
            gpidata = InstrumentData.GPI(reffile)
            calib = Calibrate(objpaths, gpidata)
            calib.write_to_oifits("dataset.oifits")
        
        instrument_data - stores the mask geometry (namely # holes),
                    instrument info, and wavelength obs mode info
                    an instance of the appropriate data class

        objpaths       - List of directory paths (e.g. [tgtpth, calpth1, calpth2, ...]
                    containing fringe observables for tgt, cal1, [cal2,...].
                    The first path is the target.  One or more calibrators follow.
                    Used to be parameter 'paths'.

        savedir     - default is in the working directory; can give path from
                      CWD or absolute path.

        extra_dimension - Dataset has extra dimesion: wavelength or polarization, for example.
                    Value None means the data files are in the "object" level directory
                    Value string file (directory??)  in each object level folder containing 
                    additional layer of data
                    Used to be parameter 'sub_dir_tag'
        
        This will load all the observations into attributes:
        cp_mean_cal ... size [ncals, naxis2, ncp]
        cp_err_cal  ... size [ncals, naxis2, ncp]
        v2_mean_cal ... size [ncals, naxis2, nbl]
        v2_err_cal  ... size [ncals, naxis2, nbl]
        cp_mean_tar ... size [naxis2, ncp]  
        cp_err_tar  ... size [naxis2, ncp]
        v2_mean_tar ... size [naxis2, nbl]
        v2_err_tar  ... size [naxis2, nbl]
        """

        if 'interactive' in kwargs.keys():
            self.interactive = kwargs['interactive']
        else:
            self.interactive = True

        # Added 10/14/2016 -- here's a visibilities flag, where you can specify a 
        # cutoff and only accept exposures with higher visibilities
        # Or maybe we should have it just reject outliers?
        # Let's have vflag be rejection of the fraction provided
        # e.g. if vflag=0.25, then exposures with the 25% lowest avg visibilities will be flagged
        if "vflag" in kwargs.keys():
            self.vflag=kwargs['vflag']
        else:
            self.vflag=0.0
    
        #if no savedir specified, default is current working directory
        if savedir ==None:
            self.savedir = os.getcwd()
        else:
            self.savedir = savedir

        try:
            os.listdir(savedir)
        except:
            os.mkdir(savedir)
        self.savedir = savedir

        # number of calibrators being used:
        self.ncals = len(objpaths) - 1 # number of calibrators, if zero, set to 1
        if self.ncals==0:# No calibrators given
            self.ncals = 1 # to avoid empty arrays
        self.nobjs = len(objpaths) # number of total objects

        self.N = len(instrument_data.mask.ctrs)
        self.nbl = int(self.N*(self.N-1)/2)
        self.ncp = int(comb(self.N, 3))
        self.instrument_data = instrument_data

        # Additional axis (e.g., wavelength axis)
        # Can be size one. Defined by instrument_data wavelength array
        self.naxis2 = instrument_data.nwav
        print("nrm_core.Calibrate: self.naxis2 = instrument_data.nwav = ", instrument_data.nwav)


        if extra_dimension == None: 
            if self.interactive==True:
                #print "!! naxis2 is set to a non-zero number but extra_layer"
                print("extra dimension is not defined, so NAXIS2 will be ignored.")
                print("results will not be stored at the object level directory")
                print("Do you want proceed anyway? (y/n)")
                ans = input()
                if ans =='y':
                    pass
                elif ans == 'n':
                    sys.exit("OK, stopping.  Try providing 'extra_dimension' keyword arg to Calibrate()")
                else:
                    sys.exit("Exiting: Provide 'extra_dimension' keyword arg to Calibrate() to access polz, wavelens directories")
            else:
                pass

        # Set up all the arrays
        # Cal arrays have ncal axis, "wavelength" axis, and ncp axis
        self.cp_mean_cal = np.zeros((self.ncals, self.naxis2, self.ncp))
        self.cp_err_cal = np.zeros((self.ncals, self.naxis2, self.ncp))
        self.v2_mean_cal = np.zeros((self.ncals, self.naxis2, self.nbl))
        self.v2_err_cal = np.zeros((self.ncals, self.naxis2, self.nbl))
        self.pha_mean_cal = np.zeros((self.ncals, self.naxis2, self.nbl))
        self.pha_err_cal = np.zeros((self.ncals, self.naxis2, self.nbl))

        # target arrays have "wavelength" axis and ncp axis
        self.cp_mean_tar = np.zeros((self.naxis2, self.ncp))
        self.cp_err_tar = np.zeros((self.naxis2, self.ncp))
        self.v2_mean_tar = np.zeros((self.naxis2, self.nbl))
        self.v2_err_tar = np.zeros((self.naxis2, self.nbl))
        self.pha_mean_tar = np.zeros((self.naxis2, self.nbl))
        self.pha_err_tar = np.zeros((self.naxis2, self.nbl))

        #self.cov_mat_cal = np.zeros(nexps*self.naxi2)

        # is there a subdirectory (e.g. for the exposure -- need to make this default)
        if extra_dimension is not None:
            self.extra_dimension = extra_dimension
            for ii in range(self.nobjs):
                exps = [f for f in os.listdir(objpaths[ii]) \
                        if self.extra_dimension in f and  \
                        os.path.isdir(os.path.join(objpaths[ii],f))]
                nexps = len(exps)
                print("DEBUG: "+str(nexps))
                amp = np.zeros((self.naxis2, nexps, self.nbl))
                pha = np.zeros((self.naxis2, nexps, self.nbl))
                cps = np.zeros((self.naxis2, nexps, self.ncp))

                # Create the cov matrix arrays
                if ii == 0:
                    self.cov_mat_tar = np.zeros((self.naxis2, nexps, nexps))
                    self.sigmasquared_tar = np.zeros((self.naxis2, nexps))
                    self.cov_mat_cal = np.zeros((self.naxis2, nexps, nexps))
                    self.sigmasquared_cal = np.zeros((self.naxis2, nexps))
                else:
                    pass
                expflag=[]
                for qq in range(nexps):
                    # nwav files
                    cpfiles = [f for f in os.listdir(objpaths[ii]+exps[qq]) if "CPs" in f] 
                    print(cpfiles)
                    ampfiles = [f for f in os.listdir(objpaths[ii]+exps[qq]) \
                                if "amplitudes" in f]
                    phafiles = [f for f in os.listdir(objpaths[ii]+exps[qq]) if "phase" in f] 
                    for slc in range(len(cpfiles)):
                        amp[slc, qq,:] = np.loadtxt(objpaths[ii]+exps[qq]+"/"+ampfiles[slc])
                        cps[slc, qq,:] = np.loadtxt(objpaths[ii]+exps[qq]+"/"+cpfiles[slc])
                        pha[slc, qq,:] = np.loadtxt(objpaths[ii]+exps[qq]+"/"+phafiles[slc])
                    # 10/14/2016 -- flag the exposure if we get amplitudes > 1
                    # Also flag the exposure if vflag is set, to reject fraction indicated
                    if True in (amp[:,qq,:]>1):
                        print('amp > 1 for {}'.format(exps[qq]))
                        expflag.append(qq)

                if self.vflag>0.0:
                    self.ncut = int(self.vflag*nexps) # how many are we cutting out
                    sorted_exps = np.argsort(amp.mean(axis=(0,-1)))
                    cut_exps = sorted_exps[:self.ncut] # flag the ncut lowest exposures
                    expflag = expflag + list(cut_exps)

                # Create the cov matrix arrays
                if ii == 0:
                    rearr = np.rollaxis(cps, -1, 0).reshape(self.naxis2*self.ncp, nexps)
                    R_i = rearr - rearr.mean(axis=1)[:,None]
                    R_j = R_i.T
                    self.cov = np.dot(R_i,R_j) / (nexps - 1)
                else:
                    rearr = np.rollaxis(cps, -1, 0).reshape(self.naxis2*self.ncp, nexps)
                    R_i = rearr - rearr.mean(axis=1)[:,None]
                    R_j = R_i.T
                    self.cov +=np.dot(R_i,R_j) / (nexps - 1)

                # Also adding a mask to calib steps
                ############################
                for slc in range(self.naxis2):
                    if ii==0:
                        # closure phases and squared visibilities
                        self.cp_mean_tar[slc,:], self.cp_err_tar[slc,:], \
                            self.v2_mean_tar[slc,:], self.v2_err_tar[slc,:], \
                            self.pha_mean_tar[slc,:], self.pha_err_tar = \
                            self.calib_steps(cps[slc,:,:], amp[slc,:,:], pha[slc,:,:], nexps, expflag=expflag)

                    else:
                        # closure phases and visibilities
                        self.cp_mean_cal[ii-1,slc, :], self.cp_err_cal[ii-1,slc, :], \
                            self.v2_mean_cal[ii-1,slc,:], self.v2_err_cal[ii-1,slc,:], \
                            self.pha_mean_cal[ii-1,slc,:], self.pha_err_cal[ii-1, slc,:] = \
                            self.calib_steps(cps[slc,:,:], amp[slc,:,:], pha[slc,:,:], nexps, expflag=expflag)

            nexp_c = self.sigmasquared_cal.shape[1]

        else:
            print("else")
            for ii in range(self.nobjs):

                cpfiles = [f for f in os.listdir(objpaths[ii]) if "CPs" in f] 
                ampfiles = [f for f in os.listdir(objpaths[ii]) if "amplitudes" in f]
                phafiles = [f for f in os.listdir(objpaths[ii]) if "phase" in f]
                nexps = len(cpfiles)
                print("nexp: "+str(nexps))

                amp = np.zeros((nexps, self.nbl))
                pha = np.zeros((nexps, self.nbl))
                cps = np.zeros((nexps, self.ncp))
                print(nexps)
                expflag=[]
                for qq in range(nexps):
                    amp[qq,:] = np.loadtxt(objpaths[ii]+"/"+ampfiles[qq])
                    if True in (amp[qq,:]>1):
                        print('amp > 1 for {}'.format(ampfiles[qq]))
                        expflag.append(qq)
                    print(cpfiles[qq])
                    pha[qq,:] = np.loadtxt(objpaths[ii]+"/"+phafiles[qq])
                    cps[qq,:] = np.loadtxt(objpaths[ii]+"/"+cpfiles[qq])

                # Covariance 06/27/2017
                if ii == 0:
                    rearr = np.rollaxis(cps, -1, 0).reshape(self.ncp, nexps)
                    R_i = rearr - rearr.mean(axis=1)[:,None]
                    R_j = R_i.T
                    self.cov = np.dot(R_i,R_j) / (nexps - 1)
                else:
                    rearr = np.rollaxis(cps, -1, 0).reshape(self.ncp, nexps)
                    R_i = rearr - rearr.mean(axis=1)[:,None]
                    R_j = R_i.T
                    self.cov +=np.dot(R_i,R_j) / (nexps - 1)

                ############################
                # Oct 14 2016 -- adding in a visibilities flag. Can't be >1 that doesn't make sense.
                # Also adding a mask to calib steps
                if ii==0:
                    # closure phases and squared visibilities
                    self.cp_mean_tar[0,:], self.cp_err_tar[0,:], \
                        self.v2_mean_tar[0,:], self.v2_err_tar[0,:], \
                        self.pha_mean_tar[0,:], self.pha_err_tar[0,:] = \
                        self.calib_steps(cps, amp, pha, nexps, expflag=expflag)
                else:
                    # Fixed clunkiness!
                    # closure phases and visibilities
                    self.cp_mean_cal[ii-1,0, :], self.cp_err_cal[ii-1,0, :], \
                        self.v2_mean_cal[ii-1,0,:], self.v2_err_cal[ii-1,0,:], \
                        self.pha_mean_cal[ii-1,0,:], self.pha_err_cal[ii-1,0,:] = \
                        self.calib_steps(cps, amp, pha, nexps, expflag=expflag)


        # Combine mean calibrator values and errors
        self.cp_mean_tot = np.zeros(self.cp_mean_cal[0].shape)
        self.cp_err_tot = self.cp_mean_tot.copy()
        self.v2_mean_tot = np.zeros(self.v2_mean_cal[0].shape)
        self.v2_err_tot = self.v2_mean_tot.copy()
        self.pha_mean_tot = np.zeros(self.pha_mean_cal[0].shape)
        self.pha_err_tot = self.pha_mean_tot.copy()
        for ww in range(self.ncals):
            self.cp_mean_tot += self.cp_mean_cal[ww]
            self.cp_err_tot += self.cp_err_cal[ww]**2
            self.v2_mean_tot += self.v2_mean_cal[ww]
            self.v2_err_tot += self.v2_err_cal[ww]**2
            self.pha_mean_tot += self.pha_mean_cal[ww]
            self.pha_err_tot += self.pha_err_cal[ww]**2
        self.cp_mean_tot = self.cp_mean_tot/self.ncals
        self.cp_err_tot = np.sqrt(self.cp_err_tot)
        self.v2_mean_tot = self.v2_mean_tot/self.ncals
        self.v2_err_tot = np.sqrt(self.v2_err_tot)
        self.pha_mean_tot = self.pha_mean_tot/self.ncals
        self.pha_err_tot = np.sqrt(self.pha_err_tot)

        # Calibrate
        self.cp_calibrated = self.cp_mean_tar - self.cp_mean_tot
        self.cp_err_calibrated =  np.sqrt(self.cp_err_tar**2 + self.cp_err_tot**2)
        self.v2_calibrated = self.v2_mean_tar/self.v2_mean_tot
        self.v2_err_calibrated = np.sqrt(self.v2_err_tar**2 + self.v2_err_tot**2)
        self.pha_calibrated = self.pha_mean_tar - self.pha_mean_tot
        self.pha_err_calibrated = np.sqrt(self.pha_err_tar**2 + self.pha_err_tot**2)

        # convert to degrees
        self.cp_calibrated_deg = self.cp_calibrated * 180/np.pi
        self.cp_err_calibrated_deg = self.cp_err_calibrated * 180/np.pi
        self.pha_calibrated_deg = self.pha_calibrated * 180/np.pi
        self.pha_err_calibrated_deg = self.pha_err_calibrated * 180/np.pi

    def calib_steps(self, cps, amps, pha, nexp, expflag=None):
        "Calculates closure phase and mean squared visibilities & standard error"
        #########################
        # 10/14/16 Change flags exposures where vis > 1 anywhere
        # Apply the exposure flag
        expflag = None
        cpmask = np.zeros(cps.shape, dtype=bool)
        blmask = np.zeros(amps.shape, dtype=bool)
        if expflag is not None:
            nexp -= len(expflag) # don't count the bad exposures
            cpmask[expflag, :] = True
            blmask[expflag, :] = True
        else:
            pass

        print('nexp after mask {:d}'.format(nexp))

        meancp = np.ma.masked_array(cps, mask=cpmask).mean(axis=0)
        meanv2 = np.ma.masked_array(amps, mask=blmask).mean(axis=0)**2
        meanpha = np.ma.masked_array(pha, mask=blmask).mean(axis=0)**2

        errcp = np.sqrt(mstats.moment(np.ma.masked_array(cps, mask=cpmask), moment=2, axis=0))/np.sqrt(nexp)
        errv2 = np.sqrt(mstats.moment(np.ma.masked_array(amps**2, mask=blmask), moment=2, axis=0))/np.sqrt(nexp)
        errpha = np.sqrt(mstats.moment(np.ma.masked_array(pha, mask=blmask), moment=2, axis=0))/np.sqrt(nexp)

        # Set cutoff accd to Kraus 2008 - 2/3 of median
        errcp[errcp < (2/3.0)*np.median(errcp)] =(2/3.0)*np.median(errcp) 
        errpha[errpha < (2/3.0)*np.median(errpha)] =(2/3.0)*np.median(errpha) 
        errv2[errv2 < (2/3.0)*np.median(errv2)] =(2/3.0)*np.median(errv2) 

        #print("input:",cps)
        #print("avg:", meancp)
        #print("exposures flagged:", expflag)
        return meancp, errcp, meanv2, errv2, meanpha, errpha


    def save_to_txt(self):
        """Saves calibrated results to text files
           If naxis2 is specified, saves results over each component in the 
           additional axis, denoted by the slice number."""
        if hasattr(self, "naxis2"):
            for slc in range(naxis2):
                tag = "_deg_{0}.txt".format(slc)
                fns = ["cps"+tag, "cperr"+tag, "v2"+tag, "v2err"+tag]
                arrs = [self.cp_calibrated_deg, self.cp_err_calibrated_deg, \
                        self.v2_calibrated, self.v2_err_calibrated, \
                        self.pha_calibrated_deg, self.pha_err_calibrated_deg]
                self._save_txt(fns, arrs)
        else:
            tag = "_deg.txt".format(slc)
            fns = ["cps"+tag, "cperr"+tag, "v2"+tag, "v2err"+tag]
            arrs = [self.cp_calibrated_deg, self.cp_err_calibrated_deg, \
                    self.v2_calibrated, self.v2_err_calibrated, \
                    self.pha_calibrated_deg, self.pha_err_calibrated_deg]
            self._save_txt(fns, arrs)

    def _save_txt(self, fns, arrays):
        """
        fns and arrays are each a list of 4 elements -- 
        fns is 4 file names and arrays is 4 arrays of calibrated observables
        """

        np.savetxt(self.savedir+"/"+fns[0],arrays[0])
        np.savetxt(self.savedir+"/"+fns[1], arrays[1])
        np.savetxt(self.savedir+"/"+fns[2], arrays[2])
        np.savetxt(self.savedir+"/"+fns[3], arrays[3])

        return None

    def save_to_oifits(self, fn_out, **kwargs):
        """
        User may wish to save to oifits
        Specify reference fits files to check header values
        can also provide oifits keywords


        Check: oifits standard -- degrees or radians?
        """
        print(kwargs)

        
        #####  AS moving to js_oifits  from .misctools.write_oifits import OIfits
        from nrm_analysis.misctools import glue_js_oifits

        # look for kwargs, e.g., phaseceil, anything else?
        if "phaseceil" in list(kwargs.keys()):
            self.phaseceil = kwargs["phaseceil"]
        else:
            # default for flagging closure phases (deg)
            self.phaseceil = 1.0e2 # degrees
        if "clip" in kwargs.keys():
            self.clip_wls = kwargs["clip"]
            nwav = self.naxis2-2*self.clip_wls
            covshape = (self.naxis2*self.ncp) - (2*self.clip_wls*self.ncp)
            clippedcov = np.zeros((covshape, covshape))
            for k in range(self.ncp):
                clippedcov[nwav*k:nwav*(k+1),nwav*k:nwav*(k+1)] = \
                    self.cov[self.naxis2*k+self.clip_wls:self.naxis2*(k+1)-self.clip_wls,
                             self.naxis2*k+self.clip_wls:self.naxis2*(k+1)-self.clip_wls]
        else:
            # default is no clipping - maybe could set instrument-dependent clip in future
            self.clip_wls = None
            clippedcov = self.cov

        if not hasattr(self.instrument_data, "parang_range"):
            self.instrument_data.parang_range = 0.0
        if not hasattr(self.instrument_data, "avparang"):
            self.instrument_data.avparang = 0.0
        self.obskeywords = {
                'path':self.savedir+"/",
                'year':self.instrument_data.year, 
                'month':self.instrument_data.month,
                'day':self.instrument_data.day,
                'TEL':self.instrument_data.telname,\
                'instrument':self.instrument_data.instrument, 
                'arrname':self.instrument_data.arrname, 
                'object':self.instrument_data.objname,
                'RA':self.instrument_data.ra, 
                'DEC':self.instrument_data.dec, \
                'PARANG':self.instrument_data.avparang, 
                'PARANGRANGE':self.instrument_data.parang_range,
                'PA':self.instrument_data.pa, 
                'phaseceil':self.phaseceil,
                'covariance':clippedcov}

        print("Calibrate.save_to_oifits(): \n",
        "\tv2",self.v2_calibrated.shape, "v2err",self.v2_err_calibrated.shape, '\n',
        "\tcps",self.cp_calibrated_deg.shape, "cperr",self.cp_err_calibrated_deg.shape, '\n',
        "\tpha",self.pha_calibrated_deg.shape, "phaerr",self.pha_err_calibrated_deg.shape,  '\n',
        "\twave",self.instrument_data.lam_c[self.instrument_data.filt], '\n',
        "\twave",self.instrument_data.lam_w[self.instrument_data.filt], '\n',
        "\tnwave",self.instrument_data.nwav, '\n',
        "\thole_size",self.instrument_data.mask.hdia, '\n',
        "\tnholes",self.instrument_data.mask.ctrs.shape[0], '\n',
        )
        print("len(ctrs):", len(self.instrument_data.mask.ctrs))
        for k in self.obskeywords.keys():
            print("\t\t%s"%k, self.obskeywords[k])

        print("\t mask.ctrs is of type: ", type(self.instrument_data.mask.ctrs))
        #oif = OIfits(self.instrument_data.mask, self.obskeywords)
        glue_js_oifits.write(
                    obskeywords=self.obskeywords,
                    v2=self.v2_calibrated[0,:], v2err=self.v2_err_calibrated[0,:],
                    cps=self.cp_calibrated_deg[0,:], cperr=self.cp_err_calibrated_deg[0,:],
                    pha=self.pha_calibrated_deg[0,:], phaerr=self.pha_err_calibrated_deg[0,:], 
                    wave=self.instrument_data.lam_c[self.instrument_data.filt],
                    bandwidth=self.instrument_data.lam_w[self.instrument_data.filt],
                    nwave=self.instrument_data.nwav,
                    hole_size=self.instrument_data.mask.hdia,
                    nholes=self.instrument_data.mask.ctrs.shape[0],
                    ctrs = self.instrument_data.mask.ctrs ,
                    )
        """ 
        interface_oifits_js_writ(v2=self.v2_calibrated, v2err=self.v2_err_calibrated, \
                    cps=self.cp_calibrated_deg, cperr=self.cp_err_calibrated_deg, \
                    pha = self.pha_calibrated_deg, phaerr = self.pha_err_calibrated_deg) 
        oif_js_write(v2=self.v2_calibrated, v2err=self.v2_err_calibrated, \
                    cps=self.cp_calibrated_deg, cperr=self.cp_err_calibrated_deg, \
                    pha = self.pha_calibrated_deg, phaerr = self.pha_err_calibrated_deg) 
        mask='jwst'
        instrument = 'NIRISS'
        """

        """
        oif.dummytables()
        # Option to clip out band edges for multiple wavelengths
        # clip can be scalar or 2-element. scalar will do symmetric clipping
        wavs = oif.wavextension(self.instrument_data.wavextension[0], \
                    self.instrument_data.wavextension[1], clip=self.clip_wls)
        oif.oi_data(read_from_txt=False, v2=self.v2_calibrated, v2err=self.v2_err_calibrated, \
                    cps=self.cp_calibrated_deg, cperr=self.cp_err_calibrated_deg, \
                    pha = self.pha_calibrated_deg, phaerr = self.pha_err_calibrated_deg) 
        print("oif.write in nrm_core: ")
        oif.write(fn_out)
        """

    def txt_2_oifits():
        """
        Calibrated data already saved to txt, want to save this to oifits
        """
        return None

    def _from_gpi_header(fitsfiles):
        """
        Things I think are important. Average the parang measurements
        """
        parang=[]
        pa = []
        for fitsfile in fitsfiles:
            f = fits.open(fitsfile)
            hdr = f[0].header
            f.close()
            ra = hdr['RA']
            dec = hdr['DEC']
            parang.append(hdr['PAR_ANG'] - 1.00) # degree pa offset from 2014 SPIE +/- 0.03
            pa.append(hdr['PA'])
        return ra, dec, np.mean(parang), np.mean(pa)

    def _from_ami_header(fitsfiles):
        return None

class BinaryAnalyze:
    def __init__(self, oifitsfn, savedir = None, extra_error=0, plot="on"):
        """
        BinaryAnalyze loads in an oifits file, also contains various methods
        for searching for a secondary point source:

        coarse_binary_search ***out of date*** - loop through a coars grid to
                             find min chi^2, feeds finer search
        detec_map *** phasing out *** - create crude "detection sensitivity" map
        chi2map              - updated coarse grid search, parallelized, 
                               much faster
        detection_limits     - Needs to be checked, but MC approach to estimating
                               detection limits.
        grid_spectrum        - holding position constant, computes contrast over 
                               set of wavelengths that minimize chi^2 (coarse 
                               spectrum extraction)
        run_emcee            - runs MCMC fit for a BINARY. 
        + various plotting routines to show the results.
        """
        self.oifitsfn = oifitsfn
        self.extra_error = extra_error

        get_data(self)
        if savedir==None:
            self.savedir=os.getcwd()
        else:
            self.savedir = savedir
        self.plot=plot


    def coarse_binary_search(self, lims, nstep=20):
        """
        For getting first guess on contrast, separation, and angle

        lims:   [(c_low, c_hi), (sep_lo, sep_hi), (pa_low, pa_hi)]
        contrast is in logspace, so provide powers for the range
        separation in mas, pa in degrees
        """
        #cons = np.linspace(lims[0][0], lims[0][1], num=nstep)
        nn = np.arange(nstep)
        r = (lims[0][-1]/lims[0][0])**(1 / float(nstep-1))
        cons = lims[0][0] * r**(nn)
        #cons = np.linspace(lims[0][0], lims[0][1], num=nstep)
        seps = np.linspace(lims[1][0], lims[1][1], num=nstep)
        angs = np.linspace(lims[2][0], lims[2][1], num=nstep)
        loglike = np.zeros((nstep, nstep, nstep))

        priors = np.array([(-np.inf, np.inf) for f in range( 3 ) ])
        constant = {"wavl": self.wavls}

        for i in range(nstep):
            for j in range(nstep):
                for k in range(nstep):
                    #params = {'con':cons[i], 'sep':seps[j], 'pa':angs[k]}
                    params = [cons[i], seps[j], angs[k]]
                    loglike[i,j,k] = cp_binary_model(params, constant, priors, None, self.uvcoords, self.cp, self.cperr)
        loglike_0 = cp_binary_model([0, 0, 0], constant, priors, None, self.uvcoords, self.cp, self.cperr)

        wheremax = np.where(loglike==loglike.max())
        print("abs max", wheremax)
        print("loglike at max axis=0",wheremax[0][0], loglike[wheremax[0][0],:,:].shape)
        print("===================")
        print("Max log likelikehood for contrast:",cons[wheremax[0]]) 
        print("Max log likelikehood for separation:", seps[wheremax[1]], "mas")
        print("Max log likelikehood for angle:", angs[wheremax[2]], "deg")
        print("===================")
        coarse_params = cons[wheremax[0]], seps[wheremax[1]], angs[wheremax[2]]

        plt.figure()
        plt.set_cmap("cubehelix")
        plt.title("separation vs. pa at contrast of {0:.1f}".format(cons[wheremax[0][0]]))
        plt.imshow(loglike[wheremax[0][0], :,:].transpose())
        plt.xticks(np.arange(nstep)[::5], np.round(seps[::5],3))
        plt.yticks(np.arange(nstep)[::5], np.round(angs[::5],3))
        plt.xlabel("Separation")
        plt.ylabel("PA")
        plt.colorbar()
        plt.savefig(self.savedir+"/sep_pa.pdf")

        plt.figure()
        plt.title("contrast vs. separation, at PA of {0:.1f} deg".format(angs[wheremax[2][0]]))
        plt.xticks(np.arange(nstep)[::5], np.round(cons[::5],3))
        plt.yticks(np.arange(nstep)[::5], np.round(seps[::5],3))
        plt.xlabel("Contrast")
        plt.ylabel("Separation")
        plt.imshow(loglike[:,:,wheremax[2][0]].transpose())
        plt.colorbar()
        plt.savefig(self.savedir+"/con_sep.pdf")

        plt.figure()
        plt.title("contrast vs. angle, at separation of {0:.1f} mas".format(seps[wheremax[1][0]]))
        plt.xticks(np.arange(nstep)[::5], np.round(cons[::5], 3))
        plt.yticks(np.arange(nstep)[::5], np.round(angs[::5], 3))
        plt.xlabel("Contrast")
        plt.ylabel("PA")
        plt.imshow(loglike[:,wheremax[1][0],:].transpose())
        plt.colorbar()
        plt.savefig(self.savedir+"/con_pa.pdf")

        """
        # chi^2 detection grid
        detec_mask = np.exp(loglike) > 25 + np.exp(loglike_0)
        con_index = np.ma.masked_array(loglike, mask=detec_mask).argmax(axis=2)
        detec_array = cons[con_index]
        print detec_array.shape
        self.loglike = loglike
        self.loglike_0 = loglike_0
        plt.figure()
        plt.title("1-sigma detection?")
        plt.imshow(detec_array, cmap="CMRmap")
        plt.xlabel("sep (mas)")
        plt.ylabel("angle (deg)")
        #plt.yticks(np.arange(nstep)[::(nstep/4)], seps[::(nstep/4)]*np.sin(angs)[::(nstep/4)])
        #plt.xticks(np.arange(nstep)[::(nstep/4)], seps[::(nstep/4)]*np.cos(angs)[::(nstep/4)])
        plt.xticks(np.arange(nstep)[::(nstep/4)], np.round(seps[::(nstep/4)], 0))
        plt.yticks(np.arange(nstep)[::(nstep/4)], angs[::(nstep/4)])
        plt.colorbar(label="Contrast")
        """
        
        if self.plot=="on":
            plt.show()
        return coarse_params

    def coarse_multi(self, lims, known, nstep=25):
        """
        For getting first guess on contrast, separation, and angle 
        for an additional companion. Iterate until suffient # of 
        companions is reached. Provide known parameters for whatever
        number have already been fit.

        lims:   [(c_low, c_hi), (sep_lo, sep_hi), (pa_low, pa_hi)]
        known: [[c_1, c_2,...], [s_1, s_2, ...], [pa_1, pa_2, ...]]

        contrast is in logspace, so provide powers for the range
        separation in mas, pa in degrees
        """
        #cons = np.linspace(lims[0][0], lims[0][1], num=nstep)
        nn = np.arange(nstep)
        r = (lims[0][-1]/lims[0][0])**(1 / float(nstep-1))
        cons = lims[0][0] * r**(nn)
        #cons = np.linspace(lims[0][0], lims[0][1], num=nstep)
        seps = np.linspace(lims[1][0], lims[1][1], num=nstep)
        angs = np.linspace(lims[2][0], lims[2][1], num=nstep)
        loglike = np.zeros((nstep, nstep, nstep))

        priors = np.array([(-np.inf, np.inf) for f in range( 3 ) ])
        constant = {"wavl": self.wavls}

        for i in range(nstep):
            for j in range(nstep):
                for k in range(nstep):
                    #params = {'con':cons[i], 'sep':seps[j], 'pa':angs[k]}
                    params = known+[cons[i], seps[j], angs[k]] # string them all together
                    loglike[i,j,k] = cp_multi_model(params, constant, priors, None, self.uvcoords, self.cp, self.cperr)
        loglike_0 = cp_binary_model([0, 0, 0], constant, priors, None, self.uvcoords, self.cp, self.cperr)

        wheremax = np.where(loglike==loglike.max())
        print("abs max", wheremax)
        print("loglike at max axis=0",wheremax[0][0], loglike[wheremax[0][0],:,:].shape)
        print("===================")
        print("Max log likelikehood for contrast:", end=' ') 
        print(cons[wheremax[0]])
        print("Max log likelikehood for separation:", end=' ') 
        print(seps[wheremax[1]], "mas")
        print("Max log likelikehood for angle:", end=' ') 
        print(angs[wheremax[2]], "deg")
        print("===================")
        coarse_params = cons[wheremax[0]], seps[wheremax[1]], angs[wheremax[2]]
        return coarse_params

    def detec_map(self, lims, nstep=50, hyp = 0, save=False):
        """
        For getting first guess on contrast, separation, and angle

        lims:   [(c_low, c_hi), (sep_lo, sep_hi), (pa_low, pa_hi)]
        contrast is in logspace, so provide powers for the range
        separation in mas, pa in degrees

        hyp: Hypothesis -- default is null
        """
        from matplotlib.colors import LogNorm
        #cons = np.linspace(lims[0][0], lims[0][1], num=nstep)
        nn = np.arange(nstep)
        r = (lims[0][-1]/lims[0][0])**(1 / float(nstep-1))
        cons = lims[0][0] * r**(nn)
        #cons = np.logspace(np.log10(lims[0][0]), np.log10(lims[0][1]), num=nstep)
        ras = np.linspace(-lims[1][1], lims[1][1], num = nstep)
        decs = np.linspace(-lims[1][1], lims[1][1], num = nstep)
        #seps = np.sqrt(ras**2 + decs**2)
        #angs = np.arctan2(decs, ras)
        #loglike = np.zeros((nstep, nstep, nstep))
        chi2cube = np.zeros((nstep, nstep, nstep))
        chi2_bin_model = np.zeros((nstep, nstep, nstep))

        priors = np.array([(-np.inf, np.inf) for f in range( 3 ) ])
        constant = {"wavl": self.wavls}

        for i in range(nstep):
            for j in range(nstep):
                for k in range(nstep):
                    #params = {'con':cons[i], 'sep':seps[j], 'pa':angs[k]}
                    ang = 180*np.arctan2(decs[k], ras[j])/np.pi
                    sep = np.sqrt(ras[j]**2 + decs[k]**2)
                    params = [cons[i], sep, ang]
                    chi2cube[i,j,k] = allvis_binary_model(params, constant, priors, None, 
                                                          self.uvcoords, self.cp, self.cperr, self.t3amp,
                                                          self.t3amperr, stat="chi2")
        chi2_0 = allvis_binary_model([0,0,0], constant, priors, None,
                                     self.uvcoords, self.cp, self.cperr, self.t3amp, self.t3amperr,
                                     stat="chi2")
        print("detecmap chi2null", chi2_0)

        # chi^2 detection grid
        #detec_mask = - np.exp(loglike) > (25 - np.exp(loglike_0))
        #if hyp==0:
        detec_mask = chi2cube >= ( 25 + chi2_0)
        #else:
        #    detec_mask = chi2cube >= ( 25 + chi2_bin_model)
        self.detec_mask = detec_mask
        con_index = np.ma.masked_array(chi2cube, mask=detec_mask).argmax(axis=0)
        #con_index = chi2cube.argmin(axis=0)
        detec_array = cons[con_index]
        print(detec_array.shape)
        #self.loglike = loglike
        #self.loglike_0 = loglike_0
        self.chi2cube = chi2cube
        self.chi2_0 = chi2_0
        plt.figure()
        plt.title("5-sigma detection threshold")
        #plt.title("chi2grid")
        #plt.imshow(-(chi2cube.min(axis=0)), cmap="CMRmap")
        plt.pcolor(detec_array, norm=LogNorm(vmin=detec_array.min(), vmax=detec_array.max()), cmap="CMRmap")
        plt.xlabel("RA (mas)")
        plt.ylabel("DEC (mas)")
        #plt.yticks(np.arange(nstep)[::(nstep/4)], seps[::(nstep/4)]*np.sin(angs)[::(nstep/4)])
        #plt.xticks(np.arange(nstep)[::(nstep/4)], seps[::(nstep/4)]*np.cos(angs)[::(nstep/4)])
        plt.xticks(np.linspace(0, nstep, 5), np.linspace(ras.min(), ras.max(), 4+1))
        plt.yticks(np.linspace(0, nstep, 5), np.linspace(decs.min(), decs.max(), 4+1))
        plt.colorbar(label="Contrast")
        
        if self.plot=="on":
            plt.show()
        if save is not False:
            plt.savefig(self.savedir+save)

    def chi2map(self, maxsep=300., clims = [0.001, 0.5], nstep=50, \
                threads=4, observables="cp"):
        """
        Makes a coarse chi^2 map at the contrast where chi^2 is minimum for each position. 
        Default cps only, but can choose observables="all" to use visibility info also.
        """

        nn = np.arange(nstep)
        r = (clims[-1]/clims[0])**(1 / float(nstep-1))
        self.cons = clims[0] * r**(nn)
        self.ras = np.linspace(-maxsep, maxsep, num = nstep)
        self.decs = np.linspace(-maxsep, maxsep, num = nstep)

        # Make position grid
        t0 = time.time()
        ras = np.tile(self.ras, (nstep, self.nwav, 1))
        ras = np.rollaxis(ras, -1, 0)
        decs = np.tile(self.decs, (nstep, self.nwav, 1))
        decs = np.rollaxis(decs, -1, 1)
        # Need to add these onto uvcoords too
        uvcoords = np.rollaxis(np.rollaxis(np.rollaxis(np.tile(self.uvcoords, (nstep, nstep, 1, 1, 1, 1)), -2,0), -2, 0), -2,0)
        t1 = time.time()
        print("took "+str(t1-t0)+"s to assemble position grids")
        print(uvcoords.shape)
        print(self.cp.shape)
        print(ras.shape)
        print(np.shape(self.wavls))

        t2 = time.time()

        # Now split this up by which set of observables we want to test
        # Either just closure phases, or visibility info also.

        if observables=="cp":
            store_dict = [{"data":self.cp, "error":self.cperr, "uvcoords":uvcoords, \
                      "params":[self.cons[i],np.sqrt(ras**2+decs**2),180*np.arctan2(decs,ras)/np.pi], \
                      "wavls":self.wavls, "dof":self.cp.shape[0] - 3} for i in range(nstep)] 
            # Calc null chi^2
            self.chi2_null = chi2_grid_loop({"params":[0,0,0],"data":self.cp, \
                                    "error":self.cperr, "uvcoords":uvcoords,\
                                     "wavls":self.wavls, "dof": self.cp.shape[0] - 3})
            if threads>0:
                pool = Pool(processes=threads)
                print("Threads:", threads)
                self.chi2grid = np.array(pool.map(chi2_grid_loop, store_dict))
            else:
                self.chi2grid = np.zeros((nstep, nstep, nstep))
                for ii in range(len(self.cons)):
                    self.chi2grid[ii] = chi2_grid_loop(store_dict[ii])

        elif observables=="all":
            data = np.concatenate((self.cp, self.t3amp))
            error = np.concatenate((self.cperr, self.t3amperr))
            store_dict = [{"data":data, "error":error, "uvcoords":uvcoords, \
                      "params":[self.cons[i],np.sqrt(ras**2+decs**2),180*np.arctan2(decs,ras)/np.pi], \
                      "wavls":self.wavls, "dof":data.shape[0] - 3} for i in range(nstep)] 
            # Calc null chi^2
            self.chi2_null = chi2_grid_loop_all({"params":[0,0,0],"data":data, \
                                    "error":error, "uvcoords":uvcoords,\
                                     "wavls":self.wavls, "dof": data.shape[0] - 3})
            if threads>0:
                pool = Pool(processes=threads)
                print("Threads:", threads)
                self.chi2grid = np.array(pool.map(chi2_grid_loop_all, store_dict))
            else:
                self.chi2grid = np.zeros((nstep, nstep, nstep))
                for ii in range(len(self.cons)):
                    self.chi2grid[ii] = chi2_grid_loop_all(store_dict[ii])


        t3 = time.time()
        pool.terminate()
        print("took "+str(t3-t2)+"s to compute all chi^2 grid points")

        chi2min = np.where(self.chi2grid == self.chi2grid.min())
        bestparams = np.array([self.cons[chi2min[0]][0], \
                               np.sqrt(self.ras[chi2min[1]]**2 + self.decs[chi2min[2]]**2)[0], \
                               np.arctan2(self.decs[chi2min[2]], self.ras[chi2min[1]])[0]*180/np.pi])
        print("Best Contrast:", self.cons[chi2min[0]])
        print("Best Separation:", np.sqrt(self.ras[chi2min[1]]**2 + self.decs[chi2min[2]]**2))
        print("Best PA:", np.arctan2(self.decs[chi2min[2]], self.ras[chi2min[1]])*180/np.pi)

        self.significance = self.chi2_null - np.min(self.chi2grid, axis=0).transpose()
        self.significance[self.significance<0] = 0

        self.chi2map_savdata = {"chi2grid":self.chi2grid, "ra":self.ras, "dec":self.decs, "con": self.cons,\
                   "significance":self.significance}
        return bestparams


    def save_chi2map(self, absolute_path_filename_save):
        #savestr = self.savedir+os.path.sep+save+"_chi2map.pick"
        #savestr = self.savedir+os.path.sep+self.oifitsfn.replace(".oifits", "")+"_chi2map.pick"
        f = open(absolute_path_filename_save, "w")
        pickle.dump(self.chi2map_savdata, f)
        f.close()
        return None


    def plot_chi2map(self, savdata, savestr=False, show=False):
        #unpack everything
        ras = savdata['ra']
        decs = savdata['dec']
        chi2grid = savdata['chi2grid']
        cons = savdata['con']
        significance = savdata['significance']
        nstep = len(ras)
        
        plt.figure()
        plt.plot(nstep/2.0 -0.5,nstep/2.0 - 0.5, marker="*", color='w', markersize=20)
        #plt.imshow(np.min(self.chi2grid, axis=0).transpose(), cmap="cubehelix")
        #significance = self.chi2_null - np.min(self.chi2grid, axis=0).transpose()
        #significance[significance<0] = 0
        
        plt.imshow(np.sqrt(significance), cmap="cubehelix", interpolation="nearest")
        plt.xlabel("RA (mas)")
        plt.ylabel("DEC (mas)")
        plt.xticks(np.linspace(0, nstep, 5), np.linspace(ras.min(), ras.max(), 4+1))
        plt.yticks(np.linspace(0, nstep, 5), np.linspace(decs.min(), decs.max(), 4+1))
        plt.colorbar()
        plt.gca().invert_yaxis()

        if savestr is not False:
            plt.savefig(savestr)
        if show==True:
            plt.show()
        plt.clf()
        return None


    def two_hyp_test():
        """
        Is my data consistent with Null Hypothesis?
        """
        pass

    def detection_limits(self, ntrials = 1, seplims = [20, 200],\
                         conlims = [0.0001, 0.99], anglims = [0,360],\
                         nsep = 24, ncon=24, nang=24, threads=4,\
                         observables="cp", scale=1.0):
        """
        Inspired by pymask code.
        ntrials: Number of times we draw randomly
        seplims: Where to search in separation space, default 20-200 mas
        conlims: Contrast bounds of search def: 1e-4 to 0.99
        anglims: Sky angle bounds (deg), should leave this 0 to 360 deg
                 This routine will average over all angles for calculation
        nsep/ncon/nang: number of each to simulate -- should be a multiple
                        of the threads set for best performance. If this
                        takes too much memory, try reducing these #s and
                        increasing ntrials, since this sets the grid size
        threads: no threads on your machine for parallel processing
        observables: default is "cp" for just considering closure phases
                     can optionally set to "all" to also consider 
                     visibility amplitudes in t3amp axis. Currently this
                     option is not working/not fully tested. 
        save: to save or not to save? Default set to false. If turned on
              will save as detection_limits .pick and .pdf. Must change
              filename separately in driver/commands if running multiple.
        scale: Error scale -- typically set to sqrt(Nholes/3) to account for
               # indepent closure phases compared to total.
        """
        pool = Pool(processes = threads)

        priors = np.array([(-np.inf, np.inf) for f in range( 3 ) ])

        self.seps = np.linspace(seplims[0], seplims[1], nsep)
        self.angs = np.linspace(anglims[0], anglims[1], nang)
        nn = np.arange(ncon)
        r = (conlims[-1]/conlims[0])**(1 / float(ncon-1))
        self.cons = conlims[0] * r**(nn)
        #self.cons = np.linspace(1.0/float(conlims[0]), 1.0/float(conlims[1]), ncon)
        #self.cons = 1.0 / self.cons

        # Set up the big grids and add a wavelength axis so this all works
        seps = np.tile(self.seps, (ncon, nang, self.nwav, 1))
        seps = np.rollaxis(seps, -1, 0)
        cons = np.tile(self.cons, (nsep, nang, self.nwav, 1))
        cons = np.rollaxis(cons, -1, 1)
        angs = np.tile(self.angs, (nsep, ncon, self.nwav, 1))
        angs = np.rollaxis(angs, -1, 2)

        # set up big uvcoordinate grid, keep wavelength axis at the end
        # should be shape (2, 3, ncp, nsep, ncon, nang, nwav)
        uvcoords = np.rollaxis(np.rollaxis(np.rollaxis(np.tile(self.uvcoords, \
                               (nsep, ncon, nang, 1, 1, 1, 1)), -2,0), -2, 0), -2, 0)
        print("Computing model cps over", nsep*ncon*nang, "parameters.")

        # Set up some random errors to add in per trial
        # Consider scaling random cperr by wavelength?
        if observables=="cp":
            randnums = np.random.randn(int(ntrials), len(self.cp), int(self.nwav))
            # randomize* the measurement errors
            # errors shape here is [ntrials, ncps, nwavs]
            errors = scale*self.cperr[None, ...]*randnums
            # errors shape here becomes [ntrials, nsep, ncon, nang, ncp, nwav]?
            # ...which is way too many for multiwav data...
            #errors = np.rollaxis(np.tile(self.cperr[None, ...]*randnums,\
            #                     (nsep, ncon, nang,1, 1, 1)),-3,0)
            print("errors shape:", errors.shape)
            t1 = time.time()
            #modelcps is shape [ncp, nsep, ncon, nang, nwav]
            modelcps = model_cp_uv(uvcoords, cons, seps, angs, 1.0/self.wavls)
            # now becomes [nsep, ncon, nang, ncp, nwav]
            modelcps = np.rollaxis(modelcps, 0, -1)
            t2 = time.time()
            #modelcps = np.rollaxis(pool.map(model_cp_uv(uvcoords, \
            #                       cons, seps, angs, 1.0/self.wavls), 0, -1)
            print("Finished computing big grid, took", t2-t1, "s")
            print("modelcps shape:", modelcps.shape)

            t3 = time.time()
            print("setting up the dictionary...")
            store_dict = [{"self":self,"ntrials":ntrials, "model":modelcps, \
                           "randerrors":errors[i], "dataerrors":self.cperr} for i in range(len(errors))]

        elif observables=="all":
            randnums = np.random.randn(int(ntrials), \
                                       len(self.cp)+len(self.t3amp),\
                                       int(self.nwav))
            # randomize* the measurement errors
            allerrors = np.concatenate((self.cperr, self.t3amperr))
            errors = scale*allerrors[None, ...]*randnums
            print("errors shape:", errors.shape)

            #detec_grid = np.zeros((len(seps), len(cons), len(angs)))
            t1 = time.time()
            modelcps = np.rollaxis(model_cp_uv(uvcoords, cons, seps,\
                                   angs, 1.0/self.wavls), 0, -1)
            modelt3 = np.rollaxis(model_t3amp_uv(uvcoords, cons, seps,\
                                  angs, 1.0/self.wavls), 0, -1)
            t2 = time.time()
            print("Finished computing big grid, took", t2-t1, "s")
            print("modelcps shape:", modelcps.shape)
            print("modelt3 shape:", modelt3.shape)
            model = np.concatenate((modelcps, modelt3), axis=3)
            print(self.t3amp)
            print("t3 errors")
            print(self.t3amperr)
            print("all")
            print(allerrors)
            print("v2")
            print(self.v2err)
            print("cp errors")
            #self.cperr

            t3 = time.time()
            print("setting up the dictionary...")
            store_dict = [{"self":self,"ntrials":ntrials, "model":model, "randerrors":errors[i],\
                           "dataerrors":allerrors} for i in range(len(errors))]

        t4 = time.time()
        print("dictionary took", t4-t3, "s to set up")
        if observables=="all":
            big_detec_grid = np.sum(pool.map(detec_calc_loop_all, store_dict),axis=0) / float(ntrials)
        else:
            big_detec_grid = np.sum(pool.map(detec_calc_loop, store_dict),axis=0) / float(ntrials)
        pool.terminate()
        t5 = time.time()
        print("Time to finish detec_calc_loop:", t5-t3, "s")

        self.detec_grid = big_detec_grid.sum(axis=-1) / float(nang)
        # Get the order right
        self.detec_grid = self.detec_grid.transpose()

        """
        for ii in range(nsep):
            for jj in range(ncon):
                for kk in range(nang)
                    bin_model = model_cp_uv(self.uvcoords, seps[ii],\
                                 cons[jj], angs[kk], 1.0/self.wavls)
                    randomize = bin_model + (errors*randnums)
                    for trial in range(ntrials):
                        # null and binary hyp here are 1-D length ntrials
                    chi2null[kk,tt] = cp_binary_model([0,0,0], \
                                     {"wavl":self.wavls}, priors, None, \
                                     self.uvcoords, self.cp, \
                                     self.cperr*randnum[:,:,trial], stat="chi2")
                    chi2_grid = cp_binary_model([seps[ii], cons[jj], \
                                     angs[kk]], {"wavl":self.wavls}, \
                                     priors, None, self.uvcoords, self.cp,\
                                     errors, stat="chi2")
                diff = chi2_grid - chi2null
                # How many detects for each separation & contrast
                # Normalize by # of points
                detec_grid[ii,jj] = (diff <0.0).sum() / float(ntrials*len(angs))
        """
        # pickle the data
        clevels = [0.5, 0.9, 0.99, 0.999]
        self.savdata_deteclims = {"clevels": clevels, "separations": self.seps, \
                   "angles":self.angs, "contrasts":self.cons, \
                   "detections":self.detec_grid}
        return self.savdata_deteclims

    def plot_deteclims(self, savdata, savestr = False, plot="off"):
        # contour plot
        clevels = savdata["clevels"]
        seps = savdata["separations"]
        cons = savdata["contrasts"]
        angs = savdata["angles"]
        detec_grid = savdata["detections"]
        colors = ['k', 'k', 'k', 'k']
        plt.figure()
        SEP, CON = np.meshgrid(seps, cons)
        contours = plt.contour(SEP, CON, detec_grid, clevels,\
                               colors=colors, linewidth=2, \
                               extent=[seps.min(), seps.max(), \
                                       cons.min(), cons.max()])
        plt.yscale('log')
        plt.clabel(contours)
        plt.contourf(SEP, CON, detec_grid, clevels, cmap=plt.cm.bone)
        plt.colorbar()
        plt.xlabel("Separation (mas)")
        plt.ylabel("Contrast Ratio")
        plt.title("Detection Limits")

        if savestr is not False:
            plt.savefig(savestr)
        return None


    def grid_spectrum(self, sep, pa, ncon=100, conlims=[1.0e-3, 0.999], plot=True):
        """ If the position is known (sep, pa), look for best 
            contrast at each wavelength."""
        nn = np.arange(ncon)
        r = (conlims[-1]/conlims[0])**(1 / float(ncon-1))
        self.cons = conlims[0] * r**(nn)
        cons = np.tile(self.cons, (self.nwav, 1))
        cons = np.rollaxis(cons, -1, 0)
        print("cons shape", cons.shape)
        uvcoords = np.rollaxis(np.rollaxis(np.rollaxis(np.tile(self.uvcoords,\
                               (ncon, 1, 1, 1, 1)), -2,0), -2, 0), -2, 0)
        print("uvcoords shape", uvcoords.shape)
        print("Computing model cps over", ncon, "parameters.")
        t1 = time.time()
        model_cps = model_cp_uv(uvcoords, cons, sep, pa, 1.0/self.wavls)
        t2 = time.time()
        print("Finished computing big grid, took", t2-t1, "s")
        print("model shape:", model_cps.shape)
        model_cps = np.rollaxis(model_cps, 0, -1)
        print("new model shape", model_cps.shape)
        datacps = np.tile(self.cp, (ncon, 1, 1))
        dataerror = np.tile(self.cperr, (ncon, 1, 1))
        print("datacps shape", datacps.shape)

        t4 = time.time()
        self.con_spectrum = np.zeros((self.nwav, len(self.cons)))
        """
        if use_covar == True:
        for nn in range(ncon):
            model_cps_con = np.rollaxis(model_cps[nn,:,:], 0,-1)
            data_con = np.rollaxis(data_con[nn,:,:], 0,-1)
            chi2 = np.sum(np.dot((model_cps_con - data_con).transpose(),np.dot(inv_covar, (model_cps_con - data_con))))
        """
        for ll in range(self.nwav):
            #dof = self.nwav*self.ncp - 1
            fac = 1.0#(1/float(dof))
            chi2 = fac*np.sum((model_cps[:,:,ll] - datacps[:,:,ll])**2 / (dataerror[:,:,ll]**2), axis=-1)
            minimum = chi2.min()
            #print "chi2:", chi2.shape
            self.con_spectrum[ll, :] = chi2#loglike
        t5 = time.time()
        print("Time to finish contrast loop:", t5-t4, "s")
        if plot:
            plt.figure()
            plt.imshow(self.con_spectrum.transpose(), cmap="rainbow")
            #plt.plot(self.wavls*1.0e6, self.con_spectrum, 'o')
            plt.title("Rough fit spectrum in contrast ratio")
            plt.xticks(np.arange(self.nwav)[::5], np.round(self.wavls*1e6, 3)[::5])
            plt.yticks(np.arange(ncon)[::10], np.round(self.cons, 3)[::10])
            plt.xlabel("Wavelength (um)")
            plt.ylabel("Contrast Ratio $\chi^2$")
            plt.axis('normal')
            plt.colorbar()
            plt.show()
        return self.con_spectrum

    def correlation_plot(self, start=[0.16, 64, 219], bnds=50):
        """
        A nice visualization to see how the data compares 
        to model solutions. Plot is adjustable.

        separation in mas, pa in degrees
        """
        from matplotlib.widgets import Slider, Button, RadioButtons
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        plt.title("Model vs. Data")
        plt.xlim(-bnds, bnds)
        plt.ylim(-bnds, bnds)

        priors = np.array([(-np.inf, np.inf) for f in range( len(start) ) ])
        #constant = {'wavl':self.wavls}

        # Data and model both have shape ncp, nwav
        modelcps = model_cp_uv(self.uvcoords, start[0], start[1],\
                               start[2], 1.0/self.wavls)
        #print "model shape:"
        #print modelcps.shape
        #print type(modelcps)
        #print "why are these closure phases so small??"
        #print self.cp
        #print self.cp.shape
        plt.plot([-bnds, bnds], [-bnds,bnds])
        #plt.errorbar(self.cp.flatten(), modelcps.flatten(), yerr = self.cperr.flatten(), fmt='.')
        l, = plt.plot(self.cp.flatten(), modelcps.flatten(), '.')
        plt.xlabel("Measured closure phase (degrees)")
        plt.ylabel("Model closure phase (degrees)")
        plt.ylim(-bnds, bnds)
        plt.xlim(-bnds, bnds)

        # Set up the widget plot from matplotlib demo
        axcolor = "lightgoldenrodyellow"
        axang = plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=axcolor)
        axcrat = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        axsep = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

        sang = Slider(axang, 'Theta', 0,360 , valinit=start[2])
        scrat = Slider(axcrat, 'Cratio', 0.001, 0.999, valinit=start[0])
        ssep = Slider(axsep, 'Sep (mas)', 20, 300, valinit=start[1])

        def update(val):
            theta = sang.val
            print("theta:", theta)
            cratio = scrat.val
            print("cratio:", cratio)
            sep = ssep.val
            print("separation:", sep)
            newparams = [cratio, sep, theta]
            modelcps = model_cp_uv(self.uvcoords, cratio, \
                                   sep, theta, 1.0/self.wavls)
            l.set_ydata(modelcps)
            fig.canvas.draw_idle()
        sang.on_changed(update)
        scrat.on_changed(update)
        ssep.on_changed(update)

        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        def reset(event):
            sang.reset()
            scrat.reset()
            ssep.reset()
        button.on_clicked(reset)

        rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
        radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
        def colorfunc(label):
            l.set_color(label)
            fig.canvas.draw_idle()
        radio.on_clicked(colorfunc)

        plt.show()

    def run_emcee(self, params, constant={}, nwalkers = 250, \
                  niter = 1000, burnin=500, spectrum_model=None, priors=None, \
                  threads=4, scale=1.0, observables="cp"):
        """
        A lot of options in this method, read carefully.

        Arguments params and constant are dictionaries. 

        For example if you wanted to search for 3 parameters, contrast, separation, and PA:
            params = {'con': cr_val, 'sep': sep_mas, 'pa': pa_deg}
            default constant contains only {'wavls': array_of_wavelengths (m)}
            can provide separation and pa if trying to fit a spectrum

        Priors are bounds for the different parameters. Default is no bounds.

        """
        import emcee
        self.constant = constant
        self.constant['wavl'] = self.wavls
        # Options are None 'slope' or 'free'
        self.spectrum_model = spectrum_model
        self.params = params

        if priors is not None:
            self.priors = priors
        else:
            self.priors = [(-np.inf, np.inf) for f in range( len(self.params.keys()) ) ]
        print("priors:")
        print(self.priors)

        #guess = np.zeros(self.ndim)
        guess = self.make_guess()
        self.ndim = len(guess)

        # check if PA guess = 0.0 to ensure that the walkers have different values
        if guess[2] == 0.0:
            guess[2] = 360.0

        p0 = np.array([guess + 0.1*guess*np.random.rand(self.ndim) for i in range(nwalkers)])
        # wrap PA and check that the random jitter doesn't move us out of the prior
        if (self.priors[2][0] == 0) & (self.priors[2][1] == 360):
            p0[...,2] = p0[...,2] % 360.0
        elif (self.priors[2][0] == -180) & (self.priors[2][1] == 180):
            p0[...,2] = ((p0[...,2]+180.) % 360.0)-180.

        for i in range(self.ndim):
            p0[:, i] = np.clip(p0[:, i], self.priors[i][0], self.priors[i][1])

        print(guess)
        print("p0", len(p0))

        t0 = time.time()
        #print "nwalkers", nwalkers, "args", self.constant, self.priors, self.spectrum_model, self.uvcoords, self.cp, self.cperr
        if observables == "cp":
            self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, cp_binary_model, threads=threads, args=[self.constant, self.priors, self.spectrum_model, self.uvcoords, self.cp, scale*self.cperr])
        elif observables == "all":
            self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, \
                            allvis_binary_model, threads=threads, \
                            args=[self.constant, self.priors, self.spectrum_model, self.uvcoords, \
                            self.cp, scale*self.cperr, self.t3amp, scale*self.t3amperr])
        elif observables == "v2":
            self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, v2_binary_model, threads=threads, args=[self.constant, self.priors, self.spectrum_model, self.uvcoords_vis, self.v2, scale*self.v2err])
        elif observables == "multiple_all":
            self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, \
                            bispec_multi_model, threads=threads, \
                            args=[self.constant, self.priors, self.spectrum_model, self.uvcoords, \
                            self.cp, scale*self.cperr, self.t3amp, scale*self.t3amperr])
        elif observables == "multiple_cp":
            self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, \
                            cp_multi_model, threads=threads, \
                            args=[self.constant, self.priors, self.spectrum_model, self.uvcoords, \
                            self.cp, scale*self.cperr])
        else:
            print("invalid choice of observable:", observables)
            print("options are 'cp', 'v2', and 'all'")

        pos, prob, state = self.sampler.run_mcmc(p0, burnin)
        self.sampler.reset()
        t2 = time.time()
        print("burn in complete, took ", t2-t0, "s")
        pos, prob, state = self.sampler.run_mcmc(pos, niter)
        t3 = time.time()
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(self.sampler.acceptance_fraction)))
        print("This number should be between ~ 0.25 and 0.5 if everything went as planned.")

        print("ran mcmc, took", t3 - t2, "s")
        self.chain = self.sampler.flatchain
        self.fullchain = self.sampler.chain

        self.mcmc_results = {}
        print("=========================")
        print("emcee found....")
        #for ii, key in enumerate(self.params.keys()):
        for ii, key in enumerate(self.keys):
            self.mcmc_results[key] = self.chain[:,ii]
            mean = np.mean(self.mcmc_results[key])
            err = np.std(self.mcmc_results[key])
            if "sep" in key:
                print(key, ":", mean, "+/-", err, "mas")
            elif "pa" in key:
                print(key, ":", mean, "+/-", err, "deg")
            else:
                print(key, ":", mean, "+/-", err)
        print("=========================")
        # To do: report separation in mas? pa in deg?
        return self.mcmc_results


    def make_guess(self):
        # A few options here, can provide:
        # 1. contrast, separation, angle -- 3 parameters to fit
        # 2. contrast_min, slope, separation, angle -- 4 parameters
        # 3. contrast_min, slope -- 2 parameters (position is given as constant)
        # 4. nwav different contrasts - nwav parameters (position is given as constant)
        if self.spectrum_model==None:
            ncomp = np.size(self.params[list(self.params.keys())[0]])
            guess = np.zeros(len(self.params)*ncomp)
            guess[0::3] = self.params['con']
            guess[1::3] = self.params['sep']
            guess[2::3] = self.params['pa']
            if len(guess) == 3:
                self.keys = ['con', 'sep', 'pa']
            else:
                self.keys = []
                self.keys = sum([self.keys+['con_{0}'.format(f), 'sep_{0}'.format(f),'pa_{0}'.format(f)] \
                             for f in range(ncomp)], [])
        elif self.spectrum_model=="slope":
            guess = np.zeros(len(self.params))
            guess[0] = self.params['con']
            guess[1] = self.params["slope"]
            guess[2] = self.params["sep"]
            guess[3] = self.params["pa"]
            self.keys = ['con', 'slope','sep', 'pa']
        elif self.spectrum_model == "free":
            guess = np.array([self.params["con"]])[0] # here con is given as an array size nwav
            self.keys = ['wl_{0:02d}'.format(f) for f in range(len(guess))]
        else:
            print("invalid spectrum_model set")
        print("keys", self.keys)
        print("params", guess)
        return guess


    def save_mcmc_results(self, absolute_path_filename_save):
        f = open(absolute_path_filename_save, "wb")
        pickle.dump(self.mcmc_results, f)
        f.close()

    def save_mcmc_chain(self, absolute_path_filename_save):
        f = open(absolute_path_filename_save, "wb")
        pickle.dump(self.chain, f)
        f.close()


    def corner_plot(self, fn, title_fmt=None):
        import corner
        plt.figure(1)
        fig = corner.corner(self.chain, labels = self.keys, bins = 50, show_titles=True, title_fmt=title_fmt)
        plt.savefig(self.savedir+fn)
        if self.plot=="on":
            plt.show()
        plt.clf()
        return None
        
    def plot_chain_convergence(self):
        samples  = self.fullchain[:, 50:, :].reshape((-1, self.ndim))
        plt.figure()
        self.chain_convergence = {}
        for ii in range(len(samples[-1])):
            plt.subplot2grid((self.ndim,1),(ii,0))
            plt.plot(samples[:,ii])
            plt.ylabel(self.keys[ii])
            plt.xlabel("step number")
            self.chain_convergence[self.keys[ii]] = samples[:,ii]
        plt.savefig(self.savedir+"/chain_convergence.pdf")
        # Pickle and save this data?
        f = open(self.savedir+"/chain_convergence.pick", "wb")
        pickle.dump(self.chain_convergence, f)
        f.close()
        plt.show()
        plt.clf()
        return self.chain_convergence


def cp_binary_model(params, constant, priors, spectrum_model, uvcoords, cp, cperr, stat="loglike"):
    # really want to be able to give this guy some general oi_data and have bm() sort it out.
    # Need to figure out how to add in the priors

    ##################################################
    # HOW DO I TUNE THIS DEPENDING ON MY OBSERVATIONS? - need a keyword or something, need help.
    # data = self.cp, self.cperr#, self.v2, self.v2err
    ##################################################

    # priors, i.e. bounds here
    # constant
    # uvcoords
    # cp
    # cperr

    for i in range(len(params)):
        #print "HERE:",len(params)
        #print "HERE:",params[i]
        if (params[i] < priors[i][0] or params[i] > priors[i][1]):  
            return -np.inf

    if spectrum_model == None:

        # Model from params
        #model_cp = model_cp_uv(self.uvcoords, params['con'], params['sep'], \
        #                   params['pa'], 1.0/self.constant['wavl'])
        model_cp = model_cp_uv(uvcoords, params[0], params[1], \
                            params[2], 1.0/constant['wavl'])
    elif spectrum_model == 'slope':
        # params needs 'con_start' starting contrast and 'slope,' sep & pa constant?
        wav_step = constant['wavl'][1] - constant['wavl'][0]
        # contrast model is con_start + slope*delta_lambda
        #contrast = params[0] + params[1]*wav_step
        band_diff = constant['wavl'][-1] - constant['wavl'][0]
        contrast = np.linspace(params[0], params[0]+(params[1]*band_diff), \
                               num = len(constant['wavl']))
        # Model from params
        model_cp = model_cp_uv(uvcoords, contrast, params[2], \
                            params[3], 1.0/constant['wavl'])
    elif spectrum_model == 'free' :
        # Model from params - params is contrast array nwav long, sep & pa constant
        model_cp = model_cp_uv(uvcoords, params, constant['sep'], \
                            constant['pa'], 1.0/constant['wavl'])
    else:
        sys.exit("Invalid spectrum model")

    chi2stat = reduced_chi2(cp, cperr, model_cp, 34.0)
    ll = logl(cp, cperr, model_cp)
    if stat == "loglike":
        return ll
    elif stat == "chi2":
        return chi2stat

def v2_binary_model(params, constant, priors, spectrum_model, uvcoords, v2, v2err, stat="loglike"):
    # really want to be able to give this guy some general oi_data and have bm() sort it out.
    # Need to figure out how to add in the priors

    ##################################################
    # HOW DO I TUNE THIS DEPENDING ON MY OBSERVATIONS? - need a keyword or something, need help.
    # data = self.cp, self.cperr#, self.v2, self.v2err
    ##################################################

    # priors, i.e. bounds here
    # constant
    # uvcoords
    # cp
    # cperr

    for i in range(len(params)):
        if (params[i] < priors[i][0] or params[i] > priors[i][1]):  
            return -np.inf

    if spectrum_model == None:

        # Model from params
        #model_cp = model_cp_uv(self.uvcoords, params['con'], params['sep'], \
        #                   params['pa'], 1.0/self.constant['wavl'])
        model_v2 = model_v2_uv(uvcoords, params[0], params[1], \
                            params[2], 1.0/constant['wavl'])
    elif spectrum_model == 'slope':
        # params needs 'con_start' starting contrast and 'slope,' sep & pa constant?
        wav_step = constant['wavl'][1] - constant['wavl'][0]
        # contrast model is con_start + slope*delta_lambda
        contrast = params[0] + params[1]*wav_step
        # Model from params
        model_v2 = model_v2_uv(uvcoords, contrast, params[2], \
                            params[3], 1.0/constant['wavl'])
    elif spectrum_model == 'free' :
        # Model from params - params is contrast array nwav long, sep & pa constant
        model_v2 = model_v2_uv(uvcoords, params, constant['sep'], \
                            constant['pa'], 1.0/constant['wavl'])
    else:
        sys.exit("Invalid spectrum model")

    chi2stat = reduced_chi2(v2, v2err, model_v2, 34.0)
    ll = logl(v2, v2err, model_v2)
    if stat == "loglike":
        return ll
    elif stat == "chi2":
        return chi2stat

def allvis_binary_model(params, constant, priors, spectrum_model, uvcoords, \
                        cp, cperr, vis, viserr, stat="loglike", dof = 1):
    """
    For now, writing a new function to do this with both cps and bisectrum visibilities
    """
    for i in range(len(params)):
        if (params[i] < priors[i][0] or params[i] > priors[i][1]):  
            return -np.inf

    if spectrum_model == None:

        # Model from params
        model_cp = model_cp_uv(uvcoords, params[0], params[1], \
                           params[2], 1.0/constant['wavl'])
        model_vis = model_t3amp_uv(uvcoords, params[0], params[1], \
                           params[2], 1.0/constant['wavl'])
        #model_cp, model_vis = model_allvis_uv(uvcoords, uvcoords_vis, params[0], params[1], \
        #                    params[2], 1.0/constant['wavl'])

        model = np.zeros((model_cp.shape[0] + model_vis.shape[0], model_cp.shape[1]))
        model[:model_cp.shape[0],...] = model_cp
        model[model_cp.shape[0]:,...] = model_vis
    else:
        sys.exit("Invalid spectrum model")

    allvisobs = np.zeros((cp.shape[0]+vis.shape[0], cp.shape[1]))
    allvisobserr = np.zeros((cp.shape[0]+vis.shape[0], cp.shape[1]))
    allvisobs[:cp.shape[0], ...] = cp
    allvisobs[cp.shape[0]:, ...] = vis
    allvisobserr[:cp.shape[0], ...] = cperr
    allvisobserr[cp.shape[0]:, ...] = viserr

    if stat == "loglike":
        ll = logl(allvisobs, allvisobserr, model)
        return ll
    elif stat == "chi2":
        chi2stat = reduced_chi2(allvisobs, allvisobserr, model, dof)
        return chi2stat

def cp_multi_model(params, constant, priors, spectrum_model, uvcoords, \
                        cp, cperr, stat="loglike", dof = 1):
    """
    For now, writing a new function to do this with both cps and bisectrum visibilities
    """
    for i in range(len(params)):
        if (params[i] < priors[i%3][0] or params[i] > priors[i%3][1]):  
            return -np.inf
    # chop up the params by type input: [c1, s1, p1, c2, s2, p2, ...]
    nsource = len(params) / 3
    cons = params[0::3]
    seps = params[1::3]
    pas = params[2::3]
    if spectrum_model == None:

        # Model from params
        model_cp, model_vis = model_bispec_uv(uvcoords, cons, seps, \
                           pas, 1.0/constant['wavl'])
        #model_cp = np.zeros(cp.shape)
        #for q in range(nsource):
        #    model_cp += model_cp_uv(uvcoords, cons[q], seps[q], pas[q], 1.0/constant['wavl'])

        #model = np.zeros((model_cp.shape[0] + model_vis.shape[0], model_cp.shape[1]))
        #model[:model_cp.shape[0],...] = model_cp
        #model[model_cp.shape[0]:,...] = model_vis
    else:
        sys.exit("Invalid spectrum model")

    """
    allvisobs = np.zeros((cp.shape[0]+vis.shape[0], cp.shape[1]))
    allvisobserr = np.zeros((cp.shape[0]+vis.shape[0], cp.shape[1]))
    allvisobs[:cp.shape[0], ...] = cp
    allvisobs[cp.shape[0]:, ...] = vis
    allvisobserr[:cp.shape[0], ...] = cperr
    allvisobserr[cp.shape[0]:, ...] = viserr
    """

    if stat == "loglike":
        ll = logl(cp, cperr, model_cp)
        return ll
    elif stat == "chi2":
        chi2stat = reduced_chi2(cp, cperr, model_cp, dof)
        return chi2stat

def bispec_multi_model(params, constant, priors, spectrum_model, uvcoords, \
                        cp, cperr, vis, viserr, stat="loglike", dof = 1):
    """
    For now, writing a new function to do this with both cps and bisectrum visibilities
    """
    for i in range(len(params)):
        if (params[i] < priors[i%3][0] or params[i] > priors[i%3][1]):  
            return -np.inf

    # chop up the params by type input: [c1, s1, p1, c2, s2, p2, ...]
    nsource = len(params) / 3
    cons = params[0::3]
    seps = params[1::3]
    pas = params[2::3]
    if spectrum_model == None:

        # Model from params
        model_cp, model_vis = model_bispec_uv(uvcoords, cons, seps, \
                           pas, 1.0/constant['wavl'])

        model = np.zeros((model_cp.shape[0] + model_vis.shape[0], model_cp.shape[1]))
        model[:model_cp.shape[0],...] = model_cp
        model[model_cp.shape[0]:,...] = model_vis
    else:
        sys.exit("Invalid spectrum model")

    allvisobs = np.zeros((cp.shape[0]+vis.shape[0], cp.shape[1]))
    allvisobserr = np.zeros((cp.shape[0]+vis.shape[0], cp.shape[1]))
    allvisobs[:cp.shape[0], ...] = cp
    allvisobs[cp.shape[0]:, ...] = vis
    allvisobserr[:cp.shape[0], ...] = cperr
    allvisobserr[cp.shape[0]:, ...] = viserr

    if stat == "loglike":
        ll = logl(allvisobs, allvisobserr, model)
        return ll
    elif stat == "chi2":
        chi2stat = reduced_chi2(allvisobs, allvisoberr, model, dof)
        return chi2stat

def get_data(self):
    # Move this function out, pass values to the object
    try:
        self.oifdata = oifits.open(self.oifitsfn)
    except:
        print("Unable to read oifits file")
    try:
        self.avparang = self.oifdata.avparang
        self.parang_range = self.oifdata.parang_range
    except:
        print("oifits has no parang info, moving on...")

    print("Apparently we opened ", self.oifitsfn, "successfully")
    #variables = self.oifdata.__dict__.keys()
    #print("\n***** \n",variables)
    #print("\n\t")
    #print(" attribute values are:", 'target', type(self.oifdata.target), self.oifdata.target, "\n\t")
    #print(" attribute values are:", 'header', type(self.oifdata.header), self.oifdata.header, "\n\t") 
    #print(" attribute values are:", 'wavelength', type(self.oifdata.wavelength), self.oifdata.wavelength, "\n\t")
    #print(" attribute values are:", 'vis2', type(self.oifdata.vis2), self.oifdata.vis2, "\n\t")
    #print(" attribute values are:", 't3', type(self.oifdata.t3), self.oifdata.t3, "\n\t")
    #print(" attribute values are:", 'vis', type(self.oifdata.vis), self.oifdata.vis, "\n\t")
    #print(" attribute values are:", 'array', type(self.oifdata.array), self.oifdata.array, "\n\t")
    #print("***** \n")

    self.telescope = list(self.oifdata.wavelength.keys())[0]
    self.ncp = len(self.oifdata.t3)
    self.nbl = len(self.oifdata.vis2)
    self.wavls = self.oifdata.wavelength[self.telescope].eff_wave
    self.eff_band = self.oifdata.wavelength[self.telescope].eff_band
    self.nwav = len(self.wavls)
    self.uvcoords = np.zeros((2, 3, self.ncp))
    self.uvcoords_vis = np.zeros((2, self.nbl))

    # Now collect fringe observables and coordinates
    self.cp = np.zeros((self.ncp, self.nwav))
    self.cperr = np.zeros((self.ncp, self.nwav))
    self.t3amp = np.zeros((self.ncp, self.nwav))
    self.t3amperr = np.zeros((self.ncp, self.nwav))
    self.v2 = np.zeros((self.nbl, self.nwav))
    self.v2err = np.zeros((self.nbl, self.nwav))
    self.pha = np.zeros((self.nbl, self.nwav))
    self.phaerr = np.zeros((self.nbl, self.nwav))

    #Now, if extra_error is specified and there is a wavelength axis, we scale the 
    # estimated error with wavelength - User specifies error at shorted wavl?
    #self.extra_error = self.extra_error*np.ones(self.nwav)*self.wavls[0] / (self.wavls)

    for ii in range(self.ncp):
        #self.cp[:,ii] = self.oifdata.t3[ii].t3phi
        #self.cperr[:,ii] = self.oifdata.t3[ii].t3phierr
        #self.uvcoords[0,:,ii] = self.oifdata.t3[ii].u1coord, self.oifdata.t3[ii].u2coord,\
        #           -(self.oifdata.t3[ii].u1coord+self.oifdata.t3[ii].u2coord)
        #self.uvcoords[1, :,ii] = self.oifdata.t3[ii].v1coord, self.oifdata.t3[ii].v2coord,\
        #           -(self.oifdata.t3[ii].v1coord+self.oifdata.t3[ii].v2coord)
        self.cp[ii, :] = self.oifdata.t3[ii].t3phi
        self.cperr[ii, :] = np.sqrt(self.oifdata.t3[ii].t3phierr**2 + self.extra_error**2)
        self.t3amp[ii, :] = self.oifdata.t3[ii].t3amp
        self.t3amperr[ii, :] = np.sqrt(self.oifdata.t3[ii].t3amperr**2 + self.extra_error**2)
        self.uvcoords[0,:,ii] = self.oifdata.t3[ii].u1coord, self.oifdata.t3[ii].u2coord,\
                    -(self.oifdata.t3[ii].u1coord+self.oifdata.t3[ii].u2coord)
        self.uvcoords[1, :,ii] = self.oifdata.t3[ii].v1coord, self.oifdata.t3[ii].v2coord,\
                    -(self.oifdata.t3[ii].v1coord+self.oifdata.t3[ii].v2coord)
        #self.t3vis[:,ii] = self.oifdata.vis2[]
    #print self.cp
    for jj in range(self.nbl):
        #self.v2[:,jj] = self.oifdata.vis2[jj].vis2data
        #self.v2err[:,jj] = self.oifdata.vis2[jj].vis2err
        self.v2[jj, :] = self.oifdata.vis2[jj].vis2data
        self.v2err[jj, :] = self.oifdata.vis2[jj].vis2err
        try:
            #self.pha[:,jj] = self.oifdata.vis[jj].visphi
            #self.phaerr[:,jj] = self.oifdata.vis[jj].visphierr
            self.pha[jj, :] = self.oifdata.vis[jj].vispha
            self.phaerr[jj, :] = self.oifdata.vis[jj].visphaerr
            self.cv = np.sqrt(self.v2)*np.exp(-1j*self.pha)
        except:
            pass
        self.uvcoords_vis[0,jj] = self.oifdata.vis2[jj].ucoord
        self.uvcoords_vis[1,jj] = self.oifdata.vis2[jj].vcoord
    # hack right now to take care of 0 values, set to some limit, 0.001 right now
    floor = 0.00001
    self.cperr[self.cperr<floor] = self.cperr[self.cperr!=0.0].mean()
    self.phaerr[self.phaerr<floor] = self.phaerr[self.phaerr!=0.0].mean()
    self.v2err[self.v2err<floor] = self.v2err[self.v2err!=0.0].mean()
    
    # replicate the uv coordinates over the wavelength axis
    self.uvcoords = np.tile(self.uvcoords, (self.nwav, 1, 1, 1))
    self.uvcoords_vis = np.tile(self.uvcoords_vis, (self.nwav, 1, 1))
    # Now uvcoords is shape (nwav, 2, 3, ncps)
    # So we move nwav axis to the end:
    self.uvcoords = np.rollaxis(self.uvcoords, 0, 4)
    self.uvcoords_vis = np.rollaxis(self.uvcoords_vis, 0, 3)
    #for q in range(self.nwav-1):
    #   self.uvcoords[:,:,:,f] = self.uvcoords[:,:,:,0]


def detec_calc_loop(dictlist):
    # ndetected should have shape (nsep, ncon, nang) -- the first 3 dimensions of the cp model
    simcps = dictlist['model'].copy()
    simcps += dictlist['randerrors']
    #chi2null = reduced_chi2(simcps, dictlist['dataerrors'], 0)
    #chi2bin = reduced_chi2(simcps, dictlist['dataerrors'], dictlist['model'])
    chi2bin_m_chi2null = np.sum( ((dictlist['model'] - simcps)**2 - (simcps**2)) /(dictlist['dataerrors']**2), axis=(-1,-2))
    #detected = (chi2bin - chi2null)<0.0
    detected = chi2bin_m_chi2null<0.0
    #ndetected /= float(dictlist['ntrials'])
    return detected

def detec_calc_loop_all(dictlist):
    # ndetected should have shape (nsep, ncon, nang) -- the first 3 dimensions of the cp model
    simcps = dictlist['model'].copy()
    simcps += dictlist['randerrors']
    #chi2null = reduced_chi2(simcps, dictlist['dataerrors'], 0)
    #chi2bin = reduced_chi2(simcps, dictlist['dataerrors'], dictlist['model'])
    nullmodel = np.zeros(simcps.shape)
    nullmodel[:,:,:,simcps.shape[3]/2:, :] = 1.0
    chi2bin_m_chi2null = np.sum( ((dictlist['model'] - simcps)**2 - ((simcps-nullmodel)**2)) /(dictlist['dataerrors']**2), axis=(-1,-2))
    #detected = (chi2bin - chi2null)<0.0
    detected = chi2bin_m_chi2null<0.0
    #ndetected /= float(dictlist['ntrials'])
    return detected


def logl(data, err, model):
    """
    Likelihood given data, errors, and the model values
    These are all shape (nobservable, nwav)
    """
    #for ii in range(len(model)):
    #   #ll += -0.5*np.log(2*np.pi)*data[2*ii].size + np.sum(-np.log(data[2*ii+1]**2)
    #return -0.5*np.log(2*np.pi) - np.sum(np.log(err)) - np.sum((model - data)**2/(2*data**2))
    #return -0.5*np.log(2*np.pi)*data.size + np.sum(-np.log(err**2) - 0.5*((model - data)/err)**2)
    chi2 = np.nansum(((data-model)/err)**2)
    loglike = -chi2/2
    #return np.sum(-np.log(err**2) - 0.5*((model - data)/err)**2)
    return loglike


def logl_cov(flatdata, invcovmat, flatmodel):
    """
    flatdata and flatmodel must be the same shape & len=1
    """
    v_i = (flatdata - flatmodel).reshape(flatdata.shape[0], 1)
    chi2 = np.dot(flatdata - flatmodel, np.dot(incovmat, (flatdata-flatmodel).T))
    loglike = -chi2/2
    return loglike

def reduced_chi2(data, err, model, dof=1.0):
    return (1/float(dof))*np.sum(((model - data)**2)/(err**2), axis=(-1,-2))


def assemble_cov_mat(self):
    meancps = np.mean(self.cp, axis=0)
    flat_mean_sub_cp = (self.cp - meancps[None,:]).flatten
    covmat = flat_mean_sub_cp[None,:]*flat_mean_sub_cp[:,None]
    return covmat

def chi2_grid_loop(args):
    # Model from data, err, uvcoords, params, wavls
    p0, p1, p2 = args['params']
    modelcps = np.rollaxis(model_cp_uv(args['uvcoords'], p0, p1, p2, 1/args['wavls']), 0, -1)
    chi2 = np.nansum( (modelcps - args['data'])**2 / args['error']**2, axis = (-1,-2))/ args["dof"]
    return chi2


def chi2_grid_loop_all(args):
    # Model from data, err, uvcoords, params, wavls
    p0, p1, p2 = args['params']
    modelcps = np.rollaxis(model_cp_uv(args['uvcoords'], p0, p1, p2, 1/args['wavls']), 0, -1)
    modelt3 = np.rollaxis(model_t3amp_uv(args['uvcoords'], p0, p1, p2, 1/args['wavls']), 0, -1)
    model = np.concatenate((modelcps, modelt3), axis=2)
    chi2 = np.nansum( (model - args['data'])**2 / args['error']**2, axis = (-1,-2))/ args["dof"]
    return chi2

class DiskAnalyze:
    def __init__(self):
        print("not finished.")

    def diffvis_model(self, params, priors):
        """
        Calibrate polz data and calculate differential visibilities - 
        Later: Forward model something from a radiative transfer code. 
               This is more work...
        """

        # priors, here we're doing a general search, so it's a good idea to have some priors
        for i in range(len(params)):
            if (params[i] < priors[i,1] or params[i] > priors[i,0]):    
                return -np.inf
            else:
                pass

    def vis_model_ellipse(self, params, priors):

        data = self.cp, self.cperr, self.v2, self.v2err

        # priors, here we're doing a general search, so it's a good idea to have some priors
        for i in range(len(params)):
            if (params[i] < priors[i,1] or params[i] > priors[i,0]):    
                return -np.inf
            else:
                pass

                model_vis = model_vis_ellipse(params['semmaj'], params['semmin'], params['inc'])

                ll = logl(data, model)
                return ll


