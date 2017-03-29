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

"""


# Standard imports
import os, sys, time
import numpy as np
from astropy.io import fits
from scipy.misc import comb
from scipy.stats import sem, mstats
import cPickle as pickle
import matplotlib.pyplot as plt
plt.ion() # Added after JA's suggestion - eventually will make plotting better.

# Module imports
from fringefitting.LG_Model import NRM_Model
import misctools.utils
from misctools.utils import mas2rad, baselinify, rad2mas
import misctools.utils as utils
from modeling.binarymodel import model_cp_uv
from multiprocessing import Pool

import oifits

class FringeFitter:
    def __init__(self, instrument_data, **kwargs):
        """
        Fit fringes in the image plane

        Takes an instance of the appropriate instrument class
        Various options can be set

        kwarg options:
        oversample - model oversampling (also how fine to measure the centering)
        centering - If you already know the subpixel centering of your data, give it here (not recommended)
        savedir - Where do you want to save the new files to? Default is working directory.
        datadir - Where is your data? Default is working directory.
        npix - How many pixels of your data do you want to use? default is 121x121
        debug - will plot the FT of your data next to the FT of a reference PSF. Needs poppy package to run
        verbose_save - saves more than the standard files

        auto_pixscale - will search for the best pixel scale value for your data given instrument geometry
        auto_rotate - will search for the best rotation value for your data given instrument geometry

        main method:
        * fit_fringes

        Idea: default interact == True. User can turn this off to have everything automatically overwrite?

        """
        self.instrument_data = instrument_data

        #######################################################################
        # Options
        if "oversample" in kwargs:
            self.oversample = kwargs["oversample"]
        else:
            #default oversampling is 3
            self.oversample = 3
        if "auto_pixscale" in kwargs:
            # can be True/False or 1/0
            self.auto_scale = kwargs["auto_pixscale"]
        else:
            self.auto_scale = False
        if "auto_rotate" in kwargs:
            # can be True/False or 1/0
            self.auto_rotate = kwargs["auto_rotate"]
        else:
            self.auto_rotate = False
        if "centering" in kwargs:
            self.hold_centering = kwargs["centering"]
        else:
            # default is auto centering
            self.hold_centering = False
        if "savedir" in kwargs:
            self.savedir = kwargs["savedir"]
        else:
            self.savedir = os.getcwd()
        if "datadir" in kwargs:
            self.datadir = kwargs["datadir"]
        else:
            self.datadir = os.getcwd()
        if "npix" in kwargs:
            self.npix = kwargs["npix"]
        else:
            self.npix = 121
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
        if 'flip' in kwargs:
            self.flip = kwargs['flip']
        else:
            self.flip = False

        #######################################################################


        #######################################################################
        # Create directories if they don't already exit
        try:
            os.mkdir(self.savedir)
        except:
            if self.interactive is True:
                print self.savedir+" Already exists, rewrite its contents? (y/n)"
                ans = raw_input()
                if ans == "y":
                    pass
                elif ans == "n":
                    sys.exit("use alternative save directory with kwarg 'savedir' when calling FringeFitter")
                else:
                    sys.exit("Invalid answer. Stopping.")
            else:
                pass

        self.refimgs = self.instrument_data.ref_imgs_dir # could be taken care of in InstrumentData?
        try:
            os.mkdir(self.refimgs)
        except:
            pass
        #######################################################################

        np.savetxt(self.savedir+"/coordinates.txt", self.instrument_data.mask.ctrs)
        np.savetxt(self.savedir+"/wavelengths.txt", self.instrument_data.wavextension[0])

        #nrm = NRM_Model(mask = self.instrument_data.mask, pixscale = self.instrument_data.pscale_rad, over = self.oversample, holeshape=self.instrument_data.holeshape)
        #print nrm.holeshape
        # In future can just pass instrument_data to NRM_Model

        #plot conditions
        if self.debug==True or self.auto_scale==True or self.auto_rotate==True:
            import matplotlib.pyplot as plt
        if self.debug==True:
            import poppy.matrixDFT as mft

    def fit_fringes(self, fns, threads=0):
        if type(fns) == str:
            fns = [fns, ]


#         store_dict = [{"data":self.cp, "error":self.cperr, "uvcoords":uvcoords, \
#                       "params":[self.cons[i],np.sqrt(ras**2+decs**2),180*np.arctan2(decs,ras)/np.pi], \
#                       "wavls":self.wavls} for i in range(nstep)] 
                      
        store_dict = [{"object":self, "file":self.datadir+"/"+fn,"id":jj} for jj,fn in enumerate(fns)] 
        
        t2 = time.time()
#         if threads>0:
#             from multiprocessing.dummy import Pool as ThreadPool 
#             pool = ThreadPool(processes=threads) 
# #             pool = Pool(processes=threads)
#             print "Running fit_fringes in parallel with %d threads" % threads
# #             self.chi2grid = np.array(pool.map(chi2_grid_loop, store_dict))
#             pool.map(fit_fringes_parallel , store_dict)
#             t3 = time.time()
#             print "Parallel with %d threads took %s s to fit all fringes" % (threads,str(t3-t2))
#       
#         else:
        if 1:
            for jj,fn in enumerate(fns):              
                fit_fringes_parallel({"object":self, "file":self.datadir+"/"+fn,"id":jj}, threads)
            t3 = time.time()
            print "Parallel with %d threads took %s s to fit all fringes" % (threads,str(t3-t2))
#             print "Linear processing        took %s s to fit all fringes" % (str(t3-t2))
                
            
            
 #                self.scidata, self.scihdr = self.instrument_data.read_data(self.datadir+"/"+fn)
# 
#                 #ctrref = utils.centerit(scidata[)
# 
#                 self.sub_dir_str = self.instrument_data.sub_dir_str
#                 try:
#                     os.mkdir(self.savedir+self.sub_dir_str)
#                 except:
#                     pass
# 
#             
#                 for slc in range(self.instrument_data.nwav):
#                     # create the reference PSF directory if doing any auto_scaling or rotation
#                     try:
#                         os.mkdir(self.refimgs+'{0:02d}'.format(slc)+'/')
#                     except:
#                         pass
# 
#                     # NRM_Model
#                     nrm = NRM_Model(mask=self.instrument_data.mask, pixscale = self.instrument_data.pscale_rad,\
#                                     holeshape=self.instrument_data.holeshape, over = self.oversample, flip=self.flip)
# 
#                     nrm.refdir=self.refimgs+'{0:02d}'.format(slc)+'/'
#                     nrm.bandpass = self.instrument_data.wls[slc]
#                     #hdr['WAVL'] = wls[slc]
# 
#                     self.ctrd = utils.centerit(self.scidata[slc, :,:], r = self.npix//2)
#                     refslice = self.ctrd.copy()
#                     if True in np.isnan(refslice):
#                         refslice=utils.deNaN(5, self.ctrd)
#                         if True in np.isnan(refslice):
#                             refslice = utils.deNaN(20,refslice)
# 
# 
#                     nrm.reference = self.ctrd
#                     if self.hold_centering == False:
#                         # this fn should be more descriptive
#                         nrm.auto_find_center(os.path.join(self.savedir,"ctrmodel.fits"))
#                         nrm.bestcenter = 0.5-nrm.over*nrm.xpos, 0.5-nrm.over*nrm.ypos
#                     else:
#                         nrm.bestcenter = self.hold_centering
# 
#                     # similar if/else routines for auto scaling and rotation
# 
#                     #print "from nrm_core, centered shape:",self.ctrd.shape[0], self.ctrd.shape[1]
#                     nrm.make_model(fov = self.ctrd.shape[0], bandpass=nrm.bandpass, over=self.oversample,
#                                    centering=nrm.bestcenter, pixscale=nrm.pixel, flip=self.flip)
#                     nrm.fit_image(self.ctrd, modelin=nrm.model)
#                     """
#                     Attributes now stored in nrm object:
# 
#                     -----------------------------------------------------------------------------
#                     soln            --- resulting sin/cos coefficients from least squares fitting
#                     fringephase     --- baseline phases in radians
#                     fringeamp       --- baseline amplitudes (flux normalized)
#                     redundant_cps   --- closure phases in radians
#                     redundant_cas   --- closure amplitudes
#                     residual        --- fit residuals [data - model solution]
#                     cond            --- matrix condition for inversion
#                     -----------------------------------------------------------------------------
#                     """
# 
#                     if self.debug==True:
#                         dataft = mft.matrix_dft(self.ctrd, 256, 512)
#                         refft = mft.matrix_dft(nrm.refpsf, 256, 512)
#                         plt.figure()
#                         plt.title("Data")
#                         plt.imshow(np.sqrt(abs(dataft)), cmap = "bone")
#                         plt.figure()
#                         plt.title("Reference")
#                         plt.imshow(np.sqrt(abs(refft)), cmap="bone")
#                         plt.show()
#                 
#                     self.save_output(slc, nrm)



    def save_output(self, slc, nrm, verbose=False):
        # cropped & centered PSF
        fits.PrimaryHDU(data=self.ctrd, header=self.scihdr).writeto(os.path.join(self.savedir+\
                self.sub_dir_str,"centered_%02d.fits"%slc), clobber=True)

        model, modelhdu = nrm.plot_model(fits_true=1)

        # default save to text files
        np.savetxt(self.savedir+self.sub_dir_str+"/solutions_{0:02d}.txt".format(slc), nrm.soln)
        np.savetxt(self.savedir+self.sub_dir_str+"/phases_{0:02d}.txt".format(slc), nrm.fringephase)
        np.savetxt(self.savedir+self.sub_dir_str+"/amplitudes_{0:02d}.txt".format(slc), nrm.fringeamp)
        np.savetxt(self.savedir+self.sub_dir_str+"/CPs_{0:02d}.txt".format(slc), nrm.redundant_cps)
        np.savetxt(self.savedir+self.sub_dir_str+"/CAs_{0:02d}.txt".format(slc), nrm.redundant_cas)

        # optional save outputs
        if self.verbose_save:
            np.savetxt(self.savedir+self.sub_dir_str+"/condition_{0:02d}.txt".format(slc), nrm.cond)
            np.savetxt(self.savedir+self.sub_dir_str+"/flux_{0:02d}.txt".format(slc), nrm.flux)

        # save to fits files
        fits.PrimaryHDU(data=nrm.residual).writeto(self.savedir+\
                    self.sub_dir_str+"/residual_{0:02d}.fits".format(slc), clobber=True)
        modelhdu.writeto(self.savedir+\
                    self.sub_dir_str+"/modelsolution_{0:02d}.fits".format(slc), clobber=True)
        
        if 1:            
            # JSA save linearfit results
            myPickleFile = os.path.join(self.savedir+self.sub_dir_str,"linearfit_result_{0:02d}.pkl".format(slc))
            pickle.dump( (nrm.linfit_result), open( myPickleFile , "wb" ) ) 
            if verbose:
                print("Wrote pickled file  %s" % myPickleFile)
            
                    

    def save_auto_figs(self, slc, nrm):
        # pixel scales
        if self.auto_scale==True:
            plt.figure()
            plt.plot(rad2mas(nrm.pixscales), nrm.pixscl_corr)
            plt.vlines(rad2mas(nrm.pixscale_optimal), nrm.pixscl_corr[0],
                        nrm.pixscl_corr[-1], linestyles='--', color='r')
            plt.text(rad2mas(nrm.pixscales[1]), nrm.pixscl_corr[1], 
                     "best fit at {0}".format(rad2mas(nrm.pixscale_optimal)))
            plt.savefig(self.savedir+self.sub_dir_str+"/pixscalecorrelation_{0:02d}.png".format(slc))
        
        # rotation
        if self.auto_rotate==True:
            plt.figure()
            plt.plot(nrm.rots, nrm.corrs)
            plt.vlines(nrm.rot_measured, nrm.corrs[0],
                        nrm.corrs[-1], linestyles='--', color='r')
            plt.text(nrm.rots[1], nrm.corrs[1], 
                     "best fit at {0}".format(nrm.rot_measured))
            plt.savefig(self.savedir+self.sub_dir_str+"/rotationcorrelation_{0:02d}.png".format(slc))



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

    def __init__(self, paths, instrument_data, savedir="calibrated", sub_dir_tag=None, **kwargs):
        """
        Initilize the class

        e.g., to run this in a driver 
            gpidata = InstrumentData.GPI(reffile)
            calib = Calibrate(paths, gpidata)
            calib.write_to_oifits("dataset.oifits")
        
        instrument_data - stores the mask geometry (namely # holes),
                    instrument info, and wavelength obs mode info
                    an instance of the appropriate data class

        paths       - paths containing target and calibrator(s) fringe 
                    observables. This is done per target. The first path is 
                    assumed to be the target, the remaining paths belong to any 
                    and all calibrators

        savedir     - default is folder called "calibrated" in the working 
                    directory

        sub_dir_tag - Does this dataset have an additional axis?
                    (e.g. wavelength or polz)
                    This is a file string in each object folder to access
                    the additional layer of data
        
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
    
        try:
            os.listdir(savedir)
        except:
            os.mkdir(savedir)
        self.savedir = savedir

        # number of calibrators being used:
        self.ncals = len(paths) - 1 # number of calibrators, if zero, set to 1
        if self.ncals==0:# No calibrators given
            self.ncals = 1 # to avoid empty arrays
        self.nobjs = len(paths) # number of total objects

        self.N = len(instrument_data.mask.ctrs)
        self.nbl = int(self.N*(self.N-1)/2)
        self.ncp = int(comb(self.N, 3))
        self.instrument_data = instrument_data

        # Additional axis (e.g., wavelength axis)
        # Can be size one. Defined by instrument_data wavelength array
        self.naxis2 = instrument_data.nwav

        # some warnings

        # safer to just let this happen, commenting out
        #if self.naxis2 == 1:
        #   if sub_dir_tag is not None:
        #       if self.interactive==True:
        #           print "!! naxis2 is set to zero but sub_dir_tag is defined !!",
        #           print "Are you sure you want to do this?",
        #           print "Will look for files only in ",
        #           print paths
        #           print "proceed anyway? (y/n)"
        #           ans = raw_input()
        #           if ans =='y':
        #               pass
        #           elif ans == 'n':
        #               sys.exit("stopping, naxis2 must be > 1 to use sub_dir_tag, see help")
        #           else:
        #               sys.exit("invalid response, stopping")
        #       else:
        #           pass
                    
        #else:
        if sub_dir_tag == None: 
            if self.interactive==True:
                #print "!! naxis2 is set to a non-zero number but extra_layer"
                print "extra_layer is not defined !! naxis2 will be ignored."
                print "results will not be stored in a subdirectory"
                print "proceed anyway? (y/n)"
                ans = raw_input()
                if ans =='y':
                    pass
                elif ans == 'n':
                    sys.exit("stopping, try providing 'sub_dir_tag' keyword arg")
                else:
                    sys.exit("invalid response, stopping. Try providing 'sub_dir_tag' keyword arg")
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
        if sub_dir_tag is not None:
            self.sub_dir_tag = sub_dir_tag
            for ii in range(self.nobjs):
                exps = [f for f in os.listdir(paths[ii]) if self.sub_dir_tag in f]
                nexps = len(exps)
                amp = np.zeros((self.naxis2, nexps, self.nbl))
                pha = np.zeros((self.naxis2, nexps, self.nbl))
                cps = np.zeros((self.naxis2, nexps, self.ncp))

                # Create the cov matrix arrays
                if ii == 0:
                    self.cov_mat_tar = np.zeros((self.naxis2, nexps, nexps))
                    self.sigmasquared_tar = np.zeros((self.naxis2, nexps))
                elif ii==1:
                    self.cov_mat_cal = np.zeros((self.naxis2, nexps, nexps))
                    self.sigmasquared_cal = np.zeros((self.naxis2, nexps))
                else:
                    pass

                for qq in range(nexps):
                    # nwav files
                    cpfiles = [f for f in os.listdir(paths[ii]+exps[qq]) if "CPs" in f] 
                    ampfiles = [f for f in os.listdir(paths[ii]+exps[qq]) \
                                if "amplitudes" in f]
                    phafiles = [f for f in os.listdir(paths[ii]+exps[qq]) if "phase" in f] 
                    expflag=[]
                    for slc in range(len(cpfiles)):
                        amp[slc, qq,:] = np.loadtxt(paths[ii]+exps[qq]+"/"+ampfiles[slc])
                        cps[slc, qq,:] = np.loadtxt(paths[ii]+exps[qq]+"/"+cpfiles[slc])
                        pha[slc, qq,:] = np.loadtxt(paths[ii]+exps[qq]+"/"+phafiles[slc])
                    # 10/14/2016 -- flag the exposure if we get amplitudes > 1
                    # Also flag the exposure if vflag is set, to reject fraction indicated
                    if True in (amp[:,qq,:]>1):
                        expflag.append(qq)
                if self.vflag>0.0:
                    self.ncut = int(self.vflag*nexps) # how many are we cutting out
                    sorted_exps = np.argsort(amp.mean(axis=(0,-1)))
                    cut_exps = sorted_exps[:self.ncut] # flag the ncut lowest exposures
                    expflag = expflag + list(cut_exps)
                ############################
                # Oct 14 2016 -- adding in a visibilities flag. Can't be >1 that doesn't make sense.
                # So we will knock out any exposure where this is true
                #blflag = np.zeros(amp.shape, dtype=bool)
                #cpflag = np.zeros(cps.shape, dtype=bool)
                #blflag[expflag, :,:] = True 
                #cpflag[expflag, :,:] = True 
                # how many exposures are we removing?
                #minusexps = len(expflag)
                # Also adding a mask to calib steps
                ############################
                for slc in range(self.naxis2):
                    if ii==0:
                        # closure phases and squared visibilities
                        self.cp_mean_tar[slc,:], self.cp_err_tar[slc,:], \
                            self.v2_mean_tar[slc,:], self.v2_err_tar[slc,:], \
                            self.pha_mean_tar[slc,:], self.pha_err_tar = \
                            self.calib_steps(cps[slc,:,:], amp[slc,:,:], pha[slc,:,:], nexps, expflag=expflag)
                        # measured cp shape: (nwav, nexps, ncp) mean cp shape: (nwav, ncp)
                        meansub = cps[slc, :, :] - \
                                  np.tile(self.cp_mean_tar[slc,:], (nexps, 1))
                        self.cov_mat_tar[slc, :,:] = np.dot(meansub, meansub.transpose()) / (nexps - 1)
                        # Uncertainties:
                        self.sigmasquared_tar[slc,:] = np.diagonal(self.cov_mat_tar[slc, :,:])

                    else:
                        # Fixed clunkiness!
                        # closure phases and visibilities
                        self.cp_mean_cal[ii-1,slc, :], self.cp_err_cal[ii-1,slc, :], \
                            self.v2_mean_cal[ii-1,slc,:], self.v2_err_cal[ii-1,slc,:], \
                            self.pha_mean_cal[ii-1,slc,:], self.pha_err_cal[ii-1, slc,:] = \
                            self.calib_steps(cps[slc,:,:], amp[slc,:,:], pha[slc,:,:], nexps, expflag=expflag)
                        print cps[slc, :,:].shape
                        print self.cp_mean_cal[ii-1, slc,:].shape
                        print np.tile(self.cp_mean_cal[ii-1,slc,:],(nexps, 1)).shape
                        meansub = cps[slc, :, :] - \
                                  np.tile(self.cp_mean_cal[ii-1, slc,:], (nexps, 1))
                        self.cov_mat_cal[slc, :,:]  += np.dot(meansub, meansub.transpose()) / (nexps - 1)
                        # Uncertainties:
                        self.sigmasquared_cal[slc,:] = np.diagonal(self.cov_mat_cal[slc, :,:])

            """
            ####################################
            # calculate closure phase cov matrix
            ####################################
            # zero mean and stack wavelength+exposures
            if ii ==0:
                flatcps = (cps-self.cp_mean_tar[:,None,:]).reshape(nexps*self.naxis2, self.ncp)
                self.cov_mat_tar = np.cov(flatcps)
            else:
                flatcps = (cps-self.cp_mean_cal[ii-1, :,None,:]).reshape(nexps*self.naxis2, self.ncp)
                self.cov_mat_cal += np.cov(flatcps)
            UPDATE: Oct 18 2016 -- trying to implement description from Kraus et al. 2008
            C_r = sum_i (phi_frame - phi_mean)^T (phi_frame - phi_mean) / (n - 1)
            Q: how do we get a "calibrated" covariance matrix?
            add to phase uncertainties:
            sig^2 = (2 sig_r^2 + (n_c - 1)sig_c*2 ) / (n_c + 1)
            """
            nexp_c = self.sigmasquared_cal.shape[1]
            self.sigmasquared = (2* self.sigmasquared_tar + \
                                 (nexp_c - 1)*self.sigmasquared_cal) / (nexp_c + 1)

        else:
            for ii in range(self.nobjs):
                cpfiles = [f for f in os.listdir(paths[ii]) if "CPs" in f] 
                ampfiles = [f for f in os.listdir(paths[ii]) if "amplitudes" in f]
                phafiles = [f for f in os.listdir(paths[ii]) if "phase" in f]
                nexps = len(cpfiles)
                amp = np.zeros((nexps, self.nbl))
                pha = np.zeros((nexps, self.nbl))
                cps = np.zeros((nexps, self.ncp))
                for qq in range(nexps):
                    amp[qq,:] = np.loadtxt(paths[ii]+"/"+ampfiles[qq])
                    if True in (amp[qq,:]>1):
                        expflag.append(qq)
                    pha[qq,:] = np.loadtxt(paths[ii]+"/"+phafiles[qq])
                    cps[qq,:] = np.loadtxt(paths[ii]+"/"+cpfiles[qq])
                ############################
                # Oct 14 2016 -- adding in a visibilities flag. Can't be >1 that doesn't make sense.
                #v2flag = np.zeros(amp.shape, dtype=bool)
                #v2flag[amp>1] = True 
                # Also adding a mask to calib steps
                if ii==0:
                    # closure phases and squared visibilities
                    self.cp_mean_tar[0,:], self.cp_err_tar[0,:], \
                        self.v2_err_tar[0,:], self.v2_err_tar[0,:], \
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
        cpmask = np.zeros(cps.shape, dtype=bool)
        blmask = np.zeros(amps.shape, dtype=bool)
        if expflag is not None:
            nexp -= len(expflag) # don't count the bad exposures
            cpmask[expflag, :] = True
            blmask[expflag, :] = True
        else:
            pass

        #meancp = np.mean(cps, axis=0)
        meancp = np.ma.masked_array(cps, mask=cpmask).mean(axis=0)
        #covmat_cps = np.cov(np.rollaxis(cps - meancp, -1,0))

        #meanv2 = np.mean(amps, axis=0)**2
        meanv2 = np.ma.masked_array(amps, mask=blmask).mean(axis=0)**2
        #covmat_v2 = np.cov(np.rollaxis(amps**2 - meanv2, -1,0))

        #meanpha = np.mean(pha, axis=0)
        meanpha = np.ma.masked_array(pha, mask=blmask).mean(axis=0)**2
        #covmat_pha = np.cov(np.rollaxis(pha - meanpha, -1,0))

        errcp = np.sqrt(mstats.moment(np.ma.masked_array(cps, mask=cpmask), moment=2, axis=0))/np.sqrt(nexp)
        errv2 = np.sqrt(mstats.moment(np.ma.masked_array(amps**2, mask=blmask), moment=2, axis=0))/np.sqrt(nexp)
        errpha = np.sqrt(mstats.moment(np.ma.masked_array(pha, mask=blmask), moment=2, axis=0))/np.sqrt(nexp)
        # Set cutoff accd to Kraus 2008 - 2/3 of median
        errcp[errcp < (2/3.0)*np.median(errcp)] =(2/3.0)*np.median(errcp) 
        errpha[errpha < (2/3.0)*np.median(errpha)] =(2/3.0)*np.median(errpha) 
        errv2[errv2 < (2/3.0)*np.median(errv2)] =(2/3.0)*np.median(errv2) 
        print "input:",cps
        print "avg:", meancp
        print "exposures flagged:", expflag
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
        print kwargs

        
        from misctools.write_oifits import OIfits
        #except:
        #   print "Need oifits.py nd write_oifits.py to use this method"
        #   return None

        # look for kwargs, e.g., phaseceil, anything else?
        if "phaseceil" in kwargs.keys():
            self.phaseceil = kwargs["phaseceil"]
        else:
            # default for flagging closure phases (deg)
            self.phaseceil = 1.0e2
        if "clip" in kwargs.keys():
            self.clip_wls = kwargs["clip"]
        else:
            # default is no clipping - maybe could set instrument-dependent clip in future
            self.clip_wls = None

        self.obskeywords = {
                'path':self.savedir+"/",
                'year':self.instrument_data.year, 
                'month':self.instrument_data.month,
                'day':self.instrument_data.day,
                'TEL':self.instrument_data.telname,\
                'arrname':self.instrument_data.arrname, 
                'object':self.instrument_data.objname,
                'RA':self.instrument_data.ra, 
                'DEC':self.instrument_data.dec, \
                'PARANG':self.instrument_data.parang, 
                'PA':self.instrument_data.pa, 
                'phaseceil':self.phaseceil}

        oif = OIfits(self.instrument_data.mask,self.obskeywords)
        oif.dummytables()
        # Option to clip out band edges for multiple wavelengths
        # clip can be scalar or 2-element. scalar will do symmetric clipping
        wavs = oif.wavextension(self.instrument_data.wavextension[0], \
                    self.instrument_data.wavextension[1], clip=self.clip_wls)
        oif.oi_data(read_from_txt=False, v2=self.v2_calibrated, v2err=self.v2_err_calibrated, \
                    cps=self.cp_calibrated_deg, cperr=self.cp_err_calibrated_deg, \
                    pha = self.pha_calibrated_deg, phaerr = self.pha_err_calibrated_deg) 
        oif.write(fn_out)

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
    def __init__(self, oifitsfn, savedir = "calibrated", extra_error=0):
        """
        What do I want to do here?
        Want to load an oifits file and look for a binary -- anything else?
        """
        self.oifitsfn = oifitsfn
        self.extra_error = extra_error

        get_data(self)
        self.savedir = savedir

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
        print "abs max", wheremax
        print "loglike at max axis=0",wheremax[0][0], loglike[wheremax[0][0],:,:].shape
        print "==================="
        print "Max log likelikehood for contrast:", 
        print cons[wheremax[0]]
        print "Max log likelikehood for separation:", 
        print seps[wheremax[1]], "mas"
        print "Max log likelikehood for angle:", 
        print angs[wheremax[2]], "deg"
        print "==================="
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
        
        plt.show()
        return coarse_params

    def detec_map(self, lims, nstep=50, hyp = 0):
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
                    chi2cube[i,j,k] = cp_binary_model(params, constant, priors, None, self.uvcoords, self.cp, self.cperr, stat="chi2")
                    #chi2_bin_model[i, j,k] = cp_binary_model([hyp, sep, ang], constant, priors, None, self.uvcoords, self.cp, self.cperr, stat="chi2")
        chi2_0 = cp_binary_model([0,0,0], constant, priors, None, self.uvcoords, self.cp, self.cperr, stat="chi2")

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
        print detec_array.shape
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
        
        plt.show()

    def chi2map(self, maxsep=300., clims = [0.001, 0.5], nstep=50, threads=4, save=True):
        """
        Makes a coarse chi^2 map at the contrast where chi^2 is minimum for each position. 
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
        print "took "+str(t1-t0)+"s to assemble position grids"
        print uvcoords.shape
        print self.cp.shape
        print ras.shape
        print np.shape(self.wavls)
        
        t2 = time.time()
        store_dict = [{"data":self.cp, "error":self.cperr, "uvcoords":uvcoords, \
                      "params":[self.cons[i],np.sqrt(ras**2+decs**2),180*np.arctan2(decs,ras)/np.pi], \
                      "wavls":self.wavls} for i in range(nstep)] 
        if threads>0:
            pool = Pool(processes=threads)
            print "Threads:", threads
            self.chi2grid = np.array(pool.map(chi2_grid_loop, store_dict))
        else:
            self.chi2grid = np.zeros((nstep, nstep, nstep))
            for ii in range(len(self.cons)):
                self.chi2grid[ii] = chi2_grid_loop(store_dict[ii])
        t3 = time.time()
        print "took "+str(t3-t2)+"s to compute all chi^2 grid points"

        plt.figure()
        plt.plot(nstep/2.0 -0.5,nstep/2.0 - 0.5, marker="*", color='w', markersize=20)
        plt.imshow(np.min(self.chi2grid, axis=0).transpose(), cmap="cubehelix")
        plt.xlabel("RA (mas)")
        plt.ylabel("DEC (mas)")
        plt.xticks(np.linspace(0, nstep, 5), np.linspace(self.ras.min(), self.ras.max(), 4+1))
        plt.yticks(np.linspace(0, nstep, 5), np.linspace(self.decs.min(), self.decs.max(), 4+1))

        chi2min = np.where(self.chi2grid == self.chi2grid.min())
        bestparams = np.array([self.cons[chi2min[0]][0], \
                               np.sqrt(self.ras[chi2min[1]]**2 + self.decs[chi2min[2]]**2)[0], \
                               np.arctan2(self.decs[chi2min[2]], self.ras[chi2min[1]])[0]*180/np.pi])
        print "Best Contrast:", self.cons[chi2min[0]]
        print "Best Separation:", np.sqrt(self.ras[chi2min[1]]**2 + self.decs[chi2min[2]]**2)
        print "Best PA:", np.arctan2(self.decs[chi2min[2]], self.ras[chi2min[1]])*180/np.pi

        savdata = {"chi2grid":self.chi2grid, "ra":self.ras, "dec":self.decs, "con": self.cons}
        if save:
            f = open(self.savedir+os.path.sep+"chi2map.pick", "w")
            pickle.dump(savdata, f)

        plt.savefig(self.savedir+"chi2map.pdf")
        plt.show()
        return bestparams

    def two_hyp_test():
        """
        Is my data consistent with Null Hypothesis?
        """

    def detection_limits(self, ntrials = 1, seplims = [20, 200], conlims = [0.0001, 0.99], anglims = [0,360], nsep = 24, ncon=24, nang=24, threads=4, save=False, scale=1.0):
        """
        Inspired by pymask code.
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

        # Some random errors to add in per trial
        # Consider scaling random cperr by wavelength?
        randnums = np.random.randn(int(ntrials), len(self.cp), int(self.nwav))
        # randomize* the measurement errors
        errors = scale*self.cperr[None, ...]*randnums
        #errors = np.rollaxis(np.tile(self.cperr[None, ...]*randnums, (nsep, ncon, nang,1, 1, 1)),-3,0)
        print "errors shape:", errors.shape

        #detec_grid = np.zeros((len(seps), len(cons), len(angs)))
        # set up big uvcoordinate grid, keep wavelength axis at the end
        # should be shape (2, 3, ncp, nsep, ncon, nang, nwav)
        uvcoords = np.rollaxis(np.rollaxis(np.rollaxis(np.tile(self.uvcoords, (nsep, ncon, nang, 1, 1, 1, 1)), -2,0), -2, 0), -2, 0)
        print "Computing model cps over", nsep*ncon*nang, "parameters."
        t1 = time.time()
        #modelcps = model_cp_uv(uvcoords, cons, seps, angs, 1.0/self.wavls)
        modelcps = np.rollaxis(model_cp_uv(uvcoords, cons, seps, angs, 1.0/self.wavls), 0, -1)
        #modelcps = np.rollaxis(pool.map(model_cp_uv(uvcoords, cons, seps, angs, 1.0/self.wavls), 0, -1)
        t2 = time.time()
        print "Finished computing big grid, took", t2-t1, "s"
        print "modelcps shape:", modelcps.shape

        """
        print "modelcps shape:", modelcps.shape
        modelcps = np.tile(modelcps, (ntrials, 1, 1, 1, 1, 1))
        print "modelcps shape:", modelcps.shape
        simcps = modelcps + errors
        print "simcps shape:", simcps.shape
        chi2null = reduced_chi2(simcps, self.cperr, 0)
        print "chi2null shape:", chi2null.shape, "and chi2null sum:", chi2null.sum()
        chi2bin = reduced_chi2(simcps, modelcps, self.cperr)
        print "chi2bin shape:", chi2bin.shape, "and chi2bin sum:", chi2bin.sum()
        self.detec_grid = ((chi2bin - chi2null)<0.0).sum(axis=(0,-1)) / float(ntrials*len(angs))
        print "detec_grid shape:", self.detec_grid.shape
        """

        t3 = time.time()
        print "setting up the dictionary..."
        store_dict = [{"self":self,"ntrials":ntrials, "model":modelcps, "randerrors":errors[i], "dataerrors":self.cperr} for i in range(len(errors))]
        t4 = time.time()
        print "dictionary took", t4-t3, "s to set up"
        big_detec_grid = np.sum(pool.map(detec_calc_loop, store_dict),axis=0) / float(ntrials)
        t5 = time.time()
        print "Time to finish detec_calc_loop:", t5-t3, "s"

        self.detec_grid = big_detec_grid.sum(axis=-1) / float(nang)
        # Get the order right
        self.detec_grid = self.detec_grid.transpose()

        """
        for ii in range(nsep):
            for jj in range(ncon):
                for kk in range(nang)
                    bin_model = model_cp_uv(self.uvcoords, seps[ii], cons[jj], angs[kk], 1.0/self.wavls)
                    randomize = bin_model + (errors*randnums)
                    for trial in range(ntrials):
                        # null and binary hyp here are 1-D length ntrials
                    chi2null[kk,tt] = cp_binary_model([0,0,0], {"wavl":self.wavls}, priors, None, self.uvcoords, self.cp, self.cperr*randnum[:,:,trial], stat="chi2")
                    chi2_grid = cp_binary_model([seps[ii], cons[jj], angs[kk]], {"wavl":self.wavls}, priors, None, self.uvcoords, self.cp, errors, stat="chi2")
                diff = chi2_grid - chi2null
                # How many detects for each separation & contrast
                # Normalize by # of points
                detec_grid[ii,jj] = (diff <0.0).sum() / float(ntrials*len(angs))
        """

        # contour plot
        clevels = [0.5, 0.9, 0.99, 0.999]
        colors = ['k', 'k', 'k', 'k']
        plt.figure()
        SEP, CON = np.meshgrid(self.seps, self.cons)
        contours = plt.contour(SEP, CON, self.detec_grid, clevels, colors=colors, linewidth=2, \
                               extent=[seplims[0], seplims[1], conlims[0], conlims[1]])
        plt.yscale('log')
        plt.clabel(contours)
        plt.contourf(SEP, CON, self.detec_grid, clevels, cmap=plt.cm.bone)
        plt.colorbar()
        plt.xlabel("Separation (mas)")
        plt.ylabel("Contrast Ratio")
        plt.title("Detection Limits")

        # pickle the data
        savdata = {"clevels": clevels, "separations": self.seps, "angles":self.angs, \
                   "contrasts":self.cons, "detections":self.detec_grid}
        if save:
            f = open(self.savedir+os.path.sep+"detection_limits.pick", "w")
            pickle.dump(savdata, f)
            f.close()
            plt.savefig(self.savedir+os.path.sep+"detection_limits.pdf")
        plt.draw()

    def grid_spectrum(self, sep, pa, ncon=100, conlims=[1.0e-3, 0.999], plot=True):
        """ If the position is known (sep, pa), look for best contrast at each wavelength."""
        nn = np.arange(ncon)
        r = (conlims[-1]/conlims[0])**(1 / float(ncon-1))
        self.cons = conlims[0] * r**(nn)
        cons = np.tile(self.cons, (self.nwav, 1))
        cons = np.rollaxis(cons, -1, 0)
        print "cons shape", cons.shape
        uvcoords = np.rollaxis(np.rollaxis(np.rollaxis(np.tile(self.uvcoords, (ncon, 1, 1, 1, 1)), -2,0), -2, 0), -2, 0)
        print "uvcoords shape", uvcoords.shape
        print "Computing model cps over", ncon, "parameters."
        t1 = time.time()
        model_cps = model_cp_uv(uvcoords, cons, sep, pa, 1.0/self.wavls)
        t2 = time.time()
        print "Finished computing big grid, took", t2-t1, "s"
        print "model shape:", model_cps.shape
        model_cps = np.rollaxis(model_cps, 0, -1)
        print "new model shape", model_cps.shape
        datacps = np.tile(self.cp, (ncon, 1, 1))
        dataerror = np.tile(self.cperr, (ncon, 1, 1))
        print "datacps shape", datacps.shape

        t4 = time.time()
        self.con_spectrum = np.zeros((self.nwav, len(self.cons)))
        for ll in range(self.nwav):
            chi2 = np.sum((model_cps[:,:,ll] - datacps[:,:,ll])**2 / (dataerror[:,:,ll]**2), axis=-1)
            minimum = chi2.min()
            #print "chi2:", chi2.shape
            self.con_spectrum[ll, :] = chi2#loglike
        t5 = time.time()
        print "Time to finish contrast loop:", t5-t4, "s"
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
        A nice visualization to see how the data compares to model solutions. Plot is adjustable.

        separation in mas, pa in degrees
        """
        from matplotlib.widgets import Slider, Button, RadioButtons
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        plt.title("Model vs. Data")
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)

        priors = np.array([(-np.inf, np.inf) for f in range( len(start) ) ])
        #constant = {'wavl':self.wavls}

        # Data and model both have shape ncp, nwav
        modelcps = model_cp_uv(self.uvcoords, start[0], start[1], start[2], 1.0/self.wavls)
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
            print "theta:", theta
            cratio = scrat.val
            print "cratio:", cratio
            sep = ssep.val
            print "separation:", sep
            newparams = [cratio, sep, theta]
            modelcps = model_cp_uv(self.uvcoords, cratio, sep, theta, 1.0/self.wavls)
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

    def run_emcee(self, params, constant={}, nwalkers = 250, niter = 1000, spectrum_model=None, priors=None, threads=4, scale=1.0, show=True):
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
        print "priors:"
        print self.priors

        #guess = np.zeros(self.ndim)
        guess = self.make_guess()
        self.ndim = len(guess)

        p0 = [guess + 0.1*guess*np.random.rand(self.ndim) for i in range(nwalkers)]
        print guess
        print "p0", len(p0)

        t0 = time.time()
        #print "nwalkers", nwalkers, "args", self.constant, self.priors, self.spectrum_model, self.uvcoords, self.cp, self.cperr
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, cp_binary_model, threads=threads, args=[self.constant, self.priors, self.spectrum_model, self.uvcoords, self.cp, scale*self.cperr])

        pos, prob, state = self.sampler.run_mcmc(p0, 100)
        self.sampler.reset()
        t2 = time.time()
        print "burn in complete, took ", t2-t0, "s"
        pos, prob, state = self.sampler.run_mcmc(pos, niter)
        t3 = time.time()
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(self.sampler.acceptance_fraction)))
        print "This number should be between ~ 0.25 and 0.5 if everything went as planned."

        print "ran mcmc, took", t3 - t2, "s"
        self.chain = self.sampler.flatchain
        self.fullchain = self.sampler.chain

        self.mcmc_results = {}
        print "========================="
        print "emcee found...."
        #for ii, key in enumerate(self.params.keys()):
        for ii, key in enumerate(self.keys):
            self.mcmc_results[key] = self.chain[:,ii]
            mean = np.mean(self.mcmc_results[key])
            err = np.std(self.mcmc_results[key])
            if key=="sep":
                print key, ":", mean, "+/-", err, "mas"
            elif key=="pa":
                print key, ":", mean, "+/-", err, "deg"
            else:
                print key, ":", mean, "+/-", err
        print "========================="
        # To do: report separation in mas? pa in deg?
        # pickle self.mcmc_results here:
        pickle.dump(self.mcmc_results, \
                    open(self.savedir+"/mcmc_results_{0}.pick".format(spectrum_model), "wb"))

        return self.mcmc_results

    def make_guess(self):
        # A few options here, can provide:
        # 1. contrast, separation, angle -- 3 parameters to fit
        # 2. contrast_min, slope, separation, angle -- 4 parameters
        # 3. contrast_min, slope -- 2 parameters (position is given as constant)
        # 4. nwav different contrasts - nwav parameters (position is given as constant)
        if self.spectrum_model==None:
            guess = np.zeros(len(self.params))
            guess[0] = self.params['con']
            guess[1] = self.params['sep']
            guess[2] = self.params['pa']
            self.keys = ['con', 'sep', 'pa']
        elif self.spectrum_model=="slope":
            guess = np.zeros(len(self.params))
            guess[0] = self.params['con']
            guess[1] = self.params["slope"]
            guess[2] = self.params["sep"]
            guess[3] = self.params["pa"]
            self.keys = ['con', 'slope','sep', 'pa']
        elif self.spectrum_model == "free":
            guess = self.params["con"] # here con is given as an array size nwav
            self.keys = ['wl_{0:02d}'.format(f) for f in range(len(guess))]
        else:
            print "invalid spectrum_model set"
        return guess
    def corner_plot(self, fn):
        import corner
        plt.figure(1)
        fig = corner.corner(self.chain, labels = self.keys, bins = 200, show_titles=True)
        plt.savefig(self.savedir+fn)
        plt.show()
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
        pickle.dump(self.chain_convergence, open(self.savedir+"/chain_convergence.pick", "wb"))
        plt.show()
        return self.chain_convergence

def diffphase_binary_model(self):
    # Figure out how to do diff phase here in Calibrate first?
    # Look for, e.g., emission features.
    return None

def plot_diffphase_uv(self):
    return None

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
        contrast = params[0] + params[1]*wav_step
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

def get_data(self):
    # Move this function out, pass values to the object
    try:
        self.oifdata = oifits.open(self.oifitsfn)
    except:
        print "Unable to read oifits file"
    self.telescope = self.oifdata.wavelength.keys()[0]
    self.ncp = len(self.oifdata.t3)
    self.nbl = len(self.oifdata.vis2)
    self.wavls = self.oifdata.wavelength[self.telescope].eff_wave
    self.eff_band = self.oifdata.wavelength[self.telescope].eff_band
    self.nwav = len(self.wavls)
    #self.ucoord = np.zeros((3, self.ncp))
    self.uvcoords = np.zeros((2, 3, self.ncp))#, self.nwav))

    # Now collect fringe observables and coordinates
    self.cp = np.zeros((self.ncp, self.nwav))
    self.cperr = np.zeros((self.ncp, self.nwav))
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
        self.uvcoords[0,:,ii] = self.oifdata.t3[ii].u1coord, self.oifdata.t3[ii].u2coord,\
                    -(self.oifdata.t3[ii].u1coord+self.oifdata.t3[ii].u2coord)
        self.uvcoords[1, :,ii] = self.oifdata.t3[ii].v1coord, self.oifdata.t3[ii].v2coord,\
                    -(self.oifdata.t3[ii].v1coord+self.oifdata.t3[ii].v2coord)
    #print self.cp
    for jj in range(self.nbl):
        #self.v2[:,jj] = self.oifdata.vis2[jj].vis2data
        #self.v2err[:,jj] = self.oifdata.vis2[jj].vis2err
        self.v2[jj, :] = self.oifdata.vis2[jj].vis2data
        self.v2err[jj, :] = self.oifdata.vis2[jj].vis2err
        try:
            #self.pha[:,jj] = self.oifdata.vis[jj].visphi
            #self.phaerr[:,jj] = self.oifdata.vis[jj].visphierr
            self.pha[jj, :] = self.oifdata.vis[jj].visphi
            self.phaerr[jj, :] = self.oifdata.vis[jj].visphierr
        except:
            pass
    # hack right now to take care of 0 values, set to some limit, 0.001 right now
    floor = 0.001
    self.cperr[self.cperr<floor] = self.cperr[self.cperr!=0.0].mean()
    self.phaerr[self.phaerr<floor] = self.phaerr[self.phaerr!=0.0].mean()
    self.v2err[self.v2err<floor] = self.v2err[self.v2err!=0.0].mean()
    
    # replicate the uv coordinates over the wavelength axis
    self.uvcoords = np.tile(self.uvcoords, (self.nwav, 1, 1, 1))
    # Now uvcoords is shape (nwav, 2, 3, ncps)
    # So we move nwav axis to the end:
    self.uvcoords = np.rollaxis(self.uvcoords, 0, 4)
    #for q in range(self.nwav-1):
    #   self.uvcoords[:,:,:,f] = self.uvcoords[:,:,:,0]

    """
    def detec_calc_loop(self, stored):
        # ndetected should have shape (nsep, ncon, nang) -- the first 3 dimensions of the cp model
        ndetected = np.zeros((stored['model'].shape[0], stored['model'].shape[1], stored['model'].shape[2]))
        for tt in range(stored['ntrials']):
            simcps = stored['model'].copy()
            simcps += stored['randerrors'][tt]
            chi2null = reduced_chi2(simcps, stored['dataerrors'], 0)
            chi2bin = reduced_chi2(simcps, stored['dataerrors'], stored['model'])
            ndetected += ((chi2bin - chi2null)<0.0)
        ndetected /= float(stored['ntrials'])
        return ndetected
    """

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

def logl(data, err, model):
    """
    Likelihood given data, errors, and the model values
    These are all shape (nobservable, nwav)
    """
    #for ii in range(len(model)):
    #   #ll += -0.5*np.log(2*np.pi)*data[2*ii].size + np.sum(-np.log(data[2*ii+1]**2)
    #return -0.5*np.log(2*np.pi) - np.sum(np.log(err)) - np.sum((model - data)**2/(2*data**2))
    #return -0.5*np.log(2*np.pi)*data.size + np.sum(-np.log(err**2) - 0.5*((model - data)/err)**2)
    return np.sum(-np.log(err**2) - 0.5*((model - data)/err)**2)

def reduced_chi2(data, err, model, dof=1.0):
    return (1/float(dof))*np.sum(((model - data)**2/(err**2)), axis=(-1,-2))

def assemble_cov_mat(self):
    meancps = np.mean(self.cp, axis=0)
    flat_mean_sub_cp = (self.cp - meancps[None,:]).flatten
    covmat = flat_mean_sub_cp[None,:]*flat_mean_sub_cp[:,None]
    return covmat

def chi2_grid_loop(args):
    # Model from data, err, uvcoords, params, wavls
    p0, p1, p2 = args['params']
    #print "In Loop:"
    #print np.shape(p0)
    #print np.shape(p1)
    #print np.shape(p2)
    #print np.shape(args['wavls'])
    #print np.shape(args['uvcoords'])
    modelcps = np.rollaxis(model_cp_uv(args['uvcoords'], p0, p1, p2, 1/args['wavls']), 0, -1)
    chi2 = np.sum( (modelcps - args['data'])**2 / args['error']**2, axis = (-1,-2))
    return chi2

class DiskAnalyze:
    def __init__(self):
        print "not finished."

    def diffvis_model(self, params, priors):
        """
        polz data - look for differential visibilities and fit something from a radiative transfer code
                    > Hyperion? Does it have polz info
                    > mcfost? - Has polz info.
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



#JSA try to parallelise fit_fringes
def fit_fringes_parallel(args,threads):
    self = args['object']
    filename = args['file']
    id_tag = args['id']
    self.scidata, self.scihdr = self.instrument_data.read_data(filename)

    #ctrref = utils.centerit(scidata[)

    self.sub_dir_str = self.instrument_data.sub_dir_str
    try:
        os.mkdir(self.savedir+self.sub_dir_str)
    except:
        pass

    store_dict = [{"object":self, "slc":slc} for slc in range(self.instrument_data.nwav)] 

    if threads>0:
#       from multiprocessing.dummy import Pool as ThreadPool 
#       pool = ThreadPool(processes=threads) 
        pool = Pool(processes=threads)
        print "Running fit_fringes in parallel with %d threads" % threads
        pool.map(fit_fringes_single_integration , store_dict)


#   print self.instrument_data.nwav
#   This is a loop over integrations
    else:
        for slc in range(self.instrument_data.nwav):
            fit_fringes_single_integration({"object":self, "slc":slc})

def fit_fringes_single_integration(args):
    self = args['object']
    slc  = args['slc']
    id_tag = args['slc']
    
    # create the reference PSF directory if doing any auto_scaling or rotation
#         try:
#             os.mkdir(self.refimgs+'{0:02d}'.format(slc)+'/')
#         except:
#             pass

    # NRM_Model
    nrm = NRM_Model(mask=self.instrument_data.mask, pixscale = self.instrument_data.pscale_rad,\
                    holeshape=self.instrument_data.holeshape, over = self.oversample, flip=self.flip)

    nrm.refdir=self.refimgs+'{0:02d}'.format(slc)+'/'
    nrm.bandpass = self.instrument_data.wls[slc]
#         print self.instrument_data.wls[slc]

    self.ctrd = utils.centerit(self.scidata[slc, :,:], r = self.npix//2)
    refslice = self.ctrd.copy()
    if True in np.isnan(refslice):
        refslice=utils.deNaN(5, self.ctrd)
        if True in np.isnan(refslice):
            refslice = utils.deNaN(20,refslice)


    nrm.reference = self.ctrd
    if self.hold_centering == False:
        # this fn should be more descriptive
        nrm.auto_find_center(os.path.join(self.savedir+self.sub_dir_str,"ctrmodel_%02d.fits"%id_tag))
        nrm.bestcenter = 0.5-nrm.over*nrm.xpos, 0.5-nrm.over*nrm.ypos
    else:
        nrm.bestcenter = self.hold_centering

    # similar if/else routines for auto scaling and rotation

    #print "from nrm_core, centered shape:",self.ctrd.shape[0], self.ctrd.shape[1]
    nrm.make_model(fov = self.ctrd.shape[0], bandpass=nrm.bandpass, over=self.oversample,
                   centering=nrm.bestcenter, pixscale=nrm.pixel, flip=self.flip)
    nrm.fit_image(self.ctrd, modelin=nrm.model)
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
        dataft = mft.matrix_dft(self.ctrd, 256, 512)
        refft = mft.matrix_dft(nrm.refpsf, 256, 512)
        plt.figure()
        plt.title("Data")
        plt.imshow(np.sqrt(abs(dataft)), cmap = "bone")
        plt.figure()
        plt.title("Reference")
        plt.imshow(np.sqrt(abs(refft)), cmap="bone")
        plt.show()

    self.save_output(slc, nrm)







