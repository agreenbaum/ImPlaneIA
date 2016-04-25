#! /usr/bin/env python

"""
by A. Greenbaum & A. Sivaramakrishnan 
April 2016 agreenba@pha.jhu.edu

Contains 

FringeFitter - fit fringe phases and amplitudes to data in the image plane

Calibrate - Calibrate target data with calibrator data

"""


# Standard imports
import os, sys
import numpy as np
from astropy.io import fits
from scipy.misc import comb
from scipy.stats import sem, mstats
import cPickle as pickle

# Module imports
from fringefitting.LG_Model import NRM_Model
import misctools.utils
from misctools.utils import mas2rad
import misctools.utils as utils

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
			self.savedir = os.get_cwd()
		if "datadir" in kwargs:
			self.datadir = kwargs["datadir"]
		else:
			self.datadir = os.get_cwd()
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

		nrm = NRM_Model(mask = self.instrument_data.mask, pixscale = self.instrument_data.pscale_rad, over = self.oversample)
		# In future can just pass instrument_data to NRM_Model

		#plot conditions
		if self.debug==True or self.auto_scale==True or self.auto_rotate==True:
			import matplotlib.pyplot as plt
		if self.debug==True:
			import poppy.matrixDFT as mft

	def fit_fringes(self, fns):
		if type(fns) == str:
			fns = [fns, ]

		for fn in fns:
			self.scidata, self.scihdr = self.instrument_data.read_data(self.datadir+"/"+fn)

			#ctrref = utils.centerit(scidata[)

			self.sub_dir_str = self.instrument_data.sub_dir_str
			try:
				os.mkdir(self.savedir+self.sub_dir_str)
			except:
				pass

			
			for slc in range(self.instrument_data.nwav):
				# create the reference PSF directory if doing any auto_scaling or rotation
				try:
					os.mkdir(self.refimgs+'{0:02d}'.format(slc)+'/')
				except:
					pass

				# NRM_Model
				nrm = NRM_Model(mask=self.instrument_data.mask, pixscale = self.instrument_data.pscale_rad,\
								over = self.oversample)

				nrm.refdir=self.refimgs+'{0:02d}'.format(slc)+'/'
				nrm.bandpass = self.instrument_data.wls[slc]
				#hdr['WAVL'] = wls[slc]

				self.ctrd = utils.centerit(self.scidata[slc, :,:], r = self.npix//2)
				refslice = self.ctrd.copy()
				if True in np.isnan(refslice):
					refslice=utils.deNaN(5, self.ctrd)
					if True in np.isnan(refslice):
						refslice = utils.deNaN(20,refslice)


				nrm.reference = self.ctrd
				if self.hold_centering == False:
					nrm.auto_find_center("ctrmodel.fits")
					nrm.bestcenter = 0.5-nrm.over*nrm.xpos, 0.5-nrm.over*nrm.ypos
				else:
					nrm.bestcenter = nrm.centering

				# similar if/else routines for auto scaling and rotation

				nrm.make_model(fov = self.ctrd.shape[0], bandpass=nrm.bandpass, over=self.oversample,
							   centering=nrm.bestcenter, pixscale=nrm.pixel)
				nrm.fit_image(self.ctrd, modelin=nrm.model)
				"""
				Attributes now stored in nrm object:

				-----------------------------------------------------------------------------
				soln 			--- resulting sin/cos coefficients from least squares fitting
				fringephase 	--- baseline phases in radians
				fringeamp		---	baseline amplitudes (flux normalized)
				redundant_cps	--- closure phases in radians
				redundant_cas 	--- closure amplitudes
				residual		--- fit residuals [data - model solution]
				cond			--- matrix condition for inversion
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

	def save_output(self, slc, nrm):
		# cropped & centered PSF
		fits.PrimaryHDU(data=self.ctrd, header=self.scihdr).writeto(self.savedir+\
				self.sub_dir_str+"/centered_"+str(slc)+".fits", clobber=True)

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
					self.sub_dir_str+"/residual{0:02d}.fits".format(slc), clobber=True)
		modelhdu.writeto(self.savedir+\
					self.sub_dir_str+"/modelsolution{0:02d}.fits".format(slc), clobber=True)

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
			calib = CalibrateNRM(paths, gpidata)
			calib.write_to_oifits("dataset.oifits")
		
		instrument_data	- stores the mask geometry (namely # holes),
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
		if self.naxis2 == 1:
			if sub_dir_tag is not None:
				if self.interactive==True:
					print "!! naxis2 is set to zero but sub_dir_tag is defined !!",
					print "Are you sure you want to do this?",
					print "Will look for files only in ",
					print paths
					print "proceed anyway? (y/n)"
					ans = raw_input()
					if ans =='y':
						pass
					elif ans == 'n':
						sys.exit("stopping, naxis2 must be > 1 to use sub_dir_tag, see help")
					else:
						sys.exit("invalid response, stopping")
				else:
					pass
					
		else:
			if sub_dir_tag == None: 
				if self.interactive==True:
					print "!! naxis2 is set to a non-zero number but extra_layer"
					print "is not defined !! naxis2 will be ignored."
					print "proceed anyway? (y/n)"
					ans = raw_input()
					if ans =='y':
						pass
					elif ans == 'n':
						sys.exit("stopping, define sub_dir_tag if naxis2>1")
					else:
						sys.exit("invalid response, stopping")
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

		# is there a subdirectory (e.g. for the exposure -- need to make this default)
		if sub_dir_tag is not None:
			self.sub_dir_tag = sub_dir_tag
			for ii in range(self.nobjs):
				exps = [f for f in os.listdir(paths[ii]) if self.sub_dir_tag in f]
				nexps = len(exps)
				amp = np.zeros((self.naxis2, nexps, self.nbl))
				cps = np.zeros((self.naxis2, nexps, self.ncp))
				for qq in range(nexps):
					# nwav files
					cpfiles = [f for f in os.listdir(paths[ii]+exps[qq]) if "CPs" in f] 
					ampfiles = [f for f in os.listdir(paths[ii]+exps[qq]) \
								if "amplitudes" in f]
					phafiles = [f for f in os.listdir(paths[ii]+exps[qq]) if "phase" in f] 
					amp = np.zeros((self.naxis2, nexps, self.nbl))
					pha = np.zeros((self.naxis2, nexps, self.nbl))
					cp = np.zeros((self.naxis2, nexps, self.ncp))
					for slc in range(len(cpfiles)):
						amp[slc, qq,:] = np.loadtxt(paths[ii]+exps[qq]+"/"+ampfiles[slc])
						cp[slc, qq,:] = np.loadtxt(paths[ii]+exps[qq]+"/"+cpfiles[slc])
						pha[slc, qq,:] = np.loadtxt(paths[ii]+exps[qq]+"/"+phafiles[slc])
				for slc in range(self.naxis2):
					if ii==0:
						# closure phases and squared visibilities
						self.cp_mean_tar[slc,:], self.cp_err_tar[slc,:], \
							self.v2_mean_tar[slc,:], self.v2_err_tar[slc,:], \
							self.pha_mean_tar[slc,:], self.pha_err_tar = \
							self.calib_steps(cp[slc,:,:], amp[slc,:,:], pha[slc,:,:], nexps)
					else:
						# Fixed clunkiness!
						# closure phases and visibilities
						self.cp_mean_cal[ii-1,slc, :], self.cp_err_cal[ii-1,slc, :], \
							self.v2_mean_cal[ii-1,slc,:], self.v2_err_cal[ii-1,slc,:], \
							self.pha_mean_cal[ii-1,slc,:], self.pha_err_cal[ii-1, slc,:] = \
							self.calib_steps(cp[slc,:,:], amp[slc,:,:], pha[slc,:,:], nexps)

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
					pha[qq,:] = np.loadtxt(paths[ii]+"/"+phafiles[qq])
					cps[qq,:] = np.loadtxt(paths[ii]+"/"+cpfiles[qq])
				if ii==0:
					# closure phases and squared visibilities
					self.cp_mean_tar[0,:], self.cp_err_tar[0,:], \
						self.v2_err_tar[0,:], self.v2_err_tar[0,:], \
						self.pha_mean_tar[slc,:], self.pha_err_tar = \
						self.calib_steps(cps, amp, pha, nexps)
				else:
					# Fixed clunkiness!
					# closure phases and visibilities
					self.cp_mean_cal[ii-1,0, :], self.cp_err_cal[ii-1,0, :], \
						self.v2_mean_cal[ii-1,0,:], self.v2_err_cal[ii-1,0,:], \
						self.pha_mean_cal[ii-1,0,:], self.pha_err_cal[ii-1,0,:] = \
						self.calib_steps(cps, amp, pha, nexps)

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

		self.cp_calibrated_deg = self.cp_calibrated * 180/np.pi
		self.cp_err_calibrated_deg = self.cp_err_calibrated * 180/np.pi
		self.pha_calibrated_deg = self.pha_calibrated * 180/np.pi
		self.pha_err_calibrated_deg = self.pha_err_calibrated * 180/np.pi

	def calib_steps(self, cps, amps, pha, nexp):
		"Calculates closure phase and mean squared visibilities & standard error"
		meancp = np.mean(cps, axis=0)
		meanv2 = np.mean(amps, axis=0)**2
		meanpha = np.mean(pha, axis=0)
		errcp = mstats.moment(cps, moment=2, axis=0)/np.sqrt(nexp)
		errv2 = mstats.moment(amps**2, moment=2, axis=0)/np.sqrt(nexp)
		errpha = mstats.moment(pha, moment=2, axis=0)/np.sqrt(nexp)
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
		#	print "Need oifits.py nd write_oifits.py to use this method"
		#	return None

		# look for kwargs, e.g., phaseceil, anything else?
		if "phaseceil" in kwargs.keys():
			self.phaseceil = kwargs["phaseceil"]
		else:
			# default for flagging closure phases
			self.phaseceil = 1.0e1
		if "clip" in kwargs.keys():
			self.clip_wls = clip
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
	def __init__(self, oifitsfn, savedir = "calibrated"):
		"""
		What do I want to do here?
		Want to load an oifits file and look for a binary -- anything else?
		"""
		self.oifitsfn = oifitsfn

		get_data(self)
		self.savedir = savedir

		import matplotlib.pyplot as plt


	def coarse_binary_search(self, lims, nstep=20):
		"""
		For getting first guess on contrast, separation, and angle
		"""
		grid = np.zeros((nstep, nstep, nstep))
		cons = np.linspace(lims[0][0], lims[0][1], num=nstep)
		seps = np.linspace(lims[1][0], lims[1][1], num=nstep)
		angs = np.linspace(lims[2][0], lims[2][1], num=nstep)
		loglike = np.zeros((nstep, nstep, nstep))

		for i in range(nstep):
			for j in range(nstep):
				for k in range(nstep):
					params = {'con':cons[i], 'sep':seps[j], 'pa':angs[k]}
					loglike[i,j,k] = cp_binary_model(params, {"wavl:":self.wavls})

		wheremax = np.where(loglike==loglike.max())
		print "abs max", wheremax
		print "loglike at max axis=0",wheremax[0][0], loglike[wheremax[0][0],:,:].shape
		print "==================="
		print "Max log likelikehood for contrast:", 
		print cons[wheremax[0]]
		print "Max log likelikehood for separation:", 
		print seps[wheremax[1]]
		print "Max log likelikehood for angle:", 
		print angs[wheremax[2]]*180./np.pi
		print "==================="

		plt.figure()
		plt.set_cmap("cubehelix")
		plt.title("separation vs. pa at contrast of {0:.1f}".format(cons[wheremax[0][0]]))
		plt.imshow(loglike[wheremax[0][0], :,:])
		plt.xticks(np.arange(nstep)[::5], np.round(seps[::5],3))
		plt.yticks(np.arange(nstep)[::5], np.round(angs[::5]*180/np.pi,3))
		plt.xlabel("Separation")
		plt.ylabel("PA")
		plt.savefig(self.savedir+"/sep_pa.pdf")

		plt.figure()
		plt.title("contrast vs. separation, at PA of {0:.1f} deg".format(angs[wheremax[2][0]]*180/np.pi))
		plt.xticks(np.arange(nstep)[::5], np.round(cons[::5],3))
		plt.yticks(np.arange(nstep)[::5], np.round(seps[::5],3))
		plt.xlabel("Contrast")
		plt.ylabel("Separation")
		plt.imshow(loglike[:,:,wheremax[2][0]])
		plt.savefig(self.savedir+"/con_sep.pdf")

		plt.figure()
		plt.title("contrast vs. angle, at separation of {0:.1f} mas".format(seps[wheremax[1][0]]))
		plt.xticks(np.arange(nstep)[::5], np.round(cons[::5], 3))
		plt.yticks(np.arange(nstep)[::5], np.round(angs[::5]*180/np.pi, 3))
		plt.xlabel("Contrast")
		plt.ylabel("PA")
		plt.imshow(loglike[:,wheremax[1][0],:])
		plt.savefig(self.savedir+"/con_pa.pdf")

		plt.show()

	def run_emcee(self, params, constant, nwalkers = 250, niter = 1000, spectrum_model=None, priors=None, threads=4):
		"""
		A lot of options in this method, read carefully.

		Arguments params and constant are dictionaries. 

		For example if you wanted to search for 3 parameters, contrast, separation, and PA:
			params = {'con': cr_val, 'sep': sep_rad, 'pa': pa_rad}
			constant = {'wavls': array_of_wavelengths}
		because  we are searching for con, sep, & pa, and we hold the wavelength constant

		Priors are bounds here. 

		"""
		import emcee
		self.ndim = len(params)
		constant = {}
		constant['wavl'] = self.wavls
		# Options are None 'slope' or 'free'
		self.spectrum_model = spectrum_model
		self.params = params

		if priors is not None:
			self.priors = priors
		else:
			self.priors = [(-np.inf, np.inf) for f in range( len(self.params.keys()) ) ]

		guess = np.zeros(self.ndim)
		q=0
		for key in self.params.keys():
			guess[q] = self.params[key]
			q+=1
		p0 = [guess + 0.1*guess*np.random(self.ndim) for i in range(nwalkers)]

		t0 = time.time()
		self.sampler = EnsembleSampler(nwalkers, self.ndim, self.cp_binary_model, threads=threads, args=[constant,])

		t2 = time.time()
		pos, prob, state = self.sampler.run_mcmc(p0, niter)
		t3 = time.time()
		print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
		print "This number should be between approximately 0.25 and 0.5 if everything went as planned."

		print "ran mcmc, took", t3 - t2, "s"
		self.chain = sampler.flatchain

		self.mcmc_results = {}
		print "========================="
		print "emcee found...."
		for ii, key in enumerate(self.params.keys()):
			self.mcmc_results[key] = self.chain[:,ii]
			mean = np.mean(self.mcmc_results[key])
			err = np.std(self.mcmc_results[key])
			print key, ":", mean, "+/-", err
		print "========================="
		# pickle self.mcmc_results here:
		pickle.dump(self.mcmc_results, open(savedir+"/mcmc_results.pick", "wb"))

		import corner
		fig = corner.corner(chain, labels = self.params.keys(), bins = 100)
		plt.savefig(self.savedir+"triangle_plot.pdf")
		plt.show()
		
	def plot_chain_convergence(self):
		samples  = self.chain[:, 50:, :].reshape((-1, self.ndim))
		plt.figure()
		self.chain_convergence = {}
		for ii in range(chain.shape[-1]):
			plt.subplot2grid((self.ndim,1),(ii,0))
			plt.plot(samples[:,ii])
			plt.ylabel(self.params.keys()[ii])
			plt.xlabel("step number")
			self.chain_convergence[self.params.keys()[ii]] = samples[:,ii]
		plt.savefig(self.savedir+"/chain_convergence.pdf")
		# Pickle and save this data?
		pickle.dump(self.chain_convergence, open(savedir+"/chain_convergence.pick", "wb"))
		plt.show()

	def diffphase_binary_model(self):
		# Figure out how to do diff phase here in Calibrate first?
		# Look for, e.g., emission features.
		return None

	def cp_binary_model(self, params, constant):
		# really want to be able to give this guy some general oi_data and have bm() sort it out.
		# Need to figure out how to add in the priors

		##################################################
		# HOW DO I TUNE THIS DEPENDING ON MY OBSERVATIONS? - need a keyword or something, need help.
		# data = self.cp, self.cperr#, self.v2, self.v2err
		##################################################
	
		# priors, i.e. bounds here
		for i in range(len(params)):
			if (params[i] < self.priors[i,1] or params[i] > self.priors[i,0]):	
				return -np.inf
			else:

				if self.spectrum_model == None:

					# Model from params
					model_cp = model_cp_uv(self.uvcoords, params['con'], params['sep'], \
										params['pa'], 1.0/constant['wavl'])
				elif spectrum == 'slope':
					# params needs 'con_start' starting contrast and 'slope,' sep & pa constant?
					wav_step = constant['wavl'][1] - constant['wavl'][0]
					# contrast model is con_start + slope*delta_lambda
					contrast = np.linspace(params['con_start'] + params['slope']*wav_step)
					# Model from params
					model_cp = model_cp_uv(self.uvcoords, contrast, params['sep'], \
										params['pa'], 1.0/constant['wavl'])
				elif spectrum == 'free' :
					# Model from params - params is contrast array nwav long, sep & pa constant
					model_cp = model_cp_uv(self.uvcoords, params['con'], constant['sep'], \
										constant['pa'], 1.0/constant['wavl'])
				else:
					sys.exit("Invalid spectrum model")

				ll = logl(self.cp, self.cperr, model_cp)
				return ll

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
		self.uvcoords = np.zeros((2, 3, ncp))#, self.nwav))

		# Now collect fringe observables and coordinates
		self.cp = np.zeros((self.nwav, self.ncp))
		self.cperr = np.zeros((self.nwav, self.ncp))
		self.v2 = np.zeros((self.nwav, self.nbl))
		self.v2err = np.zeros((self.nwav, self.nbl))
		self.pha = np.zeros((self.nwav, self.nbl))
		self.phaerr = np.zeros((self.nwav, self.nbl))

		for ii in range(self.ncp):
			self.cp[:,ii] = self.oifdata.t3[ii].t3phi
			self.cperr[:,ii] = self.oifdata.t3[ii].t3phierr
			self.uvcoords[0,:,ii] = self.oifdata.t3[ii].u1coord, self.oifdata.t3[ii].u2coord,\
						-(self.oifdata.t3[ii].u1coord+self.oifdata.t3[ii].u2coord)
			self.uvcoords[1, :,ii] = self.oifdata.t3[ii].v1coord, self.oifdata.t3.v2coord,\
						-(self.oifdata.t3.v1coord+self.oifdata.t3.v2coord)
			# replicate the uv coordinates over the wavelength axis
			self.uvcoords = np.tile(self.uvcoords, (self.nwav, 1, 1, 1))
			# Now uvcoords is shape (nwav, 2, 3, ncps)
			self.uvcoords = np.rollaxis(self.uvcoords, 0, 4)
			#for q in range(self.nwav-1):
			#	self.uvcoords[:,:,:,f] = self.uvcoords[:,:,:,0]
		for jj in range(self.instrumentdata.nbl):
			self.v2[:,jj] = self.oifdata.vis2[jj].vis2data
			self.v2err[:,jj] = self.oifdata.vis2[jj].vis2err
			self.pha[:,jj] = self.oifdata.vis2[jj].visphi
			self.phaerr[:,jj] = self.oifdata.vis2[jj].visphierr
		

def logl(data, err, model):
	"""
	data must be 2x as long as model
	if only considering cps then model is (cps,) size 1
	"""
	#for ii in range(len(model)):
	#	#ll += -0.5*np.log(2*np.pi)*data[2*ii].size + np.sum(-np.log(data[2*ii+1]**2)
	return -0.5*np.log(2*np.pi) - np.sum(np.log(err)) - np.sum((model - data)**2/(2*data**2))

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


