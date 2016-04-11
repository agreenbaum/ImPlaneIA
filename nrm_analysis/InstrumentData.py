#! /usr/bin/env python

"""
InstrumentData Class -- defines data format, wavelength info, mask geometry

Instruments/masks supported:
GPI NRM
NIRISS AMI
VISIR SAM
** we like acronyms **


"""

# Standard Imports
import numpy as np
from astropy.io import fits

# Module imports
from misctools.mask_definitions import NRM_mask_definitions # mask geometries, GPI, NIRISS, VISIR supported
from misctools import utils

um = 1.0e-6

class GPI:

	def __init__(self, reffile, **kwargs):
		"""
		Initialize GPI class

		ARGUMENTS:

		reffile - reference fits file gpi-pipeline reduced containing useful header info
		
		"""

		# only one NRM on GPI:
		self.arrname = "gpi_g10s40"
		self.pscale_mas = 14.14 
		self.pscale_rad = utils.mas2rad(self.pscale_mas)
		self.mask = NRM_mask_definitions(maskname=self.arrname)
		self.mask.ctrs = np.array(self.mask.ctrs)
		# Hard code -1.5 deg rotation in data (April 2016)
		# (can be moved to NRM_mask_definitions later)
		self.mask.ctrs = utils.rotatevectors(self.mask.ctrs, -1.5*np.pi/180.)
		# Add in hole/baseline properties ?

		# Get info from reference file 
		reffits = fits.open(reffile)
		self.hdr0 = reffits[0].header
		self.hdr1 = reffits[1].header
		self.refdata = reffits[1].data
		reffits.close()
		# instrument settings
		self.mode = self.hdr0["DISPERSR"]
		self.obsmode = self.hdr0["OBSMODE"]
		self.band = self.obsmode[-1] # K1 is two letters
		self.ref_imgs_dir = "refimgs_"+self.band+"/"

		# wavelength info: spect mode or pol more
		if "PRISM" in self.mode:
			# GPI's spectral mode
			self.nwav = self.hdr1["NAXIS3"]
			self.wls = np.linspace(self.hdr1["CRVAL3"], \
				  self.hdr1["CRVAL3"]+self.hdr1['CD3_3']*self.nwav, self.nwav)*um
			self.eff_band = um*np.ones(self.nwav)*(self.wls[-1] - self.wls[0])/self.nwav
		elif "WOLLASTON" in self.mode:
			# GPI's pol mode. Will define this for the DIFFERENTIAL VISIBILITIES
			# diff vis: two channels 0/45 and 22/67
			self.wav = 2
			band_ctrs = {"Y":(1.14-0.95)*um/2., "J":(1.35-1.12)*um/2., \
						 "H":(1.80-1.50)*um/2., "1":(2.19-1.9)*um/2., \
						 "2":(2.4-2.13)*um/2.0}
			band_wdth = {"Y":(1.14-0.95)*um, "J":(1.35-1.12)*um, "H":(1.80-1.50)*um, \
						 "1":(2.19-1.9)*um, "2":(2.4-2.13)*um}
			lam_c = band_ctrs[self.band]
			lam_w = band_wdth[self.band]
			self.wls = np.array([lam_c, lam_c])
			self.eff_band = np.array([lam_w, lam_w])
		else:
			sys.exit("Check your reference file header. "+\
					"Keywork DISPERSR='{0}' not understood".format(self.mode))
		self.wavextension = (self.wls, self.eff_band)

		# Observation info
		self.telname= "GEMINI"
		self.ra, self.dec = self.hdr0["RA"], self.hdr0["DEC"]
		self.date = self.hdr0["DATE"]
		self.month = self.date[-5:-3]
		self.day = self.date[-2:]
		self.year = self.date[:4]
		self.parang = self.hdr0["PAR_ANG"]
		self.pa = self.hdr0["PA"]
		self.objname = self.hdr0["OBJECT"]
		self.itime = self.hdr1["ITIME"]

		# Look for additional keyword arguments ?
	def read_data(self, fn):

		fitsfile = fits.open(fn)
		sci=fitsfile['SCI'].data
		hdr=fitsfile['SCI'].header
		fitsfile.close()
		#fitshdr = fitsfile[0].header
		self.sub_dir_str = fn[-21:-10]

		return sci, hdr

class VISIR:
	def __init__(self):
		"""

		--------- UNDER CONSTRUCTION --------

		Initialize VISIR class

		ARGUMENTS:
		"""
		self.mask = NRM_mask_definitions(maskname="visir_sam", holeshape="hex")
		self.mask.ctrs = np.array(self.mask.ctrs)
		# for single slice data, need to read as 3D (1, npix, npix)

		self.ref_imgs_dir = "refimgs/"

class NIRISS:
	def __init__(self, filt, **kwargs):
		"""

		--------- UNDER CONSTRUCTION --------

		Initialize NIRISS class

		ARGUMENTS:

		kwargs:
		UTR
		Or just look at the file structure
		Either user has webbpsf and filter file can be read, or this will use a tophat and give a warning
		"""
		self.mask = NRM_mask_definitions(maskname="jwst_g7s6c", holeshape="hex")
		self.mask.ctrs = np.array(self.mask.ctrs)

		# define bandpass either by tophat or webbpsf filt file
		#self.wls = np.array([self.bandpass,])
		self.filt = filt

		self.ref_imgs_dir = "refimgs_"+self.filt+"/"


	def read_data(self, fn):
		# for single slice data, need to read as 3D (1, npix, npix)
		# for utr data, need to read as 3D (ngroup, npix, npix)
		scidata=fitsfile[0].data
		hdr=fitsfile[0].header
		return data, hdr

	def _generate_filter_files()
		"""Either from WEBBPSF, or tophat, etc. A set of filter files will also be provided"""
		return None

