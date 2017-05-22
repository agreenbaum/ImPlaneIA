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
import os, sys, time

# Module imports
from misctools.mask_definitions import NRM_mask_definitions # mask geometries, GPI, NIRISS, VISIR supported
from misctools import utils

um = 1.0e-6

class GPI:

    def __init__(self, reffile, **kwargs):
        """
        Initialize GPI class

        ARGUMENTS:

        reffile - one or a list of reference fits files 
                  gpi-pipeline reduced containing useful header info
        
        optionally: 'gpifilterpath'
                - Point to a directory which contains GPI filter files
                  code will read in the relevant file and pick out a 
                  sample of wavelengths and transmissions that span the
                  list, to be used later to generate the model.
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
        self.holeshape="circ"

        # Get info from reference file 
        self.hdr0 = []
        self.hdr1 = []
        self.refdata = []
        if type(reffile)==str:
            reffile = [reffile,]
        for fn in reffile:
            reffits = fits.open(fn)
            self.hdr0.append(reffits[0].header)
            self.hdr1.append(reffits[1].header)
            self.refdata.append(reffits[1].data)
            reffits.close()
        # instrument settings
        self.mode = self.hdr0[0]["DISPERSR"]
        self.obsmode = self.hdr0[0]["OBSMODE"]
        self.band = self.obsmode[-1] # K1 is two letters
        self.ref_imgs_dir = "refimgs_"+self.band+"/"

        # wavelength info: spect mode or pol more
        if "PRISM" in self.mode:
            # GPI's spectral mode
            self.nwav = self.hdr1[0]["NAXIS3"]
            self.wls = np.linspace(self.hdr1[0]["CRVAL3"], \
                  self.hdr1[0]["CRVAL3"]+self.hdr1[0]['CD3_3']*self.nwav, self.nwav)*um
            self.eff_band = um*np.ones(self.nwav)*(self.wls[-1] - self.wls[0])/self.nwav
        elif "WOLLASTON" in self.mode:
            # GPI's pol mode. Will define this for the DIFFERENTIAL VISIBILITIES
            # diff vis: two channels 0/45 and 22/67

            self.nwav = 2

            # Define the bands in case we use a tophat filter
            band_ctrs = {"Y":(1.14-0.95)*um/2., "J":(1.35-1.12)*um/2., \
                         "H":(1.80-1.50)*um/2., "1":(2.19-1.9)*um/2., \
                         "2":(2.4-2.13)*um/2.0}
            band_wdth = {"Y":(1.14-0.95)*um, "J":(1.35-1.12)*um, "H":(1.80-1.50)*um, \
                         "1":(2.19-1.9)*um, "2":(2.4-2.13)*um}
            wghts = np.ones(15)
            wavls = np.linspace(band_ctrs[self.band]-band_wdth[self.band]/2.0, \
                                band_ctrs[self.band]+band_wdth[self.band]/2.0, num=15)

            if 'gpifilterpath' in kwargs:
                if self.band=="Y":
                    filterfile = kwargs["gpifilterpath"]+"GPI-filter-Y.fits"
                if self.band=="J":
                    filterfile = kwargs["gpifilterpath"]+"GPI-filter-J.fits"
                if self.band=="H":
                    filterfile = kwargs["gpifilterpath"]+"GPI-filter-H.fits"
                if self.band=="1":
                    filterfile = kwargs["gpifilterpath"]+"GPI-filter-K1.fits"
                if self.band=="2":
                    filterfile = kwargs["gpifilterpath"]+"GPI-filter-K2.fits"
                # Read in gpi filter file
                fitsfilter = fits.open(filterfile)[1].data
                wavls = []
                wghts = []
                # Sample the filter file so the filter is only 50 elements long
                skip = len(fitsfilter[0][0]) / 50
                for ii in range(len(fitsfilter[0][0])/skip):
                    if fitsfilter[0][1][skip*ii]>0.7:
                        wavls.append(fitsfilter[0][0][skip*ii]*1.0e-6)
                        wghts.append(fitsfilter[0][1][skip*ii])

            lam_c = band_ctrs[self.band]
            lam_w = band_wdth[self.band]

            transmission = np.array([[wghts[f], wavls[f]] for f in range(len(wghts))])
            self.wls = [transmission, transmission]
            self.eff_band = np.array([lam_w, lam_w])
        else:
            sys.exit("Check your reference file header. "+\
                    "Keywork DISPERSR='{0}' not understood".format(self.mode))

        # For OIFits structure
        self.wavextension = (self.wls, self.eff_band)

        # Observation info
        self.telname= "GEMINI"
        self.ra, self.dec = self.hdr0[0]["RA"], self.hdr0[0]["DEC"]
        try:
            self.date = self.hdr0[0]["DATE"]
        except:
            self.date = self.hdr0[0]["DATE-OBS"]
        self.month = self.date[-5:-3]
        self.day = self.date[-2:]
        self.year = self.date[:4]
        #self.parang = self.hdr0["PAR_ANG"]
        # AVPARANG added Aug 2 2016
        self.parangs = []
        self.itime = []
        for ii in range(len(reffile)):
            self.parangs.append(self.hdr1[ii]["AVPARANG"])
            self.itime.append(self.hdr1[ii]["ITIME"])
        self.avparang = np.mean(self.parangs)
        self.parang_range = abs(self.parangs[-1] - self.parangs[0])
        self.totalinttime = np.sum(self.itime)
        try:
            self.pa = self.hdr0[0]["PA"]
        except:
            self.pa = 0.0
        try:
            self.objname = self.hdr0[0]["OBJECT"]
        except:
            self.objname = "Unknown"

        # Look for additional keyword arguments ?
    def read_data(self, fn):

        fitsfile = fits.open(fn)
        sci=fitsfile[1].data
        hdr=fitsfile[1].header
        fitsfile.close()
        #fitshdr = fitsfile[0].header
        self.sub_dir_str = fn[-21:-10]

        return sci, hdr

class VISIR:
    def __init__(self, objname, src = "A0V"):
        """
        Initialize VISIR class

        ARGUMENTS:
        objname - string w/name of object observed
        src - if pysynphot is installed, can provide a guess at the stellar spectrum
        """

        self.objname = objname

        self.arrname = "visir_sam"
        self.pscale_mas = 45
        self.pscale_rad = utils.mas2rad(self.pscale_mas)
        self.mask = NRM_mask_definitions(maskname=self.arrname)
        self.mask.ctrs = np.array(self.mask.ctrs)
        self.holeshape="hex"

        # tophat filter
        # this can be swapped with an actual filter file
        self.lam_c = 11.3*1e-6 # 11.3 microns
        self.lam_w = 0.6/11.3 # 0.6 micron bandpass
        self.filt = tophatfilter(11.3*um, 0.6/11.3, npoints=10)
        try:
            self.wls = [utils.combine_transmission(self.filt, src), ]
        except:
            self.wls = [self.filt, ]
        self.wavextension = (self.lam_c, self.lam_w)
        self.nwav=1

        self.ref_imgs_dir = "refimgs/"

        #############################
        # Observation info - I don't know yet how JWST data headers will be structured
        self.telname= "VLT"
        try:
            self.ra, self.dec = self.hdr0["RA"], self.hdr0["DEC"]
        except:
            self.ra, self.dec = 00, 00
        try:
            self.date = self.hdr0["DATE"]
            self.month = self.date[-5:-3]
            self.day = self.date[-2:]
            self.year = self.date[:4]
        except:
            lt = time.localtime()
            self.date = "{0}{1:02d}{2:02d}".format(lt[0],lt[1],lt[2])
            self.month = lt[1]
            self.day = lt[2]
            self.year = lt[0]
        try:
            self.parang = self.hdr0["PAR_ANG"]
        except:
            self.parang = 00
        try:
            self.pa = self.hdr0["PA"]
        except:
            self.pa = 00
        try:
            self.itime = self.hdr1["ITIME"]
        except:
            self.itime = 00
        #############################

    def read_data(self, fn):
        # for datacube of exposures, need to read as 3D (nexp, npix, npix)
        fitsfile = fits.open(fn)
        scidata=fitsfile[0].data
        hdr=fitsfile[0].header
        #self.sub_dir_str = self.filt+"_"+objname
        self.sub_dir_str = '/' + fn.split('/')[-1].replace('.fits', '')
        self.nexp = scidata.shape[0]
        # rewrite wavextension to be same length as nexp
        self.wavextension = (self.lam_c*np.ones(self.nexp), \
                             self.lam_w*np.ones(self.nexp))
        return scidata, hdr

class NIRISS:
    def __init__(self, filt, objname="obj", src="A0V", out_dir='', **kwargs):
        """
        Initialize NIRISS class

        ARGUMENTS:

        kwargs:
        UTR
        Or just look at the file structure
        Either user has webbpsf and filter file can be read, or this will use a tophat and give a warning
        """

        # define bandpass either by tophat or webbpsf filt file
        #self.wls = np.array([self.bandpass,])
        self.filt = filt
        self.objname = objname
        #############################
        lam_c = {"F277W":2.77e-6, "F380M": 3.8e-6, "F430M": 4.3e-6, "F480M": 4.8e-6}
        lam_w = {"F277W":0.2, "F380M": 0.1, "F430M": 0.05, "F480M": 0.08}
        lam_bin = {"F277W": 50, "F380M": 20, "F430M":20,"F480M":30}
        #############################

        # only one NRM on GPI:
        self.arrname = "jwst_g7s6c"
        self.pscale_mas = 65
        self.pscale_rad = utils.mas2rad(self.pscale_mas)
        self.mask = NRM_mask_definitions(maskname=self.arrname)
        self.mask.ctrs = np.array(self.mask.ctrs)
        # Hard code any rotations? 
        # (can be moved to NRM_mask_definitions later)
        # Add in hole/baseline properties ?
        self.holeshape="hex"

        ## Get info from reference file 
        #reffits = fits.open(reffile)
        #self.hdr0 = reffits[0].header
        #reffits.close()
        ## instrument settings
        #self.mode = self.hdr0["DISPERSR"]
        #self.obsmode = self.hdr0["OBSMODE"]
        #self.band = self.obsmode[-1] # K1 is two letters
        self.ref_imgs_dir = os.path.join(out_dir,"refimgs_"+self.filt+"/")

        # Wavelength info for NIRISS bands F277W, F380M, F430M, or F480M
        try:
            # If user has webbpsf installed, this will work
            #self.throughput = utils.get_webbpsf_filter(self.filt+"_throughput.fits", \
            #    trim = (lam_c[self.filt], lam_w[self.filt]))
            self.throughput = utils.trim_webbpsf_filter(self.filt, specbin=lam_bin[self.filt])
        except:
            self.throughput = utils.tophatfilter(lam_c[self.filt], lam_w[self.filt], npoints=11)

        try:
            self.wls = [utils.combine_transmission(self.throughput, src), ]
        except:
            self.wls = [self.throughput, ]

        self.wavextension = ([lam_c[self.filt],], [lam_w[self.filt],])
        self.nwav=1

        #############################
        # Observation info - I don't know yet how JWST data headers will be structured
        self.telname= "JWST"
        try:
            self.ra, self.dec = self.hdr0["RA"], self.hdr0["DEC"]
        except:
            self.ra, self.dec = 00, 00
        try:
            self.date = self.hdr0["DATE"]
            self.month = self.date[-5:-3]
            self.day = self.date[-2:]
            self.year = self.date[:4]
        except:
            lt = time.localtime()
            self.date = "{0}{1:02d}{2:02d}".format(lt[0],lt[1],lt[2])
            self.month = lt[1]
            self.day = lt[2]
            self.year = lt[0]
        try:
            self.parang = self.hdr0["PAR_ANG"]
        except:
            self.parang = 00
        try:
            self.pa = self.hdr0["PA"]
        except:
            self.pa = 00
        try:
            self.itime = self.hdr1["ITIME"]
        except:
            self.itime = 00
        #############################

    def read_data(self, fn, mode="slice"):
        # mode options are slice or UTR
        # for single slice data, need to read as 3D (1, npix, npix)
        # for utr data, need to read as 3D (ngroup, npix, npix)
        fitsfile = fits.open(fn)
        scidata=fitsfile[0].data
        hdr=fitsfile[0].header
        #self.sub_dir_str = self.filt+""
        self.sub_dir_str = '/' + fn.split('/')[-1].replace('.fits', '')
        if len(scidata.shape)==3:
            self.nwav=scidata.shape[0]
            [self.wls.append(self.wls[0]) for f in range(self.nwav-1)]
            return scidata, hdr
        elif len(scidata.shape)==2:
            return np.array([scidata,]), hdr
        else:
            sys.exit("invalid data dimensions for NIRISS. Should have dimensionality of 2 or 3.")

    def _generate_filter_files():
        """Either from WEBBPSF, or tophat, etc. A set of filter files will also be provided"""
        return None

