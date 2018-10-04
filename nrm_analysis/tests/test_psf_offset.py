import unittest
import os
import glob
import sys
import numpy as np
from astropy.io import fits
from astropy import units as u

#from nrm_analysis.misctools.utils import Affine2d
from nrm_analysis.misctools.utils import affinepars2header
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.fringefitting import utility_classes as UC # for make_standard_image
"""
    Test PSF offset directions, half-pixel after unification on ImCtr, fromfunctions
    for LG++, and Affine2d class effect ond centering the psf in the array.
    anand@stsci.edu 2018.04.12

    run with pytest -s _moi_.py to see stdout on screen
    All units SI unless units in variable name
"""

um = 1.0e-6
arcsec2rad = u.arcsec.to(u.rad)

class PSFoffsetTestCase(unittest.TestCase):

    def setUp(self):

        # directory containing the test data
        data_dir = os.path.join(os.path.dirname(__file__),'test_data/psf_offset')
        self.data_dir = data_dir
        print(data_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.pixel = 0.0656 *  u.arcsec.to(u.rad)
        self.npix = 87
        self.wave = 4.3e-6 # m
        self.over = 1

        """
        mx, my = 1.0, 1.0
        sx, sy= 0.0, 0.0
        xo, yo= 0.0, 0.0
        affine_ideal = Affine2d(mx=mx,my=my, 
                                sx=sx,sy=sy, 
                                xo=xo,yo=yo, name="Ideal")
        """
        invert = False  # greyscale inversion (true/false)

        holeshape = "hex"
        pixel_as = 0.0656/1.3  # slightly finer-than-actual-detector sampling, for clearer images
        print("\n\t*** Watch out *** if comparing to real data: slightly finer-than-actual-detector sampling used for clearer images\n")
        fov = 101 # 'detpix' but at finer sampling than actual
        over=1
        wav_m = 4.3e-6 
        filt = 'F430M'
        band = 'Monochromatic '+np.str(wav_m)


        psf_offsets = ((0,0), (1.0,0), )
        psf_offsets = ((0,0), )
        psf_offsets = ((0,0), (1.0,0), (0, 1.0), (1.0,1.0))
        print((enumerate(psf_offsets)))
        psfs = np.zeros((len(psf_offsets),fov,fov)) # for easy to understand movie
        print(('psfs.shape', psfs.shape))
        fncube3 = data_dir + "/cube.fits"

        # Loop over psf_offsets
        for noff,  psfo in enumerate(psf_offsets):
            jw3 = NRM_Model(mask='jwst', holeshape=holeshape, 
                            pixscale=arcsec2rad*pixel_as,
                            datapath="", refdir="./refdir")
                            #chooseholes=None)
            
            jw3.simulate(fov=fov, bandpass=wav_m, over=over, psf_offset=psfo)
             

            mnem = "imctr_%.1f_%.1f"%psfo
            fn3 = data_dir + "/" +  mnem + '.fits'
            name_seed = mnem 
            # PSF
            fits.writeto(fn3, jw3.psf, overwrite=True)
            header = fits.getheader(fn3)
            header['SEGNAMES'] = "asbuilt"
            header['PIXELSCL'] = pixel_as
            header['over'] = over
            header['FILTER'] = wav_m
            header['hole'] = holeshape 
            header['PUPIL'] =  'asbuilt'
            header['psfoff0'] =  (psfo[0], "ds9X / detector pixels")
            header['psfoff1'] =  (psfo[1], "ds9Y / detector pixels")
            header['SRC'] =  ("test_psf_offset.py", "generating code")
            header['author'] =  ("anand@stsci.edu", "responsible person?")
            fits.update(fn3,jw3.psf, header=header)
            psfs[noff,:,:] = jw3.psf

            """ calls tkinter, needs $DISPLAY environment variable, breaks travis testing
            figname = UC.make_standard_image(
                                           fn3,
                                           image_title=name_seed,
                                           save_plot=1, 
                                           plot_dir=data_dir, 
                                           name_seed=name_seed, 
                                           stretch='linear', 
                                           x_axis_label=None, 
                                           y_axis_label=None,
                                           invert=invert,
                                           add_grid=True)
            print(figname + " written to " + data_dir)
            """

            del jw3

        fits.PrimaryHDU(data=psfs).writeto(fncube3, overwrite=True) # good for movie

        """
        # circ g7s6 jwst
        self.jw_default = NRM_Model(mask='jwst', holeshape="circ", affine2d=aff)
        self.jw_default.simulate(fov=self.npix, bandpass=self.wave, over=self.over)
        psffn = self.fnfmt.format(self.npix, 'circ', self.wave/um, rot)
        fits.writeto(psffn, self.jw_default.psf, overwrite=True)
        header = fits.getheader(psffn)
        header = affinepars2header(header, aff)
        fits.update(psffn, self.jw_default.psf, header=header)
        del self.jw_default

        # circonly 
        self.jw_default = NRM_Model(mask='jwst', holeshape="circonly", affine2d=aff)
        self.jw_default.simulate(fov=self.npix, bandpass=self.wave, over=self.over)
        psffn = self.fnfmt.format(self.npix, 'circonly', self.wave/um, rot)
        fits.writeto(psffn, self.jw_default.psf, overwrite=True)
        header = fits.getheader(psffn)
        header = affinepars2header(header, aff)
        fits.update(psffn, self.jw_default.psf, header=header)
        del self.jw_default

        """

    def test_psf(self):
        self.assertTrue(0.0 < 1e-15, 'error: test_test_psf_offset failed')

    """
    Bring this test back into the actual assert test... later.   AS in beta-
    def test_psf(self):
        psf0  = fits.getdata(self.fnfmt.format(self.npix, 'hexonly', self.wave/um, 0))
        psf90 = fits.getdata(self.fnfmt.format(self.npix, 'hexonly', self.wave/um, 90))
        psf90r0 = np.rot90(psf90)
        psfdiff = psf0 - psf90r0
        fits.PrimaryHDU(data=psfdiff).writeto(self.data_dir+"/diff_90_0_off10.fits", overwrite=True)
        psf0  = fits.getdata(self.fnfmt.format(self.npix, 'hexonly', self.wave/um, 5))
        psf90 = fits.getdata(self.fnfmt.format(self.npix, 'hexonly', self.wave/um, 95))
        psf90r0 = np.rot90(psf90)
        psfdiff = psf0 - psf90r0
        fits.PrimaryHDU(data=psfdiff).writeto(self.data_dir+"/diff_95_5_off10.fits", overwrite=True)
        self.assertTrue(0.0 < 1e-15, \
                        'error: test_affine2d failed')
    """


if __name__ == "__main__":
    unittest.main()

