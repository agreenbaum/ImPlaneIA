import unittest
import os
import glob
import sys
import numpy as np
from astropy.io import fits
from astropy import units as u

from nrm_analysis import find_affine2d_parameters as FAP

# To make fake image data...
import nrm_analysis.misctools.utils as utils
#import avoidhexsingularity, Affine2d
from nrm_analysis.fringefitting.LG_Model import NRM_Model

"""
    Test rotfinding  using Affine2d class rotation
    anand@stsci.edu 2018.08.15

    run with pytest -s _moi_.py to see stdout on screen
    All units SI unless units in variable name
"""

um = 1.0e-6


class Affine2dRotTestCase(unittest.TestCase):

    def setUp(self):

        self.holeshape = "hex"

        # directory containing the test data
        data_dir = os.path.join(os.path.dirname(__file__),'test_data/find_affine2d_parameters')
        self.data_dir = data_dir
        print(data_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        pixel = 0.0656 *  u.arcsec.to(u.rad)
        npix = 87
        wave = 4.3e-6 # m
        over = 3
        holeshape='hex'
        
        rotd_true = 9.25
        rotdegs = (8.0, 9.0, 10.0, 11.0, 12.0) # search in this range of rotation (degrees)

        # store the real affine to be uased to create test data
        self.affine = utils.Affine2d(rotradccw=np.pi*utils.avoidhexsingularity(rotd_true)/180.0, 
                                name="{0:.4}".format(float(rotd_true)))
        self.affine.show(label="Creating affine2d for PSF data")

        # create image data to find its rotation, as a sample test...
        imagefn = data_dir+"/imagedata.fits"
        jw = NRM_Model(mask='jwst', holeshape="hex", affine2d=self.affine)
        jw.set_pixelscale(pixel)
        jw.simulate(fov=npix, bandpass=wave, over=over)
        fits.writeto(imagefn, jw.psf, overwrite=True)
        imagedata = jw.psf.copy()
        del jw
        fits.getdata(imagefn)

        print("driver:", rotdegs)
        mx, my, sx,sy, xo,yo, = (1.0,1.0, 0.0,0.0, 0.0,0.0)
        aff_best_rot =FAP.find_rotation(imagedata,
                                        rotdegs, mx, my, sx,sy, xo,yo,
                                        pixel, npix, wave, over, holeshape, outdir=data_dir)

        print(aff_best_rot)
        self.aff_best_rot = aff_best_rot


    def test_psf(self):
        """ 
            sanity check and code development tool than a routine test.
            Look at true affine used to make data cf measured affine parameter erotation
        """
        rot_error_d = self.affine.get_rotd() - self.aff_best_rot.get_rotd()
        self.assertTrue(np.abs(rot_error_d) < 1e-2,  'error: test_find_affine2d_parameters failed')

if __name__ == "__main__":
    unittest.main()

