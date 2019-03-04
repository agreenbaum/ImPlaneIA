import unittest
import os
import glob
import sys
import numpy as np
from astropy.io import fits
from astropy import units as u

from nrm_analysis.misctools.utils import Affine2d
from nrm_analysis.misctools.utils import avoidhexsingularity
from nrm_analysis.fringefitting.LG_Model import NRM_Model

"""
    Test rotation for Affine2d class
    anand@stsci.edu 2018.06.21

    run with pytest -s _moi_.py to see stdout on screen
    All units SI unless units in variable name
"""

arcsec2rad = u.arcsec.to(u.rad)
um = 1.0e-6

def affinepars2header(hdr, affine2d):
    """ writes affine2d parameters into fits header """
    hdr['affine'] = (affine2d.name, 'Affine2d in pupil: name')
    hdr['aff_mx'] = (affine2d.mx, 'Affine2d in pupil: xmag')
    hdr['aff_my'] = (affine2d.my, 'Affine2d in pupil: ymag')
    hdr['aff_sx'] = (affine2d.sx, 'Affine2d in pupil: xshear')
    hdr['aff_sy'] = (affine2d.sx, 'Affine2d in pupil: yshear')
    hdr['aff_xo'] = (affine2d.xo, 'Affine2d in pupil: x offset')
    hdr['aff_yo'] = (affine2d.yo, 'Affine2d in pupil: y offset')
    hdr['aff_dev'] = ('analyticnrm2', 'dev_phasor')
    return hdr

class Affine2dTestCase(unittest.TestCase):

    def setUp(self):

        # directory containing the test data
        data_dir = os.path.join(os.path.dirname(__file__),'test_data/affine2d_rot_psf')
        self.data_dir = data_dir
        print(data_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.fnfmt = data_dir+'/psf_nrm_{2:.1f}_{0}_{1}_{3:.0f}.fits' # expects strings of  %(imsize,hole)

        self.fmt = "    ({0:+.3f}, {1:+.3f}) -> ({2:+.3f}, {3:+.3f})"
        self.pixel = 0.0656 *  u.arcsec.to(u.rad)
        self.npix = 87
        self.wave = 4.3e-6 # m
        self.over = 1

        mx, my = 1.0, 1.0
        sx, sy= 0.0, 0.0
        xo, yo= 0.0, 0.0
        affine_ideal = Affine2d(mx=mx,my=my, 
                                sx=sx,sy=sy, 
                                xo=xo,yo=yo, name="Ideal")

        for rot in (0,5,10,15,20,25,30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95):
            """
            print("           rot degrees pre", rot, end='')
            diagnostic = rot/15.0 - int(rot/15.0)
            print("  diagnostic", diagnostic, end='')
            rot = avoidhexsingularity(rot) # in utils
            print("   rot degrees post", rot)
            """
            rot = avoidhexsingularity(rot) # in utils
            affine_rot = Affine2d(rotradccw=np.pi*rot/180.0, name="{0:.0f}".format(rot)) # in utils
            aff = affine_rot

            # hexonly g7s6 jwst
            self.jw = NRM_Model(mask='jwst', holeshape="hexonly", affine2d=aff)
            self.jw.set_pixelscale(self.pixel*arcsec2rad)
            self.jw.simulate(fov=self.npix, bandpass=self.wave, over=self.over)
            psffn = self.fnfmt.format(self.npix, 'hexonly', self.wave/um, rot)
            fits.writeto(psffn, self.jw.psf, overwrite=True)
            header = fits.getheader(psffn)
            header = affinepars2header(header, aff)
            fits.update(psffn, self.jw.psf, header=header)
            del self.jw

            del aff
            del affine_rot


    def test_psf(self):
        """ Read in PSFs with 0, 90 degree affines, 5 and 95 degree affines, 
            Rotate one set and subtract from the smaller rot PSF - should be zero if
            everything is correctly calculated.  If we nudge the PSF centers to avoid the 
            line singularity that hextransformEE will encounter if the psf is centrally 
            placed in a pixel.
            The file names are hand-edited to reflect the oversampling and rotations,
            so this is more a sanity check and code development tool than a routine test.
        """
        psf0  = fits.getdata(self.fnfmt.format(self.npix, 'hexonly', self.wave/um, 0))
        psf90 = fits.getdata(self.fnfmt.format(self.npix, 'hexonly', self.wave/um, 90))
        psf90r0 = np.rot90(psf90)
        psfdiff = psf0 - psf90r0
        fits.PrimaryHDU(data=psfdiff).writeto(self.data_dir+"/diff_90_0_off08_ov1.fits", overwrite=True)
        psf0  = fits.getdata(self.fnfmt.format(self.npix, 'hexonly', self.wave/um, 5))
        psf90 = fits.getdata(self.fnfmt.format(self.npix, 'hexonly', self.wave/um, 95))
        psf90r0 = np.rot90(psf90)
        psfdiff = psf0 - psf90r0
        fits.PrimaryHDU(data=psfdiff).writeto(self.data_dir+"/diff_95_5_off08_ov1.fits", overwrite=True)
        self.assertTrue(0.0 < 1e-15,  'error: test_affine2d failed')

if __name__ == "__main__":
    unittest.main()

