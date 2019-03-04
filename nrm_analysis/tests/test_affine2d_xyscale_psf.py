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
    Test pitch algebra for Affine2d class
    anand@stsci.edu 2018.04.12

    run with pytest -s _moi_.py to see stdout on screen
    All units SI unless units in variable name
"""

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
    return hdr

class Affine2dTestCase(unittest.TestCase):

    def setUp(self):

        # directory containing the test data
        data_dir = os.path.join(os.path.dirname(__file__),'test_data/affine2d_xyscale_psf')
        self.data_dir = data_dir
        print(data_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.fnfmt = data_dir+'/psf_nrm_{0:d}_{1:s}_{2:.3f}um_{3:s}.fits' 
        # expects e.g. self.fnfmt.format(self.npix, 'hex', self.wave/um, aff.name)

        self.pixel = 0.0656 *  u.arcsec.to(u.rad)
        self.npix = 87
        self.wave = 4.3e-6 # m
        self.over = 1

        ################
        self.pointsfmt = "    ({0:+.3f}, {1:+.3f}) -> ({2:+.3f}, {3:+.3f})"
        self.numtestfmt = "    min: {0:+.1e},  max: {1:+.1e},  avg: {2:+.1e},  sig: {3:+.1e}" # min max avg stddev
        self.testpoints = (np.array((0,0)), np.array((1,0)), np.array((0,1)), np.array((1,1)))

        xo, yo= 0.0, 0.0 # always set these for PSFs...

        mx, my = 1.0, 1.0
        sx, sy= 0.0, 0.0
        self.affine_ideal = Affine2d(mx=mx,my=my, sx=sx,sy=sy, xo=xo,yo=yo, name="Ideal")

        mx, my = -1.0, 1.0
        sx, sy= 0.0, 0.0
        self.affine_xrev = Affine2d(mx=mx,my=my, sx=sx,sy=sy, xo=xo,yo=yo, name="Xreverse")

        mx, my = 2.0, 1.0
        sx, sy= 0.0, 0.0
        self.affine_xmag = Affine2d(mx=mx,my=my, sx=sx,sy=sy, xo=xo,yo=yo, name="Xmag")

        mx, my = 1.0, 1.0
        sx, sy= 0.5, 0.0
        self.affine_sigx = Affine2d(mx=mx,my=my, sx=sx,sy=sy, xo=xo,yo=yo, name="Xshear")

        for aff in (self.affine_ideal, self.affine_xrev, self.affine_xmag, self.affine_sigx):
           
            # circ g7s6 jwst created for info only
            self.jw = NRM_Model(mask='jwst', holeshape="circ", affine2d=aff)
            self.jw.set_pixelscale(self.pixel)
            self.jw.simulate(fov=self.npix, bandpass=self.wave, over=self.over)
            psffn = self.fnfmt.format(self.npix, 'circ', self.wave/um, aff.name)
            fits.writeto(psffn, self.jw.psf, overwrite=True)
            header = fits.getheader(psffn)
            header = affinepars2header(header, aff)
            fits.update(psffn, self.jw.psf, header=header)
            del self.jw

            # circonly g7s6 jwst created for info only
            self.jw = NRM_Model(mask='jwst', holeshape="circonly", affine2d=aff)
            self.jw.set_pixelscale(self.pixel)
            self.jw.simulate(fov=self.npix, bandpass=self.wave, over=self.over)
            psffn = self.fnfmt.format(self.npix, 'circonly', self.wave/um, aff.name)
            fits.writeto(psffn, self.jw.psf, overwrite=True)
            header = fits.getheader(psffn)
            header = affinepars2header(header, aff)
            fits.update(psffn, self.jw.psf, header=header)
            del self.jw

            # hex g7s6 jwst
            self.jw = NRM_Model(mask='jwst', holeshape="hex", affine2d=aff)
            self.jw.set_pixelscale(self.pixel)
            self.jw.simulate(fov=self.npix, bandpass=self.wave, over=self.over)
            psffn = self.fnfmt.format(self.npix, 'hex', self.wave/um, aff.name)
            fits.writeto(psffn, self.jw.psf, overwrite=True)
            header = fits.getheader(psffn)
            header = affinepars2header(header, aff)
            fits.update(psffn, self.jw.psf, header=header)
            del self.jw

            # hexonly g7s6 jwst
            self.jw = NRM_Model(mask='jwst', holeshape="hexonly", affine2d=aff)
            self.jw.set_pixelscale(self.pixel)
            self.jw.simulate(fov=self.npix, bandpass=self.wave, over=self.over)
            psffn = self.fnfmt.format(self.npix, 'hexonly', self.wave/um, aff.name)
            fits.writeto(psffn, self.jw.psf, overwrite=True)
            header = fits.getheader(psffn)
            header = affinepars2header(header, aff)
            fits.update(psffn, self.jw.psf, header=header)
            del self.jw

            # fringeonly g7s6 jwst
            self.jw = NRM_Model(mask='jwst', holeshape="fringeonly", affine2d=aff)
            self.jw.set_pixelscale(self.pixel)
            self.jw.simulate(fov=self.npix, bandpass=self.wave, over=self.over)
            psffn = self.fnfmt.format(self.npix, 'fringeonly', self.wave/um, aff.name)
            fits.writeto(psffn, self.jw.psf, overwrite=True)
            header = fits.getheader(psffn)
            header = affinepars2header(header, aff)
            fits.update(psffn, self.jw.psf, header=header)
            del self.jw

            del aff

        self.hextest = self.eval_psf_hex()
        self.fringeonlytest = self.eval_psf_fringeonly()
        self.hexonlytest = self.eval_psf_hexonly()
        self.evals = [ self.hextest, self.fringeonlytest, self.hexonlytest]
        self.sigsum = self.hextest + self.fringeonlytest + self.hexonlytest 


    """  Ideal and Xreverse_xflipped should be machine zero or close """
    def eval_psf_hex(self):
        psfIdeal  = fits.getdata(self.fnfmt.format(self.npix, 'hex', self.wave/um, "Ideal"))
        psfXrev  = fits.getdata(self.fnfmt.format(self.npix, 'hex', self.wave/um, "Xreverse"))
        psfdiff = psfIdeal - psfXrev[::-1,:]
        print("    Numerical Summary:         hex  psf: " +\
              self.numtestfmt.format(psfIdeal.min(), psfIdeal.max(), psfIdeal.mean(), psfIdeal.std()))
        print("    Numerical Summary:         hex test: " +\
              self.numtestfmt.format(psfdiff.min(), psfdiff.max(), psfdiff.mean(), psfdiff.std()))
        fits.PrimaryHDU(data=psfdiff).writeto(self.data_dir+"/diff_hex.fits", overwrite=True)
        return psfdiff.std()

    def eval_psf_fringeonly(self):
        psfIdeal  = fits.getdata(self.fnfmt.format(self.npix, 'fringeonly', self.wave/um, "Ideal"))
        psfXrev  = fits.getdata(self.fnfmt.format(self.npix, 'fringeonly', self.wave/um, "Xreverse"))
        psfdiff = psfIdeal - psfXrev[::-1,:]
        print("    Numerical Summary:  fringeonly  psf: " +\
              self.numtestfmt.format(psfIdeal.min(), psfIdeal.max(), psfIdeal.mean(), psfIdeal.std()))
        print("    Numerical Summary:  fringeonly test: " +\
              self.numtestfmt.format(psfdiff.min(), psfdiff.max(), psfdiff.mean(), psfdiff.std()))
        fits.PrimaryHDU(data=psfdiff).writeto(self.data_dir+"/diff_fringeonly.fits", overwrite=True)
        return psfdiff.std()

    def eval_psf_hexonly(self):
        psfIdeal  = fits.getdata(self.fnfmt.format(self.npix, 'hexonly', self.wave/um, "Ideal"))
        psfXrev  = fits.getdata(self.fnfmt.format(self.npix, 'hexonly', self.wave/um, "Xreverse"))
        psfdiff = psfIdeal - psfXrev[::-1,:]
        print("    Numerical Summary:     hexonly  psf: " +\
              self.numtestfmt.format(psfIdeal.min(), psfIdeal.max(), psfIdeal.mean(), psfIdeal.std()))
        print("    Numerical Summary:     hexonly test: " +\
              self.numtestfmt.format(psfdiff.min(), psfdiff.max(), psfdiff.mean(), psfdiff.std()))
        fits.PrimaryHDU(data=psfdiff).writeto(self.data_dir+"/diff_hexonly.fits", overwrite=True)
        return psfdiff.std()

    def test_psfs(self):
        print("Standard deviations of all three tests:", self.evals)
        self.assertTrue(self.sigsum < 1e-7, 'error: test_affine2d_xyscale_psf failed')



if __name__ == "__main__":
    unittest.main()


"""  Useful fits file command line viewer example...
#! /bin/csh

/Applications/SAOImage\ DS9.app/Contents/MacOS/ds9  \
                \
                -tile grid gap 2  -tile grid layout 4 5  \
                -zoom 8 \
                -scale sqrt \
                -view  colorbar no   \
                -view panner no   \
                -view magnifier yes   \
                -view buttons no  \
                -view object no  \
                -view info yes  \
                -view WCS no  \
                -view physical no  \
                -view image yes    \
            \
            psf_nrm_87_fringeonly_4.300um_Xreverse.fits    \
            psf_nrm_87_fringeonly_4.300um_Ideal.fits    \
            psf_nrm_87_fringeonly_4.300um_Xshear.fits    \
            psf_nrm_87_fringeonly_4.300um_Xmag.fits    \
            \
            psf_nrm_87_circonly_4.300um_Xreverse.fits    \
            psf_nrm_87_circonly_4.300um_Ideal.fits    \
            psf_nrm_87_circonly_4.300um_Xshear.fits    \
            psf_nrm_87_circonly_4.300um_Xmag.fits    \
            \
            psf_nrm_87_hexonly_4.300um_Xreverse.fits    \
            psf_nrm_87_hexonly_4.300um_Ideal.fits    \
            psf_nrm_87_hexonly_4.300um_Xshear.fits    \
            psf_nrm_87_hexonly_4.300um_Xmag.fits    \
            \
            psf_nrm_87_circ_4.300um_Xreverse.fits    \
            psf_nrm_87_circ_4.300um_Ideal.fits    \
            psf_nrm_87_circ_4.300um_Xshear.fits    \
            psf_nrm_87_circ_4.300um_Xmag.fits    \
            \
            psf_nrm_87_hex_4.300um_Xreverse.fits    \
            psf_nrm_87_hex_4.300um_Ideal.fits    \
            psf_nrm_87_hex_4.300um_Xshear.fits    \
            psf_nrm_87_hex_4.300um_Xmag.fits    \
            &
"""
