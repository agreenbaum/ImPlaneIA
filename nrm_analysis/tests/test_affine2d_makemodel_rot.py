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
    Test rotation for makemodel cos & sin fringes using Affine2d class rotation
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

def holes2string(hlist):
    """
    if hlist is ('b4','c5') return 'b4c5'
    """
    hstr = ""
    for hn in hlist:
        hstr = hstr+hn
    return hstr.lower()



class Affine2dMakeModelRotTestCase(unittest.TestCase):

    def setUp(self):

        allholes = ('b4','c2','b5','b2','c1','b6','c6')
        b4,c2,b5,b2,c1,b6,c6 = allholes
        self.hc = (b2,b6,b5) # holechoices
        self.hstr = holes2string(self.hc)
        self.holeshape = "hex"


        # directory containing the test data
        data_dir = os.path.join(os.path.dirname(__file__),'test_data/affine2d_makemodel_rot')
        self.data_dir = data_dir
        print(data_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
       # expects strings of  % (self.npix, self.holeshape, self.wave/um, rot, self.hstr)
        self.fnfmt = data_dir+'/psf_nrm_{2:.1f}_{0}_{1}_{3:.0f}_{4:s}.fits' 

        self.fmt = "    ({0:+.3f}, {1:+.3f}) -> ({2:+.3f}, {3:+.3f})"
        self.pixel = 0.0656 
        self.npix = 87
        self.wave = 4.3e-6 # m
        self.over = 11


        mx, my = 1.0, 1.0
        sx, sy= 0.0, 0.0
        xo, yo= 0.0, 0.0
        affine_ideal = Affine2d(mx=mx,my=my, 
                                sx=sx,sy=sy, 
                                xo=xo,yo=yo, name="Ideal")

        rots = (0,5,10,15,20,25,30, 35, 40, 45, 50, 
                       55, 60, 65, 70, 75, 80, 85, 90, 95)
        rots = (0.000, 10.000,  )
        for rot in rots:
            
            rot = avoidhexsingularity(rot) # in utils
            affine_rot = Affine2d(rotradccw=np.pi*rot/180.0, 
                                  name="{0:.0f}".format(rot)) # in utils
            aff = affine_rot

            # holeshape is hex or circ g7s6 jwst 
            # because model_array requires 
            # a primary beam for one slice of the model
            self.jw = NRM_Model(mask='jwst', 
                                holeshape=self.holeshape, 
                                #chooseholes=self.hc, 
                                affine2d=aff)
            self.jw.set_pixelscale(self.pixel*arcsec2rad)
            self.jw.simulate(fov=self.npix, 
                             bandpass=self.wave, 
                             over=self.over)
            # write psf
            psffn = self.fnfmt.format(self.npix, self.holeshape,
                                      self.wave/um, rot, self.hstr)
            fits.writeto(psffn, self.jw.psf, overwrite=True)
            header = fits.getheader(psffn)
            header = affinepars2header(header, aff)
            fits.update(psffn, self.jw.psf, header=header)
            print("test:  psf shape", self.jw.psf.shape)

            modelslices = self.jw.make_model(fov=self.npix, 
                                             bandpass=self.wave,
                                             over=self.over)
            print("test:  modelslices type", type(modelslices))
            print("test:  modelslices shape", modelslices.shape)
            modelfn = psffn.replace("psf_nrm","model")

            # write model
            model_for_fitsfile = np.zeros((modelslices.shape[2], 
                                           modelslices.shape[0], 
                                           modelslices.shape[1]))
            for sl in range(modelslices.shape[2]):
                model_for_fitsfile[sl,:,:] = modelslices[:,:,sl]
            print("test:  model_for_fitsfile type", type(model_for_fitsfile))
            print("test:  model_for_fitsfile shape", model_for_fitsfile.shape)
            fits.writeto(modelfn, model_for_fitsfile[6,:,:], overwrite=True)
            header = fits.getheader(modelfn)
            header = affinepars2header(header, aff)
            fits.update(modelfn, model_for_fitsfile[6,:,:], header=header)

            del self.jw

            del aff
            del affine_rot


    def test_psf(self):
        """ 
            sanity check and code development tool than a routine test.
        """
        self.assertTrue(0.0 < 1e-15,  'error: test_affine2d failed')

if __name__ == "__main__":
    unittest.main()

