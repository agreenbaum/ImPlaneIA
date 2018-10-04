

import unittest, os, glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u


import sys
#sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../'))
#sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))
# print(sys.path)

try:
    import oifits
    print('oifits successfully imported')
except ImportError:
    print('Module oifits not found. Please include it in your path')    
    sys.exit()


#import nrm_analysis
import nrm_analysis.nrm_core as nrm_core
import nrm_analysis.InstrumentData as InstrumentData
import nrm_analysis.find_affine2d_parameters as FAP
import nrm_analysis.misctools.utils as utils



arcsec2rad = u.arcsec.to(u.rad)

holeshape='hex'
oversample = 1
n_image = 77

rotd_true = 50
rotd_search = (49.0, 49.5, 50, 50.5, 51.0)


class FindPassAffine2dTestCase(unittest.TestCase):
    def setUp(self):
    
        # setup parameters for simulation
        verbose = 1
        overwrite = 1

        monochromatic_wavelength_m = 4.3e-6 
        mask = 'MASK_NRM'
        filter = 'F430M'
        pixel = 0.0656 
        filter_name = 'Monochromatic '+np.str(monochromatic_wavelength_m)

        self.filter = filter
        self.filter_name = filter_name
        self.monochromatic_wavelength_m = monochromatic_wavelength_m

        # directory containing the test data
        datadir = os.path.join(os.path.dirname(__file__),'test_data/find_pass_affine')
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        self.datadir = datadir

        # file containing the test data
        imagefn = datadir+"/imagedata.fits"
        # use millidegrees and integers
        imagefn = imagefn.replace(".fits","_truerotmd{0:+05d}.fits".format(int(1000.0*float(rotd_true))))
        self.imagefn = imagefn

        # savedir for reduced quantities
        self.savedir = imagefn.replace(".fits","_truerotmd{0:+05d}.fits".format(int(1000.0*float(rotd_true))))
        self.savedir = self.datadir+'/'

        print('\n')
        print('  >> >> >> >> set-up  self.imagefn %s' % self.imagefn)
        print('  >> >> >> >> set-up  self.datadir %s' % self.datadir)
        print('  >> >> >> >> set-up  self.savedir %s' % self.savedir)


        # Create test data
        
        affine = utils.Affine2d(rotradccw=np.pi*utils.avoidhexsingularity(rotd_true)/180.0, 
                                name="{0:.3}".format(float(rotd_true)))
        affine.show(label="Creating PSF data")
        #
        from nrm_analysis.fringefitting.LG_Model import NRM_Model
        jw = NRM_Model(mask='jwst',holeshape="hex", affine2d=affine)
        jw.set_pixelscale(pixel*arcsec2rad)
        jw.simulate(fov=n_image, 
                    bandpass=monochromatic_wavelength_m, 
                    over=oversample)
        # PSF without oversampling
        fits.writeto(imagefn, jw.psf, overwrite=True)
        header = fits.getheader(imagefn)
        header['PIXELSCL'] = pixel
        header['FILTER'] = filter_name
        header['PUPIL'] = mask 
        header = utils.affinepars2header(header, affine)
        fits.update(imagefn, jw.psf, header=header)
        self.imagedata = jw.psf.copy()
        del jw
        del affine

        #else:
        #    self.imagedata = fits.getdata(self.imagefn) 
    
                
    def test_find_pass_affine(self):
        
        header = fits.getheader(self.imagefn)   
        filename = os.path.basename(self.imagefn)
        datadir = os.path.dirname(self.imagefn)
        print('\n')
        print('  >> >> >> >> testing self.imagefn %s' % self.imagefn)
        print('  >> >> >> >> testing self.datadir %s' % self.datadir)
        print('  >> >> >> >> testing self.savedir %s' % self.savedir)

        fn = filename.replace('.fits','')
        print('  >> >> >> >> fn %s' % fn)

        print("driver rotd_search:", rotd_search)
        print("driver rotd_true:", rotd_true)
        mx, my, sx,sy, xo,yo, = (1.0,1.0, 0.0,0.0, 0.0,0.0)
        aff_best_rot =FAP.find_rotation(self.imagedata,
                                        rotd_search, mx, my, sx,sy, xo,yo,
                                        header['PIXELSCL']*arcsec2rad, 
                                        n_image, self.monochromatic_wavelength_m, oversample, holeshape, 
                                        outdir=None)
        print("rotd_measured:",  aff_best_rot)
        aff_best_rot.show(label="Measured ")



        if 'Monochromatic' in self.filter_name:
            nirissdata = InstrumentData.NIRISS(filt=self.filter, affine2d=aff_best_rot)
            nirissdata.wls = [self.monochromatic_wavelength_m]
            print((nirissdata.wls))

        nirissdata.pscale_mas = header['PIXELSCL'] * 1000
        nirissdata.pscale_rad = nirissdata.pscale_mas/1000. * arcsec2rad
        #nirissdata.set_affine2d(aff_best_rot)
    
        ff = nrm_core.FringeFitter(nirissdata, oversample=oversample,
                      savedir=self.savedir, datadir=datadir,
                      npix=n_image, interactive=False,
                      hold_centering=True)
        print(('FringeFitter oversampling: %d' % ff.oversample))
    
        threads = 1
        ff.fit_fringes([filename])

        CP_file = sorted(glob.glob(os.path.join(datadir,'%s/%s*.txt' % (filename.split('.')[0],'CPs_') )));
        print("CP_file: ", type(CP_file), len(CP_file))
        CP = Table.read(CP_file[0],format='ascii.no_header',names=({'closure_phase'}))

        #         perform the actual test
        self.assertTrue(np.mean(np.array(CP['closure_phase'])) < 1e-7, 
            'Simulated closure phases of point source are non-zero')  # Avoiding hex singularity in hextransformEE.py causes this sort of noise in exactly-XYaligned PSF 2018 AS

if __name__ == '__main__':
    unittest.main()
