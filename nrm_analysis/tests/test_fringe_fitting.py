from __future__ import print_function

import unittest, os, glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from matplotlib import pylab as pl

import sys, pickle
import uncertainties

sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))


try:
    import oifits
#     print('oifits successfully imported')
except ImportError:
    print('Module oifits not found. Please include it in your path')
    print('You can for instance run the shell command')    
    print('wget http://astro.ins.urfu.ru/pages/~pboley/oifits/oifits.py --directory-prefix %s' % os.path.dirname(__file__))
    print('and try again')
    sys.exit()

try: 
    import linearfit
except ImportError:
    print('linearfit module not imported, no covariances will be saved.')
    print(sys.path)
    1/0

from nrm_analysis import nrm_core, InstrumentData
from nrm_analysis.fringefitting import leastsqnrm
from nrm_analysis.fringefitting import utility_classes

reload(leastsqnrm)
reload(utility_classes)


arcsec2rad = u.arcsec.to(u.rad)

oversample = 11
n_image = 77
flip = False

class FringeFittingTestCase(unittest.TestCase):
    def setUp(self):
        '''
        Generate monochromatic image of a point source 
        '''
        # setup parameters for simulation
        verbose = 0
        overwrite = 0

        monochromatic_wavelength_m = 4.3e-6 
        mask = 'MASK_NRM'
        filter = 'F430M'
        pixelscale_arcsec = 0.0656 
        filter_name = 'Monochromatic '+np.str(monochromatic_wavelength_m)

        self.filter = filter
        self.filter_name = filter_name
        self.monochromatic_wavelength_m = monochromatic_wavelength_m
        
        # directory containing the test data
        data_dir = os.path.join(os.path.dirname(__file__),'test_data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        out_dir = data_dir

        name_seed = 'PSF_NIRISS_%s_%s'%(mask,filter)

        psf_image_name = name_seed + '_reference.fits'
        psf_image = os.path.join(data_dir,psf_image_name)
        psf_image_without_oversampling = os.path.join(data_dir,psf_image_name.replace('.fits','_without_oversampling.fits'))

        if (not os.path.isfile(psf_image_without_oversampling)) | (overwrite):
            #             Generate monochromatic image of a point source 
            from fringefitting.LG_Model import NRM_Model
            jw = NRM_Model(mask='jwst',holeshape="hex",flip=flip)
            jw.simulate(fov=n_image, bandpass=monochromatic_wavelength_m, over=oversample, pixel = pixelscale_arcsec * arcsec2rad)
            
            # optional writing of oversampled image
            if 0:
                fits.writeto(psf_image,jw.psf_over, clobber=True)
                header = fits.getheader(psf_image)
                header['PIXELSCL'] = pixelscale_arcsec/oversample
                header['FILTER'] = filter_name
                header['PUPIL'] = mask 
                fits.update(psf_image,jw.psf_over/10000./28., header=header)

            # PSF without oversampling
            fits.writeto(psf_image_without_oversampling,jw.psf, clobber=True)
            header = fits.getheader(psf_image_without_oversampling)
            header['PIXELSCL'] = pixelscale_arcsec
            header['FILTER'] = filter_name
            header['PUPIL'] = mask 
            fits.update(psf_image_without_oversampling,jw.psf, header=header)
    
    
        # list of files produced for target
        file_list = glob.glob(os.path.join(data_dir,'*%s*.fits' % 'without_oversampling' ));
            
        self.simulated_image = file_list[0]
        self.psf_image_without_oversampling = psf_image_without_oversampling
        
        
                
    def test_fringe_fitting(self,verbose = 0):
        '''
        Run fringe fitting algorithm on the point source image and test that closure phases are zero        
        '''
    
        image_file_name = self.simulated_image        
        
        if verbose:
            print('testing on %s' % image_file_name)
        header = fits.getheader(image_file_name)   
        file_name = os.path.basename(image_file_name)
        data_dir = os.path.dirname(image_file_name)
        save_dir = data_dir + '/'

        if 'Monochromatic' in self.filter_name:
            nirissdata = InstrumentData.NIRISS(filt=self.filter)
            nirissdata.wls = [self.monochromatic_wavelength_m]
            print(nirissdata.wls)
                
    
        nirissdata.pscale_mas = header['PIXELSCL'] * 1000
        print('measure_fringes: pixelscale %f'%header['PIXELSCL']) 
        nirissdata.pscale_rad = nirissdata.pscale_mas/1000. * arcsec2rad
    
    
        ff = nrm_core.FringeFitter(nirissdata, oversample=oversample, \
                      savedir=save_dir, datadir=data_dir, \
                      npix=n_image, interactive=False, flip=flip)
        print('FringeFitter oversampling: %d' % ff.oversample)
    
        threads = 0
        ff.fit_fringes([file_name],threads=threads)

        CP_file = sorted(glob.glob(os.path.join(data_dir,'%s/%s*.txt' % (file_name.split('.')[0],'CPs_') )));
        CP = Table.read(CP_file[0],format='ascii.no_header',names=({'closure_phase'}))

        #         perform the actual test
        self.assertTrue(np.mean(np.array(CP['closure_phase'])) < 1e-15, 'Simulated closure phases of point source are non-zero')

    
    def test_fringe_fitting_linearfit(self, verbose = 0, make_plot = 0):
        '''
        Compare results of default fitting (no weighting, no uncertainty propagation) 
        with results obtained from linearfit module coupled with full covariance propagation
        using the uncertainties package (linear approximation) 
        '''

        data_dir = os.path.dirname( self.simulated_image )
        save_dir = os.path.join(data_dir,'%s' % (self.simulated_image.split('.')[0]))
        ffr = utility_classes.FringeFitterResult(save_dir) 
        ffr.print_results()
        pklfile = os.path.join(save_dir,'linearfit_result_00.pkl')
        ffr.linfit_result = pickle.load(  open( pklfile, "rb" ) )

        if verbose:
            ffr.linfit_result.display_results(nformat='e')
    
        # compute CPs and CAs the default way        
        soln = ffr.linfit_result.p
        fringeamp, fringephase = leastsqnrm.tan2visibilities( soln )
        redundant_cps = leastsqnrm.redundant_cps(fringephase, N=7)
        redundant_cas = leastsqnrm.return_CAs(fringeamp, N=7)

    
        # make an informed choice on whether to use normalised or formal covariance matrix
        soln_cov = uncertainties.correlated_values( soln, np.array(ffr.linfit_result.p_normalised_covariance_matrix) )

        # computations without uncertainty propagation (regular -> reg)           
        fringephase_reg, fringeamp_reg, redundant_cps_reg, redundant_cas_reg = leastsqnrm.phases_and_amplitudes(soln)
            
        # computations with covariance propagation (covariances -> cov)            
        fringephase_cov, fringeamp_cov, redundant_cps_cov, redundant_cas_cov = leastsqnrm.phases_and_amplitudes(soln_cov)


        #         perform the actual tests
        self.assertTrue(np.mean(redundant_cps) - np.mean(redundant_cps_reg) == 0, 'Default computation done two ways disagree')
        self.assertTrue(np.mean(redundant_cps_reg) - np.mean(redundant_cps_cov).nominal_value < np.mean(redundant_cps_cov).std_dev , 'Default computation and full covariance propagation disagree significantly')

        
        #         optional figure 
        if make_plot:
            j = 0
            save_plot = 1
            fig_dir = save_dir
            name_seed2 = 'linearfit_test'
            ffr.integration[j].show_results(save_plot=save_plot,out_dir=fig_dir,name_seed=name_seed2,cps=redundant_cps_cov,cas=redundant_cas_cov,fps=fringephase,fas=fringeamp)
            pl.close('all')


if __name__ == '__main__':
    unittest.main()
    
    
    
    
    