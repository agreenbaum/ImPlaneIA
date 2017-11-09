from __future__ import print_function

# import os, sys, pdb, pickle, glob, aplpy
import os,glob
import numpy as np
import pylab as pl
from astropy.table import Table, Column
from astropy.table import hstack as tablehstack
from astropy.io import fits
# from scipy.misc import comb
from uncertainties import unumpy
import aplpy

class NrmIntegrationResult(object):
    def __init__(self, solutions , closure_quantities , baseline_quantities ):
        self.solutions = solutions
        self.closure_quantities = closure_quantities
        self.baseline_quantities = baseline_quantities

    def print_results(self,valid_index = None,number_format='f'):        
        if valid_index is None:
            valid_index = np.arange(len(self.closure_quantities))           
        for col in self.closure_quantities.colnames:
            if number_format=='f':
                print('Average %s is %3.6f +/- %3.6f (%d measurements)' % (col,np.mean(self.closure_quantities[col][valid_index]),np.std(self.closure_quantities[col][valid_index]),len(valid_index)))
            elif number_format=='e':
                print('Average %s is %3.3e +/- %3.3e (%d measurements)' % (col,np.mean(self.closure_quantities[col][valid_index]),np.std(self.closure_quantities[col][valid_index]),len(valid_index)))

    def show_results(self,save_plot=0,out_dir='',name_seed=None,cps=None,cas=None,fps=None,fas=None,number_format='f'):        
        fig=pl.figure(figsize=(12, 6),facecolor='w', edgecolor='k'); pl.clf();
        attributes = np.array(['closure_quantities','baseline_quantities'])
        y_locations = np.array([1,0])
        subplot_index = 1
        for ii,attribute in enumerate(attributes):
            for jj,col_name in enumerate(getattr(self,attribute).colnames):       
                if number_format=='f':
                    print('Average %s is %3.3f +/- %3.3f' % (col_name,np.mean(getattr(self,attribute)[col_name]),np.std(getattr(self,attribute)[col_name])))
                elif number_format=='e':
                    print('Average %s is %3.3e +/- %3.3e' % (col_name,np.mean(getattr(self,attribute)[col_name]),np.std(getattr(self,attribute)[col_name])))
                    


                pl.subplot(2,2,ii+1+jj*2)
                coord = getattr(self,attribute)[col_name]
                pl.plot(coord,'bo',label=col_name)
                pl.axhline(y=y_locations[jj],color='0.7',ls='--',zorder=-50)
                if (col_name == 'closure_phase') & (cps is not None):
                    pl.plot(np.arange(len(coord)),unumpy.nominal_values(cps),'ks',label='CP incl. covariances',zorder=-40)
                    pl.errorbar(np.arange(len(coord)),unumpy.nominal_values(cps),yerr=unumpy.std_devs(cps),ecolor='k',fmt=None,zorder=-40)
                elif (col_name == 'closure_amplitude') & (cas is not None):
                    pl.plot(np.arange(len(coord)),unumpy.nominal_values(cas),'ks',label='CA incl. covariances',zorder=-40)
                    pl.errorbar(np.arange(len(coord)),unumpy.nominal_values(cas),yerr=unumpy.std_devs(cas),ecolor='k',fmt=None,zorder=-40)
                elif (col_name == 'fringe_phase') & (fps is not None):
                    pl.plot(np.arange(len(coord)),unumpy.nominal_values(fps),'ks',label='FP incl. covariances',zorder=-40)
                    pl.errorbar(np.arange(len(coord)),unumpy.nominal_values(fps),yerr=unumpy.std_devs(fps),ecolor='k',fmt=None,zorder=-40)
                elif (col_name == 'fringe_amplitude') & (fas is not None):
                    pl.plot(np.arange(len(coord)),unumpy.nominal_values(fas),'ks',label='FA incl. covariances',zorder=-40)
                    pl.errorbar(np.arange(len(coord)),unumpy.nominal_values(fas),yerr=unumpy.std_devs(fas),ecolor='k',fmt=None,zorder=-40)

            
                pl.legend(loc='best')
                pl.ylabel('%s' %(col_name))
                if (jj==1) & (attribute=='closure_quantities'):
                    pl.xlabel('Triangle number')
                elif (jj==1) & (attribute=='baseline_quantities'):
                    pl.xlabel('Baseline number')
                subplot_index +=1
        fig.tight_layout(h_pad=0.0)     
        pl.show()
        if save_plot == 1:
            figure_name = os.path.join(out_dir,'%s_%s.pdf' % (name_seed,'fringe_quantities'))
            pl.savefig(figure_name,transparent=True,bbox_inches='tight',pad_inches=0.05)            

        

    def get_average_quantities(self,number_format='f',soln_cov_orig=None, verbose=False):        
    
        values = []
        labels = []
        attributes = np.array(['closure_quantities','baseline_quantities'])
        for ii,attribute in enumerate(attributes):
            for jj,col_name in enumerate(getattr(self,attribute).colnames):       
                mean_value = np.mean(getattr(self,attribute)[col_name])
                std_value  = np.std( getattr(self,attribute)[col_name])
                if verbose:
                    if number_format=='f':
                        print('Average %s is %3.3f +/- %3.3f' % (col_name,mean_value,std_value))
                   
                values.append(mean_value)  
                labels.append('Mean '+col_name+'_simple')  
                values.append(std_value)  
                labels.append('RMS '+col_name+'_simple')  

        if soln_cov_orig is not None:
            fringe_phase_cov, fringe_amplitude_cov, closure_phase_cov, closure_amplitude_cov = u_get_phases_and_amplitudes(soln_cov_orig)
            
            for observable in ['fringe_phase_cov', 'fringe_amplitude_cov', 'closure_phase_cov', 'closure_amplitude_cov']:
                exec('data = %s'%observable)
                mean_value = np.mean( unumpy.nominal_values(data) )
                weighted_mean_value = np.average(unumpy.nominal_values(data), weights=unumpy.std_devs(data) )
                std_value  = unumpy.nominal_values( unumpy.sqrt(np.mean(np.abs(data - np.mean(data))**2)) )
                
                values.append(mean_value)  
                labels.append('Mean '+observable)  
                values.append(weighted_mean_value)  
                labels.append('Weighted mean '+observable)  
                values.append(std_value)  
                labels.append('RMS '+observable)  
                
        return np.array(values), np.array(labels)            
            
    

        
    def show_images(self,save_plot=0,out_dir='',stretch='log',name_seed=None):
        file_number = 0
        fig = pl.figure(figsize=(20, 7),facecolor='w', edgecolor='k'); pl.clf();
        images = np.array(['centered_file','modelsolution_file','residual_file'])
        image_title = np.array(['Data','Model','Residuals'])
        if name_seed is None:
            name_seed = os.path.basename(getattr(self,images[0])[file_number]).split('.')[0]
        
        for j,attribute_name in enumerate(images):
        
            file_image = getattr(self, attribute_name)
            file_image = file_image[file_number]
            
            data = fits.getdata(file_image)
            header = fits.getheader(file_image)
            gc = aplpy.FITSFigure(data, figure=fig, north=False,subplot=(1,3,j+1))
            vmin = np.min(data)
            vmax = np.max(data)
            vmid=vmin-1
            vmin = None
            gc.show_grayscale(invert = True,aspect=1,stretch=stretch,vmin=vmin,vmid=vmid,vmax=vmax)#, stretch='log', vmid=-1)#,vmid=-1)#, aspect = pixScaleAC_mas/pixScaleAL_mas, pmax =90 )
            pl.title(image_title[j])        
            gc.add_colorbar()   
            if j==0:                
              textsize = 12
              mycol = 'k'
              gc.add_label(0.05 , 0.91, '%s\n%s\n%s' % (header['INSTRUME'],header['FILTER'],header['PUPIL']), size =textsize, relative=True,horizontalalignment='left',color=mycol) #header['STARMAG']
            if j==2:                
                gc.add_label(0.05 , 0.91, '%s %3.3f' % ('RMS', np.std(data) ), size =textsize, relative=True,horizontalalignment='left',color=mycol) 

        fig.tight_layout(h_pad=0.0)      
        pl.show()
        if save_plot == 1:
            figure_name = os.path.join(out_dir,'%s_%s.pdf' % (name_seed,'images'))
            gc.save(figure_name,dpi=300)



def make_standard_image(file, image_title='', save_plot=0, plot_dir=None, name_seed=None, stretch='linear', x_axis_label=None, y_axis_label=None):
    '''

    :param file:
    :return:
    '''

    if name_seed is None:
        name_seed = os.path.basename(file).split('.')[0]

    fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k'); pl.clf();
    data = fits.getdata(file)
    header = fits.getheader(file)
    gc = aplpy.FITSFigure(data, figure=fig, north=False)
    vmin = np.min(data)
    vmax = np.max(data)
    vmid = vmin - 1
    vmin = None
    gc.show_grayscale(invert=True, aspect=1, stretch=stretch, vmin=vmin, vmid=vmid,
                      vmax=vmax)
    pl.title(image_title)
    gc.add_colorbar()
    textsize = 12
    mycol = 'k'
    if 0:
        gc.add_label(0.05, 0.91, '%s\n%s\n%s' % (header['INSTRUME'], header['FILTER'], header['PUPIL']), size=textsize,
                     relative=True, horizontalalignment='left', color=mycol)  # header['STARMAG']
    if 1:
        gc.add_label(0.05, 0.91, '%s %3.3f' % ('RMS', np.std(data)), size=textsize, relative=True,
                     horizontalalignment='left', color=mycol)

    if x_axis_label is None:
        x_axis_label = 'PIXEL (AXIS1)'
    if y_axis_label is None:
        y_axis_label = 'PIXEL (AXIS2)'

    gc.axis_labels.set_xtext(x_axis_label)
    gc.axis_labels.set_ytext(y_axis_label)


    fig.tight_layout(h_pad=0.0)
    # pl.show()

    figure_name = os.path.join(plot_dir, '%s_%s.pdf' % (name_seed, 'standard_image'))
    if save_plot == 1:
        gc.save(figure_name, dpi=300)

    return figure_name



class FringeFitterResult(object):
    """a structure class for nrm_core.FringeFitter.fit_fringes results"""
    def __init__(self, data_dir ):
        self.data_dir = data_dir              # directory containing the results

        print('Loading results saved in %s' % (self.data_dir))
        # text files        
        self.solutions_file  = sorted(glob.glob(os.path.join(self.data_dir,'%s*.txt' % 'solutions_' )));
        self.amplitudes_file = sorted(glob.glob(os.path.join(self.data_dir,'%s*.txt' % 'amplitudes_' )));
        self.CA_file         = sorted(glob.glob(os.path.join(self.data_dir,'%s*.txt' % 'CAs_' )));
        self.CP_file         = sorted(glob.glob(os.path.join(self.data_dir,'%s*.txt' % 'CPs_' )));
        self.phases_file     = sorted(glob.glob(os.path.join(self.data_dir,'%s*.txt' % 'phases_' )));
    
        # fits images
        self.centered_file   = sorted(glob.glob(os.path.join(self.data_dir,'%s*.fits' % 'centered_' )));
        self.modelsolution_file = sorted(glob.glob(os.path.join(self.data_dir,'%s*.fits' % 'modelsolution_' )));
        self.residual_file   = sorted(glob.glob(os.path.join(self.data_dir,'%s*.fits' % 'residual_' )));
        
        NINT = len(self.solutions_file)
        self.NINT = NINT

        # object arrays to hold results of each integration     
        integration = np.ndarray((self.NINT,),dtype=np.object)

        for j,file_number in enumerate(np.arange(self.NINT)):
            solutions = Table.read(self.solutions_file[file_number],format='ascii.no_header', guess=False,names=({'soln'}))
            CA = Table.read(self.CA_file[file_number],format='ascii.no_header',names=({'closure_amplitude'}))
            CP = Table.read(self.CP_file[file_number],format='ascii.no_header',names=({'closure_phase'}))
            closure_quantities = tablehstack((CA,CP))
            A = Table.read(self.amplitudes_file[file_number],format='ascii.no_header',names=({'fringe_amplitude'}))
            P = Table.read(self.phases_file[file_number],format='ascii.no_header',names=({'fringe_phase'}))
            baseline_quantities = tablehstack((A,P))
            integration[j] = NrmIntegrationResult( solutions , closure_quantities , baseline_quantities )

        self.integration = integration

        
    def print_results(self,valid_index = None,number_format='f'): 
        for j in np.arange(self.NINT):
            print('Integration %02d' % j)
            self.integration[j].print_results(valid_index = valid_index,number_format=number_format)           
        
