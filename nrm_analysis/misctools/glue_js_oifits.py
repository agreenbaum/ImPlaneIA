import numpy as np
import scipy
import SAM_pip.sam_tools as sam_tools
import SAM_pip.js_oifits as oifits

def weighted_avg_and_std(values, weights):
    import numpy as np
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


def ff(coeff, P2VM, BB):
    import numpy as np
    yy = np.dot(P2VM, coeff) - BB
    return np.dot(yy, yy)


"""
def red_SAM(data_file, PSF_im, uwindow, bl_x, bl_y, wave, nwave, filter_gain, hole_size, imsize, px_scale, nholes, nbl, ncp, \
           baselines, closure_phases, x, y, air, cube_bl, cube_bl_sin, xcoord, ycoord, source, bandwidth, instrument, \
           oversample, arrname = 'SIM'):
"""
def bls(ctrs):
    N = len(ctrs)
    nbl = N*(N-1)//2
    nbl = int(nbl)
    # labels uv points by holes they came from
    u = np.zeros(nbl)
    v = np.zeros(nbl)
    nn=0
    for ii in range(N-1):
        for jj in range(N-ii-1):
            u[nn+jj] = ctrs[ii,0] - ctrs[ii+jj+1,0]
            v[nn+jj] = ctrs[ii,1] - ctrs[ii+jj+1,1]
        nn = nn+jj+1
    return u,v

def write( obskeywords=None, 
           v2=None, v2err=None,
           cps=None, cperr=None, pha=None, phaerr=None, 
           wave=None, bandwidth=None, nwave=None,
           hole_size=None, nholes=None,
           ctrs=None):
    """
        obskeywords keys = { 'path' 'year' 'month' 'day' 'TEL' 'arrname' 'object' 'RA' 'DEC' 'PARANG'
                'PARANGRANGE' 'PA' 'phaseceil' 'covariance' 
         mask attributes: activeD ctrs hdia instrument 'NIRISS'  maskname 'jwst_g7s6c' 
        obskeywords
        object obj
	PARANG 0.0
	day 25
	TEL JWST
	arrname jwst_g7s6c
	path ../example_data/noise//
	year 2019
	PA 0
	month 9
	RA 0
        DEC 0
	PARANGRANGE 0.0
	phaseceil 100.0
    """
    bl_x, bl_y = ctrs[:,0], ctrs[:,1]
    nbl = int(scipy.misc.comb(nholes,2))
    ncp = int(scipy.misc.comb(nholes,3))
    print("comb nbl: ", nbl)
    print("comb ncp: ", ncp)
    baselines = bls(ctrs)
    closure_phases = cps
    ami2oif(bl_x, bl_y, wave, nwave, hole_size, nholes, nbl, ncp,
           baselines, closure_phases, ctrs[:,0], ctrs[:,1], obskeywords['object'], bandwidth, obskeywords["instrument"],
           obskeywords['PARANG'],
           v2, v2err, pha, phaerr, 
           arrname = obskeywords['arrname'])

def ami2oif(bl_x, bl_y, wave, nwave, hole_size, nholes, nbl, ncp,
           baselines, closure_phases, xcoord, ycoord, source, bandwidth, instrument,
           parang,
           v2, v2err, pha, phaerr, 
           arrname):

    from skimage.restoration import unwrap_phase
    import astropy.io.fits as pyfits
    import matplotlib.pyplot as plt
    import pdb; pdb.set_trace()
    from astropy.time import Time
    from datetime import datetime
    import scipy.optimize as optimize

    VIS_aver = np.sqrt(v2)
    VIS_aver_err = v2err / (2 * VIS_aver)  # if y=x^2; y + dy = x^2 + 2*x*dx;  so dx = y/(2x) 
    PHASE_aver = pha
    PHASE_aver_err = phaerr
    # number of slices in cube - use nwave for starters to get oifits written
    sz = 1 #np.shape(data_cube)

    mean_parang = parang

    bl_xp = bl_x * np.cos(np.deg2rad(mean_parang)) - bl_y * np.sin(np.deg2rad(mean_parang))
    bl_yp = bl_x * np.sin(np.deg2rad(mean_parang)) + bl_y * np.cos(np.deg2rad(mean_parang))
    bl_x1_cp, bl_y1_cp, bl_x2_cp, bl_y2_cp, t3amp_mod, t3phi_mod = sam_tools.compute_closure_phases(nbl, ncp, baselines, closure_phases, \
                                                                               bl_xp, bl_yp, np.sqrt(v2), \
                                                                               PHASE_aver)
    ###########################################
    ####### Save the OIFITS file ##############

    oi_file = oifits.oifits()
    if arrname == 'SIM':
        oi_arrname = instrument+'_'+arrname
        oi_target = source+'_'+arrname
        oi_ra = 0.0
        oi_dec = 0.0
        oi_pmra = 0.0
        oi_pmdec = 0.0
        oi_time = np.zeros([nbl]) + 2000.0
        tjd = Time(datetime.now().strftime('%Y-%m-%d'))
        oi_dateobs = datetime.now().strftime('%Y-%m-%d')
        oi_mjd = np.zeros([nbl]) + tjd.mjd
        oi_inttime = np.zeros([nbl]) + 10.0
        oi_visflag = np.zeros([nbl], dtype='i1')
        oi_v2flag = np.zeros([nbl], dtype='i1')
        oi_targetid = np.zeros([nbl]) + 1
        oi_t3targetid = np.zeros([ncp]) + 1
        oi_t3flag = np.zeros([ncp], dtype='i1')
        oi_t3inttime = np.zeros([ncp]) + 10.0
        oi_t3time = np.zeros([ncp]) + 2000.0
        oi_t3mjd = np.zeros([ncp]) + tjd.mjd
        catg = 'SIM_DATA'
        equinox = 2000.0
        radvel = 0.0
        parallax = 0.0

    oi_telname = np.zeros([nholes], dtype='S16')
    oi_staname = np.zeros([nholes], dtype='S16')
    oi_staindex = np.zeros([nholes], dtype='>i2')
    oi_sta_coord = np.zeros([nholes, 3], dtype='>f8')
    oi_size = np.zeros([nholes], dtype='f4')
    for i in range(nholes):
        oi_telname[i] = 'Hole' + str(i + 1)
        oi_staname[i] = 'P' + str(i + 1)
        oi_staindex[i] = i + 1
        oi_size[i] = hole_size
        oi_sta_coord[i, 0] = xcoord[i, 0]
        oi_sta_coord[i, 1] = ycoord[i, 0]
        oi_sta_coord[i, 2] = 0

    oi_file.array = oifits.OI_ARRAY(1, oi_arrname, 'GEOCENTRIC', 0.,0.,0.,oi_telname, oi_staname, oi_staindex, oi_size, oi_sta_coord)

    oi_file.target = oifits.OI_TARGET(1, np.array([1]), np.array([oi_target]), np.array([oi_ra]), np.array([oi_dec]), \
                                      np.array([equinox]), np.array([0.]), np.array([0.]), np.array([radvel]), \
                                      np.array(['UNKNOWN']), np.array(['OPTICAL']), np.array([oi_pmra]), np.array([oi_pmdec]), \
                                      np.array([0.]), np.array([0.]), np.array([parallax]), \
                                      np.array([0.]), np.array(['UNKNOWN']))
    oi_file.wavelength = oifits.OI_WAVELENGTH(1,instrument, np.array([wave]), np.array([bandwidth]))
    oi_file.vis = oifits.OI_VIS(1, oi_dateobs, oi_arrname, instrument, oi_targetid, oi_time, oi_mjd, oi_inttime, VIS_aver, \
                                VIS_aver_err, PHASE_aver, PHASE_aver_err, bl_xp, bl_yp, baselines+1, oi_visflag)

    oi_file.vis2 = oifits.OI_VIS2(1, oi_dateobs, oi_arrname, instrument, oi_targetid, oi_time, oi_mjd, oi_inttime, V2_aver,\
                                  V2_aver_err, bl_xp, bl_yp, baselines+1, oi_v2flag)

    oi_file.t3 = oifits.OI_T3(1, oi_dateobs, oi_arrname, instrument, oi_t3targetid, oi_t3time, oi_t3mjd, oi_t3inttime, \
                              T3AMP_aver, T3AMP_aver_err, CP_aver, CP_aver_err, bl_x1_cp, bl_y1_cp, bl_x2_cp, bl_y2_cp,\
                              closure_phases+1, oi_t3flag)

    oi_file.write(catg+'_uncalib_'+data_file[:-5]+'.oifits')
    return
