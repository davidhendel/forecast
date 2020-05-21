import os
import sys
import glob
import numpy as np
import scipy
from astropy.table import Table
from scipy.interpolate import interp1d
import dill
import string
import astropy.io.fits as pyfits
_ISOCHRONE_DIR  =  os.environ['_ISOCHRONE_DIR']
_DATADIR  =  os.environ['_FORECAST_DATA_DIR']
_LOCALDIR =  os.environ['_FORECAST_LOCAL_DIR']
        
def get_closest(array, values):
    #`values` should be sorted
    sortindex = np.argsort(values)
    svalues = values[sortindex]

    #make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, svalues, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(svalues - array[np.maximum(idxs-1, 0)]) < \
                                              np.fabs(svalues - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1

    #ii will return the indexes to the original order
    ii = values.argsort().argsort()

    return array[idxs[ii]]


def getIsoCurve(iso1, iso2, magstep=0.01):
    """
    Returns a (dense) list of points sampling along the isochrone

    Arguments:
    ---------
    iso1, iso2:
        interp1d instances from isodict
    magstep: float(optional)
        The step in magntidues along the isochrone
    Returns:
    -------
    gcurve,rcurve: Tuple of numpy arrays
        The tupe of arrays of magnitudes in g and r going along the isochrone
    """
    mini = iso1.x
    mag1 = iso1.y
    mag2 = iso2.y
    out1, out2 = [], []
    for i in range(len(mini) - 1):
        l_1, l_2, r_1, r_2 = mag1[i], mag2[i], mag1[i + 1], mag2[i + 1]
        maggap = max(abs(r_1 - l_1), abs(r_2 - l_2))
        if maggap < .2:
            npt = maggap / magstep + 2
            mag1grid = np.linspace(l_1, r_1, npt)
            mag2grid = np.linspace(l_2, r_2, npt)
            out1.append(mag1grid)
            out2.append(mag2grid)
        else: 
            npt = 2
            mag1grid = np.linspace(l_1, r_1, npt)
            mag2grid = np.linspace(l_2, r_2, npt)
            out1.append(mag1grid)
            out2.append(mag2grid)

    out1, out2 = [np.concatenate(_) for _ in [out1, out2]]
    return out1, out2

def betw(x, x1, x2): 
    return (x >= x1) & (x <= x2)

def calc_star_bg(stream_config, survey_config, 
                 iso = '-10.06-0.0008', magerror_mod=1., minerr=0.003, maxerr=0.1, thresh=2):
    dm = (5*np.log10(stream_config.obs.dist()*1e3)-5)
    F = pyfits.open(stream_config.bgfname)[1].data
    apmags = survey_config.get_mag(F['smass'], F['age'], F['feh'], rs = F['rad'], \
                              bands=survey_config.filters, interpdict = survey_config.isointerp)
    g, r = apmags[survey_config.filters[0]], apmags[survey_config.filters[1]]
    gc_isochrone1 = survey_config.isointerp[survey_config.filters[0]+iso]
    gc_isochrone2 = survey_config.isointerp[survey_config.filters[1]+iso]

    gcurve, rcurve = getIsoCurve(gc_isochrone1,gc_isochrone2)
    gcurve, rcurve = [_ + dm for _ in [gcurve, rcurve]]

    mincol, maxcol = -1., 2.
    minmag, maxmag = 17, 28
    colbin = 0.01
    magbin = 0.01

    colbins = np.arange(mincol, maxcol, colbin)
    magbins = np.arange(minmag, maxmag, magbin)

    grgrid, rgrid = np.mgrid[mincol:maxcol:colbin, minmag:maxmag:magbin]
    ggrid = grgrid + rgrid

    arr0 = np.array([(ggrid).flatten(), rgrid.flatten()]).T
    arr = np.array([gcurve, rcurve]).T
    tree = scipy.spatial.cKDTree(arr)
    D, xind = tree.query(arr0)
    
    gerr = survey_config.em[survey_config.filters[0]](ggrid.flatten()).reshape(ggrid.shape)
    rerr = survey_config.em[survey_config.filters[1]](ggrid.flatten()).reshape(ggrid.shape)

    maglim_g = survey_config.get_mag_limit(survey_config.filters[0])#getMagLimit('g', survey)
    maglim_r = survey_config.get_mag_limit(survey_config.filters[1])#getMagLimit('r', survey)

    errfactor=magerror_mod
    gerr = gerr*errfactor
    rerr = rerr*errfactor
    gerr, rerr = [np.maximum(_, minerr) for _ in [gerr, rerr]]

    dg = ggrid - gcurve[xind].reshape(ggrid.shape)
    dr = rgrid - rcurve[xind].reshape(rgrid.shape)


    mask = (np.abs(dg / gerr) < thresh) & (np.abs(dr / rerr) <
                                           thresh) & (rgrid < maglim_r) & (ggrid < maglim_g)

    colid = np.digitize(g - r, colbins) - 1
    magid = np.digitize(r, magbins) - 1
    xind = betw(colid, 0, grgrid.shape[0] - 1) & betw(magid, 0, grgrid.shape[1])
    xmask = np.zeros(len(g), dtype=bool)
    xmask[xind] = mask[colid[xind], magid[xind]]
    nbgstars = xmask.sum()
    bgdens = nbgstars

    return bgdens





class survey():
    def __init__(self, name, filters, isodirname, emfname, offsets=None):
        self.name = name
        self.isodirname = isodirname
        self.files = np.sort(glob.glob(isodirname+'/*'))
        self.zs = np.array([file.split('_')[1].split('.txt')[0] for file in self.files]).astype(float)
        example = scipy.genfromtxt(self.files[0], names = True, skip_header = 11)
        self.ages = np.unique(example['logAge'])
        self.filters=list(filters.keys())
        self.isofilterdict = filters
        self.offsets=offsets
        self.em = {}
        t = Table.read(emfname, format='ascii')
        for filt in filters:
            self.em[filt]=interp1d(t[filt],t[filt+'_err'],bounds_error=False,fill_value="extrapolate")
    
    def load_iso_interps(self, remake=False, save=False, maxlabel=2):
        
        if remake==False: 
            self.isointerp = dill.load( \
                open(_DATADIR+"iso_interps/"+self.name+"_iso_interp_dict_maxlabel%i.pkl"%maxlabel, "rb" ))
        
        if remake==True:
            interpdict={}
            for i, file in enumerate(self.files):
                isodata = scipy.genfromtxt(file, names = True, skip_header = 11)
                for j, age in enumerate(self.ages):
                    for band in self.filters:
                        sel = ((isodata['logAge']==age)&(isodata['label']<=maxlabel))
                        
                        interpdict[band+'-%1.2f'%(age)+'-%1.4f'%(float(self.zs[i]))] = \
                                interp1d(isodata['Mini'][sel], isodata[self.isofilterdict[band]][sel], \
                                fill_value=99, bounds_error=False)
                    if self.offsets is not None:
                        for band in self.filters:
                            b1 = interpdict[self.offsets[band][0]+'-%1.2f'%(age)+'-%1.4f'%(float(self.zs[i]))].y
                            b2 = interpdict[self.offsets[band][1]+'-%1.2f'%(age)+'-%1.4f'%(float(self.zs[i]))].y
                            interpdict[band+'-%1.2f'%(age)+'-%1.4f'%(float(self.zs[i]))] = \
                                interp1d(isodata['Mini'][sel], self.offsets[band][2](b1,b2), \
                                fill_value=99, bounds_error=False)

            self.isointerp=interpdict
        if save==True:
            dill.dump(interpdict,\
                      open(_DATADIR+"iso_interps/"+self.name+"_iso_interp_dict_maxlabel%i.pkl"%maxlabel,'wb'))
        
    def get_mag(self, masses, ages, fehs, rs = None, bands=None, interpdict=None, verbose=False):
        mags={}
        zs = 10**fehs*0.019
        for band in bands:
            mags[band] = np.zeros(len(masses))
            zs_to_use   = get_closest((self.zs), zs)
            ages_to_use = get_closest((self.ages), ages)
            for ii in self.ages:
                psel = (ages_to_use==ii)
                if np.sum(psel)>0:
                    for jj in self.zs:
                        sel = ((ages_to_use==ii)&(np.round(zs_to_use,4)==jj))
                        if verbose==True: print( ii, jj, np.sum(sel))
                        mags[band][sel] = interpdict[band+'-%1.2f'%(ii)+'-%1.4f'%jj](masses[sel])
            if rs is not None:
                mags[band] = mags[band] + (5*np.log10(rs*1e3)-5)
        return mags
    
    def get_mag_limit(self, band, maxerr=0.1):
        xgrid = np.linspace(self.em[band].x[0],self.em[band].x[-1], 1000)
        err = self.em[band](xgrid)
        xid = np.argmax(err * (err < maxerr))
        return xgrid[xid]
                
            
###############################
#SDSS
sdss_filters = {'g':'gmag', 'r':'rmag' }#{'umag':'u', 'gmag':'g', 'rmag':'r', 'imag':'i', 'zmag':'z'}
sdss_errormodel = _DATADIR+'survey_errormodels/sdss_errormodel.txt'
sdss_survey = survey('sdss', sdss_filters, _ISOCHRONE_DIR+'sdss2mass/', sdss_errormodel)

###############################
#CFHT - need something special for g-r correction
cfht_filters = {'g':'gmag', 'r':'rmag' }#{'umag':'u', 'gmag':'g', 'rmag':'r', 'imag':'i', 'zmag':'z'}
def cfht_g(g,r):
    return g-.185*(g-r)
def cfht_r(g,r):
    return r-0.024*(g-r)
cfht_offsets = {'g':['g', 'r', cfht_g],'r':['g','r', cfht_r]}
cfht_errormodel = _DATADIR+'survey_errormodels/cfht_errormodel.txt'
cfht_survey = survey('cfht', cfht_filters,_ISOCHRONE_DIR+'sdss2mass/', cfht_errormodel, offsets=cfht_offsets)

###############################
#DES
des_filters = {'g':'gmag', 'r':'rmag' }#{'umag':'u', 'gmag':'g', 'rmag':'r', 'imag':'i', 'zmag':'z', 'Ymag':'Y'}
des_errormodel = _DATADIR+'survey_errormodels/des_errormodel.txt'
des_survey = survey('des', des_filters, _ISOCHRONE_DIR+'DES/', des_errormodel)
       
###############################
#LSST
lsst_filters = {'g':'gmag', 'r':'rmag' }#{'umag':'u', 'gmag':'g', 'rmag':'r', 'imag':'i', 'zmag':'z', 'Ymag':'Y'}
lsst_errormodel = _DATADIR+'survey_errormodels/lsst_errormodel.txt'
lsst_survey = survey('lsst', lsst_filters, _ISOCHRONE_DIR+'lsstwfirst/', lsst_errormodel)

###############################
#LSST10
lsst10_filters = {'g':'gmag', 'r':'rmag' }#{'umag':'u', 'gmag':'g', 'rmag':'r', 'imag':'i', 'zmag':'z', 'Ymag':'Y'}
lsst10_errormodel = _DATADIR+'survey_errormodels/lsst10_errormodel.txt'
lsst10_survey = survey('lsst10', lsst10_filters, _ISOCHRONE_DIR+'lsstwfirst/', lsst10_errormodel)

###############################
#WFIRST
wfirst_filters = {'Z':'Z087mag', 'H':'H158mag'}
            #{'R062mag':'r', 'Z087mag':'z', 'Y106mag':'Y', 'J129mag':'J', 'H158mag':'H', 'F184mag':'F', 'W149mag':'W'}
wfirst_errormodel = _DATADIR+'survey_errormodels/wfirst_errormodel.txt'
wfirst_survey = survey('wfirst', wfirst_filters, _ISOCHRONE_DIR+'lsstwfirst/', wfirst_errormodel)