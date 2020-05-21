import os
import numpy as np
import pickle
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.actionAngle import estimateBIsochrone
from galpy.orbit import Orbit
from galpy.util import bovy_conversion, bovy_coords

from galpy.df import streamdf, streamgapdf
from streampepperdf import streampepperdf

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose

_DATADIR  =  os.environ['_FORECAST_DATA_DIR']
_LOCALDIR =  os.environ['_FORECAST_LOCAL_DIR']

R0,V0= 8., 220.

def parse_times(times,age):
    if 'sampling' in times:
        nsam= int(times.split('sampling')[0])
        return [float(ti)/bovy_conversion.time_in_Gyr(V0,R0)
                for ti in np.arange(1,nsam+1)/(nsam+1.)*age]
    return [float(ti)/bovy_conversion.time_in_Gyr(V0,R0)
            for ti in times.split(',')]

class stream_config():
    """
    Simple stream model configuration helper for automation
    
    """
    def __init__(self,obs=None, sigv=0.5, age=5.0, R=None, R_coord=None, R_name=None, ntimes='64sampling', leading=True, name=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize the DF of a stellar stream peppered with impacts

        INPUT:
            
            obs: galpy Orbit instance representing the stream's progenitor
            
            sigv: ~ internal velocity dispersion in km/s,
            
            age: stream age in Gyr
            
            R: array, a rotation matrix from R_coord to the stream coordinates
        
            R_coord: the Astropy frame R is trasforming from (probably coord.Galctic or coord.ICRS)
        
            R_name: string, a name for the new coordinate system   
            
            ntimes: number of times a subhalo could have interacted, evenly spaced through
            
            leading: if True, model the leading tail, otherwise model the trailing tail
            
        """
        self.obs=obs
        self.sigv=sigv
        self.age=age
        self.R=R
        self.R_coord=R_coord
        self.R_name=R_name
        self.ntimes=ntimes
        self.timpact=parse_times(ntimes, age)
        self.leading=leading
        self.name=name
    
    def load(self):
        """
        NAME:

           load

        PURPOSE:

           Load pickled streamdf model & add coordinate transformations to Astropy frame graph, if relevant

        INPUT:
            
            none
            
        OUTPUT:
        
            none
            
        """
        self.sdf={}
        if os.path.exists(_DATADIR+'model_pickles/'+self.name+'_'+self.ntimes+'_leading.pkl'):
            self.sdf['leading']  = pickle.load(
                open(_DATADIR+'model_pickles/'+self.name+'_'+self.ntimes+'_leading.pkl', 'rb'))
            print('Leading DF loaded')
        else: print('path not found!')
        if os.path.exists(_DATADIR+'model_pickles/'+self.name+'_'+self.ntimes+'_trailing.pkl'):
            self.sdf['trailing']  = pickle.load(
                open(_DATADIR+'model_pickles/'+self.name+'_'+self.ntimes+'_trailing.pkl', 'rb'))
            print('Trailing DF loaded')
        if (self.R is not None):
            add_stream_coord(self.R, self.R_coord, self.R_name, sdf=self)
            
            
        

class stream_coord(coord.BaseCoordinateFrame):
    """
    Based on the Sgr example in the Astropy docs, this creates a new 
    coordinate system for the stream model

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    Lambda : `~astropy.coordinates.Angle`, optional, must be keyword
        The longitude-like angle corresponding to the direction along Phoenix.
    Beta : `~astropy.coordinates.Angle`, optional, must be keyword
        The latitude-like angle corresponding to the direction perpendicular to Phoenix.
    distance : `Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.
    pm_Lambda_cosBeta : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion along the stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_Beta : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in Declination for this object (``pm_ra_cosdec`` must
        also be given).
    radial_velocity : :class:`~astropy.units.Quantity`, optional, must be keyword
        The radial velocity of this object.

    """

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'Lambda'),
            coord.RepresentationMapping('lat', 'Beta'),
            coord.RepresentationMapping('distance', 'distance')]
    }
    
    _default_wrap_angle = 180*u.deg
    
    def __init__(self, *args, **kwargs):
        wrap = kwargs.pop('wrap_longitude', True)
        super().__init__(*args, **kwargs)
        if wrap and isinstance(self._data, (coord.UnitSphericalRepresentation,
                                            coord.SphericalRepresentation)):
            self._data.lon.wrap_angle = self._default_wrap_angle

    def represent_as(self, base, s='base', in_frame_units=False):
        r = super().represent_as(base, s=s, in_frame_units=in_frame_units)
        r.lon.wrap_angle = self._default_wrap_angle
        return r

def setup_streammodel(
    obs=None,
    pot = MWPotential2014,
    leading=False,
    timpact=None,
    hernquist=True,
    age=5.,
    sigv=.5,
    singleImpact=False,
    length_factor=1.,
    vsun=[-11.1,V0+24.,7.25],
    b=None,
    **kwargs):
    '''
    NAME:

       setup_streammodel
    
    PURPOSE:

        Initialize a streamdf or streampepperdf instance of stellar stream, depending on its impact history

    INPUT:

        obs: Orbit instance for progenitor position

        pot: host potential
        
        age: stream age in Gyr
        
        sigv: ~ internal velocity dispersion in km/s, controls the stream length in proportion to the age
        
        b: fit parameter for the isochrone approximation, if None it is set automatically
        
        R, R_coord: R_name: a rotation matrix for transformation to stream coordinates,the frame they are
            transforming from, and a name for the new coordinate system
        
        custom_transform: depreciated, superseded by the Astropy implementation below

        leading: if True, use leading tail, use trailing tail otherwise

        hernquist: if True, use Hernquist spheres for subhalos; Plummer otherwise

        singleImpact: force use of the streamgapdf instead of streampepperdf

        length_factor: consider impacts up to length_factor x length of the stream

        streamdf kwargs
    
    OUTPUT:

       object

    HISTORY:

       2020-05-08 - Started - Hendel (UofT)

    

    '''

    #automatically set up potential model
    if b==None: 
        obs.turn_physical_off()
        b = estimateBIsochrone(pot, obs.R(), obs.z())
        obs.turn_physical_on()
        print('Using isochrone approxmation parameter of %1.3f, should typically be between 0.5 and 1'%b)
    aAI= actionAngleIsochroneApprox(pot=pot,b=b)
    
    if timpact is None:
        sdf= streamdf(sigv/V0,progenitor=obs,
                      pot=pot,aA=aAI,
                      leading=leading,nTrackChunks=11,
                      tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                      ro=R0,vo=V0,R0=R0,
                      vsun=vsun,
                      custom_transform=None)
    elif singleImpact:
        sdf= streamgapdf(sigv/V0,progenitor=obs,
                         pot=pot,aA=aAI,
                         leading=leading,nTrackChunks=11,
                         tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                         ro=R0,vo=V0,R0=R0,
                         vsun=vsun,
                         custom_transform=None,
                         timpact= 0.3/bovy_conversion.time_in_Gyr(V0,R0),
                         spline_order=3,
                         hernquist=hernquist,
                         impact_angle=0.7,
                         impactb=0.,
                         GM= 10.**-2./bovy_conversion.mass_in_1010msol(V0,R0),
                         rs= 0.625/R0,
                         subhalovel=np.array([6.82200571,132.7700529,14.4174464])/V0,
                         **kwargs)
    else:
        sdf= streampepperdf(sigv/V0,progenitor=obs,
                            pot=pot,aA=aAI,
                            leading=leading,nTrackChunks=101,
                            tdisrupt=age/bovy_conversion.time_in_Gyr(V0,R0),
                            ro=R0,vo=V0,R0=R0,
                            vsun=vsun,
                            custom_transform=None,
                            timpact=timpact,
                            spline_order=3,
                            hernquist=hernquist,
                            length_factor=length_factor)
    sdf.turn_physical_off()  
    return sdf


def add_stream_coord(R, R_coord, R_name, sdf=None):
    """
    NAME:

       setup_streammodel
    
    PURPOSE:

        Add an Astropy frame with a givens static matrix trasformation; it becomes part of the frame graph
        so it can be transformed to or from any other frame.

    INPUT:

        R: array, a rotation matrix from R_coord to the stream coordinates
        
        R_coord: the Astropy frame R is trasforming from (probably coord.Galctic or coord.ICRS)
        
        R_name: string, a name for the new coordinate system   
        
        sdf: object, if set add the transformation as an object method for convenience
    
    OUTPUT:

       none

    HISTORY:

       2020-05-09 - Started - Hendel (UofT)
    
    """
    newcoord = type(R_name, (stream_coord,), {})
    if sdf is not None:
        setattr(sdf,R_name,newcoord)
        setattr(sdf,'stream_coord',newcoord)

    @frame_transform_graph.transform(coord.StaticMatrixTransform, R_coord, newcoord)
    def Rcoord_to_streamcoord():
        """ Compute the transformation matrix from the input frame coordinates of R
            heliocentric Phoenix coordinates. R_coord should be e.g. coord.Galactic or coord.ICRS
        """
        return R

    @frame_transform_graph.transform(coord.StaticMatrixTransform, newcoord, R_coord)
    def streamcoord_to_Rcoord():
        """ Compute the transformation matrix from heliocentric stream coordinates to
            the input frame coordinates of R. R_coord should be e.g. coord.Galactic or coord.ICRS
        """
        return matrix_transpose(R)

    
    
################################################################################
################################################################################
################################################################################
# Subhalo impact model

def nsubhalo(m):
    return 0.3*(10.**6.5/m)
def rs(m,plummer=False,rsfac=1.):
    if plummer:
        return 1.62*rsfac/R0*(m/10.**8.)**0.5
    else:
        return 1.05*rsfac/R0*(m/10.**8.)**0.5
def dNencdm(sdf_pepper,m,Xrs=3.,plummer=False,rsfac=1.,sigma=120.):
    return sdf_pepper.subhalo_encounters(\
        sigma=sigma/V0,nsubhalo=nsubhalo(m),
        bmax=Xrs*rs(m,plummer=plummer,rsfac=rsfac))
def powerlaw_wcutoff(massrange,cutoff):
    accept= False
    while not accept:
        prop= (10.**-(massrange[0]/2.)+(10.**-(massrange[1]/2.)\
                         -10.**(-massrange[0]/2.))\
                   *np.random.uniform())**-2.
        if np.random.uniform() < np.exp(-10.**cutoff/prop):
            accept= True
    return prop/bovy_conversion.mass_in_msol(V0,R0)