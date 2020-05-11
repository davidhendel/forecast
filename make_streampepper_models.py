import sys
import numpy as np
from multiprocessing import Pool
import streammodel_util
from galpy.orbit import Orbit
from galpy.util import save_pickles
from galpy.util import bovy_conversion, bovy_coords
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
R0, V0= 8., 220.

'''
Call with e.g. python3 make_streampepper_models.py pal5 phx gd1

Scripting setup of the models used to calculate mock streams.
Using 64 time samplings is expensive but makes a big difference.
New streams can be added with additional configuration sections as below; 
changes needed are the orbit, age, sigv to reproduce your desired streams.

There is also the option to add some observational components; a stream 
coordinate frame can be defined (see streammodel_util.add_stream_coord) and 
leading and trailing observation windows in stream coordinates are allowed but
be cautious of coordinate wraps!

'''

configs = []

###############################
##Pal 5
##note that this is for the xi-eta system of Ibata et al. 2016
if 'pal5' in sys.argv:
    pal5name = 'pal5'
    pal5obs= Orbit([229.018,-0.124,23.2,-2.296,-2.257,-58.7],
                   radec=True,ro=R0,vo=V0,
                   solarmotion=[-11.1,24.,7.25])
    pal5sigv= 0.5
    pal5age = 5.0

    pal5R = np.array([[-0.65582036, -0.75491388, -0.00216421],
        [ 0.75491565, -0.6558219 ,  0.        ],
        [-0.00141933, -0.00163379,  0.99999766]])
    pal5R_coord = coord.ICRS
    pal5R_name = 'pal5_coord'

    pal5_config = streammodel_util.stream_config(obs=pal5obs, age=pal5age, sigv=pal5sigv, 
                                R=pal5R, R_coord=pal5R_coord, R_name=pal5R_name, 
                                ntimes='64sampling', name=pal5name)
    configs += [pal5_config]


###############################
##Phoenix
if 'phx' in sys.argv:
    phxname='phx'
    phxobs = Orbit([27.60969888973802*u.deg,
                    -43.54155350527743*u.deg,
                    17.624735997461556*u.kpc,
                    2.19712949699425679*(u.mas/u.yr),
                    -0.5240686072157521*(u.mas/u.yr),
                    19.93208180482703*(u.km/u.s)],radec=True,ro=8,vo=220, 
                   solarmotion=[-11.1,24.,7.25])
    phxsigv=0.76667
    phxage = 1.5

    phxR = np.array([[ 0.86864881, -0.49542835,  0.], 
                     [ 0.15106312,  0.26486333,  0.95237984], 
                     [ 0.47183597,  0.82728361, -0.30491417]])
    phxR_coord = coord.Galactic
    phxR_name = 'phx_coord'

    phx_config = streammodel_util.stream_config(obs=phxobs, age=phxage, sigv=phxsigv, 
                                R=phxR, R_coord=phxR_coord, R_name=phxR_name, 
                                ntimes='64sampling',name=phxname)
    configs += [phx_config]


###############################
##GD-1
if 'gd1' in sys.argv:
    gd1name='gd1'
    gd1obs= Orbit([1.56148083,0.35081535,-1.15481504,0.88719443,
                    -0.47713334,0.12019596],ro=8,vo=220,
                  solarmotion=[-11.1,24.,7.25])
    gd1sigv=0.1825
    gd1age=9.

    gd1R= np.array([[-0.4776303088, -0.1738432154, 0.8611897727],
                  [0.510844589, -0.8524449229, 0.111245042],
                  [0.7147776536, 0.4930681392, 0.4959603976]])
    gd1R_coord = coord.ICRS
    gd1R_name = 'gd1_coord'

    gd1_config = streammodel_util.stream_config(obs=gd1obs, age=gd1age, sigv=gd1sigv, 
                                R=gd1R, R_coord=gd1R_coord, R_name=gd1R_name, 
                                ntimes='64sampling',name=gd1name)
    configs += [gd1_config]

########################################################################
########################################################################
########################################################################


def save_stream_model_pickles(fname, config, leading):
    model = streammodel_util.setup_streammodel(obs=config.obs, age=config.age, sigv=config.sigv,
                                               timpact=config.timpact, leading=leading)
    save_pickles(fname, model)
    return

if __name__ == '__main__':
    nmodels = 2*len(configs) #for leading and trailing
    print('Streams:')
    for i in sys.argv: print(i)
    print('using ', nmodels, ' models/processes')
    p = Pool(nmodels)
    for i, config in enumerate(configs):
        print('working on '+config.name+' leading' )
        fname = config.name+'_'+config.ntimes+'_leading.pkl'
        p.apply_async(save_stream_model_pickles(fname, config, True))
        print('working on '+config.name+' trailing' )
        fname = config.name+'_'+config.ntimes+'_trailing.pkl'
        p.apply_async(save_stream_model_pickles(fname, config, False))    






