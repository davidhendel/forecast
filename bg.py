import os
import sys
import numpy as np
import subprocess
_DATADIR  =  os.environ['_FORECAST_DATA_DIR']
_LOCALDIR =  os.environ['_FORECAST_LOCAL_DIR']


def run_galaxia(ebf_fname, param_fname, outputDir = './', appMagLimit=27, lat=0, lon=0, area=1, fSample=1, r_max=1000, make_fits=True):
    '''make galaxia parameter file; there are more options than used here if desired.
    be careful of huge output files! the defaults here give 1.7M at (lat, lon) = (45,120) but 100G at (0,0)!'''
    paramstring =    '''outputFile                          %s
outputDir                           %s
photoSys                            SDSS
magcolorNames                       g,g-r
appMagLimits[0]                     0
appMagLimits[1]                     %1.2f
absMagLimits[0]                     -1000
absMagLimits[1]                     1000
colorLimits[0]                      -1000
colorLimits[1]                      1000
geometryOption                      1
longitude                           %1.4f
latitude                            %1.4f
surveyArea                          %i
fSample                             %1.4f
popID                               -1
warpFlareOn                         1
seed                                12
r_max                               %1.0f
starType                            0
photoError                          0
'''%(ebf_fname,
    outputDir,
    appMagLimit,
    lon,
    lat,
    area,
    fSample,
    r_max)

    #write the parameter file
    with open(outputDir+param_fname, 'w') as param_file:
        param_file.write(paramstring)
    
    #spawn a subprocess to run galaxia
    completed = subprocess.run(['galaxia', '-r', outputDir+param_fname])
    print('returncode:', completed.returncode)
    
    if make_fits:
        print("The ebf package is (as of now) python 2 only! \\
                  To use this you will need to pip install it in python2!")
        subprocess.run(['python2','ebfconverter.py',outputDir+ebf_fname])


    
