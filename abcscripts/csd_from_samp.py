import numpy
import numpy as np
import scipy
from optparse import OptionParser
import os, os.path
import glob
import csv
import time
import pickle
from optparse import OptionParser
from numpy.polynomial import Polynomial
from scipy import interpolate, signal
from galpy.util import save_pickles, bovy_conversion, bovy_coords
import simulate_streampepper
import bispectrum
import pal5_util
from gd1_util import R0,V0
_DATADIR  =  os.environ['_FORECAST_DATA_DIR']

#Run abc analysis on first n streams of an abcfile containing the power spectra

def get_options():
	usage = "usage: %prog [options]"
	parser = OptionParser(usage=usage)
	parser.add_option("--csdfile",dest='csdfilename',
					  help="Name of input file, i.e. rows of power spectra")
	parser.add_option("--csdfile",dest='csdfilename',
					  help="Name of output file for computed power spectra")
	parser.add_option("--nstreams",dest='nstreams',default=None,
					  help="Number of rows to run; default = all")
	parser.add_option("--resamples",dest='resamples',default=1,
					  help="Number of instatiations of background counts")
	parser.add_option("--polydeg",dest='polydeg',default=1,
					  type='int',
					  help="Polynomial order to fit to smooth stream density")
	return parser


if __name__ == '__main__':
	parser= get_options()
	options,args= parser.parse_args()
	samp_data = scipy.genfromtxt(_DATADIR+options.infilename,delimiter=',', loose=True, invalid_raise=False)
	rates = samp_data[1:,0]
	bins = samp_data[0]
	xixi = (bins[1:] + bins[:-1]) / 2.
	samp_arr = samp_data[1:,1:]
	csdfile = open(_DATADIR+options.csdfilename,'w')
	csdwriter= csv.writer(csdfile,delimiter=',')
	csdwriter.writerow(list(bins))
	csdfile.flush()

	for n in np.arange(int(options.nstreams)):
		for i in np.arange(int(options.resamples)):
			tdens = samp_arr[n] + np.random.poisson(int(options.nbg), size=len(samp_arr[n]))
			tdens = np.maximum(tdens - int(options.nbg), np.zeros_like(tdens))
			pp= Polynomial.fit(xixi,tdens,deg=int(options.polydeg),w=1./np.sqrt(tdens+1.))
			tdens = tdens/pp(xixi)
			# Compute power spectrum
			tcsd= signal.csd(tdens,tdens,fs=1./(xixi[1]-xixi[0]),
						scaling='spectrum',nperseg=len(xixi))[1].real
			power= np.sqrt(tcsd*(xixi[-1]-xixi[0]))
			write_row= [rates[n]]
			write_row.extend(list(power))
			csdwriter.writerow(write_row)
			csdfile.flush()


'''
if analyising line-by-line this could be helpful

def getstuff(filename, criterion):
	with open(filename, "rb") as csvfile:
		datareader = csv.reader(csvfile)
		yield next(datareader)  # yield the header row
		count = 0
		for row in datareader:
			if row[3] == criterion:
				yield row
				count += 1
			elif count:
				# done when having read a consecutive series of rows 
				return

def getdata(infile):
	for row in getline(infile):
		print(row[0], len(row))

def getline(infile):
	with open(infile,'r') as read_infile:
		datareader = csv.reader(read_infile)
		for row in datareader:
			if len(row)==78:
				yield row
'''










