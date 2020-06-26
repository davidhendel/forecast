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
	parser.add_option("--infile",dest='infilename',
					  help="Name of input file, i.e. rows of binned star counts")
	parser.add_option("--csdfile",dest='csdfilename',
					  help="Name of output file for computed power spectra")
	parser.add_option("--nstreams",dest='nstreams',default=None,
					  help="Number of rows to run; default = all")
	parser.add_option("--resamples",dest='resamples',default=1,
					  help="Number of instatiations of background counts")
	parser.add_option("--nbg",dest='nbg',default=0,
					  help="background counts per bin")
	parser.add_option("--polydeg",dest='polydeg',default=1,
					  type='int',
					  help="Polynomial order to fit to smooth stream density")
	return parser

#for rate in [-1.5,1]
def kde_epanechnikov(x,h,ydata):
	"""ydata= ln[(1+xdata)/(1-xdata)]"""
	h= numpy.ones_like(x)*h
	h[x < -1.]= h[x < -1.]*(-1.*(x[x < -1.]+1.)+1.) # use slightly wider kernel at small values
	y= numpy.log((1.35+(x+.25))/(1.35-(x+.25)))
	#r= numpy.fabs(numpy.tile(y,(len(ydata),1)).T-ydata)/h
	r= numpy.fabs(numpy.tile(y,(len(ydata),1)).T-ydata)/numpy.tile(h,(len(ydata),1)).T
	r[r > 1.]= 1. # Gets around multi-D slicing
	return numpy.sum(0.75*(1.-r**2.),axis=1)/h*(1./(1.35+(x+.25))+1./(1.35-(x+.25)))

def compute_abc(data, dataarr, eps=5., deps=.85, faccept=.05):
	sindx= (np.abs(dataarr[:,1]-data[1]) < eps)*(np.abs(dataarr[:,2]-data[2]) < eps)*(np.abs(dataarr[:,3]-data[3]) < (eps))\
							*(np.abs(dataarr[:,4]-data[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	while sum(sindx) > len(d)*faccept:
		eps=eps*deps
		#print mod, eps, teps, sum(sindx)
		sindx= (np.abs(dataarr[:,1]-data[1]) < eps)*(np.abs(dataarr[:,2]-data[2]) < eps)*(np.abs(dataarr[:,3]-data[3]) < (eps))\
								*(np.abs(dataarr[:,4]-data[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	scale=1.
	kernel_width=.25
	rate_full= dataarr[sindx][:,0]
	#print len(rate_full)
	xxs= numpy.linspace(-1.5,1.,151)
	#kdey_full= kde_epanechnikov(xxs,kernel_width,numpy.log((1.6+rate_full)/(1.6-rate_full)))\
	kdey_full= kde_epanechnikov(xxs,kernel_width,numpy.log((1.35+(rate_full+.25))/(1.35-(rate_full+.25))))\
		+numpy.random.uniform(size=len(xxs))*0.000001
	kdey_full/= numpy.sum(kdey_full)*(xxs[1]-xxs[0])  
	return xxs, kdey_full

def upper_limit(xxs, kdey_full):
	cp= np.cumsum(kdey_full)/numpy.sum(kdey_full)
	cp = cp+(.000001*(np.arange(len(kdey_full))))
	uplim_95= scipy.interpolate.InterpolatedUnivariateSpline(cp,xxs,k=1)(0.95)
	return uplim_95

def credible_interval(xxs, kdey_full, frac=.68):
	# Get peak and 68% around the peak
	kdey_full = kdey_full + (.000001*np.flip(np.arange(len(xxs))))
	bf= xxs[numpy.argmax(kdey_full)]
	if np.sum(kdey_full[((xxs<bf)&(xxs>(np.min(xxs)+.05)))]<(.5*numpy.max(kdey_full)))==0:
		raise NameError('Not two-sided enough')
	sindx= numpy.argsort(-kdey_full) # minus reverses sort
	cp= numpy.cumsum((kdey_full/numpy.sum(kdey_full))[sindx])
	m= xxs[sindx][cp > frac]
	uplim= numpy.amin(m[m > bf])
	lowlim= numpy.amax(m[m < bf])
	return lowlim, bf, uplim


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




			data = dataarr[n]
			rates = rates+[data[0]]
			xxs, kdey_full = compute_abc(data, dataarr)
			#plt.plot(xxs,kdey_full,label=labels[i],color=colors[i])#,'-',lw=3.,color=color,zorder=zorder,overplot=overplot)
			try: 
				lowlim, bf, uplim = credible_interval(xxs, kdey_full)
				#print(labels[i]+': constrained, %1.3f, [%1.3f, %1.3f], %r'%(bf, lowlim, uplim, ((d[n,0]<uplim)&(d[n,0]>lowlim))))
				abcwriter.writerow(list(['constrained', format(data[0],'.4f'), format(bf,'.4f'), format(lowlim,'.4f'), format(uplim,'.4f'), ((d[n,0]<uplim)&(d[n,0]>lowlim))]))
				csvabc.flush()
			except:
				uplim_95 = upper_limit(xxs, kdey_full)
				#print(labels[i]+': upper limit, <%1.3f, %r'%(uplim_95, ((d[n,0]<uplim_95))))
				abcwriter.writerow(list(['upperlimit', format(data[0],'.4f'), '0', '0', format(uplim_95,'.4f'),((d[n,0]<uplim_95))]))
				csvabc.flush()
		csvabc.close()


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










