#import iso_handling
#import pal5_mock_ps
import scipy
import scipy.signal
import csv

labels=['CFHT','LSST 1-year','LSST 10-year','WFIRST']
colors=['seagreen','firebrick', 'slateblue', 'darkorange']

###################
# Start with CDM-like 
dire = '/Users/hendel/projects/streamgaps/streampepper/'

xi_wfirst = scipy.genfromtxt(dire+'csd_13323_5-9_003bg.dat', delimiter=',', loose=True, invalid_raise=False,skip_header=1)
xi_cfht =   scipy.genfromtxt(dire+'csd_3200_5-9_013bg.dat',  delimiter=',', loose=True, invalid_raise=False,skip_header=1)
xi_lsst =   scipy.genfromtxt(dire+'csd_5318_5-9_005bg.dat',  delimiter=',', loose=True, invalid_raise=False,skip_header=1)
xi_lsst10 = scipy.genfromtxt(dire+'csd_7101_5-9_004bg.dat',  delimiter=',', loose=True, invalid_raise=False,skip_header=1)

ratearrs = [xi_cfht[:,0],xi_lsst[:,0],xi_lsst10[:,0],xi_wfirst[:,0]]
dataarrs = [xi_cfht[:,1:],xi_lsst[:,1:],xi_lsst10[:,1:],xi_wfirst[:,1:]]

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


def compute_abc(data, dataarr, rates, eps=5., deps=.85, faccept=.05):
	sindx= (np.abs(dataarr[:,1]-data[1]) < eps)*(np.abs(dataarr[:,2]-data[2]) < eps)*(np.abs(dataarr[:,3]-data[3]) < (eps))\
							*(np.abs(dataarr[:,4]-data[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	while sum(sindx) > len(dataarr)*faccept:
		eps=eps*deps
		#print mod, eps, teps, sum(sindx)
		sindx= (np.abs(dataarr[:,1]-data[1]) < eps)*(np.abs(dataarr[:,2]-data[2]) < eps)*(np.abs(dataarr[:,3]-data[3]) < (eps))\
								*(np.abs(dataarr[:,4]-data[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	scale=1.
	kernel_width=.25
	rate_full= rates[sindx]
	#print len(rate_full)
	xxs= numpy.linspace(-1.5,1.,151)
	#kdey_full= kde_epanechnikov(xxs,kernel_width,numpy.log((1.6+rate_full)/(1.6-rate_full)))\
	kdey_full= kde_epanechnikov(xxs,kernel_width,numpy.log((1.35+(rate_full+.25))/(1.35-(rate_full+.25))))\
		+numpy.random.uniform(size=len(xxs))*0.000001
	kdey_full/= numpy.sum(kdey_full)*(xxs[1]-xxs[0])  
	return xxs, kdey_full


def compute_twostream_abc(data1, data2, dataarr, rates, eps=5., deps=.85, faccept=.05):
	sindx1= (np.abs(dataarr[:,1]-data1[1]) < eps)*(np.abs(dataarr[:,2]-data1[2]) < eps)*(np.abs(dataarr[:,3]-data1[3]) < (eps))\
							*(np.abs(dataarr[:,4]-data1[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	sindx2= (np.abs(dataarr[:,1]-data2[1]) < eps)*(np.abs(dataarr[:,2]-data2[2]) < eps)*(np.abs(dataarr[:,3]-data2[3]) < (eps))\
							*(np.abs(dataarr[:,4]-data2[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	sindx=(sindx1&sindx2)						
	while sum(sindx) > len(dataarr)*faccept:
		eps=eps*deps
		#print mod, eps, teps, sum(sindx)
		sindx1= (np.abs(dataarr[:,1]-data1[1]) < eps)*(np.abs(dataarr[:,2]-data1[2]) < eps)*(np.abs(dataarr[:,3]-data1[3]) < (eps))\
								*(np.abs(dataarr[:,4]-data1[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
		sindx2= (np.abs(dataarr[:,1]-data2[1]) < eps)*(np.abs(dataarr[:,2]-data2[2]) < eps)*(np.abs(dataarr[:,3]-data2[3]) < (eps))\
								*(np.abs(dataarr[:,4]-data2[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
		sindx=(sindx1&sindx2)		
	scale=1.
	kernel_width=.25
	rate_full= rates[sindx]
	#print len(rate_full)
	xxs= numpy.linspace(-1.5,1.,151)
	kdey_full= kde_epanechnikov(xxs,kernel_width,numpy.log((1.35+(rate_full+.25))/(1.35-(rate_full+.25))))\
		+numpy.random.uniform(size=len(xxs))*0.000001
	kdey_full/= numpy.sum(kdey_full)*(xxs[1]-xxs[0])  
	return xxs, kdey_full


def ecdf(data):
    xp = np.sort(data)
    yp = np.arange(1, len(data)+1) / len(data)
    def ecdf_instance(x):
        return np.interp(x, xp, yp, left=0, right=1)
    return ecdf_instance


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
#Power spectra
bins=np.linspace(0,15,151)
centroids = (bins[1:] + bins[:-1]) / 2.
px, py= scipy.signal.csd(np.ones(151),np.ones(151), fs=1./(centroids[1]-centroids[0]),scaling='spectrum',nperseg=len(centroids))
px= 1./px

plt.figure(figsize=(9,4))
#Compare CDM-like and much lower
plt.subplot(121)
for i, d in enumerate(dataarrs):
	plt.loglog(px,np.nanmedian(d[(ratearrs[i]> -.02)&(ratearrs[i]<.02)],axis=0), color = colors[i], label = labels[i])
	plt.fill_between(px, np.nanpercentile(d[(ratearrs[i]> -.02)&(ratearrs[i]<.02)], (25), axis=0),  y2= np.nanpercentile(d[(ratearrs[i]> -.02)&(ratearrs[i]<.02)], (75), axis=0), color=colors[i], alpha=0.2) 
plt.legend(loc='upper left')
plt.title('CDM rate ')
plt.xlabel(r'$k_{\xi}$ [deg]')#plt.xlabel('Scale [deg]')
plt.ylabel(r'$\sqrt{P_{\delta\delta}(k_{\xi})}$')#plt.ylabel('Power')
plt.ylim(0.05,0.8)

plt.subplot(122)
for i, d in enumerate(dataarrs):
	plt.loglog(px,np.nanmedian(d[(ratearrs[i]> -.42)&(ratearrs[i]<-.38)],axis=0), color = colors[i], label = labels[i]) 
	plt.fill_between(px, np.nanpercentile(d[(ratearrs[i]> -.42)&(ratearrs[i]<-.38)], (25), axis=0),  y2= np.nanpercentile(d[(ratearrs[i]> -.42)&(ratearrs[i]<-.38)], (75), axis=0), color=colors[i], alpha=0.2)
plt.legend(loc='upper left')
plt.title('0.4x CDM rate ')
plt.xlabel(r'$k_{\xi}$ [deg]')#plt.xlabel('Scale [deg]')
plt.ylabel(r'$\sqrt{P_{\delta\delta}(k_{\xi})}$')#plt.ylabel('Power')
plt.ylim(0.05,0.8)


plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/ps_vs_rate_comparison.png',dpi=300,bbox_inches='tight')

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
#Constraints by survey

#single stream PDFs
csvabc= open(dire+'5-9_constraints_onestream.csv','w')
abcwriter= csv.writer(csvabc, delimiter=',')
for n in np.arange(10000)*10:
	for i, dataarr in enumerate(dataarrs):
		data = dataarr[n]
		xxs, kdey_full = compute_abc(data, dataarr, ratearrs[i])
		#plt.plot(xxs,kdey_full,label=labels[i],color=colors[i])#,'-',lw=3.,color=color,zorder=zorder,overplot=overplot)

		try: 
			lowlim, bf, uplim = credible_interval(xxs, kdey_full)
			#print(labels[i]+': constrained, %1.3f, [%1.3f, %1.3f], %r'%(bf, lowlim, uplim, ((d[n,0]<uplim)&(d[n,0]>lowlim))))
			abcwriter.writerow(list(['constrained', format(ratearrs[i][n],'.4f'), format(bf,'.4f'), format(lowlim,'.4f'), format(uplim,'.4f'), ((ratearrs[i][n]<uplim)&(ratearrs[i][n]>lowlim))]))
			csvabc.flush()
		except:
			uplim_95 = upper_limit(xxs, kdey_full)
			#print(labels[i]+': upper limit, <%1.3f, %r'%(uplim_95, ((d[n,0]<uplim_95))))
			abcwriter.writerow(list(['upperlimit', format(ratearrs[i][n],'.4f'), '0', '0', format(uplim_95,'.4f'),((ratearrs[i][n]<uplim_95))]))
			csvabc.flush()

csvabc.close()



####save double stream PDFs
csvabc= open(dire+'5-9_constraints_twostream.csv','w')
abcwriter= csv.writer(csvabc,delimiter=',')
for n in np.arange(1000)*10:
	nrate, noff = np.divmod(n,10)
	for i, dataarr in enumerate(dataarrs):
		rates = ratearrs[i]
		ratesort = np.argsort(rates)
		unsort = np.argsort(ratesort)
		data1 = dataarr[n]
		data2 = dataarr[(ratesort[np.searchsorted(rates[ratesort],rates[nrate],side='right')])+noff]
		xxs, kdey_full = compute_twostream_abc(data1, data2, dataarr, ratearrs[i])

		try: 
			lowlim, bf, uplim = credible_interval(xxs, kdey_full)
			#print(labels[i]+': constrained, %1.3f, [%1.3f, %1.3f], %r'%(bf, lowlim, uplim, ((d[n,0]<uplim)&(d[n,0]>lowlim))))
			abcwriter.writerow(list(['constrained', format(ratearrs[i][n],'.4f'), format(bf,'.4f'), format(lowlim,'.4f'), format(uplim,'.4f'), ((ratearrs[i][n]<uplim)&(ratearrs[i][n]>lowlim))]))
			csvabc.flush()
		except:
			uplim_95 = upper_limit(xxs, kdey_full)
			#print(labels[i]+': upper limit, <%1.3f, %r'%(uplim_95, ((d[n,0]<uplim_95))))
			abcwriter.writerow(list(['upperlimit', format(ratearrs[i][n],'.4f'), '0', '0', format(uplim_95,'.4f'),((ratearrs[i][n]<uplim_95))]))
			csvabc.flush()

csvabc.close()


########################################
########################################
########################################
####Pal5 ABC plots

onestream_types =                       	scipy.genfromtxt(dire+'5-9_constraints_onestream.csv', unpack=True, delimiter=',', comments=None, usecols=(0), dtype=None)
onestream_types =							onestream_types.astype('str')
onestream_tf =                           	scipy.genfromtxt(dire+'5-9_constraints_onestream.csv', unpack=True, delimiter=',', comments=None, usecols=(5), dtype=None)
onestream_rates, onestream_maxs, onestream_lowlims, onestream_uplims = 	scipy.genfromtxt(dire+'5-9_constraints_onestream.csv', unpack=True, delimiter=',', comments=None, usecols=(1,2,3,4))

####Scatter true vs infered
plt.figure(figsize=(4,4))
plt.subplot(111,aspect='equal')
for i, dataarr in enumerate([xi_cfht[:,1:],xi_lsst[:,1:],xi_lsst10[:,1:],xi_wfirst[:,1:]]):
	plt.scatter(onestream_rates[i::4][onestream_types[i::4]=='constrained'], onestream_maxs[i::4][onestream_types[i::4]=='constrained'], 
		s=3, color = colors[i], label = labels[i], alpha = .3)
	#plt.errorbar(onestream_rates[i::4][onestream_types[i::4]=='upperlimit'],  onestream_uplims[i::4][onestream_types[i::4]=='upperlimit'], 
	#	uplims=True, yerr=.03, linestyle='none', marker=None, color = colors[i], alpha =0.3)
plt.legend(loc='upper left')
plt.plot([-1.5,1],[-1.5,1],c='k', lw=2)
plt.xlabel('True rate')
plt.ylabel('Inferred rate')
plt.title('One stream - Pal5')
for i, label in enumerate(labels):
	s_res = np.sqrt(np.sum((onestream_rates[i::4][onestream_types[i::4]=='constrained']- onestream_maxs[i::4][onestream_types[i::4]=='constrained'])**2)/(sum(onestream_types[i::4]=='constrained')-2))
	plt.text(1,-1.4+.1*i, label+' residual standard deviation = %1.2f dex'%(s_res), horizontalalignment='right', verticalalignment='center')
plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/onestream_accuracy.png',dpi=300,bbox_inches='tight')


####Histograms of constrained vs upper limits

rates = onestream_rates
types = onestream_types
tf = onestream_tf
plt.figure(figsize=(9,5))
ratebins = np.linspace(-1.5,1,25)
for i in np.arange(4):
	plt.subplot(2,2,i+1)
	h, be = np.histogram(rates, bins=ratebins)
	plt.hist(rates[i::4][(types[i::4]=='upperlimit' )&(tf[i::4]==True)], bins=ratebins, histtype='step', color=colors[i], label=(labels[i]+' limit'), lw=2,linestyle=':', cumulative=False)
	plt.hist(rates[i::4][(types[i::4]=='constrained')&(tf[i::4]==True)], bins=ratebins, histtype='step', color=colors[i], label=(labels[i]+' constraint'), lw=2, cumulative=False)
	#plt.hist(rates[i::4], weights=(.5*np.ones(len(rates[i::4]))), bins=ratebins, histtype='step', color='k', label=(labels[i]+' 50\%'), lw=1, cumulative=False)
	plt.hist(rates[i::4][tf[i::4]==False], bins=ratebins, histtype='step', color='r', label=(labels[i]+' error'), lw=1, cumulative=False)

	plt.legend(fontsize=8)
	if i in [2,3]:plt.xlabel('True impact rate')
	if i in [0,2]:plt.ylabel('Number of streams')
	plt.ylim(0,250)
plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/onestream_constraint.png',dpi=300,bbox_inches='tight')



####Determine where 50% are constrained
plt.figure(figsize=(4,4))
for i in np.arange(4):
	cnecdf = diff(ecdf(rates[i::4][(types[i::4]=='constrained' )&(tf[i::4]==True)])(np.linspace(-1.5,1.,51)))
	ulecdf = diff(ecdf(rates[i::4][(types[i::4]=='upperlimit' )&(tf[i::4]==True)])(np.linspace(-1.5,1.,51)))
	plt.plot(np.linspace(-1.5,1.,50),cnecdf/(ulecdf+cnecdf), lw=2, label=labels[i]+' %1.2f'%(min(np.linspace(-1.5,1.,50)[cnecdf/(ulecdf+cnecdf)>.5])) ,color = colors[i])
plt.legend()
plt.plot([-1.5,1],[.5,.5],c='k',linestyle=':')
plt.xlim(-1.5,1)
plt.xlabel('True impact rate')
plt.ylabel('Fraction constrained')
plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/onestream_expectedconstraint.png',dpi=300,bbox_inches='tight')



########### for 2 pal5s if interested
twostream_types =                       	scipy.genfromtxt(dire+'5-9_constraints_twostream.csv', unpack=True, delimiter=',', comments=None, usecols=(0), dtype=None)
twostream_types =							twostream_types.astype('str')
twostream_tf =                           	scipy.genfromtxt(dire+'5-9_constraints_twostream.csv', unpack=True, delimiter=',', comments=None, usecols=(5), dtype=None)
twostream_rates, twostream_maxs, twostream_lowlims, twostream_uplims = 	scipy.genfromtxt(dire+'5-9_constraints_twostream.csv', unpack=True, delimiter=',', comments=None, usecols=(1,2,3,4))


plt.subplot(122,aspect='equal')
#plt.subplot(111,aspect='equal')
for i, dataarr in enumerate([xi_cfht[:,1:],xi_lsst[:,1:],xi_lsst10[:,1:],xi_wfirst[:,1:]]):
	plt.scatter (twostream_rates[i::4][twostream_types[i::4]=='constrained'], twostream_maxs[i::4][twostream_types[i::4]=='constrained'], 
		s=3, color = colors[i], label = labels[i])
	#plt.errorbar(twostream_rates[i::4][twostream_types[i::4]=='upperlimit'],  twostream_uplims[i::4][twostream_types[i::4]=='upperlimit'], 
	#	uplims=True, yerr=.03, linestyle='none', marker=None, color = colors[i], alpha =0.3)
plt.legend()
plt.plot([-1.5,1],[-1.5,1],c='k', lw=2)
plt.xlabel('True rate')
plt.ylabel('Inferred rate')
plt.title('Two stream - Two Pal5s')