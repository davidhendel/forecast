import iso_handling
import pal5_mock_ps
import scipy
model = pal5_mock_ps.pal5_stream_instance()
#model.add_smooth()
#model.sample()
#model.assign_masses()

errormodels = iso_handling.load_errormodels()
isodict = iso_handling.load_iso_interps(remake=False)
try: 
	bgdict = np.load('/Users/hendel/projects/streamgaps/streampepper/bgdict.npy').item()
except:
	bgdict = iso_handling.gen_bg_counts_interp(isodict=isodict,errormodels=errormodels, verbose=True)
	#bg_interp = iso_handling.gen_bg_counts_interp(surveys=['SDSS','CFHT','LSST','LSST10','CASTOR'], 
	# bands = ['gr','gr','gr','gr','ug'], isodict=isodict, errormodels=errormodels)

######################################################
######################################################
######################################################
#Figure 1: compare error models
plt.figure(figsize=(4,4))
mags = np.linspace(18,28,1000)
surveys=['CFHT', 'LSST', 'LSST10', 'WFIRST']
bands={'CFHT':['g','r'],'LSST':['g','r'],'LSST10':['g','r'],'WFIRST':['z','h']}
colors=['seagreen','firebrick', 'slateblue', 'darkorange']
linestyles = ['-',':']
linewidths = [2,2,2,2]
for i, key in enumerate(surveys):
	for j, band in enumerate(bands[key]):
		errs = errormodels[key][band](mags)
		plt.plot(mags[errs<.2], errs[errs<.2], label=(key+' '+band),color=colors[i], linestyle=linestyles[j], linewidth=linewidths[i])
plt.plot([18,28],[.2,.2],c='k', lw=1, linestyle='--')
plt.text(21.5,.2,r'$\mathrm{5\sigma\ detection}$', horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', edgecolor='none',alpha=1))
plt.plot([18,28],[.1,.1],c='k', lw=1, linestyle='--')
plt.text(21.5,.1,r'$\mathrm{10\sigma\ detection}$', horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', edgecolor='none',alpha=1))
plt.xlim(20,28)
plt.ylim(0,.28)
plt.xlabel('Magnitude [mag]')
plt.ylabel('Magnitude uncertainty [mag]')
plt.legend(loc='upper right', ncol=2, fontsize='small')
plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/fig_photerr.png',dpi=300,bbox_inches='tight')

######################################################
######################################################
######################################################
#Figure 1: fraction of stars with xi<12.6
plt.figure(figsize=(4,4))
d = np.loadtxt('./data/abcsamples/samp_10k_5-9_000bg_cdm.dat',skiprows=0,delimiter=',', max_rows=1)
counts = np.loadtxt('./data/abcsamples/samp_10k_5-9_000bg_cdm.dat',skiprows=1,delimiter=',')
plt.hist(np.sum(counts[:,0:127],axis=1)/10000.,bins=np.linspace(0,1,100),density=True,color='0.1')
plt.xlim(0.4,0.9)
plt.xlabel(r'$\mathrm{Fraction\ of\ stars\ with\ \xi<12.6^\circ}$')
plt.ylabel(r'$\mathrm{PDF}$')
plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/fig_fstars.png',dpi=300,bbox_inches='tight')


######################################################
######################################################
######################################################
#Stream and background counts
#g = gSDSS - 0.185(gSDSS - rSDSS)
import scipy
import iso_handling
errormodels = iso_handling.load_errormodels()
isodict = iso_handling.load_iso_interps(remake=False, maxlabel=4)
dm = (5*np.log10(23.2*1e3)-5)
ms = np.linspace(0,0.8273221254,1000)
gSDSS=isodict['sdss_g-10.06-0.0008'](ms) + dm
rSDSS=isodict['sdss_r-10.06-0.0008'](ms) + dm
gCFHT = gSDSS - 0.185*(gSDSS - rSDSS)
mag_lim_gSDSS = gSDSS[np.argmax(gCFHT<23.5)]
m_lim_CFHT = ms[np.argmax(gCFHT<23.5)]

maglim_gLSST  =iso_handling.getMagLimit('g',survey='LSST')
maglim_rLSST  =iso_handling.getMagLimit('r',survey='LSST')
maglim_gLSST10=iso_handling.getMagLimit('g',survey='LSST10')
maglim_rLSST10=iso_handling.getMagLimit('r',survey='LSST10')
maglim_hWFIRST=iso_handling.getMagLimit('h',survey='WFIRST')
maglim_zWFIRST=iso_handling.getMagLimit('z',survey='WFIRST')

gLSST  =isodict['lsst_g-10.06-0.0008'](ms) + dm
rLSST  =isodict['lsst_r-10.06-0.0008'](ms) + dm
gLSST10=isodict['lsst_g-10.06-0.0008'](ms) + dm
rLSST10=isodict['lsst_r-10.06-0.0008'](ms) + dm
hWFIRST=isodict['wfirst_h-10.06-0.0008'](ms) + dm
zWFIRST=isodict['wfirst_z-10.06-0.0008'](ms) + dm

m_lim_gLSST  = ms[np.argmax(gLSST   < maglim_gLSST  )]
m_lim_rLSST  = ms[np.argmax(rLSST   < maglim_rLSST  )]
m_lim_gLSST10= ms[np.argmax(gLSST10 < maglim_gLSST10)]
m_lim_rLSST10= ms[np.argmax(rLSST10 < maglim_rLSST10)]
m_lim_hWFIRST= ms[np.argmax(hWFIRST < maglim_hWFIRST)]
m_lim_zWFIRST= ms[np.argmax(zWFIRST < maglim_zWFIRST)]

m_lim_LSST  =np.max((m_lim_gLSST,m_lim_rLSST))
m_lim_LSST10=np.max((m_lim_gLSST10,m_lim_rLSST10))
m_lim_WFIRST=np.max((m_lim_hWFIRST,m_lim_zWFIRST))

CFHT_integral   = scipy.integrate.quad(lambda x: x**(-0.5),m_lim_CFHT,  0.8273221254)[0]
LSST_integral   = scipy.integrate.quad(lambda x: x**(-0.5),m_lim_LSST,  0.8273221254)[0]
LSST10_integral = scipy.integrate.quad(lambda x: x**(-0.5),m_lim_LSST10,0.8273221254)[0]
WFIRST_integral = scipy.integrate.quad(lambda x: x**(-0.5),m_lim_WFIRST,0.8273221254)[0]

galbg = iso_handling.gen_gal_counts_interp(dms =[16.,dm],surveys=['CFHT', 'DES', 'LSST', 'LSST10', 'WFIRST'],
	bands=['gr', 'gr', 'gr', 'gr', 'zh'], isodict=isodict, errormodels=errormodels,verbose=True)
starbg = iso_handling.gen_bg_counts_interp(dms=[16.,dm],surveys=['CFHT', 'DES', 'LSST', 'LSST10', 'WFIRST'], 
	bands=['gr', 'gr', 'gr', 'gr', 'zh'], isodict=isodict, errormodels=errormodels,verbose=True)

#isodict_max2 = iso_handling.load_iso_interps(remake=True, maxlabel=2)
#galbg_max2 = iso_handling.gen_gal_counts_interp(dms =[16.,dm],surveys=['CFHT', 'LSST', 'LSST10', 'WFIRST'],
#	bands=['gr', 'gr', 'gr', 'zh'], isodict=isodict_max2, errormodels=errormodels,verbose=True)
#starbg_max2 = iso_handling.gen_bg_counts_interp(dms=[16.,dm],surveys=['CFHT', 'LSST', 'LSST10', 'WFIRST'], 
#	bands=['gr', 'gr', 'gr', 'zh'], isodict=isodict_max2, errormodels=errormodels,verbose=True)


data = [
('CFHT',   2000*1.6*CFHT_integral/CFHT_integral  , np.round(starbg['CFHT']((dm,23.5          ))+galbg['CFHT']((dm,23.5          )))),
('LSST',   2000*1.6*LSST_integral/CFHT_integral  , np.round(starbg['LSST']((dm,maglim_gLSST  ))+galbg['LSST']((dm,maglim_gLSST  )))),
('LSST 10',2000*1.6*LSST10_integral/CFHT_integral, np.round(starbg['LSST10']((dm,maglim_gLSST10))+galbg['LSST10']((dm,maglim_gLSST10)))),
('WFIRST', 2000*1.6*WFIRST_integral/CFHT_integral, np.round(starbg['WFIRST']((dm,maglim_zWFIRST))+galbg['WFIRST']((dm,maglim_zWFIRST))))
]

from astropy.table import Table
table = Table(rows=data,names = ('Survey', 'Samples', 'Background'), dtype=('S','i','i'))
table.write('/Users/hendel/projects/streamgaps/streampepper/paper_figures/samptable_cmd33.tex',format='latex',overwrite=True)




#####################################################
#####################################################
#####################################################
#plot CMDs for CFHT, LSST, LSST10, WFIRST
from scipy.interpolate import interp1d
isodata = scipy.genfromtxt('/Users/hendel/data/isochrones/newsdsspal5.txt', names = True, skip_header = 11)
sel = ((isodata['logAge']==10.06005))#&(isodata['label']<=maxlabel))
pal5sdssiso_g = interp1d(isodata['Mini'][sel], isodata['gmag'][sel], fill_value=99, bounds_error=False)
pal5sdssiso_r = interp1d(isodata['Mini'][sel], isodata['rmag'][sel], fill_value=99, bounds_error=False)

surveys=['CFHT', 'LSST', 'LSST10', 'WFIRST']

model = pal5_mock_ps.pal5_stream_instance()
model.sample(nsample=13323)
model.assign_masses(n=13323, maxmass=pal5sdssiso_g.x[-2])

plt.figure(figsize=(9,5))
plt.subplot(1,4,1)
mlim = m_lim_CFHT
mag1 = isodict['cfht_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['CFHT']['g'](mag1)
mag2 = isodict['cfht_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['CFHT']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
plt.scatter(omag1-omag2, omag1, c='k', alpha=0.3,s=1)
plt.ylim(28,14)
plt.xlabel('g-r')
plt.ylabel('g')
plt.xlim(-0,1.5)
plt.gca().set_title('CFHT')

plt.subplot(1,4,2)

mlim = m_lim_LSST
mag1 = isodict['lsst_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['LSST']['g'](mag1)
mag2 = isodict['lsst_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['LSST']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
plt.scatter(omag1-omag2, omag1, c='k', alpha=0.3,s=1)
plt.ylim(28,14)
plt.xlabel('g-r')
plt.ylabel('g')
plt.xlim(0,1.5)
plt.gca().set_title('LSST')

plt.subplot(1,4,3)
mlim = m_lim_LSST10
mag1 = isodict['lsst_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['LSST10']['g'](mag1)
mag2 = isodict['lsst_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['LSST10']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
plt.scatter(omag1-omag2, omag1, c='k', alpha=0.3,s=1)
plt.ylim(28,14)
plt.xlabel('g-r')
plt.ylabel('g')
plt.xlim(0,1.5)
plt.gca().set_title('LSST 10')

plt.subplot(1,4,4)
mlim = m_lim_WFIRST
mag1 = isodict['wfirst_h-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['WFIRST']['h'](mag1)
mag2 = isodict['wfirst_z-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['WFIRST']['z'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
plt.scatter(omag2-omag1, omag1, c='k', alpha=0.3,s=1)
plt.ylim(28,14)
plt.xlabel('z-h')
plt.ylabel('h')
plt.xlim(.25,1.75)
plt.gca().set_title('WFIRST')
plt.subplots_adjust(wspace=.3)

plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/fig_cmds.png',dpi=300,bbox_inches='tight')

#####################################################
#####################################################
#####################################################
#plot sky histograms for CFHT, LSST, LSST10, WFIRST

model.sample(nsample=13323)
model.assign_masses(n=13323, maxmass=isodict['cfht_g-10.06-0.0008'].x[-2])

from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, (ax1, ax2,ax3,ax4) = plt.subplots(ncols=1, nrows=4, figsize=(3,8))

#plt.figure(figsize=(4,8))
#plt.subplot(4,1,1)
mlim = m_lim_CFHT
mag1 = isodict['cfht_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['CFHT']['g'](mag1)
mag2 = isodict['cfht_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['CFHT']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
xi  = model.spxi[:,0][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
eta = model.spxi[:,1][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
print xi.shape
ax1.hist2d(
	np.concatenate((xi,15*numpy.random.uniform(size=15*7*844))),
	np.concatenate((eta,7*numpy.random.uniform(size=15*7*844))),
	bins=[np.linspace(0,15,150),np.linspace(0,7,70)],
	vmin=0,vmax=40, cmap='gray')
ax1.set_aspect('equal', adjustable='box')
ax1.set_xticks([])
#ax1.set_yticks([])
ax1.set_ylabel(r'$\eta$ [deg]')
ax1.text(1,5,'CFHT',color='w', bbox=dict(facecolor='k', alpha=0.5))

#plt.subplot(4,1,2)
mlim = m_lim_LSST
mag1 = isodict['lsst_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['LSST']['g'](mag1)
mag2 = isodict['lsst_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['LSST']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
xi  = model.spxi[:,0][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
eta = model.spxi[:,1][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
print xi.shape
ax2.hist2d(
	np.concatenate((xi,15*numpy.random.uniform(size=15*7*324))),
	np.concatenate((eta,7*numpy.random.uniform(size=15*7*324))),
	bins=[np.linspace(0,15,150),np.linspace(0,7,70)],
	vmin=0,vmax=40, cmap='gray')
ax2.set_aspect('equal', adjustable='box')
ax2.set_xticks([])
#plt.gca().set_yticks([])
ax2.set_ylabel(r'$\eta$ [deg]')
ax2.text(1,5,'LSST',color='w', bbox=dict(facecolor='k', alpha=0.5))

#plt.subplot(4,1,3)
mlim = m_lim_LSST10
mag1 = isodict['lsst_g-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['LSST10']['g'](mag1)
mag2 = isodict['lsst_r-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['LSST10']['r'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
xi  = model.spxi[:,0][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
eta = model.spxi[:,1][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
print xi.shape
ax3.hist2d(
	np.concatenate((xi,15*numpy.random.uniform(size=15*7*245))),
	np.concatenate((eta,7*numpy.random.uniform(size=15*7*245))),
	bins=[np.linspace(0,15,150),np.linspace(0,7,70)],
	vmin=0,vmax=40, cmap='gray')
ax3.set_aspect('equal', adjustable='box')
ax3.set_xticks([])
#plt.gca().set_yticks([])
ax3.set_ylabel(r'$\eta$ [deg]')
ax3.text(1,5,'LSST 10',color='w', bbox=dict(facecolor='k', alpha=0.5))

#plt.subplot(4,1,4)
mlim = m_lim_WFIRST
mag1 = isodict['wfirst_h-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror1 = errormodels['WFIRST']['h'](mag1)
mag2 = isodict['wfirst_z-10.06-0.0008'](model.masses[model.masses>mlim]) + (5*np.log10(model.spc.distance.value[model.masses>mlim]*1e3)-5)
magerror2 = errormodels['WFIRST']['z'](mag2)
omag1 = (mag1+np.random.normal(size=len(mag1))*magerror1)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
omag2 = (mag2+np.random.normal(size=len(mag2))*magerror2)[((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
xi  = model.spxi[:,0][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
eta = model.spxi[:,1][model.masses>mlim][((magerror1<.1)&(magerror2<.1)&(model.spxi[:,0][model.masses>mlim]<15.))]
print xi.shape
h,xe,ye,im = ax4.hist2d(
	np.concatenate((xi,15*numpy.random.uniform(size=15*7*187))),
	np.concatenate((eta,7*numpy.random.uniform(size=15*7*187))),
	bins=[np.linspace(0,15,150),np.linspace(0,7,70)],
	vmin=0,vmax=40, cmap='gray')
ax4.set_aspect('equal', adjustable='box')
#plt.gca().set_xticks([])
ax4.set_xlabel(r'$\xi$ [deg]')
#plt.gca().set_yticks([])
ax4.set_ylabel(r'$\eta$ [deg]')
ax4.text(1,5,'WFIRST',color='w', bbox=dict(facecolor='k', alpha=0.5))
plt.subplots_adjust(hspace=.01)
cbar=fig.colorbar(im,  ax=[ax1,ax2,ax3,ax4], orientation='horizontal', pad=.1)
cbar.set_label('Counts')
plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/fig_hists_cbar.png',dpi=300,bbox_inches='tight')


#####################################################
#####################################################
#####################################################
#Power spectra

def xi_csd(d, bins=np.linspace(0,15,151), nbg=0., binned=False):
	from scipy import signal
	from numpy.polynomial import Polynomial
	np.random.seed(seed=42)
	if binned==False:
		actcounts, bins = np.histogram(d, bins=bins)
		counts = actcounts + np.random.poisson(nbg, size=len(d))
	else:counts = d + np.random.poisson(nbg, size=len(d))
	counts = np.maximum(counts - nbg, np.zeros_like(counts))
	centroids = (bins[1:] + bins[:-1]) / 2.
	err = np.sqrt(counts+nbg)#*(counts/actcounts)
	bkg=0
	degree=1
	pp=Polynomial.fit(centroids,counts,degree,w=1./numpy.sqrt(counts+1.))
	tdata= counts/pp(centroids)
	terr = err/pp(centroids)
	px, py= signal.csd(tdata,tdata, fs=1./(centroids[1]-centroids[0]),scaling='spectrum',nperseg=len(centroids))
	py= py.real
	px= 1./px
	py= numpy.sqrt(py*(centroids[-1]-centroids[0]))

	nerrsim= 1000
	ppy_err= numpy.empty((nerrsim,len(px)))
	for ii in range(nerrsim):
		tmock= terr*numpy.random.normal(size=len(centroids))
		ppy_err[ii]= signal.csd(tmock,tmock,
								fs=1./(centroids[1]-centroids[0]),scaling='spectrum',
								nperseg=len(centroids))[1].real
	py_err= numpy.sqrt(numpy.median(ppy_err,axis=0)*(centroids[-1]-centroids[0]))
	#np.save('/Users/hendel/Desktop/pscrosscheck_xi_csd.npy',(d,bins,counts,tdata,terr,px,py,py_err))
	return px,py,py_err

nruns = 10
#xi_cfht =   scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_3200_7-9_008bg.dat',delimiter=',', loose=True, invalid_raise=False)
#xi_lsst =   scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_6144_7-9_006bg.dat',delimiter=',', loose=True, invalid_raise=False)
#xi_lsst10 = scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_8245_7-9_004bg.dat',delimiter=',', loose=True, invalid_raise=False)
#xi_wfirst = scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_15577_7-9_004bg.dat',delimiter=',',loose=True, invalid_raise=False)
#samp_cfht = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_3200_7-9_008bg.dat',delimiter=',', skiprows=1, max_rows=10000)
#samp_lsst = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_6144_7-9_006bg.dat',delimiter=',', skiprows=1, max_rows=10000)
#samp_lsst10 = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_8245_7-9_004bg.dat',delimiter=',', skiprows=1, max_rows=10000)
#samp_wfirst = np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_15577_7-9_004bg.dat',delimiter=',', skiprows=1, max_rows=10000)

xi_cfht =   scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_3200_5-9_013bg.dat',delimiter=',', loose=True, invalid_raise=False)
xi_lsst =   scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_5318_5-9_005bg.dat',delimiter=',', loose=True, invalid_raise=False)
xi_lsst10 = scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_7101_5-9_004bg.dat',delimiter=',', loose=True, invalid_raise=False)
xi_wfirst = scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/abc_13323_5-9_003bg.dat',delimiter=',',loose=True, invalid_raise=False)
samp_cfht =       np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_3200_5-9_013bg.dat',delimiter=',', skiprows=1)
samp_lsst =       np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_5318_5-9_005bg.dat',delimiter=',', skiprows=1)
samp_lsst10 =     np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_7101_5-9_004bg.dat',delimiter=',', skiprows=1)
samp_wfirst =     np.loadtxt('/Users/hendel/projects/streamgaps/streampepper/data/abcsamples/samp_13323_5-9_003bg.dat',delimiter=',', skiprows=1)


px,py,py_err = xi_csd(samp_cfht[3,1:],binned=True)

# Do a Epanechnikov KDE estimate of the PDF in the transformed y=(1+x)/(1-x) space
#old, wrong
def kde_epanechnikov(x,h,ydata):
	"""ydata= ln[(1+xdata)/(1-xdata)]"""
	h= numpy.ones_like(x)*h
	h[x < -0.5]= h[x < -0.5]*(-2.*(x[x < -0.5]+0.5)+1.) # use slightly wider kernel at small values
	y= numpy.log((1.6+x)/(1.6-x))
	#r= numpy.fabs(numpy.tile(y,(len(ydata),1)).T-ydata)/h
	r= numpy.fabs(numpy.tile(y,(len(ydata),1)).T-ydata)/numpy.tile(h,(len(ydata),1)).T
	r[r > 1.]= 1. # Gets around multi-D slicing
	return numpy.sum(0.75*(1.-r**2.),axis=1)/h*(1./(1.6+x)+1./(1.6-x))

#for rate in [-1,1]
def kde_epanechnikov(x,h,ydata):
    """ydata= ln[(1+xdata)/(1-xdata)]"""
    h= numpy.ones_like(x)*h
    h[x < -0.5]= h[x < -0.5]*(-2.*(x[x < -0.5]+0.5)+1.) # use slightly wider kernel at small values
    y= numpy.log((1.1+x)/(1.1-x))
    #r= numpy.fabs(numpy.tile(y,(len(ydata),1)).T-ydata)/h
    r= numpy.fabs(numpy.tile(y,(len(ydata),1)).T-ydata)/numpy.tile(h,(len(ydata),1)).T
    r[r > 1.]= 1. # Gets around multi-D slicing
    print x
    print h*(1./(1.1+x)+1./(1.1-x))
    return numpy.sum(0.75*(1.-r**2.),axis=1)/h*(1./(1.1+x)+1./(1.1-x))

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

def plot_abc_densities(n):
	plt.figure(figsize=(8,4))
	labels=['cfht','lsst','lsst10','wfirst']
	colors=['seagreen','firebrick', 'slateblue', 'darkorange']
	rates = samp_cfht[:,0]
	for i, d in enumerate([samp_cfht[:,1:],samp_lsst[:,1:],samp_lsst10[:,1:],samp_wfirst[:,1:]]):
		plt.plot(np.linspace(0,15,150),d[n],label=labels[i],color=colors[i])
	plt.legend()
	plt.xlim(0,15)
	#plt.ylim(.0,1.2)
	plt.xlabel(r'$\xi$')
	plt.ylabel('Counts')
	plt.title('n=%i, rate=%1.2f'%(n,rates[n]))

def plot_abc_ps(n, newplot=False, plotmedian=True,title=True):
	if newplot==True: plt.figure(figsize=(8,4))
	pxx = np.ones(76)*15./(np.arange(76))
	labels=['cfht','lsst','lsst10','wfirst']
	colors=['seagreen','firebrick', 'slateblue', 'darkorange']
	for i, d in enumerate([xi_cfht[:,1:],xi_lsst[:,1:],xi_lsst10[:,1:],xi_wfirst[:,1:]]):
		data = d[n,1:]#np.nanmedian(d[:,1:],axis=0)#d[n,1:]
		if newplot==True:
			plt.loglog(px,data, color = colors[i], label=labels[i],lw=2)
			if plotmedian==True:
				perc25, perc50, perc75 = np.nanpercentile(d[(d[:,0]<(d[n,0]+.1)) & (d[:,0]>(d[n,0]-.1))][:,1:], (25,50,75), axis=0)
				plt.loglog(px,perc50,color = colors[i], linestyle=':')
				plt.fill_between(px, perc25, y2=perc75, color=colors[i], alpha=0.2)
		else: plt.loglog(px,data, color = colors[i], label=labels[i])

	if newplot==True: plt.legend()
	#plt.ylim(.0,1.2)
	plt.xlabel(r'$k_{\xi}$ [deg]')#plt.xlabel('Scale [deg]')
	plt.ylabel(r'$\sqrt{P_{\delta\delta}(k_{\xi})}$')#plt.ylabel('Power')
	if newplot==True: 
		if title==True:plt.title('n=%i, rate=%1.2f'%(n,d[n,0]))












# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# compare varying lower mass limits
csd5 = scipy.genfromtxt('./csd_5318_5-9_005bg.dat',delimiter=',', loose=True, invalid_raise=False, skip_header=1)
csd6 = scipy.genfromtxt('./csd_5318_6-9_005bg.dat',delimiter=',', loose=True, invalid_raise=False, skip_header=1)
csd7 = scipy.genfromtxt('./csd_5318_7-9_005bg.dat',delimiter=',', loose=True, invalid_raise=False, skip_header=1)
csd8 = scipy.genfromtxt('./csd_5318_8-9_005bg.dat',delimiter=',', loose=True, invalid_raise=False, skip_header=1)
csd5n = np.hstack(((np.ones(len(csd5))*5).reshape(len(csd5),1),csd5))
csd6n = np.hstack(((np.ones(len(csd5))*6).reshape(len(csd5),1),csd6))
csd7n = np.hstack(((np.ones(len(csd5))*7).reshape(len(csd5),1),csd7))
csd8n = np.hstack(((np.ones(len(csd5))*8).reshape(len(csd5),1),csd8))

# compute abc - xxs, kdey_full = compute_abc(data, dataarr)
#def plot_abc_pdf_varylim_comparison(data, dataarr)
dataarrall = np.vstack((csd5n, csd6n, csd7n, csd8n))

def compute_abc_2d(data, dataarr, rates, mins, eps=5., deps=.85, faccept=.05):
	sindx= (np.abs(dataarr[:,1]-data[1]) < eps)*(np.abs(dataarr[:,2]-data[2]) < eps)*(np.abs(dataarr[:,3]-data[3]) < (eps))\
							*(np.abs(dataarr[:,4]-data[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	while sum(sindx) > len(dataarr)*faccept:
		eps=eps*deps
		#print mod, eps, teps, sum(sindx)
		sindx= (np.abs(dataarr[:,1]-data[1]) < eps)*(np.abs(dataarr[:,2]-data[2]) < eps)*(np.abs(dataarr[:,3]-data[3]) < (eps))\
								*(np.abs(dataarr[:,4]-data[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	rate_full= rates[sindx]
	#print len(rate_full)
	outbins = np.zeros((4,15))
	xxs = numpy.linspace(-1.5,1.,16)
	for i in np.arange(4):
		outbins[i] = np.histogram(rates[sindx][mins[sindx]==(i+5)], bins=xxs)[0]
	return outbins, sindx


def plot_abc_2d(n, dataarr= dataarrall[:,2:],rates= dataarrall[:,1], mins=dataarrall[:,0]):
	plt.clf()

	outbins, sindx = compute_abc_2d(dataarr[n,2:], dataarr, rates, mins)
	plt.subplot(131)
	plt.imshow(outbins, origin='lower', extent=[-1.5,1,5,9])
	plt.ylabel('Min msub')
	plt.xlabel('Rate')

	plt.subplot(132)
	cou,bins,ed =plt.hist(rates[sindx], bins= numpy.linspace(-1.5,1.,16), histtype='step', lw=4,color='k')
	for i in np.arange(4):
		plt.
		plt.hist(rates[sindx][(mins[sindx]==(i+5))], bins= numpy.linspace(-1.5,1.,16), histtype='step', lw=2)
	plt.xlabel('Rate')
	plt.plot([rates[n],rates[n]],[0,np.max(cou)], color='.5',linestyle=':')

	plt.subplot(133)
	cou,bins,ed = plt.hist(mins[sindx], bins= numpy.linspace(5,9,5),histtype='step', lw=4,color='k')
	#for i in np.arange(4):
	#	plt.hist(mins[sindx][mins[sindx]==(i+5)], bins= numpy.linspace(5,9,5), histtype='step', lw=2)
	plt.xlabel('Min msub')
	plt.plot([mins[n]+.5,mins[n]+.5],[0,np.max(cou)], color='.5',linestyle=':')


#check rates
for i in np.arange(4):
	plt.hist(rates[mins==(i+5)], bins= numpy.linspace(-1.5,1.,16), histtype='step', lw=2, label = '[%i,9]'%(i+5))
plt.xlabel('Rate')
plt.ylabel('Number in sample')
plt.legend()

rates = [71.09356623949175,21.990731280494593,6.463051476317431,1.5527679804177161]
for i in np.arange(4):
	rr = ((rates[i])*10**np.random.uniform(low=-1.5,high=1,size=100000))
	pp = np.random.poisson(rr)
	plt.hist(np.log10(rr[pp>0]/rates[i]), bins= np.linspace(-1.5,1.,16), histtype='step', lw=3)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

how does scale of consideration affect this?
kernel smooth in 1d?
seems unlkely that the min subhalo mass will be constrained
what is the diversity of slopes? will you ever get a closed contour?
two streams?

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

xxs, kdey_full_5 = compute_abc(csd5[0,1:], csd5[:,1:], csd5[:,0])
xxs, kdey_full_6 = compute_abc(csd5[0,1:], csd6[:,1:], csd6[:,0])
xxs, kdey_full_7 = compute_abc(csd5[0,1:], csd7[:,1:], csd7[:,0])
xxs, kdey_full_8 = compute_abc(csd5[0,1:], csd8[:,1:], csd8[:,0])
plt.plot(xxs, kdey_full_5, label = 'msub in [5,9]')
plt.plot(xxs, kdey_full_6, label = 'msub in [6,9]')
plt.plot(xxs, kdey_full_7, label = 'msub in [7,9]')
plt.plot(xxs, kdey_full_8, label = 'msub in [8,9]')
plt.legend()
plt.xlim(-1.5,1)
#plt.ylim(.0,1.2)
plt.xlabel('log10 rate / CDM rate')
plt.ylabel('PDF')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 











####save single stream PDFs
import csv
csvabc= open('/Users/hendel/projects/streamgaps/streampepper/all_abc_results_fixed.csv','w')
abcwriter= csv.writer(csvabc,delimiter=',')

for n in np.arange(9990)*10+10:
	for i, dataarr in enumerate([xi_cfht[:,1:],xi_lsst[:,1:],xi_lsst10[:,1:],xi_wfirst[:,1:]]):

		data = dataarr[n]
		rates = rates+[data[0]]
		xxs, kdey_full = compute_abc(data, dataarr)
		#plt.plot(xxs,kdey_full,label=labels[i],color=colors[i])#,'-',lw=3.,color=color,zorder=zorder,overplot=overplot)

		try: 
			lowlim, bf, uplim = credible_interval(xxs, kdey_full)
			#print(labels[i]+': constrained, %1.3f, [%1.3f, %1.3f], %r'%(bf, lowlim, uplim, ((d[n,0]<uplim)&(d[n,0]>lowlim))))
			abcwriter.writerow(list(['constrained', format(data[0],'.4f'), format(bf,'.4f'), format(lowlim,'.4f'), format(uplim,'.4f'), ((d[n,0]<uplim)&(d[n,0]>lowlim))]))
			csvabc.flush()
			vals = vals+[bf]
		except:
			uplim_95 = upper_limit(xxs, kdey_full)
			#print(labels[i]+': upper limit, <%1.3f, %r'%(uplim_95, ((d[n,0]<uplim_95))))
			abcwriter.writerow(list(['upperlimit', format(data[0],'.4f'), '0', '0', format(uplim_95,'.4f'),((d[n,0]<uplim_95))]))
			csvabc.flush()

csvabc.close()



####save double stream PDFs
import csv
csvabc= open('/Users/hendel/projects/streamgaps/streampepper/all_twostream_abc_results_fixed.csv','w')
abcwriter= csv.writer(csvabc,delimiter=',')

for n in np.arange(9990)*10+10:
	nrate, noff = np.divmod(n,10)
	rates = samp_cfht[:,0]
	ratesort = np.argsort(rates)
	unsort = np.argsort(ratesort)

	for i, dataarr in enumerate([xi_cfht[:,1:],xi_lsst[:,1:],xi_lsst10[:,1:],xi_wfirst[:,1:]]):
		data1 = dataarr[n]
		data2 = dataarr[(ratesort[np.searchsorted(rates[ratesort],rates[nrate],side='right')])*10+noff]
		xxs, kdey_full = compute_twostream_abc(data1, data2, dataarr)

		try: 
			lowlim, bf, uplim = credible_interval(xxs, kdey_full)
			#print(labels[i]+': constrained, %1.3f, [%1.3f, %1.3f], %r'%(bf, lowlim, uplim, ((d[n,0]<uplim)&(d[n,0]>lowlim))))
			abcwriter.writerow(list(['constrained', format(data1[0],'.4f'), format(bf,'.4f'), format(lowlim,'.4f'), format(uplim,'.4f'), ((d[n,0]<uplim)&(d[n,0]>lowlim))]))
			csvabc.flush()
			vals = vals+[bf]
		except:
			uplim_95 = upper_limit(xxs, kdey_full)
			#print(labels[i]+': upper limit, <%1.3f, %r'%(uplim_95, ((d[n,0]<uplim_95))))
			abcwriter.writerow(list(['upperlimit', format(data1[0],'.4f'), '0', '0', format(uplim_95,'.4f'),((d[n,0]<uplim_95))]))
			csvabc.flush()

csvabc.close()









types =                       	scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/all_abc_results_fixed.csv', unpack=True, delimiter=',', comments=None, usecols=(0), dtype=None)
types =							types.astype('str')
tf =                           	scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/all_abc_results_fixed.csv', unpack=True, delimiter=',', comments=None, usecols=(5), dtype=None)
rates, maxs, lowlims, uplims = 	scipy.genfromtxt('/Users/hendel/projects/streamgaps/streampepper/all_abc_results_fixed.csv', unpack=True, delimiter=',', comments=None, usecols=(1,2,3,4))


plt.subplot(111,aspect='equal')
labels=['cfht','lsst','lsst10','wfirst']
colors=['seagreen','firebrick', 'slateblue', 'darkorange']
for i, dataarr in enumerate([xi_cfht[:,1:],xi_lsst[:,1:],xi_lsst10[:,1:],xi_wfirst[:,1:]]):
	plt.scatter (rates[i::4][types[i::4]=='constrained'], vals[i::4][types[i::4]=='constrained'], s=3, color = colors[i], label = labels[i])
	plt.errorbar(rates[i::4][types[i::4]=='upperlimit'], vals[i::4][types[i::4]=='upperlimit'], uplims=True, yerr=.03, linestyle='none', marker=None, color = colors[i])
plt.legend()
plt.plot([-1.5,1],[-1.5,1],c='k', lw=2)
plt.xlabel('true rate')
plt.ylabel('inferred rate')


ratebins = np.linspace(-1.5,1,25)
labels=['cfht','lsst','lsst10','wfirst']
colors=['seagreen','firebrick', 'slateblue', 'darkorange']
for i in np.arange(4):
	plt.subplot(2,2,i+1)
	h, be = np.histogram(rates, bins=ratebins)
	plt.hist(rates[i::4][(types[i::4]=='upperlimit' )&(tf[i::4]==True)], bins=ratebins, histtype='step', color=colors[i], label=(labels[i]+' limit'), lw=2,linestyle=':', cumulative=False)
	plt.hist(rates[i::4][(types[i::4]=='constrained')&(tf[i::4]==True)], bins=ratebins, histtype='step', color=colors[i], label=(labels[i]+' constraint'), lw=2, cumulative=False)
	#plt.hist(rates[i::4], weights=(.5*np.ones(len(rates[i::4]))), bins=ratebins, histtype='step', color='k', label=(labels[i]+' 50\%'), lw=1, cumulative=False)
	plt.hist(rates[i::4][tf[i::4]==False], bins=ratebins, histtype='step', color='r', label=(labels[i]+' error'), lw=1, cumulative=False)

	plt.legend(fontsize=8)
	plt.xlabel('True impact rate')
	plt.ylabel('Number of streams')
	plt.ylim(0,650)


labels=['cfht','lsst','lsst10','wfirst']
colors=['seagreen','firebrick', 'slateblue', 'darkorange']
for i in np.arange(4):
	cnecdf = diff(ecdf(rates[i::4][(types[i::4]=='constrained' )&(tf[i::4]==True)])(np.linspace(-1.5,1.,51)))
	ulecdf = diff(ecdf(rates[i::4][(types[i::4]=='upperlimit' )&(tf[i::4]==True)])(np.linspace(-1.5,1.,51)))
	plt.plot(np.linspace(-1.5,1.,50),cnecdf/(ulecdf+cnecdf), lw=2, label=labels[i]+' %1.2f'%(min(np.linspace(-1.5,1.,50)[cnecdf/(ulecdf+cnecdf)>.5])) ,color = colors[i])
plt.legend()
plt.plot([-1.5,1],[.5,.5],c='k',linestyle=':')
plt.xlim(-1.5,1)
plt.xlabel('True impact rate')
plt.ylabel('Fraction constrained')

labels=['cfht','lsst','lsst10','wfirst']
colors=['seagreen','firebrick', 'slateblue', 'darkorange']
for i in np.arange(4):
	#plt.errorbar(rates[i::4][(types[i::4]=='constrained' )&(tf[i::4]==True)],maxs[i::4][(types[i::4]=='constrained' )&(tf[i::4]==True)])
	plt.scatter(
		  maxs[i::4][(types[i::4]=='constrained' )&(tf[i::4]==True)],
		uplims[i::4][(types[i::4]=='constrained' )&(tf[i::4]==True)]-lowlims[i::4][(types[i::4]=='constrained' )&(tf[i::4]==True)],
		s=1,c=colors[i], label=labels[i])
	#h,xe,ye = np.histogram2d(maxs[i::4][(types[i::4]=='constrained' )&(tf[i::4]==True)],
	#	uplims[i::4][(types[i::4]=='constrained' )&(tf[i::4]==True)]-lowlims[i::4][(types[i::4]=='constrained' )&(tf[i::4]==True)],bins=np.linspace(-1.5,1.5,50))

plt.legend()
plt.xlim(-1.5,1)
plt.xlabel('Measured rate')
plt.ylabel('68\% CL range [dex]')

def ecdf(data):
    xp = np.sort(data)
    yp = np.arange(1, len(data)+1) / len(data)
    def ecdf_instance(x):
        return np.interp(x, xp, yp, left=0, right=1)
    return ecdf_instance

def plot_abc_pdf(n, faccept=0.05):
	plt.figure(figsize=(4,4))
	labels=['cfht','lsst','lsst10','wfirst']
	colors=['seagreen','firebrick', 'slateblue', 'darkorange']
	for i, dataarr in enumerate([xi_cfht[:,1:],xi_lsst[:,1:],xi_lsst10[:,1:],xi_wfirst[:,1:]]):
		data = dataarr[n]
		xxs, kdey_full = compute_abc(data, dataarr)
		plt.plot(xxs,kdey_full,label=labels[i],color=colors[i])#,'-',lw=3.,color=color,zorder=zorder,overplot=overplot)
		try: 
			lowlim, bf, uplim = credible_interval(xxs, kdey_full)
			print(labels[i]+': constrained, %1.3f, [%1.3f, %1.3f], %r'%(bf, lowlim, uplim, ((d[n,0]<uplim)&(d[n,0]>lowlim))))
		except:
			uplim_95 = upper_limit(xxs, kdey_full)
			print(labels[i]+': upper limit, <%1.3f, %r'%(uplim_95, ((d[n,0]<uplim_95))))

	plt.legend()
	plt.xlim(-1.5,1)
	#plt.ylim(.0,1.2)
	plt.xlabel('log10 rate / CDM rate')
	plt.ylabel('PDF')
	plt.ylim(0,1.2)
	plt.title('n=%i, rate=%1.2f'%(n,d[n,0]))
	plt.plot([d[n,0],d[n,0]],[-10,10],c='k')

def compute_abc(data, dataarr, eps=5., deps=.85, faccept=.05):
	sindx= (np.abs(dataarr[:,1]-data[1]) < eps)*(np.abs(dataarr[:,2]-data[2]) < eps)*(np.abs(dataarr[:,3]-data[3]) < (eps))\
							*(np.abs(dataarr[:,4]-data[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	while sum(sindx) > len(dataarr)*faccept:
		eps=eps*deps
		#print mod, eps, teps, sum(sindx)
		sindx= (np.abs(dataarr[:,1]-data[1]) < eps)*(np.abs(dataarr[:,2]-data[2]) < eps)*(np.abs(dataarr[:,3]-data[3]) < (eps))\
								*(np.abs(dataarr[:,4]-data[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	scale=1.
	kernel_width=.25
	rate_full= dataarr[sindx][:,0]
	print 'len', len(rate_full)
	#print len(rate_full)
	xxs= numpy.linspace(-1.5,1.,151)
	#kdey_full= kde_epanechnikov(xxs,kernel_width,numpy.log((1.6+rate_full)/(1.6-rate_full)))\
	kdey_full= kde_epanechnikov(xxs,kernel_width,numpy.log((1.35+(rate_full+.25))/(1.35-(rate_full+.25))))\
		+numpy.random.uniform(size=len(xxs))*0.000001
	kdey_full/= numpy.sum(kdey_full)*(xxs[1]-xxs[0])  
	return xxs, kdey_full


def plot_twostream_abc_pdf(n, faccept=0.05):
	nrate, noff = np.divmod(n,10)
	rates = samp_cfht[:,0]
	ratesort = np.argsort(rates)
	unsort = np.argsort(ratesort)
	plt.figure(figsize=(4,4))
	labels=['cfht','lsst','lsst10','wfirst']
	colors=['seagreen','firebrick', 'slateblue', 'darkorange']
	for i, dataarr in enumerate([xi_cfht[:,1:],xi_lsst[:,1:],xi_lsst10[:,1:],xi_wfirst[:,1:]]):
		data1 = dataarr[n]
		data2 = dataarr[(ratesort[np.searchsorted(rates[ratesort],rates[nrate],side='right')])*10+noff]
		xxs, kdey_full = compute_twostream_abc(data1, data2, dataarr)
		plt.plot(xxs,kdey_full,label=labels[i],color=colors[i])#,'-',lw=3.,color=color,zorder=zorder,overplot=overplot)

		try: 
			lowlim, bf, uplim = credible_interval(xxs, kdey_full)
			print(labels[i]+': constrained, %1.3f, [%1.3f, %1.3f], %r'%(bf, lowlim, uplim, ((d[n,0]<uplim)&(d[n,0]>lowlim))))
		except:
			uplim_95 = upper_limit(xxs, kdey_full)
			print(labels[i]+': upper limit, <%1.3f, %r'%(uplim_95, ((d[n,0]<uplim_95))))

	plt.legend()
	plt.xlim(-1.5,1)
	#plt.ylim(.0,1.2)
	plt.xlabel('log10 rate / CDM rate')
	plt.ylabel('PDF')
	plt.ylim(0,1.2)
	plt.title('n=%i, rate=%1.2f'%(n,d[n,0]))
	plt.plot([d[n,0],d[n,0]],[-10,10],c='k')

def compute_twostream_abc(data1, data2, dataarr, eps=5., deps=.85, faccept=.05):
	sindx1= (np.abs(dataarr[:,1]-data1[1]) < eps)*(np.abs(dataarr[:,2]-data1[2]) < eps)*(np.abs(dataarr[:,3]-data1[3]) < (eps))\
							*(np.abs(dataarr[:,4]-data1[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	sindx2= (np.abs(dataarr[:,1]-data2[1]) < eps)*(np.abs(dataarr[:,2]-data2[2]) < eps)*(np.abs(dataarr[:,3]-data2[3]) < (eps))\
							*(np.abs(dataarr[:,4]-data2[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
	sindx=(sindx1&sindx2)						
	while sum(sindx) > len(d)*faccept:
		eps=eps*deps
		#print mod, eps, teps, sum(sindx)
		sindx1= (np.abs(dataarr[:,1]-data1[1]) < eps)*(np.abs(dataarr[:,2]-data1[2]) < eps)*(np.abs(dataarr[:,3]-data1[3]) < (eps))\
								*(np.abs(dataarr[:,4]-data1[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
		sindx2= (np.abs(dataarr[:,1]-data2[1]) < eps)*(np.abs(dataarr[:,2]-data2[2]) < eps)*(np.abs(dataarr[:,3]-data2[3]) < (eps))\
								*(np.abs(dataarr[:,4]-data2[4]) < (eps))#*(np.abs(d[:,5]-data[4]) < eps)
		sindx=(sindx1&sindx2)		
	scale=1.
	kernel_width=.25
	rate_full= dataarr[sindx][:,0]
	#print len(rate_full)
	xxs= numpy.linspace(-1.5,1.,151)
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



#wfirst ~ -.79, lsst10 ~-.65, lsst ~-.55

plt.savefig('/Users/hendel/projects/streamgaps/streampepper/paper_figures/fig_abc1.png',dpi=300,bbox_inches='tight')








plt.loglog(px,np.nanmedian(xi_cfht[:,2:],axis=0),  marker='o',markersize=5, label ='CFHT', color = 'orange') 
eps = np.nanmedian(np.nanmedian(xi_cfht[:,2:],axis=0)[20:-20])
plt.plot([.001,100],[eps,eps],linestyle='--',  color = 'orange')
plt.fill_between(px, np.nanpercentile(xi_cfht[:,2:], (25), axis=0),  y2= np.nanpercentile(xi_cfht[:,2:], (75), axis=0), color='orange', alpha=0.2)
plt.loglog(px,np.nanmedian(xi_lsst[:,2:],axis=0),  marker='o',markersize=5, label ='LSST', color = 'seagreen') 
eps = np.nanmedian(np.nanmedian(xi_lsst[:,2:],axis=0)[20:-20])
plt.plot([.001,100],[eps,eps],linestyle='--',  color = 'seagreen')
plt.fill_between(px, np.nanpercentile(xi_lsst[:,2:], (25), axis=0),  y2= np.nanpercentile(xi_lsst[:,2:], (75), axis=0), color='seagreen', alpha=0.2)
plt.loglog(px,np.nanmedian(xi_lsst10[:,2:],axis=0),marker='o',markersize=5, label ='LSST 10', color = 'darkslateblue')
eps = np.nanmedian(np.nanmedian(xi_lsst10[:,2:],axis=0)[20:-20])
plt.plot([.001,100],[eps,eps],linestyle='--',  color = 'darkslateblue') 
plt.fill_between(px, np.nanpercentile(xi_lsst10[:,2:], (25), axis=0),  y2= np.nanpercentile(xi_lsst10[:,2:], (75), axis=0), color='darkslateblue', alpha=0.2)
plt.loglog(px,np.nanmedian(xi_wfirst[:,2:],axis=0),marker='o',markersize=5, label ='WFIRST', color = 'tomato') 
eps = np.nanmedian(np.nanmedian(xi_wfirst[:,2:],axis=0)[20:-20])
plt.plot([.001,100],[eps,eps],linestyle='--',  color = 'tomato')
plt.fill_between(px, np.nanpercentile(xi_wfirst[:,2:], (25), axis=0),  y2= np.nanpercentile(xi_wfirst[:,2:], (75), axis=0), color='tomato', alpha=0.2)
plt.legend(loc='upper left')
plt.ylabel('Density power')
plt.xlabel(r'$\xi$ [deg]')
plt.xlim(0.5,15)
plt.ylim(.06,.6)






