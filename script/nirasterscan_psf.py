#!/usr/bin/env python

import os
import argparse
from argparse import ArgumentParser

import pandas as pd 
from scipy import interpolate

import numpy as np
import yaml 
import astropy.io.fits as fits

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy
from scipy.ndimage.filters import gaussian_filter

__author__ = 'Teruaki Enoto'
__version__ = '0.01'
# v0.01 : 2021-05-02 : test sctipt

### Read vignetting curve
df = pd.read_csv('data/nicer_vignetting_curve_1p4867keV.csv',
	names=['angle','area'])
#f_vig = interpolate.interp1d(df['angle'],df['area'],
#	kind="quadratic",fill_value=(46.4,0.0),bounds_error=False) 	
f_vig = interpolate.interp1d(df['angle'],df['area'],
	kind="linear",fill_value=(46.4,0.0),bounds_error=False) 	

class Hist1D(object):
	def __init__(self, nbins, xlow, xhigh):
		self.nbins = nbins
		self.xlow  = xlow
		self.xhigh = xhigh
		self.hist, self.edges = np.histogram([], bins=nbins, range=(xlow, xhigh))
		self.bins = (self.edges[:-1] + self.edges[1:]) / 2.

	def fill(self, value):
		hist, edges = np.histogram([value], bins=self.nbins, range=(self.xlow, self.xhigh))
		self.hist += hist

	@property
	def data(self):
		return self.bins, self.hist

class Hist2D(object):
	# https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
	def __init__(self,xnbins,xlow,xhigh,ynbins,ylow,yhigh):
		self.xnbins = xnbins
		self.xlow  = xlow
		self.xhigh = xhigh
		self.ynbins = ynbins
		self.ylow  = ylow
		self.yhigh = yhigh		
		self.hist, self.xedges, self.yedges = np.histogram2d([],[],
			bins=(self.xnbins,self.ynbins),
			range=((self.xlow,self.xhigh),(self.ylow,self.yhigh)))
		self.xbins = (self.xedges[:-1] + self.xedges[1:]) / 2.
		self.ybins = (self.yedges[:-1] + self.yedges[1:]) / 2.		

	def fill(self, xvalue, yvalue, weights=None):
		if weights == None:
			hist, xedges, yedges = np.histogram2d([xvalue],[yvalue], 
				bins=(self.xnbins,self.ynbins), 
				range=((self.xlow,self.xhigh),(self.ylow,self.yhigh)))
		else:
			hist, xedges, yedges = np.histogram2d([xvalue], [yvalue], 
				weights=[weights],
				bins=(self.xnbins,self.ynbins), 
				range=((self.xlow,self.xhigh),(self.ylow,self.yhigh)))			
		self.hist += hist

	def get_total_weights(self):
		total_weights = 0.0
		for i in range(len(self.hist)):
			for j in range(len(self.hist[i])):		
				total_weights += self.hist[i][j]
		return total_weights

	def normalized(self,norm=1.0):
		total_weights = self.get_total_weights()
		for i in range(len(self.hist)):
			for j in range(len(self.hist[i])):		
				self.hist[i][j] = norm * self.hist[i][j]/total_weights

	def plot(self,outpdf,title,xlabel,ylabel,zlabel,
		clim_min=None,clim_max=None,smooth=False,sigma=1.0,
		max_ra=None,max_dec=None,max_circle_radius=None,
		target_ra=None,target_dec=None,
		flag_show_maxpoint=False):

		if smooth:
			target_hist = gaussian_filter(self.hist,sigma)			
		else:
			target_hist = self.hist
		max_i, max_j = np.unravel_index(target_hist.argmax(),target_hist.shape)
		max_rate = np.max(target_hist)
		max_ra = 0.5*(self.xedges[max_i]+self.xedges[max_i+1])
		max_dec = 0.5*(self.yedges[max_j]+self.yedges[max_j+1])	
		print(title,max_i,max_j,max_ra,max_dec,max_rate)
	
		if smooth:
			H = self.hist.T 
		else:
			H = np.ma.masked_where(self.hist==0.0,self.hist).T			
		fig = plt.figure(figsize=(8,7),tight_layout=True)
		ax = fig.add_subplot(111,title=title)
		ax.set_xlabel(xlabel)		
		ax.set_ylabel(ylabel)
		if smooth:
			H=gaussian_filter(H,sigma)
		img = plt.imshow(H, 
			aspect='equal',
			#cmap=plt.cm.PuRd, 
			cmap=plt.cm.Reds, 
			interpolation='nearest', 
			origin='lower',
			extent=[self.xedges[0],self.xedges[-1],
			self.yedges[0],self.yedges[-1]])
		plt.grid(color='#979A9A', linestyle='--', linewidth=1)
		#img = plt.imshow(H, 
		#	interpolation='nearest', 
		#	origin='lower',
		#	extent=[self.xedges[0],self.xedges[-1],
		#	self.yedges[0],self.yedges[-1]])
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.08)			
		fig.colorbar(img, cax=cax, label=zlabel)
		if clim_min != None and clim_max != None:
			plt.clim(clim_min,clim_max)	
		if target_ra != None and target_dec != None:
			ax.scatter(target_ra,target_dec,s=200,marker="*",color='k')
		if flag_show_maxpoint:
			if max_ra != None and max_dec != None:
				ax.scatter(max_ra,max_dec,s=50,marker="o",color='k')			
				circle = plt.Circle((max_ra,max_dec),max_circle_radius,fill=False,color='k')
				ax.add_patch(circle)
		fig.savefig(outpdf)

		return max_i,max_j,max_ra,max_dec,max_rate

	@property
	def data(self):
		return self.hist, self.xedges, self.yedges		

def fill_psf(hist2d,x_pointing,y_pointing,weights=1.0):
	#print(x_pointing,y_pointing)
	tmp_hist2d = Hist2D(
		xnbins=hist2d.xnbins, 
		xlow=hist2d.xlow, 
		xhigh=hist2d.xhigh,
		ynbins=hist2d.ynbins,
		ylow=hist2d.ylow,
		yhigh=hist2d.yhigh)

	for i in range(len(hist2d.hist)):
		for j in range(len(hist2d.hist[i])):
			x = hist2d.xbins[i]
			y = hist2d.ybins[j]
			xdiff = x - x_pointing
			ydiff = y - y_pointing
			offaxis_angle = np.sqrt((xdiff*60.0)**2 + (ydiff*60.0)**2)
			area = f_vig(offaxis_angle)
			#print(i,j,x,y,xdiff,ydiff,offaxis_angle,area)
			tmp_hist2d.fill(x,y,weights=area)
	
	#print(tmp_hist2d.get_total_weights())
	tmp_hist2d.normalized(norm=weights)
	#print(tmp_hist2d.get_total_weights())	
	#tmp_hist2d.plot(
	#	outpdf="test.pdf",
	#	title="tmp",
	#	xlabel="RA J2000 (deg)",ylabel="DEC J2000 (deg)", 
	#	zlabel='Exposure (sec)',flag_show_maxpoint=False)
	for i in range(len(hist2d.hist)):
		for j in range(len(hist2d.hist[i])):	
			hist2d.hist[i][j] += tmp_hist2d.hist[i][j]
	return hist2d 

def plot_raster_scan(mkffile,yamlfile):

	param = yaml.load(open(yamlfile),Loader=yaml.SafeLoader)
	print(param)

	result = {}

	basename = os.path.splitext(os.path.basename(mkffile))[0]
	outdir = '%s_scan_psf' % basename

	hdu = fits.open(mkffile)
	nitime = hdu['PREFILTER'].data['TIME']-hdu['PREFILTER'].data['TIME'][0]
	ra = hdu['PREFILTER'].data['RA']
	dec = hdu['PREFILTER'].data['DEC']
	rate = hdu['PREFILTER'].data[param['rate_column']]

	cmd = 'rm -rf %s; mkdir -p %s;' % (outdir,outdir)
	print(cmd);os.system(cmd)

	scan_xnbins = round(60.0*(param['scan_xhigh']-param['scan_xlow'])/param['scan_xbinsize'])
	scan_ynbins = round(60.0*(param['scan_yhigh']-param['scan_ylow'])/param['scan_ybinsize'])

	hist2d_expmap = Hist2D(
		xnbins=scan_xnbins, 
		xlow=param['scan_xlow'], 
		xhigh=param['scan_xhigh'],
		ynbins=scan_ynbins,
		ylow=param['scan_ylow'],
		yhigh=param['scan_yhigh'])
	hist2d_cntmap = Hist2D(
		xnbins=scan_xnbins,
		xlow=param['scan_xlow'],
		xhigh=param['scan_xhigh'],
		ynbins=scan_ynbins,
		ylow=param['scan_ylow'],
		yhigh=param['scan_yhigh'])

	for i in range(len(nitime)):
	#for i in range(10,15):
		print("%d/%d,%.1f" % (i,len(nitime),rate[i]))
		hist2d_expmap = fill_psf(hist2d_expmap,x_pointing=ra[i],y_pointing=dec[i])
		hist2d_cntmap = fill_psf(hist2d_cntmap,x_pointing=ra[i],y_pointing=dec[i],weights=rate[i])

	hist2d_expmap.plot(
		outpdf="%s/%s_expmap.pdf" % (outdir,basename),
		title="Exposure map (%s)" % (os.path.basename(mkffile)),
		xlabel="RA J2000 (deg)",ylabel="DEC J2000 (deg)", 
		zlabel='Exposure (sec)',flag_show_maxpoint=False)

	hist2d_cntmap.plot(
		outpdf="%s/%s_cntmap.pdf" % (outdir,basename),
		title="Raw count map (%s,%s)" % (os.path.basename(mkffile),param['rate_column']),
		xlabel="RA J2000 (deg)",ylabel="DEC J2000 (deg)", 
		zlabel='Counts',flag_show_maxpoint=False)

	hist2d_ratemap = Hist2D(
		xnbins=scan_xnbins,
		xlow=param['scan_xlow'],
		xhigh=param['scan_xhigh'],
		ynbins=scan_ynbins,
		ylow=param['scan_ylow'],
		yhigh=param['scan_yhigh'])
	for i in range(len(hist2d_expmap.hist)):
		for j in range(len(hist2d_expmap.hist[i])):
			if hist2d_expmap.hist[i][j] > 0.0 and hist2d_cntmap.hist[i][j] > 0.0:
				hist2d_ratemap.hist[i][j] = float(hist2d_cntmap.hist[i][j]) / float(hist2d_expmap.hist[i][j])		
				#hist2d_ratemap.hist[i][j] -= rate_offset			
			else:
				hist2d_ratemap.hist[i][j] = 0.0	

	flag = np.logical_and(rate>param['offset_th_min'],rate<param['offset_th_max'])
	rate_offset = np.mean(rate[flag])
	result['rate_offset'] = rate_offset

	max_i,max_j,max_ra,max_dec,max_rate = hist2d_ratemap.plot(
		outpdf="%s/%s_ratemap.pdf" % (outdir,basename),
		title="Count rate map  (%s,%s)" % (os.path.basename(mkffile),param['rate_column']),
		clim_min=rate_offset,clim_max=np.max(hist2d_ratemap.hist),
		max_circle_radius=param['nicer_fov_diam_arcmin']/60.0,
		target_ra=param['target_ra'],target_dec=param['target_dec'],
		xlabel="RA J2000 (deg)",ylabel="DEC J2000 (deg)", zlabel='Rate (counts/sec)',
		flag_show_maxpoint=True)
	result['max_i']	= max_i 
	result['max_j']	= max_j
	result['max_ra'] = max_ra
	result['max_dec'] = max_dec
	result['max_rate'] = max_rate

	max_i_sm,max_j_sm,max_ra_sm,max_dec_sm,max_rate_sm = hist2d_ratemap.plot(
		outpdf="%s/%s_cntmap_smooth.pdf" % (outdir,basename),
		title="Count rate map after smoothing  (%s,%s)" % (os.path.basename(mkffile),param['rate_column']),
		clim_min=rate_offset,clim_max=np.max(hist2d_ratemap.hist),
		max_circle_radius=param['nicer_fov_diam_arcmin']/60.0,		
		smooth=True,sigma=param['gaussian_sigma'],
		target_ra=param['target_ra'],target_dec=param['target_dec'],		
		xlabel="RA J2000 (deg)",ylabel="DEC J2000 (deg)", zlabel='Rate (counts/sec)',
		flag_show_maxpoint=True)
	result['max_i_smooth']	= max_i_sm
	result['max_j_smooth']	= max_j_sm
	result['max_ra_smooth'] = max_ra_sm
	result['max_dec_smooth'] = max_dec_sm
	result['max_rate_smooth'] = max_rate_sm

def get_parser():
	"""
	Creates a new argument parser.
	"""
	parser = argparse.ArgumentParser('nirasterscan.py',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		description="""
		This script run a analysis pipeline for the NICER raster scan. 
		"""
		)
	version = '%(prog)s ' + __version__
	parser.add_argument('--version', '-v', action='version', version=version,
		help='show version of this command')
	parser.add_argument('--mkffile', '-i', type=str, default="nicer_target_segment_table.csv", 
		help='mkffile')	
	parser.add_argument('--yamlfile', '-p', type=str, default="nicer_target_segment_table.csv", 
		help='yamlfile')		
	return parser

def main(args=None):
	parser = get_parser()
	args = parser.parse_args(args)
	plot_raster_scan(args.mkffile,args.yamlfile)

if __name__=="__main__":
	main()