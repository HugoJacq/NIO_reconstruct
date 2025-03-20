"""
Tool module for "reconstruct_inertial.py"
"""
from math import factorial
import scipy as sp
import numpy as np
from scipy.fftpack import fft
import warnings
import xarray as xr
from xgcm import Grid as xGrid
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from joblib import Parallel
import tqdm
from dask.callbacks import Callback


def PSD(time_vect, signal_vect):
    """This function automates the computation of the Power Spectral Density of a signal.
    """
    # Same as amplitude spectrum START ===================
    total_time = time_vect[-1] - time_vect[0]
    dt = time_vect[1] - time_vect[0]
    freq_resolution = 1.0 / total_time
    freq_nyquist = 1. / dt
    
    # samples
    N = len(time_vect)
    
    # build frequency
    frequency = np.arange(0, freq_nyquist, freq_resolution, dtype=float)
    frequency = frequency[: int(N / 2) - 1]  # limit to the first half
    
    raw_fft = np.fft.fft(signal_vect, norm="backward")
    raw_fft /= N
    
    # Takes half, but double the content, excepted the first component
    amplitude_spectrum = 2*np.absolute(raw_fft)[:int(N/2) - 1] 
    amplitude_spectrum[0] /= 2
    # Same as amplitude spectrum END===================
    
    power_spectral_density = 1/2 * np.absolute(
        amplitude_spectrum
        *np.conjugate(amplitude_spectrum)
    )/freq_resolution 
    
    power_spectral_density[0] *= 2
    
    return frequency, power_spectral_density

def detrended_PSD(time_vect,signal_vect,order=1):
	"""
	This is a function that computes the PSD with before a detrending of signal_vect
	By default the detrending is linear (removing ax+b)
	"""
	poly = np.polynomial.Polynomial.fit(time_vect,signal_vect,order)

	detrended = signal_vect - poly(time_vect)
	return PSD(time_vect,detrended)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
	The Savitzky-Golay filter removes high frequency noise from data.
	It has the advantage of preserving the original shape and
	features of the signal better than other types of filtering
	approaches, such as moving averages techniques.
	 Parameters
	----------
	y : array_like, shape (N,)
		the values of the time history of the signal.
	window_size : int
		the length of the window. Must be an odd integer number.
	order : int
		the order of the polynomial used in the filtering.
		Must be less then `window_size` - 1.
	deriv: int
		the order of the derivative to compute (default = 0 means only smoothing)
	Returns
	-------
	ys : ndarray, shape (N)
		the smoothed signal (or it's n-th derivative).
	Notes
	-----
	The Savitzky-Golay is a type of low-pass filter, particularly
	suited for smoothing noisy data. The main idea behind this
	approach is to make for each point a least-square fit with a
	polynomial of high order over a odd-sized window centered at
	the point.
	Examples
	--------
	t = np.linspace(-4, 4, 500)
	y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
	ysg = savitzky_golay(y, window_size=31, order=4)
	import matplotlib.pyplot as plt
	plt.plot(t, y, label='Noisy signal')
	plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
	plt.plot(t, ysg, 'r', label='Filtered signal')
	plt.legend()
	plt.show()
	References
	----------
	.. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
	Data by Simplified Least Squares Procedures. Analytical
	Chemistry, 1964, 36 (8), pp 1627-1639.
	[2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
	W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
	Cambridge University Press ISBN-13: 9780521880688

	Hugo Jacquet 19 november 2024:
	- changed np.mat to np.asmatrix for numpy 2.0 compatibility
	"""
	try:
		window_size = np.abs(int(window_size))
		order = np.abs(int(order))	
	except ValueError:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.asmatrix([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] ) # not cyclic condition
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')

def score_RMSE(U,Ut):
    """
    RMSE between U and true Ut
    
    INPUT:
		- U: 1D array, shape of Ut
		- Ut: 1D array
    OUTPUT:
		- scalar, RMSE
    """
    nt = len(U)
    return np.sqrt( np.sum( (U-Ut)**2 )/ nt )

def score_RMSE_2D(U,Ut):
    """
    RMSE between U and true Ut
    
    INPUT:
		- U: 2D array, shape of Ut
		- Ut: 2D array (time,y,x)
    OUTPUT:
		- scalar, RMSE
    """
    nt = len(U)
    timeRMSE = np.sqrt( np.sum( (U-Ut)**2, axis=0 )/ nt, axis=0 )
    return np.mean(timeRMSE)

def nearest(array,value):
	"""
	Array is 1D,
	value is 0D
	"""
	return np.argmin(np.abs(array-value))	

def find_indices(points,lon,lat,tree=None):
    if tree is None:
        lon,lat = lon.T,lat.T
        lonlat = np.column_stack((lon.ravel(),lat.ravel()))
        tree = sp.spatial.cKDTree(lonlat)
    dist,idx = tree.query(points,k=1)
    ind = np.column_stack(np.unravel_index(idx,lon.shape))
    return [(i,j) for i,j in ind]

def find_indices_ll(L_points, lon, lat, N_CPU):
	"""
	Parallel version of 'find_indices'
	
	returns a list of indices (tuples)
	"""
	# for ik in range(len(L_points)):
	# 	LON = L_points[ik]
	# 	indx,indy = find_indices([LON,LAT],oldlon,oldlat)[0]
	list_results = Parallel(n_jobs=N_CPU)(delayed(find_indices)(L_points[ik], lon, lat) for ik in range(len(L_points)))
	return list_results

def find_indices_gridded_latlon(LON, LON_bounds, LAT, LAT_bounds):
	"""
	From a gridded product, find the indx,indy corresponding to 'LON_bounds' and 'LAT_bounds'

	INPUT:
		- LON		: Longitude 2D grid (y,x)
		- LON_bounds: Longitude bounds (list of 2, °E)
		- LAT		: Latitude 2D grid (y,x)
		- LAT_bounds: Latitude bounds (list of 2, °N)
	OUTPUT:
		- indxmin, indxmax, indymin, indymax : tuple of 4 indices

	Note:
		- finding a square in a latlon grid is not possible. Here the square represented by
		the indices do not represent a square !
		- if the model is on C grid, I advise to use the rho-point lat/lon grid.,
			but use the indices found also for other grid.
	"""
	indxmin, indxmax = LON.shape[1],0
	indymin, indymax = LON.shape[0],0
	for pointLAT in LAT_bounds:
		for pointLON in LON_bounds:
			L = find_indices([pointLON,pointLAT],LON,LAT)
			indx,indy = L[0]
			if indx < indxmin: indxmin = indx
			if indx > indxmax: indxmax = indx
			if indy < indymin: indymin = indy
			if indy > indymax: indymax = indy
	return (indxmin,indxmax,indymin,indymax)
    

def psd1d(hh,dx=1.,tap=0.05, detrend=True):
	"""
 	1D PSD of real or complex array 'hh'	
 	"""

	hh=hh-np.mean(hh)
	nx=np.shape(hh)[0]

	if detrend:
		hh=sp.signal.detrend(hh)

	if tap>0:  
		ntaper = int(tap * nx + 0.5)
		taper = np.zeros(nx)+1.
		taper[:ntaper]=np.cos(np.arange(ntaper)/(ntaper-1.)*np.pi/2+3*np.pi/2)
		taper[-ntaper:] = np.cos(-np.arange(-ntaper+1,1)/(ntaper-1.)*np.pi/2+3*np.pi/2)
		hh=hh*taper

	ss=fft(hh)
	ff=np.arange(1,nx/2-1)/(nx*dx)

	PSD=2*dx/(nx)*np.abs(ss[1:int(nx/2-1)])**2

	return ff, PSD

def rotary_spectra(dt,Ua,Va,U,V):
	"""
	Computes the clockwise and counter clockwise spectra
	INPUT:
		- Ua,Va : estimated currents (1D)
		- U,V : True currents (1D)
	OUTPUT:
		- frequency array
		- tuple of 4 (MPr2,Mpr1,MPe2,MPe1):
			- counter clockwise and clockwise spectra of truth
			- counter clockwise and clockwise spectra of estimate
	"""
	nt = U.shape[0]
	nf = 200
	print(nt,nf)
	count = 0
	ensit = np.arange(0,nt-nf,int(nf/2))
	for it in ensit:
		ff,Pr1 = psd1d(U[it:it+nf]+1j*V[it:it+nf],dx=dt, detrend=True, tap=0.2)
		ff,Pr2 = psd1d(U[it:it+nf]-1j*V[it:it+nf],dx=dt, detrend=True, tap=0.2)
	if count==0:
		MPr1 = +Pr1
		MPr2 = +Pr2
	else:
		MPr1 += Pr1
		MPr2 += Pr2 


	ff, Pe1 = psd1d((Ua[it:it+nf]-U[it:it+nf]) +1j* (Va[it:it+nf]-V[it:it+nf]) ,dx=dt, detrend=True, tap=0.2)
	ff, Pe2 = psd1d((Ua[it:it+nf]-U[it:it+nf]) -1j* (Va[it:it+nf]-V[it:it+nf]) ,dx=dt, detrend=True, tap=0.2)
	if count==0:
		MPe1 = +Pe1
		MPe2 = +Pe2
	else:
		MPe1 += Pe1
		MPe2 += Pe2 


	count += 1
	MPr1 /= count
	MPr2 /= count
	MPe1 /= count
	MPe2 /= count
	return ff, MPr2, MPr1, MPe2, MPe1
 
def print_memory_array(ds,namevar):
    """
    print in terminal the amount of memory used by a dask array
    """ 
    total_memory = ds[namevar].nbytes
    print("Total memory used by the DataArray "+namevar+f": {total_memory / 1024**2:.2f} MB")
    
def print_memory_chunk(ds,namevar):
    """
    print in the terminal the memory used by a chunk of 'dataArray'.
    It is assumed that all chunk have a similar size.
    """
    nelements = np.prod(ds[namevar].data.chunksize)
    chunk_memory = nelements * ds[namevar].data.itemsize
    print("Memory used by each chunk of the DataArray "+namevar+f" {chunk_memory / 1024**2:.2f} MB")
    
    
def open_croco_sfx_file(file_list, lazy=True, chunks=None):
	"""
	This function opens an output of Croco simulations, renames some variables and then return the dataset and its xgcm grid
	"""
	
	if chunks==None:
		size_chunk=-1
	else:
		size_chunk=chunks
	
 
	# opening files
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		# warning are about chunksize, because not all chunk have the same size.
		if lazy==True:
			ds = xr.open_mfdataset(file_list,chunks=size_chunk)
		else:
			ds = xr.open_mfdataset(file_list)
			# ,chunks={'time_counter': -1,
			#     'x_rho': size_chunk,
			#     'y_rho': size_chunk,
			#     'y_u': size_chunk, 
			#     'x_u': size_chunk,
			#     'y_v': size_chunk, 
			#     'x_v': size_chunk,})

	# removing used variables
	ds = ds.drop_vars(['bvstr','bustr','ubar','vbar','hbbl','h',
	'time_instant','time_instant_bounds','time_counter_bounds'])

	# rename redundant dimensions
	_dims = (d for d in ['x_v', 'y_u', 'x_w', 'y_w'] if d in ds.dims)
	for d in _dims:
		ds = ds.rename({d: d[0]+'_rho'})

	# renaming variables
	ds = ds.rename({'zeta':'SSH',
			'sustr':'oceTAUX',
			'svstr':'oceTAUY',
			'shflx':'Heat_flx_net',
			'swflx':'frsh_water_net',
			'swrad':'SW_rad',
			'hbl':'MLD',
			'u':'U',
			'v':'V',
			'time_counter':'time'})
	if 'nav_lat_rho' in ds.variables:
		ds = ds.rename({'nav_lat_rho':'lat_rho',
			'nav_lon_rho':'lon_rho',
			'nav_lat_u':'lat_u',
			'nav_lat_v':'lat_v',
			'nav_lon_u':'lon_u',
			'nav_lon_v':'lon_v'})

	# building xgcm grid
	coords={'x':{'center':'x_rho',  'inner':'x_u'}, 
		'y':{'center':'y_rho', 'inner':'y_v'}}    
	grid = xGrid(ds, 
			coords=coords,
			boundary='extend')
	return ds, grid

def open_croco_sfx_file_at_point_loc(file_list, point_loc, interp_var='currents', lazy=True, chunks=None):
	"""
	This function opens an output of Croco simulations, renames some variables and then return the dataset and its xgcm grid
	"""
	
	if chunks==None:
		size_chunk=-1
	else:
		size_chunk=chunks
	
 
	# opening files
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		# warning are about chunksize, because not all chunk have the same size.
		if lazy==True:
			#ds = xr.open_mfdataset(file_list,chunks=size_chunk)
			ds =  xr.open_mfdataset(file_list ,chunks={'time_counter': 1,
													'x_rho': size_chunk,
													'y_rho': size_chunk,
													'y_u': size_chunk, 
													'x_u': size_chunk,
													'y_v': size_chunk, 
													'x_v': size_chunk,},parallel=True)
		else:
			ds = xr.open_mfdataset(file_list)
			

	# removing used variables
	ds = ds.drop_vars(['bvstr','bustr','ubar','vbar','hbbl','h',
	'time_instant','time_instant_bounds','time_counter_bounds'])

	# rename redundant dimensions
	_dims = (d for d in ['x_v', 'y_u', 'x_w', 'y_w'] if d in ds.dims)
	for d in _dims:
		ds = ds.rename({d: d[0]+'_rho'})

	# renaming variables
	ds = ds.rename({'zeta':'SSH',
			'sustr':'oceTAUX',
			'svstr':'oceTAUY',
			'shflx':'Heat_flx_net',
			'swflx':'frsh_water_net',
			'swrad':'SW_rad',
			'hbl':'MLD',
			'u':'U',
			'v':'V',
			'time_counter':'time'})
	if 'nav_lat_rho' in ds.variables:
		ds = ds.rename({'nav_lat_rho':'lat_rho',
			'nav_lon_rho':'lon_rho',
			'nav_lat_u':'lat_u',
			'nav_lat_v':'lat_v',
			'nav_lon_u':'lon_u',
			'nav_lon_v':'lon_v'})

	# reduce dataset size around point_loc
	# edge = 0.1 # °
	# LON_bounds = [point_loc[0]-edge,point_loc[0]+edge]
	# LAT_bounds = [point_loc[1]-edge,point_loc[1]+edge]
	# indices = find_indices_gridded_latlon(ds.lon_rho.values, LON_bounds, ds.lat_rho.values, LAT_bounds)
	# indxmin, indxmax, indymin, indymax = indices
	edge = 2 # number of points
	indx,indy = find_indices(point_loc,ds.lon_rho.values, ds.lat_rho.values)[0]
	indxmin, indxmax,= indx-edge,indx+edge 
	indymin, indymax = indy-edge,indy+edge 
 
	# smaller domain
	ds = ds.isel(x_rho=slice(indxmin,indxmax),
				y_rho=slice(indymin,indymax),
				x_u=slice(indxmin,indxmax),
				y_v=slice(indymin,indymax) )
	# load variables that we need to interp
	ds.U.load()
	ds.V.load()
	ds.oceTAUX.load()
	ds.oceTAUY.load()
	# building a new xgcm grid on the reduced dataset
	coords={'x':{'center':'x_rho',  'right':'x_u'}, 
			'y':{'center':'y_rho', 'right':'y_v'}}    
	grid = xGrid(ds, 
		coords=coords,
		boundary='extend')


	# interp at mass point
	if interp_var=='flux':
		L_u = ['oceTAUX']
		L_v = ['oceTAUY']
	elif interp_var=='currents':
		L_u = ['U','oceTAUX']
		L_v = ['V','oceTAUY']
	elif interp_var=='all':
		L_u = ['U','oceTAUX']
		L_v = ['V','oceTAUY']
	else:
		print("open_croco_sfx_file_at_point_loc: I dont know what variable you want to interp, i'll do all")
		L_u = ['U','oceTAUX']
		L_v = ['V','oceTAUY']
  
	for var in L_u:
		attrs = ds[var].attrs
		ds[var] = grid.interp(ds[var], 'x')
		ds[var].attrs = attrs
	for var in L_v:
		attrs = ds[var].attrs
		ds[var] = grid.interp(ds[var], 'y')
		ds[var].attrs = attrs
	# remove non-rho coordinates
	ds = ds.drop_vars({'lat_u','lat_v','lon_u','lon_v'})
	# all onvariables are now on rho points:  
	ds = ds.rename({'lon_rho':'lon','lat_rho':'lat'})

	# reducing dataset at the 'point_loc'
	indx,indy = find_indices(point_loc,ds.lon.values,ds.lat.values)[0]
	ds = ds.isel(x_rho=indx,y_rho=indy)
 
	return ds

class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar

	from https://github.com/joblib/joblib/issues/972#issuecomment-1623366702
 	
    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    desc: str, default: None
        the description used in the tqdm progressbar.

    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    Removed parameters:
    -------------------
    verbose: will be ignored


    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]

    """

    def __init__(
        self,
        *,
        total_tasks: int | None = None,
        desc: str | None = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. "
                "Use disable_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: tqdm.tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)
        