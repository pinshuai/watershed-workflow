"""Download DayMet data in a box and convert it to Amanzi-style HDF5.

DayMet is downloaded in box mode based on lat-lon bounds, then converted to
hdf5 files that ATS knows how to read.
"""

import requests
import datetime
import logging
import h5py, netCDF4
import sys, os
import numpy as np
import time
import workflow
import rasterio
# import scipy
from scipy.signal import savgol_filter

VALID_YEARS = (1980,2019) # changed upper limit
VALID_VARIABLES = ['tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe', 'dayl']
URL = "http://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/1328/{year}/daymet_v3_{variable}_{year}_na.nc4"


def boundsStr(bounds):
    """Returns a lat-lon id for use in filenames"""
    fmt_str = "_".join(['{:,4f}',]*4)
    return fmt_str.format(*bounds).replace('.','p')

def getFilename(tmp, bounds, year, var):
    """Returns a tmp filename for a single download file."""
    filename = 'daymet_{var}_{year}_{north}x{west}_{south}x{east}.nc'.format(year=year, var=var,
                                                                             north=bounds[3], east=bounds[2],
                                                                             west=bounds[0], south=bounds[1])
    return os.path.join(tmp, filename)

def downloadFile(tmp, bounds, year, var, force=False):
    """Gets file for a single year and single variable.

    Parameters
    ----------
    var : list of strings or None
      Name the variable, see table in the class documentation.
    year : int
      A year in the valid range (currently 1980-2018)
    bounds : [xmin, ymin, xmax, ymax]
      Collect a file that covers this shape or bounds in lat/lon.

    Returns
    -------
    filename : str
      Path to the data file.
    """
    
    if year > VALID_YEARS[1] or year < VALID_YEARS[0]:
        raise ValueError("DayMet data is available from {} to {} (does not include {})".format(VALID_YEARS[0], VALID_YEARS[1], year))
    if var not in VALID_VARIABLES:
        raise ValueError("DayMet data supports variables: {} (not {})".format(', '.join(VALID_VARIABLES), var))

    # get the target filename
    filename = getFilename(tmp, bounds, year, var)

    if not os.path.exists(filename) or force:
        url_dict = {'year':str(year),
                    'variable':var}
        url = URL.format(**url_dict)
        logging.info("  Downloading: {}".format(url))
        logging.info("      to file: {}".format(filename))

        request_params = [('var', 'lat'),
                          ('var', 'lon'),
                          ('var', var),
                          ('west', str(bounds[0])),
                          ('south', str(bounds[1])),
                          ('east', str(bounds[2])),
                          ('north', str(bounds[3])),
                          ('horizStride', '1'),
                          ('time_start', '{}-01-01T12:00:00Z'.format(year)),
                          ('time_end', '{}-12-31T12:00:00Z'.format(year)),
                          ('timeStride', '1'),
                          ('accept', 'netcdf')
                          ]

        r = requests.get(url,params=request_params)
        r.raise_for_status()

        with open(filename, 'wb') as fid:
            fid.write(r.content)
    else:
        logging.info("  Using existing: {}".format(filename))

    return filename
        
class Date:
    """Struct to store day of year and year."""
    def __init__(self, doy, year):
        self.doy = doy
        self.year = year

    def __repr__(self):
        return '{}-{}'.format(self.doy, self.year)
    
def stringToDate(s):
    """convert string to Date format (s.doy, s.year)"""
    if len(s) == 4:
        return Date(1, int(s))

    doy_year = s.split('-')
    if len(doy_year) != 2 or len(doy_year[1]) != 4:
        raise RuntimeError('Invalid date format: {}, should be DOY-YEAR'.format(s))

    return Date(int(doy_year[0]), int(doy_year[1]))

def numDays(start, end):
    """Time difference -- assumes inclusive end date."""
    return 365 * (end.year + 1 - start.year) - (start.doy-1) - (365-end.doy)

def loadFile(fname, var):
    with netCDF4.Dataset(fname, 'r') as nc:
        x = nc.variables['x'][:] * 1000. # km to m; raw netCDF file has km unit 
        y = nc.variables['y'][:] * 1000. # km to m
        time = nc.variables['time'][:]
        assert(len(time) == 365)
        val = nc.variables[var][:]
    return x,y,val

def initData(d, vars, num_days, nx, ny):
    for v in vars:
        # d[v] has shape (nband, nrow, ncol)
        d[v] = np.zeros((num_days, ny, nx),'d')

def collectDaymet(tmpdir, bounds, start, end, vars=None, force=False):
    """Calls the DayMet Rest API to get data and save raw data.
    Parameters:
    vars: list or None
        list of strings that are in VALID_VARIABLES. Default is use all available variables.
    
    """
    T0 = time.time()
    if vars == None:
        vars = VALID_VARIABLES
        logging.info(f"downloading variables: {VALID_VARIABLES}")

    dat = dict()
    d_inited = False

    for year in range(start.year, end.year+1):
        for var in vars:
            fname = downloadFile(tmpdir, bounds, year, var, force)
            x,y,v = loadFile(fname, var) # returned v.shape(nband, nrow, ncol)
            if not d_inited:
                initData(dat, vars, numDays(start,end), len(x), len(y))
                d_inited = True

            # stuff v in the right spot
            if year == start.year and year == end.year:
                dat[var][:,:,:] = v[start.doy-1:end.doy,:,:]
            elif year == start.year:
                dat[var][0:365-start.doy+1,:,:] = v[start.doy-1:,:,:]
            elif year == end.year:
                dat[var][-end.doy:,:,:] = v[-end.doy:,:,:]
            else:
                my_start = 365 * (year - start.year) - start.doy + 1
                dat[var][my_start:my_start+365,:,:] = v
    logging.info(f'seconds to write: {time.time()-T0} s')
    return dat, x, y

def reproj_Daymet(x, y, raw, dst_crs, dst_nodata = None, resolution = None):
    """
    reproject daymet raw data to watershed CRS.
    """
    var_list = list(raw.keys())
    logging.debug(f"variables: {var_list}")
    logging.debug(f"raw shape in (nband, nrow, ncol): {raw[var_list[0]].shape}")
    
    if raw[var_list[0]].ndim == 3:
        nband = raw[var_list[0]].shape[0]   
    else:
        nband = 1 
    
    daymet_crs = workflow.crs.daymet_crs()
    logging.debug(f'daymet crs: {daymet_crs}')
    
    # make sure tranform function is consistent with the unit used in CRS
    unit = daymet_crs.to_dict()['units']
    if unit == 'km':
        dx = dy = 1.0 # km
        transform = (x.min()/1000 - dx/2, dx, 0.0, y.max()/1000 + dy/2, 0.0, -dy) # accepted format(xmin, dx, 0, ymax, 0, -dy)
        affine = rasterio.transform.from_origin(x.min() - dx/2, y.max() + dy/2, dx, dy)
    elif unit == 'm':
        dx = dy = 1000.0 # m
        transform = (x.min() - dx/2, dx, 0.0, y.max() + dy/2, 0.0, -dy)
        affine = rasterio.transform.from_origin(x.min() - dx/2, y.max() + dy/2, dx, dy)
    else: 
        raise RuntimeError(f'Daymet CRS unit: {unit} is not recognized! Supported units are m or km.')
    logging.debug(f'transform: {transform}')
    logging.debug(f'Affine: {affine}') 
    
    daymet_profile = {
        'driver': 'GTiff', 
        'dtype': 'float32', 
        'nodata': -9999.0, 
        'width': len(x), 
        'height': len(y), 
        'count': nband, 
        'crs':daymet_crs,
        'transform':affine,
        'tiled': False, 
        'interleave': 'pixel'
    }

    logging.info(f'daymet profile: {daymet_profile}') 
    
    logging.info(f'reprojecting to new crs: {dst_crs}') 
    new_dat = {}
    for var in var_list:
        # if input raw array has the shape of (bands, cols, rows), need to transpose to (bands, rows, cols) for warp.raster()! 
        # idat = raw[var].swapaxes(1,2)
        idat = raw[var]
        dst_profile, dst_raster = workflow.warp.raster(src_profile=daymet_profile, src_array=idat, 
                                    dst_crs=dst_crs, dst_nodata = dst_nodata, resolution = resolution)

        # dst_array has shape of (bands, rows, cols) need to flip back to (bands, cols, rows)
        # dst_raster = dst_raster.swapaxes(1,2)
        new_dat[var] = dst_raster    
    
    logging.info(f"new profile: {dst_profile}")
    new_extent = rasterio.transform.array_bounds(dst_profile['height'], dst_profile['width'], dst_profile['transform']) # (x0, y0, x1, y1)

    logging.info(f"new extent[xmin, ymin, xmax, ymax]: {new_extent}")
    
    new_x, new_y = xy_from_profile(dst_profile)
    
    return new_x, new_y, new_extent, new_dat, daymet_profile

def smoothRaw(raw, smooth_filter = True, nyears = None):

    logging.info("averaging daymet by taking the average for each day across the actual years.")
    var_list = list(raw.keys())
    if nyears == None:
        nyears = raw[var_list[0]].shape[0]//365 
    # reshape dat
    smooth_dat = dict()
    for ivar in var_list:
        idat = raw[ivar]
        if nyears*365 != idat.shape[0]:
            idat = idat[0:nyears*365, :, :]
        idat = idat.reshape(nyears, 365, idat.shape[1], idat.shape[2])
        # # average over years so that the dat has shape of (365, nx, ny)
        inew = idat.mean(axis = 0)
        
        # apply smooth filter
        if smooth_filter:
            window = 61
            poly_order = 2
            logging.info(f"smoothing {ivar} using savgol filter, window = {window} d, poly order = {poly_order}")
            inew = savgol_filter(inew, window, poly_order, axis = 0, mode = 'wrap')  
        # repeat this for nyears
        smooth_dat[ivar] = np.tile(inew, (nyears, 1, 1))   

    return smooth_dat

def daymetToATS(dat, smooth = False, smooth_filter = True, nyears = None):
    """Accepts a numpy named array of DayMet data and returns a dictionary ATS data."""
    logging.info(f"input dat shape: {dat[list(dat.keys())[0]].shape}")
    dout = dict()
    logging.info('Converting to ATS met input')
    
    if smooth:
        dat = smoothRaw(dat, smooth_filter = smooth_filter, nyears = nyears)
        logging.info(f"shape of smoothed dat is {dat[list(dat.keys())[0]].shape}")
    mean_air_temp_c = (dat['tmin'] + dat['tmax'])/2.0
    precip_ms = dat['prcp'] / 1.e3 / 86400. # mm/day --> m/s
    
    # Sat vap. press o/water Dingman D-7 (Bolton, 1980)
    sat_vp_Pa = 611.2 * np.exp(17.67 * mean_air_temp_c / (mean_air_temp_c + 243.5))

    time = np.arange(0, dat[list(dat.keys())[0]].shape[0], 1)*86400.

    dout['air temperature [K]'] = 273.15 + mean_air_temp_c # K
    dout['incoming shortwave radiation [W m^-2]'] = dat['srad'] # Wm2
    dout['relative humidity [-]'] = np.minimum(1.0, dat['vp']/sat_vp_Pa) # -
    dout['precipitation rain [m s^-1]'] = np.where(mean_air_temp_c >= 0, precip_ms, 0)
    dout['precipitation snow [m SWE s^-1]'] = np.where(mean_air_temp_c < 0, precip_ms, 0)
    dout['wind speed [m s^-1]'] = 4. * np.ones_like(dout['relative humidity [-]'])

    logging.debug(f"output dout shape: {dout['incoming shortwave radiation [W m^-2]'].shape}")
    return time, dout

def getAttrs(bounds, start, end):
    # set the wind speed height, which is made up
    attrs = dict()
    attrs['wind speed reference height [m]'] = 2.0
    attrs['DayMet latitude min [deg]'] = bounds[1]
    attrs['DayMet longitude min [deg]'] = bounds[0]
    attrs['DayMet latitude max [deg]'] = bounds[3]
    attrs['DayMet longitude max [deg]'] = bounds[2]
    attrs['DayMet start date'] = str(start)
    attrs['DayMet end date'] = str(end)
    return attrs    

def writeATS(time, dat, x, y, attrs, filename):
    """Accepts a dictionary of ATS data and writes it to HDF5 file."""

    logging.info('Writing ATS file: {}'.format(filename))

    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    with h5py.File(filename, 'w') as fid:
        fid.create_dataset('time [s]', data=time)
        assert(len(x.shape) == 1)
        assert(len(y.shape) == 1)
       
        # ATS requires increasing order for y
        rev_y = y[::-1]
        fid.create_dataset('row coordinate [m]', data=rev_y) # should it be y?
        fid.create_dataset('col coordinate [m]', data=x)

        for key in dat.keys():
            # dat has shape (nband, nrow, ncol) 
            assert(dat[key].shape[0] == time.shape[0])
            assert(dat[key].shape[1] == y.shape[0])
            assert(dat[key].shape[2] == x.shape[0])
            # dat[key] = dat[key].swapaxes(1,2) # reshape to (nband, nrow, ncol)
            grp = fid.create_group(key)
            for i in range(len(time)):
                idat = dat[key][i,:,:]
                # flip rows to match the order of y, so it starts with (x0,y0) in the upper left
                rev_idat = np.flip(idat, axis=0)
                
                grp.create_dataset(str(i), data=rev_idat)

        for key, val in attrs.items():
            fid.attrs[key] = val

    return

def validBounds(bounds):
    return True

def getArgumentParser():
    """Gets an argparse parser for use in main"""
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('bounds', type=float, nargs=4,
                        help='Longitude, latitude bounding box: xmin, ymin, xmax, ymax')
   
    parser.add_argument('-s', '--start', type=stringToDate, default=Date(1,VALID_YEARS[0]),
                        help='Start date, in the form DOY-YEAR (e.g. 274-2018 for Oct 1, 2018)')
    parser.add_argument('-e', '--end', type=stringToDate, default=Date(365,VALID_YEARS[1]),
                        help='End date, in the form DOY-YEAR (e.g. 274-2018 for Oct 1, 2018)')

    parser.add_argument('--download-only', action='store_true',
                        help='Only download raw data.')
    parser.add_argument('--force-download', action='store_true',
                        help='Re-download all files, even if they already exist.')
    parser.add_argument('-o', '--outfile', type=str,
                        help='Output HDF5 filename.')
    return parser

def feather_bounds(bounds, buffer = 0.01):
    """
    slightly enlarge the bound so that it covers the entire watershed.
    """
    feather_bounds = list(bounds[:])
    feather_bounds[0] = feather_bounds[0] - buffer
    feather_bounds[1] = feather_bounds[1] - buffer
    feather_bounds[2] = feather_bounds[2] + buffer
    feather_bounds[3] = feather_bounds[3] + buffer
    logging.info(f"added {buffer} deg to get new bounds: {feather_bounds}")
    
    return feather_bounds

def xy_from_profile(profile):
    """
    get x, y coord from raster profile.
    """
    xmin = profile['transform'][2]
    ymax = profile['transform'][5]
    dx = profile['transform'][0]
    dy = -profile['transform'][4]
    nx = profile['width']
    ny = profile['height']

    x = xmin + dx/2 + np.arange(nx) * dx
    y = ymax - dy/2 - np.arange(ny) * dy
    
    return x, y

if __name__ == '__main__':
    parser = getArgumentParser()
    args = parser.parse_args()

    validBounds(args.bounds)
    
    if args.outfile is None:
        args.outfile = './daymet_{}_{}_{}.h5'.format(args.start, args.end, boundsStr(args.bounds))

    tmpdir = args.outfile+".tmp"
    os.makedirs(tmpdir, exist_ok=True)

    raw, x, y = collectDaymet(tmpdir, args.bounds, args.start, args.end, args.force_download)

    if not args.download_only:
        time, dout = daymetToATS(raw)
        writeATS(time, dout, x, y, getAttrs(args.bounds, args.start, args.end), args.outfile)

    sys.exit(0)
    
