"""Deals with collecting and curating data files."""

import sys,os
import numpy as np
import requests
import requests.exceptions
import zipfile
import logging
import shutil
import urllib.parse

import fiona

import workflow.conf
from workflow.conf import rcParams as rc

class Names:
    """File system meta data for downloading a file."""
    def __init__(self, name, url, base_folder, folder_template, file_template,
                 download_template=None, zip=True):
        self.name = name
        self._url = url
        self.base_folder = base_folder
        self.folder_template = folder_template
        self.file_template = file_template

        if download_template is None:
            self.download_template = folder_template
        else:
            self.download_template = download_template
        self.zip = zip

    def data_dir(self):
        return os.path.join(rc['data dir'], self.base_folder)
    
    def zip_dir(self):
        return os.path.join(self.data_dir(), 'zips')

    def folder_name(self, *args):
        if self.folder_template is not None:
            args = self.format_args(*args)
            return os.path.join(self.data_dir(), self.folder_template.format(*args))
        else:
            return self.data_dir()

    def download_base(self, *args):
        args = self.format_args(*args)
        fname = self.download_template.format(*args)
        if self.zip:
            fname += '.zip'
        return fname

    def download(self, *args):
        return os.path.join(self.zip_dir(), self.download_base(*args))

    def url(self, *args):
        return urllib.parse.urljoin(self._url, self.download_base(*args))

    def file_name_base(self, *args):
        args = self.format_args(*args)
        return self.file_template.format(*args)

    def file_name(self, *args):
        return os.path.join(self.folder_name(*args), self.file_name_base(*args))
    
class HucNames(Names):
    """A FileSystem class based on HUC digits."""
    def __init__(self, digits, *args, **kwargs):
        super(HucNames,self).__init__(*args, **kwargs)
        self.digits = digits

    def format_args(self, *args):
        assert(len(args) > 0)
        huc = workflow.conf.huc_str(args[0])
        if len(args) > 1:
            size = args[1]
        else:
            size = len(args[0])
        if len(huc) < self.digits:
            raise ValueError("HUCs are organized by %d-digit identifiers, so cannot retrieve a single %d-digit HUC file."%(self.digits,len(huc)))
        return huc[0:self.digits], size


class LatLonNames(Names):
    """A FileSystem class based upon lat/lon"""
    def format_args(self, lat, lon):
        if type(lat) is int:
            lat = 'n%i'%lat
        if type(lon) is int:
            lon = 'w%i'%lon
        return [lat,lon]

class FileManager:
    """A class that actually manages the files."""
    def __init__(self, names):
        self.names = names

    def download(self, *args, force=False):
        filename = self.names.file_name(*args)
        logging.debug("Attempting to download '%s'"%filename)
        try:
            if not os.path.exists(filename) or force:
                url = self.names.url(*args)
                downloadfile = self.names.download(*args)
                _download(url, downloadfile, force)
                if self.names.zip:
                    _unzip(downloadfile, self.names.folder_name(*args))
                else:
                    _move(downloadfile, self.names.folder_name(*args))
        except Exception as err:
            logging.info(str(err))
        else:
            logging.info('success')
            return self.names.file_name(*args)

    def file_name(self, *args):
        fname = self.names.file_name(*args)
        if os.path.exists(fname):
            logging.debug("   found file '%s'"%fname)
            return fname
        logging.debug("   not downloaded or found'%s'"%fname)
        return None


class NHDFileManager(FileManager):
    def __init__(self):
        names = HucNames(name='NHD High Resolution Water Boundary Data',
                         url='https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/',
                         base_folder='hydrography',
                         folder_template='NHD_H_{0}_HU8_Shape',
                         file_template='Shape',
                         digits=8)
        super(NHDFileManager,self).__init__(names)
    
    def load_huc(self, huc, digits=7):
        """Reads a HUC file in NHD Shapefile format"""
        hucstr = workflow.conf.huc_str(huc)
        container = self.names.file_name(huc)
        filename = os.path.join(container, 'WBDHU%d.shp'%len(hucstr))
        logging.debug("Opening '%s' for HUC '%s'"%(filename, hucstr))
        with fiona.open(filename, 'r') as fid:
            matching = [h for h in fid if h['properties']['HUC%i'%len(hucstr)] == hucstr]
            profile = fid.profile
        if len(matching) is not 1:
            raise RuntimeError("Invalid collection of HUC?")
        return profile, _normalize(matching, digits)[0]

    def load_hucs_in(self, huc, size, digits=7):
        """Reads a HUC file in NHD Shapefile format to get all sub-HUCs of a given size"""
        hucstr = workflow.conf.huc_str(huc)
        container = self.names.file_name(huc)
        filename = os.path.join(container, 'WBDHU%d.shp'%size)
        logging.debug("Opening '%s' for HUC '%s' subhucs of size %d"%(filename, hucstr, size))
        with fiona.open(filename, 'r') as fid:
            matching = [h for h in fid if h['properties']['HUC%i'%size].startswith(hucstr)]
            profile = fid.profile
        return profile, _normalize(matching, digits)

    def load_hydro(self, huc):
        """Returns the path to hydrography in this huc."""
        hucstr = workflow.conf.huc_str(huc)
        container = self.names.file_name(huc)
        filename = os.path.join(container, 'NHDFlowline.shp')
        logging.debug("Opening '%s' for streams in HUC '%s'"%(filename, hucstr))
        with fiona.open(filename, 'r') as fid:
            profile = fid.profile
            shps = [s for s in fid]
        return profile, _normalize(shps)


class NHDPlusFileManager(FileManager):
    def __init__(self):
        names = HucNames(name='NHD Plus High Resoluton Hydrography',
                         url='https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlus/HU4/HighResolution/GDB/',
                         base_folder='hydrography',
                         folder_template='NHDPlus_H_{0}_GDB',
                         file_template='NHDPlus_H_{0}_GDB.gdb',
                         digits=4)
        super(NHDPlusFileManager,self).__init__(names)
        
    def load_huc(self, huc, digits=7):
        """Reads a HUC file in NHD Shapefile format"""
        hucstr = workflow.conf.huc_str(huc)
        container = self.names.file_name(huc)
        layer = 'WBDHU%d'%len(hucstr)
        logging.debug("Opening '%s' layer '%s' for HUC '%s'"%(filename, layer, hucstr))
        with fiona.open(filename, mode='r', layer=layer) as fid:
            matching = [h for h in fid if h['properties']['HUC%i'%len(hucstr)] == hucstr]
            profile = fid.profile
        if len(matching) is not 1:
            raise RuntimeError("Invalid collection of HUC?")
        return profile, _normalize(matching, digits)[0]

    def load_hucs_in(self, huc, size, digits=7):
        """Reads a HUC file in NHD Shapefile format to get all sub-HUCs of a given size"""
        hucstr = workflow.conf.huc_str(huc)
        container = self.names.file_name(huc)
        layer = 'WBDHU%d'%len(size)
        logging.debug("Opening '%s' layer '%s' for HUC '%s'"%(filename, layer, hucstr))
        with fiona.open(filename, mode='r', layer=layer) as fid:
            matching = [h for h in fid if h['properties']['HUC%i'%size].startswith(hucstr)]
            profile = fid.profile
        return profile, _normalize(matching, digits)

    def load_hydro(self, huc):
        """Returns the path to hydrography in this huc."""
        hucstr = workflow.conf.huc_str(huc)
        container = self.names.file_name(huc)
        filename = os.path.join(container, 'NHDFlowlines.shp')
        logging.debug("Opening '%s' layer '%s' for streams in HUC '%s'"%(filename, layer, hucstr))
        with fiona.open(filename, mode='r', layer=layer) as fid:
            profile = fid.profile
            shps = [s for s in fid]
        return profile, _normalize(shps)
    
    
class TiledFileManager(FileManager):
    """A class that manages tiled downloads."""
    def download(self, bounds, force=False):
        logging.info('Collecting tiles in: "%r"'%(bounds))
        latlons = []
        dems = []
        for west in range(bounds[0], bounds[2]):
            for north in range(bounds[1]+1, bounds[3]+1):
                dems.append(super(TiledFileManager,self).download(north, -west, force=force))
        logging.info('  collected: "%r"'%(dems))
        return dems

class NEDFileManager(TiledFileManager):
    def __init__(self):
        names = LatLonNames(name='National Elevation Dataset',
                            url='https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/IMG/',
                            base_folder='dem',
                            folder_template=None,
                            file_template='USGS_NED_13_{0}{1}_IMG.img',
                            download_template='USGS_NED_13_{0}{1}_IMG.zip')
        super(NEDFileManager,self).__init__(names)
        

def _download(url, location, force=False):
    """Download a file from a URL to a location.  If force, clobber whatever is there."""
    if os.path.isfile(location):
        if force:
            os.remove(location)
        else:
            return True
    try:
        logging.info('Downloading: "%s" \n  to: "%s"'%(url, location))
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(location, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    except requests.exceptions.HTTPError as err:
        logging.info('  ERROR: %r'%err)
        raise err
    else:
        logging.info('  SUCCESS')
    return os.path.isfile(location)

def _unzip(filename, to_location):
    """Unzip the corresponding, assumed to exist, zipped DEM into the DEM directory."""
    logging.info('Unzipping: "%s" \n  to: "%s"'%(filename, to_location))
    
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(to_location)
    return to_location

def _move(filename, to_location):
    """Move a file to a folder."""
    shutil.move(filename, to_location)

def _normalize(list_of_shps, digits=7):
    """Rounds and standardizes shapefile formats to deal with multiple or single entry files."""
    for shp in list_of_shps:
        assert(type(shp['geometry']['coordinates']) is list)
        if len(shp['geometry']['coordinates']) is 0:
            continue
        if type(shp['geometry']['coordinates'][0]) is tuple:
            # single object
            coords = np.array(shp['geometry']['coordinates'], 'd').round(digits)
            if coords.shape[-1] is 3:
                coords = coords[:,0:2]
            assert(len(coords.shape) is 2)
            shp['geometry']['coordinates'] = coords
        else:
            # object collection
            for i,c in enumerate(shp['geometry']['coordinates']):
                coords = np.array(c,'d').round(digits)
                assert(len(coords.shape) is 2)
                if coords.shape[-1] is 3:
                    coords = coords[:,0:2]
            shp['geometry']['coordinates'][i] = coords
    return list_of_shps

    
def get_sources(args):
    sources = dict()
    if hasattr(args, 'source_huc'):
        if args.source_huc.upper() == 'NHD':
            sources['HUC'] = NHDFileManager()
        elif args.source_huc.upper() == 'NHDPLUS':
            sources['HUC'] = NHDPlusFileManager()
        else:
            raise ValueError("Unknown HUC source '%s'"%args.source_huc)

    if hasattr(args, 'source_dem'):
        if args.source_dem.upper() == 'NED':
            sources['DEM'] = NEDFileManager()
        else:
            raise ValueError("Unknown DEM source '%s'"%args.source_dem)
    return sources


