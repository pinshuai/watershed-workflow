"""Basic manager for interacting with raster files.
"""

import attr
import rasterio
import workflow
import numpy as np
import logging

@attr.s
class FileManagerRaster:
    """A simple class for reading rasters.

    Parameter
    ---------
    filename : str
      Path to the raster file.
    """
    _filename = attr.ib(type=str)
    
    def get_raster(self, band=1):
        """Gets a raster from the file.

        Parameter
        ---------
        band : int,optional
          Default is 1, the first band (1-indexed).
        """
        with rasterio.open(self._filename, 'r') as fid:
            profile = fid.profile
            raster = fid.read(band)
        return profile, raster

    def get_clipped_raster(self, shape, crs, band=1, nodata = np.nan):
        """Gets a raster from the file.

        Parameter
        ---------
        band : int,optional
          Default is 1, the first band (1-indexed).
        """
        # with rasterio.open(self._filename, 'r') as fid:
        #     profile = fid.profile
        #     raster = fid.read(band)
        datasets = [rasterio.open(self._filename)]
        profile = datasets[0].profile
        # logging.info(f"src CRS: {profile['crs']}")

        if type(shape) is dict:
            shape = workflow.utils.shply(shape)
        
        # warp to source file crs
        shply = workflow.warp.shply(shape, crs, profile['crs'])

        # get the bounds and download
        bounds = shply.bounds
        feather_bounds = list(bounds[:])
        feather_bounds[0] = feather_bounds[0] - .01
        feather_bounds[1] = feather_bounds[1] - .01
        feather_bounds[2] = feather_bounds[2] + .01
        feather_bounds[3] = feather_bounds[3] + .01
        # logging.info(f"feather bounds: {feather_bounds}")

        dest, output_transform = rasterio.merge.merge(datasets, bounds=feather_bounds, nodata=nodata)
        # dest = np.where(dest < -1.e-10, np.nan, dest)

        # set the profile
        profile['transform'] = output_transform
        profile['height'] = dest.shape[1]
        profile['width'] = dest.shape[2]
        profile['count'] = dest.shape[0]
        profile['nodata'] = nodata

        return profile, dest[0]        


        # return profile, raster